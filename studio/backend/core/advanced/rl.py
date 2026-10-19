# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Policy-alignment math (DPO and GRPO).

The UI exposes one Policy Alignment mode with two objectives:

* **DPO** -- learns from ``(prompt, chosen, rejected)`` triples. The policy's
  log-probability of each completion is compared against a frozen reference
  model, and the pair is pushed apart with a beta-scaled log-sigmoid loss.
* **GRPO** -- samples a group of completions per prompt, scores them with a
  reward function, standardizes the rewards *within the group* to get
  advantages, and applies a PPO-clipped policy-gradient step. No value/critic
  network is needed; the group mean replaces the baseline.

Everything in this module is dependency-free numpy so the objectives can be
unit-tested on a CPU-only host. Torch log-prob extraction lives in
:mod:`core.advanced.rl_trainer`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

RL_METHODS = ("dpo", "grpo")
REWARD_PRESETS = ("exact_match", "contains", "length", "format", "reasoning_tags")


@dataclass(frozen=True)
class RlConfig:
    method: str = "dpo"
    beta: float = 0.1
    label_smoothing: float = 0.0
    learning_rate: float = 5e-6
    num_generations: int = 4
    max_completion_length: int = 512
    temperature: float = 0.9
    top_p: float = 0.95
    clip_ratio: float = 0.2
    kl_coefficient: float = 0.0
    reward_preset: str = "exact_match"
    reward_weights: dict[str, float] | None = None
    reference_model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _numpy():
    import numpy as np

    return np


def _logsigmoid(x):
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    # log(sigmoid(x)) = -softplus(-x), stable for large |x|.
    return -np.logaddexp(0.0, -x)


def dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    *,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> dict[str, Any]:
    """Direct Preference Optimization loss (Rafailov et al., 2023).

    Returns the mean loss plus the reward margin and pairwise accuracy, which the
    run report plots next to the loss.
    """
    np = _numpy()
    if beta <= 0:
        raise ValueError("DPO beta must be positive.")
    if not 0.0 <= label_smoothing < 0.5:
        raise ValueError("DPO label_smoothing must be in [0, 0.5).")

    chosen = np.asarray(policy_chosen_logps, dtype=np.float64)
    rejected = np.asarray(policy_rejected_logps, dtype=np.float64)
    ref_chosen = np.asarray(reference_chosen_logps, dtype=np.float64)
    ref_rejected = np.asarray(reference_rejected_logps, dtype=np.float64)
    if not (chosen.shape == rejected.shape == ref_chosen.shape == ref_rejected.shape):
        raise ValueError("DPO log-probability arrays must share one shape.")

    chosen_reward = beta * (chosen - ref_chosen)
    rejected_reward = beta * (rejected - ref_rejected)
    margin = chosen_reward - rejected_reward
    # Label smoothing turns the hard 0/1 target into {ls, 1-ls}.
    loss = -(
        (1.0 - label_smoothing) * _logsigmoid(margin)
        + label_smoothing * _logsigmoid(-margin)
    )
    accuracy = float(np.mean(chosen_reward > rejected_reward))
    return {
        "loss": float(np.mean(loss)),
        "chosen_reward": float(np.mean(chosen_reward)),
        "rejected_reward": float(np.mean(rejected_reward)),
        "reward_margin": float(np.mean(margin)),
        "accuracy": accuracy,
        "per_example_loss": loss.tolist(),
    }


def sequence_logprobs_from_logits(logits, labels, mask=None) -> Any:
    """Sum token log-probabilities of `labels` under `logits` (numpy).

    ``logits`` is ``[batch, seq, vocab]``; ``labels`` is ``[batch, seq]`` with
    ``-100`` marking ignored positions, matching HF conventions.
    """
    np = _numpy()
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("Expected logits [B, T, V] and labels [B, T].")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels disagree on batch or sequence length.")
    # Shift so position t predicts token t+1.
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    max_logits = shifted_logits.max(axis=-1, keepdims=True)
    log_z = max_logits + np.log(
        np.exp(shifted_logits - max_logits).sum(axis=-1, keepdims=True)
    )
    safe_labels = np.where(shifted_labels >= 0, shifted_labels, 0)
    gathered = np.take_along_axis(
        shifted_logits, safe_labels[..., None], axis=-1
    ).squeeze(-1)
    token_logps = gathered - log_z.squeeze(-1)
    valid = shifted_labels >= 0
    if mask is not None:
        valid = valid & np.asarray(mask, dtype=bool)[:, 1:]
    token_logps = np.where(valid, token_logps, 0.0)
    return token_logps, token_logps.sum(axis=-1)


def grpo_advantages(rewards, group_size: int | None = None, *, eps: float = 1e-4) -> Any:
    """Group-relative advantages: ``(r - mean(group)) / std(group)``.

    ``rewards`` may be flat ``[B*G]`` (with ``group_size``) or ``[B, G]``.
    """
    np = _numpy()
    array = np.asarray(rewards, dtype=np.float64)
    if array.ndim == 1:
        if group_size is None:
            group_size = array.shape[0]
        if group_size <= 0 or array.shape[0] % group_size != 0:
            raise ValueError("Flat rewards must divide evenly into groups.")
        array = array.reshape(-1, group_size)
    elif array.ndim != 2:
        raise ValueError("rewards must be [B*G] or [B, G].")
    mean = array.mean(axis=1, keepdims=True)
    std = array.std(axis=1, keepdims=True)
    return (array - mean) / np.maximum(std, eps)


def grpo_policy_loss(
    logps,
    old_logps,
    advantages,
    *,
    clip_ratio: float = 0.2,
    kl=None,
    kl_coefficient: float = 0.0,
) -> dict[str, Any]:
    """PPO-clipped GRPO surrogate with an optional KL penalty.

    ``logps`` / ``old_logps`` are per-token or per-sequence log-probabilities;
    ``advantages`` is broadcast over the last axis.
    """
    np = _numpy()
    if clip_ratio <= 0:
        raise ValueError("clip_ratio must be positive.")
    new = np.asarray(logps, dtype=np.float64)
    old = np.asarray(old_logps, dtype=np.float64)
    adv = np.asarray(advantages, dtype=np.float64)
    if new.shape != old.shape:
        raise ValueError("logps and old_logps must share a shape.")
    if adv.size != new.shape[0]:
        adv = adv.reshape(-1)
        if adv.size != new.shape[0]:
            raise ValueError("advantages must have one value per sequence.")
        adv = adv.reshape(-1, *([1] * (new.ndim - 1)))
    ratio = np.exp(new - old)
    unclipped = ratio * adv
    clipped = np.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
    surrogate = np.minimum(unclipped, clipped)
    loss = -surrogate.mean()
    result: dict[str, Any] = {
        "loss": float(loss),
        "mean_ratio": float(ratio.mean()),
        "clip_fraction": float(np.mean(np.abs(ratio - 1.0) > clip_ratio)),
    }
    if kl is not None and kl_coefficient:
        kl_array = np.asarray(kl, dtype=np.float64)
        result["kl"] = float(kl_array.mean())
        result["loss"] = float(loss + kl_coefficient * kl_array.mean())
    return result


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    return [token for token in str(text).strip().split() if token]


def reward_exact_match(completion: str, reference: str | None = None) -> float:
    if reference is None:
        return 0.0
    return 1.0 if str(completion).strip() == str(reference).strip() else 0.0


def reward_contains(completion: str, reference: str | None = None) -> float:
    if reference is None:
        return 0.0
    return 1.0 if str(reference).strip().lower() in str(completion).lower() else 0.0


def reward_length(completion: str, reference: str | None = None) -> float:
    """Smooth length score: 1.0 at 64-512 tokens, decaying outside that band."""
    count = len(_tokenize(completion))
    if 64 <= count <= 512:
        return 1.0
    if count < 64:
        return count / 64.0
    return max(0.0, 1.0 - (count - 512) / 1024.0)


def reward_format(completion: str, reference: str | None = None) -> float:
    """Reward a single ``<answer>...</answer>`` block with content inside."""
    text = str(completion)
    opening = text.count("<answer>")
    closing = text.count("</answer>")
    if opening == 1 and closing == 1 and text.index("<answer>") < text.index("</answer>"):
        inner = text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
        return 1.0 if inner else 0.25
    return 0.0


def reward_reasoning_tags(completion: str, reference: str | None = None) -> float:
    """Reward a ``<reasoning>`` block followed by an ``<answer>`` block."""
    text = str(completion)
    score = 0.0
    if text.count("<reasoning>") == 1 and text.count("</reasoning>") == 1:
        score += 0.4
    if text.count("<answer>") == 1 and text.count("</answer>") == 1:
        score += 0.4
    if score == 0.8 and text.index("<reasoning>") < text.index("<answer>"):
        score += 0.2
    return score


REWARD_FUNCTIONS: dict[str, Callable[[str, str | None], float]] = {
    "exact_match": reward_exact_match,
    "contains": reward_contains,
    "length": reward_length,
    "format": reward_format,
    "reasoning_tags": reward_reasoning_tags,
}


def score_completions(
    completions: Sequence[str],
    references: Sequence[str | None] | None = None,
    *,
    preset: str = "exact_match",
    weights: Mapping[str, float] | None = None,
) -> list[float]:
    """Score a group of completions with one or more built-in reward presets."""
    names = list(weights) if weights else [preset]
    for name in names:
        if name not in REWARD_FUNCTIONS:
            raise ValueError(
                f"Unknown reward preset {name!r}; expected one of {', '.join(REWARD_PRESETS)}."
            )
    if references is not None and len(references) != len(completions):
        raise ValueError("references and completions must have equal length.")
    scores: list[float] = []
    for index, completion in enumerate(completions):
        reference = references[index] if references is not None else None
        if weights:
            total = sum(
                float(weight) * REWARD_FUNCTIONS[name](completion, reference)
                for name, weight in weights.items()
            )
            denominator = sum(abs(float(weight)) for weight in weights.values()) or 1.0
            scores.append(total / denominator)
        else:
            scores.append(REWARD_FUNCTIONS[preset](completion, reference))
    return scores


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_rl_config(
    method: str,
    config: Mapping[str, Any] | None = None,
    *,
    dataset_columns: Sequence[str] | None = None,
    num_rows: int | None = None,
) -> dict[str, Any]:
    """Validate a Policy Alignment payload and return the normalised config.

    Raises ``ValueError`` for anything that would fail mid-run, and returns a
    ``warnings`` list for combinations that work but are usually a mistake.
    """
    method = str(method).lower()
    if method not in RL_METHODS:
        raise ValueError(f"Unsupported RL method {method!r}; expected dpo or grpo.")
    payload = dict(config or {})
    warnings: list[str] = []

    beta = float(payload.get("beta", 0.1))
    if not 0 < beta <= 1.0:
        raise ValueError("beta must be in (0, 1].")
    lr = float(payload.get("learning_rate", 5e-6))
    if not 0 < lr <= 1e-2:
        raise ValueError("Policy-alignment learning_rate must be in (0, 1e-2].")
    generations = int(payload.get("num_generations", 4))
    if not 2 <= generations <= 64:
        raise ValueError("num_generations must be between 2 and 64.")
    max_length = int(payload.get("max_completion_length", 512))
    if not 16 <= max_length <= 32768:
        raise ValueError("max_completion_length must be between 16 and 32768.")
    clip = float(payload.get("clip_ratio", 0.2))
    if not 0 < clip < 1:
        raise ValueError("clip_ratio must be in (0, 1).")
    temperature = float(payload.get("temperature", 0.9))
    if not 0.1 <= temperature <= 2.0:
        raise ValueError("temperature must be between 0.1 and 2.0.")
    preset = str(payload.get("reward_preset", "exact_match"))
    if preset not in REWARD_FUNCTIONS:
        raise ValueError(f"Unknown reward preset {preset!r}.")
    weights = payload.get("reward_weights")
    if weights is not None:
        if not isinstance(weights, Mapping):
            raise ValueError("reward_weights must be an object of preset -> weight.")
        unknown = [name for name in weights if name not in REWARD_FUNCTIONS]
        if unknown:
            raise ValueError(f"Unknown reward weights: {', '.join(unknown)}.")

    columns = set(dataset_columns or [])
    if dataset_columns is not None:
        if method == "dpo":
            missing = [
                name
                for name in ("prompt", "chosen", "rejected")
                if name not in columns
            ]
            if missing:
                raise ValueError(
                    "DPO needs columns prompt, chosen, rejected; missing: "
                    + ", ".join(missing)
                )
        else:
            if "prompt" not in columns:
                raise ValueError("GRPO needs a prompt column.")
            if "reward" not in columns and preset in ("exact_match", "contains"):
                warnings.append(
                    "GRPO is using the "
                    f"{preset!r} reward, but the dataset has no reward/reference column; "
                    "every group will score identically. Add a reference column or pick a "
                    "reference-free reward such as format or length."
                )
    if num_rows is not None:
        if num_rows < generations:
            raise ValueError(
                f"The dataset has {num_rows} rows, fewer than num_generations={generations}."
            )
        if num_rows < 8:
            warnings.append("Very small RL dataset; expect noisy rewards.")

    normalized = RlConfig(
        method=method,
        beta=beta,
        label_smoothing=float(payload.get("label_smoothing", 0.0)),
        learning_rate=lr,
        num_generations=generations,
        max_completion_length=max_length,
        temperature=temperature,
        top_p=float(payload.get("top_p", 0.95)),
        clip_ratio=clip,
        kl_coefficient=float(payload.get("kl_coefficient", 0.0)),
        reward_preset=preset,
        reward_weights=dict(weights) if weights else None,
        reference_model=payload.get("reference_model"),
    ).to_dict()
    normalized["warnings"] = warnings
    return normalized
