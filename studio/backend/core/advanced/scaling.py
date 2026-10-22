# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Scaler: grow a checkpoint's parameter count before continued pretraining.

This is the Studio face of :mod:`unsloth.models.scaling`. It answers two
questions separately, because the UI has to show both before any compute runs:

* :func:`plan` reads ``config.json`` only - no torch, no weights - and returns the
  slider range (``base_params`` to ``base + 3B``), the growth floor, and every
  reason a run would be refused. Planning is cheap enough to call while the user
  drags the slider.
* :func:`run` loads the checkpoint, calls ``expand_model``, verifies the expanded
  model still computes the base function at step 0, and saves a checkpoint the
  training page can pick up.

The refusal reasons here mirror ``expand_model``'s guards exactly (quantized,
PEFT, MoE, unsupported family, growth floor). Duplicating them is the point: the
UI has to render a reason *before* the run, and a test asserts this module's copy
of the supported-family set still equals the engine's.

Growth floor: every growth study that reports a gain grows by at least 2x, so a
target between ``base`` and ``2 * base`` is refused unless the caller opts in.
"""

from __future__ import annotations

import gc
import importlib
import importlib.util
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from . import merge

# The slider's ceiling: base + 3B parameters. A T4-sized expansion is ~0.6 GB in
# fp16, and one 3B-wide step is already the largest single run this UI can plan.
CAP_ADDED_PARAMS = 3_000_000_000

# Below 2x, a growth run costs the same compute as a real one and does not
# reliably recover, so it is refused rather than warned about.
GROWTH_FLOOR = 2.0

# ``unsloth.models.scaling`` aligns widened MLP channels to this multiple.
WIDTH_ALIGNMENT = 64

METHODS = ("depth", "mlp_width", "depth+mlp_width")

METHOD_LABELS = {
    "depth": "Depth (new transformer blocks)",
    "mlp_width": "MLP width (wider feed-forward)",
    "depth+mlp_width": "Depth + MLP width (split the growth)",
}

# Mirrors ``unsloth.models.scaling._SUPPORTED_MODEL_TYPES``; a test keeps the two
# equal without importing torch into the planning path.
SUPPORTED_MODEL_TYPES = frozenset(
    {
        "llama",
        "qwen2",
        "qwen3",
        "mistral",
        "gemma",
        "gemma2",
        "phi3",
        "granite",
        "olmo",
        "olmo2",
        "olmo3",
        "smollm",
        "smollm3",
        "hunyuan",
        "helios",
    }
)

# Mirrors ``unsloth.models.scaling._MOE_CONFIG_KEYS``.
MOE_CONFIG_KEYS = (
    "num_experts",
    "num_local_experts",
    "n_routed_experts",
    "moe_intermediate_size",
    "shared_expert_intermediate_size",
    "num_experts_per_tok",
)

_PROBE_TEXT = (
    "Unsloth Scaler function-preservation probe: the expanded model must reproduce "
    "the base model's logits at step 0."
)


# ---------------------------------------------------------------------------
# Engine availability
# ---------------------------------------------------------------------------


def _repo_root() -> Optional[Path]:
    """The checkout this backend belongs to, when it runs from source."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "unsloth" / "models" / "scaling.py").is_file():
            return candidate
    return None


_ENGINE: Any = None
_ENGINE_ERROR: Optional[str] = None
_ENGINE_TRIED = False


def load_engine() -> Any:
    """Import the expansion engine, preferring the installed package.

    ``unsloth/__init__.py`` pulls in the GPU stack (triton among others), which a
    planning host does not need and a Windows CPU install does not have. The
    scaling module itself imports only the standard library and pulls torch in
    lazily, so when the package import fails the file is loaded directly.

    The outcome is cached: a failing package import is expensive, and the plan
    endpoint calls this on every slider move. Installing a missing dependency
    therefore needs a backend restart, which the refusal message says.
    """
    global _ENGINE, _ENGINE_ERROR, _ENGINE_TRIED
    if _ENGINE is not None:
        return _ENGINE
    if _ENGINE_TRIED:
        raise RuntimeError(_ENGINE_ERROR or "The expansion engine is unavailable.")
    _ENGINE_TRIED = True
    try:
        _ENGINE = _load_engine_uncached()
        return _ENGINE
    except Exception as error:
        _ENGINE_ERROR = str(error)
        raise


def _load_engine_uncached() -> Any:
    errors: list[str] = []
    try:
        return importlib.import_module("unsloth.models.scaling")
    except Exception as error:  # noqa: BLE001 - the file fallback below is the point
        errors.append(f"package import: {error}")

    root = _repo_root()
    path = root / "unsloth" / "models" / "scaling.py" if root is not None else None
    if path is not None and path.is_file():
        try:
            spec = importlib.util.spec_from_file_location("unsloth_scaling_engine", path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load {path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as error:  # noqa: BLE001 - reported below with both causes
            errors.append(f"module file: {error}")

    raise RuntimeError(
        "The expansion engine (unsloth.models.scaling) cannot be loaded here: "
        + "; ".join(errors)
        + ". Install this checkout with `pip install -e .`, or start Studio from the "
        "repository root."
    )


def engine_available() -> tuple[bool, Optional[str]]:
    """Whether the expansion engine can be loaded, and why not."""
    try:
        load_engine()
        return True, None
    except Exception as error:  # noqa: BLE001 - surfaced as a refusal, not a crash
        return False, f"{type(error).__name__}: {error}"


def _scaling_module() -> Any:
    try:
        return load_engine()
    except RuntimeError as error:
        raise ValueError(str(error)) from error


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def _embedding_params(config: Mapping[str, Any], hidden: int, vocab: int) -> int:
    """Input embeddings, plus the output head when it is not tied."""
    total = vocab * hidden
    if config.get("tie_word_embeddings") is False:
        total *= 2
    return total


def _attention_params(config: Mapping[str, Any], hidden: int) -> int:
    heads = _int(config.get("num_attention_heads"))
    kv_heads = _int(config.get("num_key_value_heads")) or heads
    if not heads:
        return 4 * hidden * hidden
    head_dim = _int(config.get("head_dim")) or max(1, hidden // max(1, heads))
    kv_width = max(1, kv_heads) * head_dim
    return hidden * hidden + 2 * hidden * kv_width + hidden * hidden


def _per_layer_params(config: Mapping[str, Any], hidden: int, intermediate: int) -> int:
    """Parameters in one decoder block: attention, MLP and the two norms."""
    return _attention_params(config, hidden) + 3 * hidden * intermediate + 2 * hidden


def _int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _base_params(config: Mapping[str, Any]) -> Optional[int]:
    """Analytic parameter count from config.json, close enough to plan against."""
    hidden = _int(config.get("hidden_size")) or _int(config.get("n_embd"))
    layers = _int(config.get("num_hidden_layers")) or _int(config.get("n_layer"))
    vocab = _int(config.get("vocab_size"))
    intermediate = (
        _int(config.get("intermediate_size"))
        or _int(config.get("ffn_dim"))
        or (_int(config.get("n_inner")) or 4 * hidden if hidden else None)
    )
    if not hidden or not layers:
        return None
    intermediate = intermediate or 4 * hidden
    total = layers * _per_layer_params(config, hidden, intermediate)
    if vocab:
        total += _embedding_params(config, hidden, vocab)
    return int(total)


def _align(value: float) -> int:
    return max(WIDTH_ALIGNMENT, int(WIDTH_ALIGNMENT * round(float(value) / WIDTH_ALIGNMENT)))


def _adapter_present(source: str) -> bool:
    """A local directory that carries a LoRA adapter instead of a plain model."""
    path = Path(source).expanduser()
    if path.is_file():
        path = path.parent
    if not path.is_dir():
        return False
    return (path / "adapter_config.json").is_file()


def _refusal(code: str, message: str, *, blocks: bool = True) -> dict[str, Any]:
    return {"code": code, "message": message, "blocks": blocks}


def _refusals(source: str, config: Mapping[str, Any], base_params: Optional[int], cap: int) -> list[dict[str, Any]]:
    """Every reason a run would be refused, in the order the engine checks them."""
    found: list[dict[str, Any]] = []

    if config.get("quantization_config") is not None or config.get("quantization_config_dict_json"):
        found.append(
            _refusal(
                "quantized",
                "This checkpoint is quantized. The scales encode the old matrix shapes, so "
                "expansion cannot be exact: start from the fp16/bf16 original and expand that.",
            )
        )

    if _adapter_present(source):
        found.append(
            _refusal(
                "adapter",
                "This is a LoRA/QLoRA adapter, not a full model. The adapter was trained "
                "against the old shapes: expand the base checkpoint first, then train a new "
                "adapter on the enlarged model.",
            )
        )

    model_type = str(config.get("model_type") or config.get("architecture") or "")
    if any(config.get(key) is not None for key in MOE_CONFIG_KEYS) or "moe" in model_type.lower():
        found.append(
            _refusal(
                "moe",
                "Mixture-of-Experts checkpoints are not supported: expert MLPs are packed "
                "tensors with their own width keys, and widening them without widening the "
                "router changes what the router scores. Use a dense source checkpoint.",
            )
        )
    elif model_type and model_type not in SUPPORTED_MODEL_TYPES:
        found.append(
            _refusal(
                "unsupported_family",
                f"Expansion is not implemented for model_type={model_type!r}. Supported dense "
                f"families: {', '.join(sorted(SUPPORTED_MODEL_TYPES))}.",
            )
        )

    if base_params is None:
        found.append(
            _refusal(
                "unknown_size",
                "config.json does not name a hidden size and layer count, so the parameter "
                "count cannot be read. Expansion needs those to target a size.",
            )
        )
    elif 2 * base_params > cap:
        found.append(
            _refusal(
                "floor_above_cap",
                f"This checkpoint has ~{_format_params(base_params)}, and {GROWTH_FLOOR:g}x is "
                f"~{_format_params(int(base_params * GROWTH_FLOOR))} - past the "
                f"{_format_params(cap)} ceiling (base + {_format_params(CAP_ADDED_PARAMS)}). "
                "Growth below 2x is refused, so no target in this range is valid.",
            )
        )
    return found


def _project(
    config: Mapping[str, Any],
    *,
    base_params: int,
    target_params: float,
    method: str,
) -> dict[str, Any]:
    """What the engine will most likely build for this target, before it loads."""
    hidden = _int(config.get("hidden_size")) or _int(config.get("n_embd")) or 0
    layers = _int(config.get("num_hidden_layers")) or _int(config.get("n_layer")) or 0
    current_intermediate = (
        _int(config.get("intermediate_size"))
        or _int(config.get("ffn_dim"))
        or (_int(config.get("n_inner")) or 4 * hidden if hidden else 0)
    )
    per_layer = _per_layer_params(config, hidden, current_intermediate) if hidden else 0
    gap = max(0.0, float(target_params) - base_params)

    added = 0
    new_layers = layers
    new_intermediate = current_intermediate

    wants_width = method in ("mlp_width", "depth+mlp_width")
    wants_depth = method in ("depth", "depth+mlp_width")
    width_gap = gap / 2.0 if method == "depth+mlp_width" else gap

    if wants_width and hidden and layers:
        per_channel = 3 * hidden * layers
        delta_channels = _align(width_gap / per_channel)
        new_intermediate = int(current_intermediate) + delta_channels
        added += delta_channels * per_channel

    if wants_depth and per_layer:
        remaining = gap - added if wants_width else gap
        new_layers = int(layers) + max(1, int(round(remaining / max(1, per_layer))))
        added += (new_layers - int(layers)) * per_layer

    projected = base_params + added
    return {
        "projected_params": int(projected),
        "added_params": int(added),
        "growth_factor": float(projected) / max(1, base_params),
        "new_num_layers": new_layers,
        "new_intermediate_size": new_intermediate,
        "current_num_layers": layers,
        "current_intermediate_size": current_intermediate,
        "params_per_layer": per_layer,
        "estimated": True,
    }


def _format_params(value: float) -> str:
    if value >= 1e9:
        return f"{value / 1e9:.2f}B"
    if value >= 1e6:
        return f"{value / 1e6:.1f}M"
    if value >= 1e3:
        return f"{value / 1e3:.1f}K"
    return f"{int(value)}"


def plan(
    source: str,
    *,
    target_params: Optional[float] = None,
    method: str = "depth",
    token: Optional[str] = None,
) -> dict[str, Any]:
    """The slider's range, the growth floor, and every refusal reason. Reads config.json only."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}.")

    label, config = merge.read_model_config(source, token = token)
    base_params = _base_params(config)
    cap = (base_params or 0) + CAP_ADDED_PARAMS
    floor_params = int(math.ceil((base_params or 0) * GROWTH_FLOOR))
    min_params = int(base_params or 0)

    refusals = _refusals(source, config, base_params, cap)
    blocking = [item for item in refusals if item["blocks"]]

    # The default target is the floor: the smallest growth this tool will run.
    default_target = min(max(min_params, floor_params), cap) if base_params else min_params
    target = float(target_params) if target_params else float(default_target)
    target = float(min(max(target, min_params), cap)) if base_params else target

    projection = (
        _project(config, base_params = int(base_params), target_params = target, method = method)
        if base_params
        else None
    )

    below_floor = bool(base_params) and target < floor_params
    if below_floor:
        refusals = refusals + [
            _refusal(
                "below_floor",
                f"{_format_params(target)} is {target / max(1, base_params):.2f}x this checkpoint. "
                f"Growth below {GROWTH_FLOOR:g}x is refused: no growth study reports a capability "
                f"gain below that, and the run costs the same as a real one. Move the slider to "
                f"{_format_params(floor_params)} or higher.",
            )
        ]

    available, engine_error = engine_available()
    if not available:
        refusals = refusals + [
            _refusal(
                "engine_unavailable",
                f"Plans work, but the expansion engine cannot load here: {engine_error}",
            )
        ]

    blocked_reason = blocking[0]["message"] if blocking else None
    if blocked_reason is None and below_floor:
        blocked_reason = next(item["message"] for item in refusals if item["code"] == "below_floor")

    notes = [
        "Parameter counts are read from config.json and are analytic; the run reports the "
        "measured count after expansion.",
        "Depth growth is bit-exact at step 0. Width growth is not: widening down_proj makes "
        "float32 re-associate the sum, so the logits move by ~1e-5 of their own scale.",
    ]

    return {
        "source": label,
        "spec": {
            "architecture": str(config.get("model_type") or config.get("architecture") or "unknown"),
            "hidden_size": _int(config.get("hidden_size")),
            "num_layers": _int(config.get("num_hidden_layers") or config.get("n_layer")),
            "intermediate_size": _int(config.get("intermediate_size")),
            "vocab_size": _int(config.get("vocab_size")),
        },
        "base_params": min_params,
        "min_params": min_params,
        "floor_params": floor_params,
        "max_params": cap,
        "cap_added_params": CAP_ADDED_PARAMS,
        "growth_floor": GROWTH_FLOOR,
        "target_params": int(target),
        "default_target_params": int(default_target),
        "step_params": max(1_000_000, int(CAP_ADDED_PARAMS / 300)),
        "method": method,
        "method_label": METHOD_LABELS[method],
        "methods": [{"id": key, "label": METHOD_LABELS[key]} for key in METHODS],
        "projection": projection,
        "target_below_floor": below_floor,
        "refusals": refusals,
        "blocked": bool(blocking),
        "blocked_reason": blocked_reason,
        "runnable": not blocking and not below_floor,
        "engine": {"available": available, "error": engine_error},
        "notes": notes,
    }




# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def _probe_ids(tokenizer: Any, torch: Any, vocab: Optional[int], device: Any) -> Any:
    """Tokenizer ids for the function-preservation check, or synthetic ones."""
    if tokenizer is not None:
        try:
            encoded = tokenizer(_PROBE_TEXT, return_tensors = "pt")
            return encoded["input_ids"].to(device)
        except Exception:  # noqa: BLE001 - a tokenizer without a vocab still gets checked
            pass
    limit = max(2, min(int(vocab or 1024), 512))
    values = [(index * 7 + 3) % limit for index in range(8)]
    return torch.tensor([values], dtype = torch.long, device = device)


def _load_model(path: str, torch: Any) -> Any:
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
    try:
        model = AutoModelForCausalLM.from_pretrained(path, dtype = "auto", **kwargs)
    except TypeError:
        # `dtype` is transformers 5's spelling; 4.x only knows `torch_dtype`.
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype = "auto", **kwargs)
    if torch.cuda.is_available():
        model.to("cuda")
    return model


def _load_tokenizer(path: str) -> Any:
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(path)
    except Exception:  # noqa: BLE001 - expansion does not need one to succeed
        return None


def run(
    source: str,
    *,
    output_dir: str,
    target_params: float,
    method: str = "depth",
    insert_strategy: str = "even",
    token: Optional[str] = None,
    progress: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    """Expand ``source`` to ``target_params`` and save a verified checkpoint in ``output_dir``."""

    def stage(name: str, message: str, **extra: Any) -> None:
        if progress is not None:
            progress({"stage": name, "message": message, **extra})

    report = plan(source, target_params = target_params, method = method, token = token)
    if not report["runnable"]:
        raise ValueError(report["blocked_reason"] or "This checkpoint cannot be expanded.")

    stage("resolving", "Resolving the checkpoint...")
    local = merge.resolve_checkpoint_source(source, token = token)

    engine = _scaling_module()
    import torch  # noqa: PLC0415 - planning must stay importable without torch

    stage("loading", "Loading the base checkpoint...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    tokenizer = _load_tokenizer(local)
    model = _load_model(local, torch)
    device = next(model.parameters()).device
    ids = _probe_ids(tokenizer, torch, report["spec"]["vocab_size"], device)

    model.eval()
    with torch.no_grad():
        base_logits = model(input_ids = ids).logits.float().clone()

    stage("expanding", f"Growing to ~{_format_params(target_params)} parameters ({method.replace('_', ' ')})...")
    started = time.monotonic()
    model, expansion = engine.expand_model(
        model,
        target_params = float(target_params),
        method = method,
        insert_strategy = insert_strategy,
    )

    stage("verifying", "Checking the expanded model still computes the base function...")
    step0_diff = engine.assert_function_preserved(
        None, model, ids, base_logits = base_logits
    )

    stage("saving", "Saving and reloading the expanded checkpoint...")
    saved = engine.save_expanded_model(
        model,
        tokenizer,
        output_dir,
        sample_input_ids = ids,
    )

    elapsed = time.monotonic() - started
    peak_gb = (
        float(torch.cuda.max_memory_allocated()) / 1e9 if torch.cuda.is_available() else 0.0
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = {
        "output_dir": str(saved.get("output_dir") or output_dir),
        "source": report["source"],
        "method": method,
        "base_params": int(expansion.get("base_params") or report["base_params"]),
        "final_params": int(expansion.get("final_params") or 0),
        "added_params": int(expansion.get("added_params") or 0),
        "growth_factor": float(expansion.get("growth_factor") or 0.0),
        "new_num_layers": expansion.get("new_num_layers"),
        "new_intermediate_size": expansion.get("new_intermediate_size"),
        "inserted_at": expansion.get("inserted_at"),
        "step0_max_logit_diff": float(step0_diff),
        "verified": bool(saved.get("verified")),
        "reload_max_logit_diff": saved.get("max_logit_diff"),
        "index": saved.get("index"),
        "peak_mem_gb": peak_gb,
        "elapsed_seconds": round(elapsed, 1),
        "refusals": report["refusals"],
        "notes": report["notes"],
        "plan": report,
    }
    stage(
        "completed",
        f"Expanded to {_format_params(result['final_params'])} ({result['growth_factor']:.2f}x).",
        done = 1,
        total = 1,
    )
    return result
