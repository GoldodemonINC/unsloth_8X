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

Growth floor: every growth study that *reports* a gain grows by at least 2x, but
that is evidence about where a gain was measured, not a reason to refuse a
smaller step. The floor is therefore 1x - any target above the base is allowed -
and the plan says so plainly instead of hiding the unmeasured range.
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

# Growth must add something, and that is the whole floor: below 2x the literature
# has not measured a gain, which the plan reports as a note rather than a refusal.
GROWTH_FLOOR = 1.0
MEASURED_GROWTH_FLOOR = 2.0

# ``unsloth.models.scaling`` aligns widened MLP channels to this multiple.
WIDTH_ALIGNMENT = 64

METHODS = ("depth", "mlp_width", "depth+mlp_width")

METHOD_LABELS = {
    "depth": "Depth (new transformer blocks)",
    "mlp_width": "MLP width (wider feed-forward)",
    "depth+mlp_width": "Depth + MLP width (split the growth)",
}

# Shrinking keeps at least this fraction of the decoder blocks; below it the
# embeddings dominate and a smaller base checkpoint is the better start.
SHRINK_FLOOR = 0.5

SHRINK_METHODS = ("depth", "mlp_width")

SHRINK_METHOD_LABELS = {
    "depth": "Depth (drop whole transformer blocks)",
    "mlp_width": "MLP width (narrow the feed-forward)",
}

# Mirrors ``unsloth.models.scaling.WIDTH_FLOOR``: a width shrink keeps at least
# this fraction of each MLP's channels. A test keeps the two constants equal.
WIDTH_SHRINK_FLOOR = 0.25

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

# Calibration text for ranking blocks by how little they change the residual
# stream. Generic on purpose: the ranking only needs representative activations,
# and a built-in sample keeps the run free of a dataset dependency.
_SHRINK_CALIBRATION = (
    "The quick brown fox jumps over the lazy dog. A transformer block reads the residual "
    "stream, attends to the tokens before it, and writes a correction back into that "
    "stream. Removing one block removes one correction: the model still runs, but it has "
    "lost a step of computation, and rescaling the remaining weights cannot restore what "
    "that step was doing. Continued pretraining on the data the model is meant to serve "
    "is what lets the surviving blocks take over the work."
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


def _refusals(
    source: str,
    config: Mapping[str, Any],
    base_params: Optional[int],
    *,
    operation: str = "expand",
) -> list[dict[str, Any]]:
    """Every reason a run would be refused, in the order the engine checks them."""
    found: list[dict[str, Any]] = []
    shrinking = operation == "shrink"

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
                f"{'Shrinking' if shrinking else 'Expansion'} is not implemented for "
                f"model_type={model_type!r}. Supported dense families: "
                f"{', '.join(sorted(SUPPORTED_MODEL_TYPES))}.",
            )
        )

    if base_params is None:
        found.append(
            _refusal(
                "unknown_size",
                f"config.json does not name a hidden size and layer count, so the parameter "
                f"count cannot be read. {'Shrinking' if shrinking else 'Expansion'} needs those "
                "to target a size.",
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


def _shrink_projection(
    config: Mapping[str, Any],
    *,
    base_params: int,
    target_params: float,
) -> dict[str, Any]:
    """How many blocks the run will drop to land near ``target_params``."""
    hidden = _int(config.get("hidden_size")) or _int(config.get("n_embd")) or 0
    layers = _int(config.get("num_hidden_layers")) or _int(config.get("n_layer")) or 0
    intermediate = (
        _int(config.get("intermediate_size"))
        or _int(config.get("ffn_dim"))
        or (_int(config.get("n_inner")) or 4 * hidden if hidden else 0)
    )
    per_layer = _per_layer_params(config, hidden, intermediate) if hidden else 0
    keep_floor = max(2, int(math.ceil(layers * SHRINK_FLOOR))) if layers else 0
    max_remove = max(0, layers - keep_floor)
    gap = max(0.0, float(base_params) - float(target_params))
    removed = max(1, int(round(gap / max(1, per_layer)))) if per_layer else 0
    removed = min(removed, max_remove)
    projected = base_params - removed * per_layer
    return {
        "projected_params": int(projected),
        "removed_params": int(removed * per_layer),
        "shrink_factor": float(projected) / max(1, base_params),
        "removed_layers": int(removed),
        "new_num_layers": int(layers - removed),
        "current_num_layers": int(layers),
        "params_per_layer": int(per_layer),
        "max_removable_layers": int(max_remove),
        "keep_floor_layers": int(keep_floor),
        "estimated": True,
    }


def _shrink_width_plan(
    label: str,
    source: str,
    config: Mapping[str, Any],
    base_params: Optional[int],
    *,
    target_params: Optional[float],
    token: Optional[str],
) -> dict[str, Any]:
    """A shrink plan that narrows each MLP instead of dropping blocks.

    One channel costs the same everywhere (a gate row, an up row and a down
    column, each ``hidden`` wide), so the slider's step is that cost times the
    layers, and the floor is a fraction of the channels rather than of the depth.
    """
    refusals = _refusals(source, config, base_params, operation = "shrink")
    blocking = [item for item in refusals if item["blocks"]]

    available, engine_error = engine_available()
    if not available:
        refusals = refusals + [
            _refusal(
                "engine_unavailable",
                f"Plans work, but the shrinking engine cannot load here: {engine_error}",
            )
        ]

    hidden = _int(config.get("hidden_size")) or _int(config.get("n_embd")) or 0
    layers = _int(config.get("num_hidden_layers")) or _int(config.get("n_layer")) or 0
    intermediate = (
        _int(config.get("intermediate_size"))
        or _int(config.get("ffn_dim"))
        or (_int(config.get("n_inner")) or 4 * hidden if hidden else 0)
    )
    per_channel = 3 * hidden * max(1, layers) if hidden else 0
    keep_floor = max(1, int(math.ceil(intermediate * WIDTH_SHRINK_FLOOR))) if intermediate else 0
    removable = max(0, intermediate - keep_floor)

    min_params = int(base_params or 0)
    max_params = int(base_params or 0)
    default_target = int(base_params or 0)
    target = float(target_params or 0.0)
    projection: Optional[dict[str, Any]] = None
    below_floor = False
    too_large = False

    if base_params and intermediate and removable >= 1:
        max_params = int(base_params - per_channel)
        min_params = int(base_params - removable * per_channel)
        default_removals = max(1, min(removable, int(round(0.1 * removable))))
        default_target = int(base_params - default_removals * per_channel)

        if target_params is None:
            target = float(default_target)
        target = float(max(min_params, min(max_params, target)))
        removed = int(round((base_params - target) / max(1, per_channel)))
        removed = max(1, min(removable, removed))
        projection = {
            "current_intermediate_size": intermediate,
            "removed_channels": removed,
            "new_intermediate_size": intermediate - removed,
            "params_per_channel": per_channel,
        }
    elif base_params and intermediate and removable < 1:
        refusals = refusals + [
            _refusal(
                "no_room",
                f"A {intermediate}-channel MLP cannot lose a channel and still keep "
                f"{WIDTH_SHRINK_FLOOR:.0%} of its width. Start from a smaller checkpoint.",
            )
        ]

    if base_params and intermediate and target_params is not None:
        if float(target_params) >= float(base_params):
            too_large = True
            refusals = refusals + [
                _refusal(
                    "not_smaller",
                    f"{_format_params(float(target_params))} is not smaller than this "
                    f"checkpoint's {_format_params(base_params)}. Use the growth direction for "
                    "a larger target.",
                    blocks = False,
                )
            ]
        elif removable >= 1 and float(target_params) > max_params:
            too_large = True
            refusals = refusals + [
                _refusal(
                    "not_smaller",
                    f"The smallest width shrink removes one channel (~{_format_params(per_channel)}), "
                    f"to {_format_params(max_params)}. Ask for that or lower.",
                    blocks = False,
                )
            ]
        elif removable >= 1 and float(target_params) < min_params:
            below_floor = True
            refusals = refusals + [
                _refusal(
                    "below_shrink_floor",
                    f"Width shrinking keeps at least {WIDTH_SHRINK_FLOOR:.0%} of the MLP "
                    f"channels, which is {keep_floor} of {intermediate} here - about "
                    f"{_format_params(min_params)}. Below that the FFN stops being a FFN.",
                    blocks = False,
                )
            ]

    blocked_reason = blocking[0]["message"] if blocking else None
    if blocked_reason is None:
        first_soft = next((item for item in refusals if not item["blocks"]), None)
        blocked_reason = first_soft["message"] if first_soft else None

    notes = [
        "Parameter counts are read from config.json and are analytic; the run reports the "
        "measured count after the channels are removed.",
        "Narrowing does not preserve the function: the output is a starting point for "
        "recovery training, and the run reports the drift it measured.",
        "Channels are ranked per block by the magnitude of their gate, up and down weights; "
        "the lowest-scoring channels go first, and at least "
        f"{WIDTH_SHRINK_FLOOR:.0%} of the width is kept.",
    ]

    return {
        "source": label,
        "direction": "shrink",
        "spec": {
            "architecture": str(config.get("model_type") or config.get("architecture") or "unknown"),
            "hidden_size": _int(config.get("hidden_size")),
            "num_layers": _int(config.get("num_hidden_layers") or config.get("n_layer")),
            "intermediate_size": _int(config.get("intermediate_size")),
            "vocab_size": _int(config.get("vocab_size")),
        },
        "base_params": int(base_params or 0),
        "min_params": min_params,
        "floor_params": min_params,
        "max_params": max_params,
        "cap_added_params": CAP_ADDED_PARAMS,
        "growth_floor": GROWTH_FLOOR,
        "shrink_floor": WIDTH_SHRINK_FLOOR,
        "target_params": int(target),
        "default_target_params": int(default_target),
        "min_added_params": int(min_params - (base_params or 0)),
        "max_added_params": int(max_params - (base_params or 0)),
        "added_params": int(target) - int(base_params or 0),
        "default_added_params": int(default_target) - int(base_params or 0),
        "floor_added_params": int(min_params - (base_params or 0)),
        "step_params": max(1, int(per_channel)),
        "method": "mlp_width",
        "method_label": SHRINK_METHOD_LABELS["mlp_width"],
        "methods": [{"id": key, "label": SHRINK_METHOD_LABELS[key]} for key in SHRINK_METHODS],
        "projection": projection,
        "target_below_floor": below_floor,
        "refusals": refusals,
        "blocked": bool(blocking),
        "blocked_reason": blocked_reason,
        "runnable": not blocking and not below_floor and not too_large,
        "engine": {"available": available, "error": engine_error},
        "notes": notes,
    }


def _shrink_plan(
    label: str,
    source: str,
    config: Mapping[str, Any],
    base_params: Optional[int],
    *,
    target_params: Optional[float],
    method: str,
    token: Optional[str],
) -> dict[str, Any]:
    """A plan whose target is smaller than the checkpoint: drop blocks or narrow MLPs."""
    if method == "mlp_width":
        return _shrink_width_plan(
            label, source, config, base_params, target_params = target_params, token = token
        )
    if method != "depth":
        raise ValueError(
            "Shrinking needs method 'depth' or 'mlp_width'; "
            f"got {method!r}."
        )

    refusals = _refusals(source, config, base_params, operation = "shrink")
    blocking = [item for item in refusals if item["blocks"]]

    available, engine_error = engine_available()
    if not available:
        refusals = refusals + [
            _refusal(
                "engine_unavailable",
                f"Plans work, but the shrinking engine cannot load here: {engine_error}",
            )
        ]

    hidden = _int(config.get("hidden_size")) or _int(config.get("n_embd")) or 0
    layers = _int(config.get("num_hidden_layers")) or _int(config.get("n_layer")) or 0
    intermediate = (
        _int(config.get("intermediate_size"))
        or _int(config.get("ffn_dim"))
        or (_int(config.get("n_inner")) or 4 * hidden if hidden else 0)
    )
    per_layer = _per_layer_params(config, hidden, intermediate) if hidden else 0
    keep_floor = max(2, int(math.ceil(layers * SHRINK_FLOOR))) if layers else 0
    max_remove = max(0, layers - keep_floor)

    min_params = int(base_params or 0)
    max_params = int(base_params or 0)
    default_target = int(base_params or 0)
    target = float(target_params or 0.0)
    projection: Optional[dict[str, Any]] = None
    below_floor = False
    too_large = False

    if base_params and layers:
        # Removing one block is the smallest shrink; removing every allowed block is the floor.
        max_params = int(base_params - per_layer)
        min_params = int(base_params - max_remove * per_layer)
        default_removals = max(1, min(max_remove, int(round(0.1 * layers))))
        default_target = int(base_params - default_removals * per_layer)

        if max_remove < 1:
            refusals = refusals + [
                _refusal(
                    "no_room",
                    f"A {layers}-block model cannot lose a block and still keep "
                    f"{SHRINK_FLOOR:.0%} of its depth. Start from a smaller checkpoint.",
                )
            ]
        else:
            if target_params is None:
                target = float(default_target)
            # Clamp only for the projection; an out-of-range request still keeps its refusal.
            target = float(max(min_params, min(max_params, target)))
            projection = _shrink_projection(
                config, base_params = int(base_params), target_params = target
            )

    if base_params and layers and target_params is not None:
        if float(target_params) >= float(base_params):
            too_large = True
            refusals = refusals + [
                _refusal(
                    "not_smaller",
                    f"{_format_params(float(target_params))} is not smaller than this "
                    f"checkpoint's {_format_params(base_params)}. Use the growth direction for "
                    "a larger target.",
                    blocks = False,
                )
            ]
        elif max_remove >= 1 and float(target_params) > max_params:
            too_large = True
            refusals = refusals + [
                _refusal(
                    "not_smaller",
                    f"The smallest shrink drops one block (~{_format_params(per_layer)}), to "
                    f"{_format_params(max_params)}. Ask for that or lower.",
                    blocks = False,
                )
            ]
        elif max_remove >= 1 and float(target_params) < min_params:
            below_floor = True
            refusals = refusals + [
                _refusal(
                    "below_shrink_floor",
                    f"Shrinking keeps at least {SHRINK_FLOOR:.0%} of the blocks, which is "
                    f"{keep_floor} of {layers} here - about {_format_params(min_params)}. Below "
                    "that the embeddings dominate; start from a smaller checkpoint instead.",
                    blocks = False,
                )
            ]

    blocked_reason = blocking[0]["message"] if blocking else None
    if blocked_reason is None:
        first_soft = next(
            (item for item in refusals if not item["blocks"]), None
        )
        blocked_reason = first_soft["message"] if first_soft else None

    notes = [
        "Parameter counts are read from config.json and are analytic; the run reports the "
        "measured count after the blocks are dropped.",
        "Dropping blocks does not preserve the function. The output is a starting point for "
        "recovery training, not a drop-in replacement for the base checkpoint.",
        "The first and last blocks are protected, and at least "
        f"{SHRINK_FLOOR:.0%} of the depth is kept.",
    ]

    return {
        "source": label,
        "direction": "shrink",
        "spec": {
            "architecture": str(config.get("model_type") or config.get("architecture") or "unknown"),
            "hidden_size": _int(config.get("hidden_size")),
            "num_layers": _int(config.get("num_hidden_layers") or config.get("n_layer")),
            "intermediate_size": _int(config.get("intermediate_size")),
            "vocab_size": _int(config.get("vocab_size")),
        },
        "base_params": int(base_params or 0),
        "min_params": min_params,
        "floor_params": min_params,
        "max_params": max_params,
        "cap_added_params": CAP_ADDED_PARAMS,
        "growth_floor": GROWTH_FLOOR,
        "shrink_floor": SHRINK_FLOOR,
        "target_params": int(target),
        "default_target_params": int(default_target),
        "min_added_params": int(min_params - (base_params or 0)),
        "max_added_params": int(max_params - (base_params or 0)),
        "added_params": int(target) - int(base_params or 0),
        "default_added_params": int(default_target) - int(base_params or 0),
        "floor_added_params": int(min_params - (base_params or 0)),
        "step_params": max(1, int(per_layer)),
        "method": "depth",
        "method_label": SHRINK_METHOD_LABELS["depth"],
        "methods": [{"id": key, "label": SHRINK_METHOD_LABELS[key]} for key in SHRINK_METHODS],
        "projection": projection,
        "target_below_floor": below_floor,
        "refusals": refusals,
        "blocked": bool(blocking),
        "blocked_reason": blocked_reason,
        "runnable": not blocking and not below_floor and not too_large,
        "engine": {"available": available, "error": engine_error},
        "notes": notes,
    }


def growth_policy(
    base_params: Optional[int],
    *,
    target_params: Optional[float] = None,
    added_params: Optional[float] = None,
) -> dict[str, Any]:
    """Report growth targets, limits, and warnings in parameter counts.

    With a known nonzero base, ``added_params`` takes precedence over an
    absolute ``target_params``. If neither is supplied, the default adds 10%
    (at least one parameter), capped at the growth ceiling. The returned
    ``requested_params`` is unclamped; ``target_params`` is clamped to the
    base and ceiling. ``over_cap`` describes the request, while ``below_floor``
    and ``refusal`` mark a clamped target that adds nothing. ``measured_note``
    describes targets below the measured growth range, including no growth.

    An unknown or zero base is treated as zero: added parameters are ignored,
    and the requested target is returned without clamping or a floor refusal.
    """
    base = int(base_params or 0)
    cap = base + CAP_ADDED_PARAMS
    floor_params = int(math.ceil(base * GROWTH_FLOOR))
    default = min(max(base + max(1, int(0.1 * base)), floor_params), cap) if base else base
    if added_params is not None and base:
        requested = float(base) + float(added_params)
    elif target_params is not None:
        requested = float(target_params)
    else:
        requested = float(default)
    target = float(min(max(requested, base), cap)) if base else requested
    below_floor = bool(base) and target <= base
    return {
        "min_params": base,
        "max_params": cap,
        "floor_params": floor_params,
        "default_target_params": default,
        "target_params": target,
        "requested_params": requested,
        "over_cap": requested > cap,
        "below_floor": below_floor,
        "refusal": (
            f"{_format_params(target)} is not larger than this checkpoint's "
            f"{_format_params(base)}. Add at least one parameter, or plan a shrink with a "
            "target below the base."
            if below_floor
            else None
        ),
        "measured_note": (
            f"{_format_params(target)} is {target / base:.2f}x this checkpoint, below the "
            f"{MEASURED_GROWTH_FLOOR:g}x where growth gains have been measured: no study "
            "reports a gain this small, so the step is an experiment rather than a safe bet."
            if base and target < base * MEASURED_GROWTH_FLOOR
            else None
        ),
    }


def plan(
    source: str,
    *,
    target_params: Optional[float] = None,
    added_params: Optional[float] = None,
    method: str = "depth",
    direction: Optional[str] = None,
    token: Optional[str] = None,
) -> dict[str, Any]:
    """The slider's range, the growth floor, and every refusal reason. Reads config.json only.

    The slider speaks in *added* parameters (0 to 3B), because that is the quantity the
    control actually sets; ``target_params`` is still accepted for callers that think in
    absolute sizes.

    ``direction`` is inferred from the target when omitted: a target below the base
    checkpoint is a shrink, anything else is growth.

    Growth defaults to 10% above the base and clamps requests to the base and
    growth ceiling. The returned plan reports refusals and whether it is runnable;
    an unavailable engine is reported as a refusal. Invalid methods, directions,
    conflicting targets, or an unreadable checkpoint config raise ValueError.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}.")
    if target_params is not None and added_params is not None:
        raise ValueError("Give target_params or added_params, not both.")
    if direction not in (None, "grow", "shrink"):
        raise ValueError(f"direction must be 'grow', 'shrink' or None, got {direction!r}.")

    label, config = merge.read_model_config(source, token = token)
    base_params = _base_params(config)

    shrinking = direction == "shrink" or (
        direction is None
        and target_params is not None
        and base_params is not None
        and float(target_params) < float(base_params)
    )
    if shrinking:
        if added_params is not None:
            raise ValueError(
                "Shrinking is planned in absolute parameters; give target_params, not added_params."
            )
        return _shrink_plan(
            label,
            source,
            config,
            base_params,
            target_params = target_params,
            method = method,
            token = token,
        )

    refusals = _refusals(source, config, base_params)
    blocking = [item for item in refusals if item["blocks"]]

    policy = growth_policy(base_params, target_params = target_params, added_params = added_params)
    target = policy["target_params"]
    min_params = policy["min_params"]
    cap = policy["max_params"]
    floor_params = policy["floor_params"]
    default_target = policy["default_target_params"]

    projection = (
        _project(config, base_params = int(base_params), target_params = target, method = method)
        if base_params
        else None
    )

    if policy["refusal"] is not None:
        refusals = refusals + [_refusal("no_growth", policy["refusal"])]

    available, engine_error = engine_available()
    if not available:
        refusals = refusals + [
            _refusal(
                "engine_unavailable",
                f"Plans work, but the expansion engine cannot load here: {engine_error}",
            )
        ]

    blocked_reason = blocking[0]["message"] if blocking else None
    if blocked_reason is None and policy["below_floor"]:
        blocked_reason = policy["refusal"]

    notes = [
        "Parameter counts are read from config.json and are analytic; the run reports the "
        "measured count after expansion.",
        "Depth growth is bit-exact at step 0. Width growth is not: widening down_proj makes "
        "float32 re-associate the sum, so the logits move by ~1e-5 of their own scale.",
        f"Any growth above the base is allowed. Below {MEASURED_GROWTH_FLOOR:g}x no study "
        "reports a measured gain, so a small step is a real experiment rather than a safe bet.",
    ]

    return {
        "source": label,
        "direction": "grow",
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
        "min_added_params": 0,
        "max_added_params": CAP_ADDED_PARAMS,
        "added_params": max(0, int(target) - min_params),
        "default_added_params": max(0, int(default_target) - min_params),
        "floor_added_params": max(0, floor_params - min_params),
        "step_params": max(1_000_000, int(CAP_ADDED_PARAMS / 300)),
        "method": method,
        "method_label": METHOD_LABELS[method],
        "methods": [{"id": key, "label": METHOD_LABELS[key]} for key in METHODS],
        "projection": projection,
        "target_below_floor": policy["below_floor"],
        "refusals": refusals,
        "blocked": bool(blocking),
        "blocked_reason": blocked_reason,
        "runnable": not blocking and not policy["below_floor"],
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


def _run_shrink(
    source: str,
    *,
    output_dir: str,
    target_params: float,
    drop_strategy: str = "importance",
    method: str = "depth",
    token: Optional[str] = None,
    progress: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    """Shrink ``source`` and save the smaller checkpoint in ``output_dir``.

    Unlike growth there is no function to preserve: the saved model is a warm start
    for recovery training, and the result reports the drift it measured so nobody
    mistakes it for a drop-in replacement.
    """

    def stage(name: str, message: str, **extra: Any) -> None:
        if progress is not None:
            progress({"stage": name, "message": message, **extra})

    report = plan(
        source, target_params = target_params, method = method, direction = "shrink", token = token
    )
    if not report["runnable"]:
        raise ValueError(report.get("blocked_reason") or "This checkpoint cannot be shrunk.")

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

    calibration = None
    if tokenizer is not None:
        try:
            encoded = tokenizer(
                _SHRINK_CALIBRATION, return_tensors = "pt", truncation = True, max_length = 256
            )
            calibration = encoded["input_ids"].to(device)
        except Exception:  # noqa: BLE001 - the weight proxy is a valid fallback
            calibration = None

    model.eval()
    with torch.no_grad():
        base_logits = model(input_ids = ids).logits.float().clone()

    projection = report["projection"] or {}
    started = time.monotonic()
    if method == "mlp_width":
        stage(
            "shrinking",
            f"Narrowing the MLP from {projection.get('current_intermediate_size', '?')} to "
            f"{projection.get('new_intermediate_size', '?')} channels in every block...",
        )
        model, shrink = engine.shrink_model_width(
            model, target_params = float(report["target_params"])
        )
    else:
        stage(
            "shrinking",
            f"Dropping {projection.get('removed_layers', '?')} of "
            f"{projection.get('current_num_layers', '?')} blocks...",
        )
        model, shrink = engine.shrink_model(
            model,
            target_params = float(report["target_params"]),
            drop_strategy = drop_strategy,
            calibration_input_ids = calibration,
        )

    stage("measuring", "Measuring how far the smaller model drifts from the base...")
    with torch.no_grad():
        shrunk_logits = model(input_ids = ids).logits.float()
    drift = float((base_logits - shrunk_logits).abs().max().item())
    scale = float(base_logits.abs().max().item())

    stage("saving", "Saving and reloading the smaller checkpoint...")
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
        "direction": "shrink",
        "method": method,
        "base_params": int(shrink["base_params"]),
        "final_params": int(shrink["final_params"]),
        "added_params": int(shrink["final_params"]) - int(shrink["base_params"]),
        "removed_params": int(shrink["removed_params"]),
        "removed_layers": shrink.get("removed_layers"),
        "removed_channels": shrink.get("removed_channels"),
        "new_num_layers": shrink.get("new_num_layers"),
        "growth_factor": float(shrink["shrink_factor"]),
        "shrink_factor": float(shrink["shrink_factor"]),
        "new_intermediate_size": shrink.get("new_intermediate_size"),
        "scored_by": shrink["scored_by"],
        "step0_max_logit_diff": drift,
        "relative_drift": drift / max(1e-6, scale),
        "function_preserved": False,
        "recovery_required": True,
        "verified": bool(saved.get("verified")),
        "reload_max_logit_diff": saved.get("max_logit_diff"),
        "peak_mem_gb": peak_gb,
        "elapsed_seconds": round(elapsed, 1),
        "refusals": report["refusals"],
        "notes": report["notes"],
        "plan": report,
    }
    stage(
        "completed",
        f"Shrunk to {_format_params(result['final_params'])} "
        f"({result['shrink_factor']:.2f}x); recovery training is the next step.",
        done = 1,
        total = 1,
    )
    return result


def run(
    source: str,
    *,
    output_dir: str,
    target_params: float,
    method: str = "depth",
    insert_strategy: str = "even",
    force_small_growth: bool = False,
    drop_strategy: str = "importance",
    token: Optional[str] = None,
    progress: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    """Grow or shrink ``source`` to ``target_params`` and save a checkpoint in ``output_dir``.

    A target below the base checkpoint dispatches to :func:`_run_shrink`; anything
    larger is depth/width expansion, which is function-preserving and needs CPT to
    unlock the new capacity. ``force_small_growth`` is still accepted for the CMT
    path, which adds capacity sized by its datasets.
    """

    def stage(name: str, message: str, **extra: Any) -> None:
        if progress is not None:
            progress({"stage": name, "message": message, **extra})

    report = plan(source, target_params = target_params, method = method, token = token)
    if report.get("direction") == "shrink":
        return _run_shrink(
            source,
            output_dir = output_dir,
            target_params = float(report["target_params"]),
            drop_strategy = drop_strategy,
            method = method,
            token = token,
            progress = progress,
        )
    if report["blocked"] or (not report["runnable"] and not force_small_growth):
        raise ValueError(report["blocked_reason"] or "This checkpoint cannot be expanded.")
    if report["target_below_floor"] and force_small_growth:
        report.setdefault("notes", []).append(
            f"Growth is {float(report['target_params']) / max(1, report['base_params']):.2f}x, "
            f"below the evidenced {MEASURED_GROWTH_FLOOR:g}x floor; this run opted in deliberately."
        )

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
        force_small_growth = force_small_growth,
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
