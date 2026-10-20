# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Layer-wise learning-rate customization.

A transformer is not one learning-rate surface. Embeddings, early attention
blocks, late blocks and the output head learn at very different useful speeds.
This module maps every parameter to a structural zone and multiplies the base
learning rate per zone (with an optional per-layer override), so a run can keep
the foundation almost frozen while letting the head move quickly.

Zone layout (by depth, when the model exposes ``N`` decoder layers)::

    embedding   embed_tokens / wte / positional embeddings
    early       layers [0, N/3)
    middle      layers [N/3, 2N/3)
    late        layers [2N/3, N)
    head        lm_head / final norm / score head
    adapter     any parameter PEFT added (lora_*, modules_to_save)

Classification is name-based on purpose: it works for every architecture
Studio loads (llama, qwen, mistral, gemma, phi, ...) without instantiating the
model, which keeps planning cheap and testable.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

ZONES = ("embedding", "early", "middle", "late", "head", "adapter")

_EMBED_MARKERS = (
    "embed_tokens",
    "wte",
    "wpe",
    "word_embeddings",
    "position_embeddings",
    "token_embedding",
    "embedding",
)
_HEAD_MARKERS = ("lm_head", "score", "classifier", "output_proj", "final_layernorm")
_ADAPTER_MARKERS = ("lora_", "modules_to_save", "dora_", "lora_magnitude_vector")

# A decoder layer is named by a container component followed by its index:
# ``model.layers.0.…``, ``transformer.h.0.…``, ``encoder.block.0.…``. Matching the
# *components* rather than one hard-coded path is what keeps GPT-2-style ``h.0``
# names from falling through to the embedding zone.
_LAYER_CONTAINERS = frozenset({"layer", "layers", "block", "blocks", "h"})
# Terminal normalisation that closes the stack: the model's final norm belongs
# with the head, not with the embeddings.
_FINAL_NORM_RE = re.compile(r"(?:^|\.)(?:norm|ln_f|final_layernorm|final_norm)(?:\.|$)")
# Structural tags reported by the zone map. They do not change the zone; they
# make the preview explain *why* a parameter landed where it did.
_MOE_ROUTER_MARKERS = ("mlp.gate", "router", "block_sparse_moe.gate", "gate.weight")
_MOE_EXPERT_MARKERS = ("experts.", "expert_", "block_sparse_moe.experts")
_HYBRID_MARKERS = ("mamba", ".ssm", "ssm_", "linear_attn", "recurrent", "conv1d", "dt_proj")


@dataclass(frozen=True)
class LayerLrConfig:
    base_learning_rate: float
    # Multiplier per zone; defaults make the classic "protect the base" recipe.
    zone_multipliers: dict[str, float] = field(
        default_factory=lambda: {
            "embedding": 0.25,
            "early": 0.25,
            "middle": 0.5,
            "late": 0.75,
            "head": 1.0,
            "adapter": 1.0,
        }
    )
    # Per-layer multiplier overrides keyed by layer index (0-based).
    layer_multipliers: dict[int, float] = field(default_factory=dict)
    # Optional smooth depth scaling applied on top of the zone multiplier:
    # early layers * start, final layers * end. None disables it.
    depth_curve: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["layer_multipliers"] = {
            str(k): v for k, v in self.layer_multipliers.items()
        }
        payload["depth_curve"] = list(self.depth_curve) if self.depth_curve else None
        return payload


def _validate_multiplier(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a number.") from error
    if not (number >= 0.0 and number <= 10.0):
        raise ValueError(f"{label} must be between 0 and 10.")
    return number


def validate_layer_lr_config(
    base_learning_rate: float, config: Mapping[str, Any] | None
) -> LayerLrConfig:
    base = float(base_learning_rate)
    if not (base > 0 and base <= 1.0):
        raise ValueError("Base learning rate must be in (0, 1].")
    payload = dict(config or {})
    zone_values = payload.get("zone_multipliers") or {}
    if not isinstance(zone_values, Mapping):
        raise ValueError("zone_multipliers must be an object of zone -> multiplier.")
    defaults = LayerLrConfig(base_learning_rate=base).zone_multipliers
    zones: dict[str, float] = {}
    for zone in ZONES:
        zones[zone] = _validate_multiplier(
            zone_values.get(zone, defaults[zone]), f"{zone} multiplier"
        )
    unknown = [zone for zone in zone_values if zone not in ZONES]
    if unknown:
        raise ValueError(f"Unknown layer zones: {', '.join(unknown)}.")
    per_layer_raw = payload.get("layer_multipliers") or {}
    if not isinstance(per_layer_raw, Mapping):
        raise ValueError("layer_multipliers must be an object of layer index -> multiplier.")
    per_layer: dict[int, float] = {}
    for key, value in per_layer_raw.items():
        try:
            index = int(key)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid layer index {key!r}.") from error
        if index < 0:
            raise ValueError("Layer indices must be non-negative.")
        per_layer[index] = _validate_multiplier(value, f"layer {index} multiplier")
    curve = payload.get("depth_curve")
    curve_tuple = None
    if curve is not None:
        if not isinstance(curve, (list, tuple)) or len(curve) != 2:
            raise ValueError("depth_curve must be [start, end].")
        curve_tuple = (
            _validate_multiplier(curve[0], "depth_curve start"),
            _validate_multiplier(curve[1], "depth_curve end"),
        )
    return LayerLrConfig(
        base_learning_rate=base,
        zone_multipliers=zones,
        layer_multipliers=per_layer,
        depth_curve=curve_tuple,
    )


def layer_index(name: str) -> int | None:
    """Depth index of a parameter, or None for terminal (non-block) parameters.

    Reads name *components*: the index must be a bare integer whose preceding
    component is a block container. That skips expert indices
    (``…experts.5.w1``) and attention-head indices, and it recognises the
    ``transformer.h.0`` and ``encoder.block.0`` spellings.
    """
    parts = name.split(".")
    for position, part in enumerate(parts[:-1]):
        if position == 0 or not part.isdigit():
            continue
        if parts[position - 1] in _LAYER_CONTAINERS:
            return int(part)
    return None


def param_kind(name: str) -> str:
    """Structural role of a parameter, for the zone-map preview."""
    lowered = name.lower()
    if any(marker in lowered for marker in _ADAPTER_MARKERS):
        return "adapter"
    if any(marker in lowered for marker in _MOE_EXPERT_MARKERS):
        return "moe_expert"
    if any(marker in lowered for marker in _MOE_ROUTER_MARKERS):
        return "moe_router"
    if any(marker in lowered for marker in _HYBRID_MARKERS):
        return "hybrid_block"
    if _FINAL_NORM_RE.search(lowered) or any(m in lowered for m in _HEAD_MARKERS):
        return "head"
    if any(marker in lowered for marker in _EMBED_MARKERS):
        return "embedding"
    if lowered.endswith("norm.weight") or lowered.endswith("norm.bias") or ".norm." in lowered:
        return "norm"
    if any(token in lowered for token in ("q_proj", "k_proj", "v_proj", "o_proj", "attn")):
        return "attention"
    if any(token in lowered for token in ("mlp", "ffn", "feed_forward")):
        return "mlp"
    return "other"


def classify_param_zone(name: str, num_layers: int | None = None) -> str:
    """Zone of one parameter name, by depth when the depth is known."""
    lowered = name.lower()
    if any(marker in lowered for marker in _ADAPTER_MARKERS):
        return "adapter"
    index = layer_index(lowered)
    if index is None:
        # Terminal parameters: the head (lm_head, score, the final norm) or the
        # input embeddings closed the stack; anything else sits beside them.
        if any(marker in lowered for marker in _HEAD_MARKERS) or _FINAL_NORM_RE.search(lowered):
            return "head"
        return "embedding"
    if num_layers:
        if index >= num_layers:
            # Off-the-end index (a padded or mis-counted config): clamp rather
            # than inventing a zone that no multiplier covers.
            index = num_layers - 1
        return zone_for_layer(index, num_layers)
    return "middle"  # refined by zone_for_layer once the depth is known


def zone_for_layer(index: int, num_layers: int) -> str:
    if num_layers <= 0:
        raise ValueError("num_layers must be positive.")
    if index < num_layers / 3.0:
        return "early"
    if index < (2.0 * num_layers) / 3.0:
        return "middle"
    return "late"


def zone_breakdown(num_layers: int) -> list[dict[str, Any]]:
    """Ranges for the UI's visual zone strip."""
    if num_layers <= 0:
        raise ValueError("num_layers must be positive.")
    buckets: dict[str, list[int]] = {zone: [] for zone in ZONES}
    for index in range(num_layers):
        buckets[zone_for_layer(index, num_layers)].append(index)
    return [
        {
            "zone": zone,
            "layers": buckets[zone],
            "start": buckets[zone][0] if buckets[zone] else None,
            "end": buckets[zone][-1] if buckets[zone] else None,
        }
        for zone in ZONES
        if buckets[zone]
    ]


def resolve_multiplier(
    name: str, config: LayerLrConfig, num_layers: int | None
) -> tuple[str, float]:
    """Return ``(zone, multiplier)`` for one parameter name."""
    index = layer_index(name)
    if any(marker in name.lower() for marker in _ADAPTER_MARKERS):
        return "adapter", config.zone_multipliers.get("adapter", 1.0)
    if index is None:
        lowered = name.lower()
        if any(marker in lowered for marker in _HEAD_MARKERS) or _FINAL_NORM_RE.search(lowered):
            zone = "head"
        else:
            zone = "embedding"
        return zone, config.zone_multipliers.get(zone, 1.0)
    if num_layers and index >= num_layers:
        index = num_layers - 1
    if index in config.layer_multipliers:
        zone = zone_for_layer(index, num_layers or index + 1)
        return zone, config.layer_multipliers[index]
    if not num_layers:
        return "middle", config.zone_multipliers.get("middle", 1.0)
    zone = zone_for_layer(index, num_layers)
    multiplier = config.zone_multipliers.get(zone, 1.0)
    if config.depth_curve and num_layers > 1:
        start, end = config.depth_curve
        progress = index / (num_layers - 1)
        multiplier *= start + (end - start) * progress
    return zone, multiplier


def build_param_groups(
    named_parameters: Iterable[tuple[str, Any]],
    config: LayerLrConfig,
    *,
    num_layers: int | None = None,
    weight_decay: float = 0.0,
) -> list[dict[str, Any]]:
    """Build optimizer param groups with per-zone learning rates.

    Every parameter lands in exactly one group. Groups are keyed by
    ``(zone, multiplier)`` so the optimizer sees a small, stable list instead of
    one group per parameter.
    """
    groups: dict[tuple[str, float], dict[str, Any]] = {}
    counts: dict[tuple[str, float], int] = {}
    for name, parameter in named_parameters:
        if not getattr(parameter, "requires_grad", True):
            continue
        zone, multiplier = resolve_multiplier(name, config, num_layers)
        lr = config.base_learning_rate * multiplier
        key = (zone, round(lr, 12))
        group = groups.get(key)
        if group is None:
            group = {
                "lr": lr,
                "weight_decay": weight_decay,
                "zone": zone,
                "multiplier": multiplier,
                "param_names": [],
                "params": [],
            }
            groups[key] = group
            counts[key] = 0
        group["params"].append(parameter)
        # Keep a bounded sample of names for the run report.
        if len(group["param_names"]) < 8:
            group["param_names"].append(name)
        counts[key] += 1
    result = []
    for key, group in groups.items():
        group["param_count"] = counts[key]
        result.append(group)
    return result


def _numel(parameter: Any) -> int:
    """Element count of a parameter, a tensor, or a bare integer."""
    if isinstance(parameter, (int, float)) and not isinstance(parameter, bool):
        return int(parameter)
    counter = getattr(parameter, "numel", None)
    if callable(counter):
        try:
            return int(counter())
        except Exception:  # noqa: BLE001 - a fake tensor in a test
            return 0
    shape = getattr(parameter, "shape", None)
    if shape is not None:
        total = 1
        for size in shape:
            total *= int(size)
        return total
    return 0


def _requires_grad(parameter: Any) -> bool:
    return bool(getattr(parameter, "requires_grad", True))


def zone_map(
    named_parameters: Iterable[tuple[str, Any]],
    config: LayerLrConfig,
    *,
    num_layers: int | None = None,
    sample_limit: int = 8,
    include_frozen: bool = False,
) -> dict[str, Any]:
    """Every parameter's zone, with proof that each one landed in exactly one.

    ``build_param_groups`` groups by ``(zone, lr)``, so a parameter that silently
    matched no zone -- or two -- was previously invisible until the loss curve
    looked wrong. This returns the full classification plus a coverage block:
    ``exact_one_zone`` is the assertion the test suite checks, and
    ``unclassified`` names anything that fell through.

    Frozen parameters are excluded from the optimizer groups by design; they are
    still classified here (``include_frozen``) so the preview can show them.
    """
    entries: list[dict[str, Any]] = []
    per_zone: dict[str, dict[str, Any]] = {
        zone: {
            "zone": zone,
            "multiplier": config.zone_multipliers.get(zone, 1.0),
            "lr": config.base_learning_rate * config.zone_multipliers.get(zone, 1.0),
            "params": 0,
            "numel": 0,
            "kinds": {},
            "sample": [],
            "trainable": 0,
        }
        for zone in ZONES
    }
    seen: set[str] = set()
    duplicates: list[str] = []
    unclassified: list[str] = []
    total_numel = 0
    assigned_numel = 0

    for name, parameter in named_parameters:
        trainable = _requires_grad(parameter)
        size = _numel(parameter)
        total_numel += size
        if name in seen:
            duplicates.append(name)
        seen.add(name)
        if not trainable and not include_frozen:
            continue
        zone, multiplier = resolve_multiplier(name, config, num_layers)
        if zone not in per_zone:
            unclassified.append(name)
            continue
        kind = param_kind(name)
        bucket = per_zone[zone]
        bucket["params"] += 1
        bucket["numel"] += size
        if trainable:
            bucket["trainable"] += 1
        bucket["kinds"][kind] = bucket["kinds"].get(kind, 0) + 1
        if len(bucket["sample"]) < sample_limit:
            bucket["sample"].append(name)
        assigned_numel += size
        entries.append(
            {
                "name": name,
                "zone": zone,
                "kind": kind,
                "layer": layer_index(name),
                "multiplier": multiplier,
                "lr": config.base_learning_rate * multiplier,
                "numel": size,
                "trainable": trainable,
            }
        )

    zones = [bucket for bucket in per_zone.values() if bucket["params"] or bucket["zone"] in ("head", "embedding")]
    exact_one_zone = (
        not duplicates and not unclassified and all(entry["zone"] in per_zone for entry in entries)
    )
    return {
        "zones": zones,
        "entries": entries,
        "coverage": {
            "params": len(entries),
            "unique_names": len(seen),
            "exact_one_zone": exact_one_zone,
            "duplicates": duplicates,
            "unclassified": unclassified,
            "total_numel": total_numel,
            "assigned_numel": assigned_numel,
            "covers_all_params": assigned_numel == total_numel,
        },
        "num_layers": num_layers,
        "zone_definition_hash_inputs": {
            "zone_multipliers": dict(config.zone_multipliers),
            "layer_multipliers": {str(k): v for k, v in config.layer_multipliers.items()},
            "depth_curve": list(config.depth_curve) if config.depth_curve else None,
        },
    }


def zone_preview(
    config: LayerLrConfig,
    *,
    num_layers: int | None,
    param_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """The plan endpoint's zone map: the depth strip plus, when given, real names."""
    payload: dict[str, Any] = {
        "layer_strip": zone_breakdown(num_layers) if num_layers else [],
        "multipliers": dict(config.zone_multipliers),
        "layer_multipliers": {str(k): v for k, v in config.layer_multipliers.items()},
        "depth_curve": list(config.depth_curve) if config.depth_curve else None,
    }
    if param_names:
        mapped = zone_map(
            ((name, 1) for name in param_names), config, num_layers = num_layers
        )
        payload["zone_map"] = mapped["zones"]
        payload["coverage"] = mapped["coverage"]
        payload["entries"] = mapped["entries"]
    return payload


def summarize_groups(groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "groups": [
            {
                "zone": group.get("zone"),
                "lr": group.get("lr"),
                "multiplier": group.get("multiplier"),
                "param_count": group.get("param_count"),
                "param_names": group.get("param_names", [])[:8],
            }
            for group in groups
        ],
        "total_params": sum(int(group.get("param_count", 0)) for group in groups),
    }
