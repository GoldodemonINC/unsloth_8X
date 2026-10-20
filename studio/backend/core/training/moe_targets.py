# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which modules a MoE finetune trains.

Unsloth resolves a fused-expert MoE's expert weights from the leaf names it is
given: naming ``gate_proj``/``up_proj``/``down_proj`` adds ``mlp.experts.gate_up_proj``
and ``down_proj`` to PEFT's ``target_parameters``, so the experts train as a block
rather than one module at a time. Studio's default list therefore already reaches
the experts; this module exists so the choice is explicit and so "experts only" or
"router only" can hand the loader a different leaf list.

Nothing here touches a model: the caller decides the scope, this decides the
leaves, and the worker checks the scope against the model it actually loaded.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

# The scopes the API accepts. ``None`` keeps Studio's default target list.
MOE_EXPERT_SCOPES = (
    "experts_only",
    "experts_and_attention",
    "experts_and_router",
    "router_only",
)

ATTENTION_LEAVES = ("q_proj", "k_proj", "v_proj", "o_proj")
# gate_proj/up_proj map onto a fused expert's gate_up_proj; down_proj onto its down_proj.
EXPERT_LEAVES = ("gate_proj", "up_proj", "down_proj")
# A MoE router is a plain Linear named `gate` (Mixtral's block_sparse_moe.gate, Qwen3-MoE's
# mlp.gate). PEFT matches suffixes, so "gate" never swallows "gate_proj".
ROUTER_LEAVES = ("gate",)

SCOPE_TARGETS: dict[str, tuple[str, ...]] = {
    "experts_only": EXPERT_LEAVES,
    "experts_and_attention": ATTENTION_LEAVES + EXPERT_LEAVES,
    "experts_and_router": EXPERT_LEAVES + ROUTER_LEAVES,
    "router_only": ROUTER_LEAVES,
}

MOE_FULL_FINETUNE_REFUSED = (
    "moe_expert_scope only applies to LoRA/QLoRA: full finetuning already trains every "
    "expert. Drop the scope, or switch to LoRA."
)

MOE_SCOPE_TARGETS_CONFLICT = (
    "moe_expert_scope '{}' replaces the target list, but target_modules already names {}. "
    "Clear target_modules, or use a scope that adds to it."
)


def normalize_scope(scope: Optional[str]) -> Optional[str]:
    """The scope in its canonical spelling, or None when it was not set."""
    if scope is None:
        return None
    value = str(scope).strip().lower()
    if not value:
        return None
    if value not in MOE_EXPERT_SCOPES:
        raise ValueError(
            f"Unknown moe_expert_scope {scope!r}; expected one of {', '.join(MOE_EXPERT_SCOPES)}."
        )
    return value


def scope_trains_experts(scope: Optional[str]) -> bool:
    normalized = normalize_scope(scope) if scope else None
    if normalized is None:
        return False
    return bool(set(SCOPE_TARGETS[normalized]) & set(EXPERT_LEAVES))


def scope_trains_router(scope: Optional[str]) -> bool:
    normalized = normalize_scope(scope) if scope else None
    if normalized is None:
        return False
    return bool(set(SCOPE_TARGETS[normalized]) & set(ROUTER_LEAVES))


def _leaf(name: str) -> str:
    return str(name).rsplit(".", 1)[-1]


def resolve_target_modules(
    scope: Optional[str],
    base_targets: Optional[Sequence[str]] = None,
    *,
    training_type: str = "LoRA/QLoRA",
) -> Optional[list[str]]:
    """The leaf list a scope asks for, or None when the scope is unset.

    ``experts_and_attention`` and ``experts_and_router`` keep whatever the caller
    already targeted and add to it (the scope names the *extra* family, not a
    replacement); ``experts_only`` and ``router_only`` replace the list, and say so
    if the caller had already named something rather than dropping it silently.
    """
    normalized = normalize_scope(scope)
    if normalized is None:
        return None
    if training_type.strip() == "Full Finetuning":
        raise ValueError(MOE_FULL_FINETUNE_REFUSED)
    requested = [target for target in (base_targets or ()) if isinstance(target, str)]
    requested = [target for target in requested if target not in ("all-linear",)]
    leaves = list(SCOPE_TARGETS[normalized])
    if normalized in ("experts_only", "router_only") and requested:
        raise ValueError(
            MOE_SCOPE_TARGETS_CONFLICT.format(normalized, ", ".join(sorted(set(requested))))
        )
    # "all-linear" would defeat experts_only: PEFT turns it into every linear layer.
    if normalized in ("experts_only", "router_only") and "all-linear" in (base_targets or ()):
        raise ValueError(
            MOE_SCOPE_TARGETS_CONFLICT.format(normalized, "all-linear")
        )
    combined = list(dict.fromkeys([*requested, *leaves]))
    return combined


def scope_selectors(scope: Optional[str]) -> dict[str, Any]:
    """The layer-family selectors that match a scope, for the VLM/audio branches."""
    normalized = normalize_scope(scope)
    if normalized is None:
        return {}
    trains_attention = normalized == "experts_and_attention"
    return {
        "finetune_language_layers": True,
        # experts_only still needs the MLP family on, since the experts live under mlp.
        "finetune_mlp_modules": True,
        "finetune_attention_modules": trains_attention,
    }


def is_moe_model(model: Any) -> bool:
    """Whether a loaded model has MoE experts, without importing torch at module load."""
    try:
        from unsloth.models._utils import is_moe_model as _is_moe
    except Exception:
        return False
    try:
        return bool(_is_moe(model))
    except Exception:
        return False


def expert_parameter_targets(model: Any, target_modules: Iterable[str]) -> list[str]:
    """The fused expert parameter names PEFT will train for these leaves.

    Empty on a non-MoE model, and on a MoE model whose experts are ModuleLists rather
    than Parameters (those are reached through target_modules instead).
    """
    try:
        from unsloth.models._utils import get_moe_target_parameters
    except Exception:
        return []
    try:
        return list(get_moe_target_parameters(model, list(target_modules)) or [])
    except Exception:
        return []


def describe_scope(scope: Optional[str], model: Any = None) -> str:
    """A log line naming what the scope does, with the expert count when known."""
    normalized = normalize_scope(scope)
    if normalized is None:
        return "MoE expert scope not set; Studio's default target list applies"
    parts = [f"MoE expert scope: {normalized}"]
    config = getattr(model, "config", model) if model is not None else None
    if config is not None:
        for attr in ("num_experts", "n_routed_experts", "num_local_experts"):
            count = getattr(config, attr, None)
            if isinstance(count, int) and count > 0:
                parts.append(f"{count} experts")
                break
        for attr in ("num_experts_per_tok", "top_k", "num_selected_experts"):
            top_k = getattr(config, attr, None)
            if isinstance(top_k, int) and top_k > 0:
                parts.append(f"top-{top_k}")
                break
    return "; ".join(parts)
