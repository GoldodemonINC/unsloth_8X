# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Merge-of-Experts (MMoE) router for cross-architecture merges.

Instead of averaging weights that do not share a shape, MMoE keeps each source
model's feed-forward block as a frozen **expert** and trains a small router that
sends each token to the best-equipped experts::

    hidden ──► router (mean-pool ─► Linear(H, E)) ──► top-k gate
                 │
                 ├─ expert 0: in_proj(H→H0) ─► FFN0 ─► out_proj(H0→H)
                 ├─ expert 1: in_proj(H→H1) ─► FFN1 ─► out_proj(H1→H)
                 └─ ...
    output = Σ gate_i · expert_i(hidden)

The per-expert projections are what make Llama 2B + Qwen 2B, or 0.8B + 2B of the
same family, runnable side by side without a single weight-space average. Only
the router and projections train; expert weights stay frozen (or keep their
existing LoRA adapters).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

__all__ = ["ExpertBlock", "MoELayer", "router_balance_loss", "build_moe_layer"]


def _torch():
    import torch

    return torch


class ExpertBlock:
    """Factory for the projection-wrapped expert used by :class:`MoELayer`."""

    @staticmethod
    def build(hidden_size: int, expert_hidden: int, ffn: Any | None = None):
        torch = _torch()
        import torch.nn as nn

        class _Expert(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.in_proj = nn.Linear(hidden_size, expert_hidden, bias=False)
                self.out_proj = nn.Linear(expert_hidden, hidden_size, bias=False)
                self.ffn = ffn if ffn is not None else _default_ffn(expert_hidden)
                self.activation = nn.GELU()

            def forward(self, hidden):
                projected = self.in_proj(hidden)
                if self.ffn is None:
                    mixed = self.activation(projected)
                else:
                    mixed = self.ffn(projected)
                return self.out_proj(mixed)

        return _Expert()


def _default_ffn(width: int):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(width, width * 2, bias=False),
        nn.GELU(),
        nn.Linear(width * 2, width, bias=False),
    )


class MoELayer:
    """Top-k routed mixture over heterogeneous experts (torch.nn.Module)."""

    def __new__(cls, *args, **kwargs):
        # Defined through a factory so importing this module needs no torch.
        return build_moe_layer(*args, **kwargs)


def build_moe_layer(
    hidden_size: int,
    expert_widths: Sequence[int],
    *,
    top_k: int = 2,
    temperature: float = 1.0,
    router_hidden: int | None = None,
    ffn_factories: Sequence[Any] | None = None,
):
    torch = _torch()
    import torch.nn as nn
    import torch.nn.functional as F

    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive.")
    if not expert_widths:
        raise ValueError("At least one expert is required.")
    if top_k < 1 or top_k > len(expert_widths):
        raise ValueError(f"top_k must be between 1 and {len(expert_widths)}.")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    class _MoELayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden_size = int(hidden_size)
            self.top_k = int(top_k)
            self.temperature = float(temperature)
            router_dim = int(router_hidden or hidden_size)
            self.router_norm = nn.LayerNorm(hidden_size)
            self.router = nn.Linear(hidden_size, router_dim, bias=False)
            self.gate = nn.Linear(router_dim, len(expert_widths), bias=False)
            factories = list(ffn_factories or [None] * len(expert_widths))
            self.experts = nn.ModuleList(
                [
                    ExpertBlock.build(hidden_size, int(width), factories[i] if i < len(factories) else None)
                    for i, width in enumerate(expert_widths)
                ]
            )
            self.last_aux_loss = torch.zeros((), dtype = torch.float32)

        def forward(self, hidden):
            batch, tokens, _ = hidden.shape
            flat = hidden.reshape(batch * tokens, -1)
            pooled = self.router_norm(flat)
            logits = self.gate(torch.tanh(self.router(pooled)))
            scores = F.softmax(logits / self.temperature, dim = -1)
            top_scores, top_indices = scores.topk(self.top_k, dim = -1)
            # Renormalize the selected gate mass so the mixture is a convex blend.
            top_scores = top_scores / top_scores.sum(dim = -1, keepdim = True).clamp_min(1e-8)

            output = torch.zeros_like(flat)
            flat_index = torch.arange(flat.shape[0], device = flat.device)
            for slot in range(self.top_k):
                expert_ids = top_indices[:, slot]
                weights = top_scores[:, slot]
                for expert_index, expert in enumerate(self.experts):
                    mask = expert_ids == expert_index
                    if not bool(mask.any()):
                        continue
                    routed = expert(flat[mask])
                    output.index_add_(
                        0,
                        flat_index[mask],
                        routed * weights[mask].unsqueeze(-1),
                    )

            # Switch-Transformer style load balancing: encourage uniform expert
            # usage so a 2B expert does not starve a 0.8B one.
            fraction = torch.zeros(len(self.experts), device = flat.device)
            for expert_index in range(len(self.experts)):
                fraction[expert_index] = (top_indices == expert_index).float().mean()
            importance = scores.mean(dim = 0)
            uniform = torch.full_like(importance, 1.0 / len(self.experts))
            self.last_aux_loss = (
                len(self.experts)
                * torch.sum(fraction * importance)
                - torch.sum(uniform * importance)
            ).abs()
            return output.reshape(batch, tokens, -1)

        def aux_loss(self):
            return self.last_aux_loss

        def export_config(self) -> dict[str, Any]:
            return {
                "type": "mmoe",
                "hidden_size": self.hidden_size,
                "expert_widths": [module.in_proj.out_features for module in self.experts],
                "top_k": self.top_k,
                "temperature": self.temperature,
                "router_dim": self.router.out_features,
            }

    return _MoELayer()


def router_balance_loss(
    routing_probabilities,
    top_indices,
    expert_count: int,
    *,
    coefficient: float = 0.01,
):
    """Standalone auxiliary loss, for callers that keep the router separate."""
    torch = _torch()

    probabilities = routing_probabilities.float()
    importance = probabilities.mean(dim = 0)
    counts = torch.zeros(expert_count, dtype = torch.float32, device = probabilities.device)
    for index in range(expert_count):
        counts[index] = (top_indices == index).float().mean()
    fraction = counts / counts.sum().clamp_min(1e-8)
    return coefficient * expert_count * torch.sum(fraction * importance)


def write_moe_bundle(
    output_dir: str,
    plan: Mapping[str, Any],
    *,
    hidden_size: int,
    seed: int = 3407,
) -> dict[str, Any]:
    """Materialize an MMoE merge plan as a loadable router bundle.

    The expert FFNs stay in the source checkpoints; this writes the router,
    per-expert input/output projections, and a manifest that maps each expert to
    its checkpoint. Reassembling the runtime model is the inference loader's
    job (a later milestone); the bundle is deliberately explicit about that.
    """
    import json
    from pathlib import Path

    import numpy as np
    from safetensors.numpy import save_file

    torch = _torch()
    torch.manual_seed(seed)
    experts = list(plan.get("experts") or [])
    if len(experts) < 2:
        raise ValueError("An MMoE bundle needs at least two experts.")
    widths = [
        int(expert.get("intermediate_size") or hidden_size * 2) for expert in experts
    ]
    layer = build_moe_layer(
        hidden_size,
        widths,
        top_k = int(plan.get("top_k", 2)),
        temperature = float(plan.get("router_temperature", 1.0)),
        router_hidden = int(plan.get("router_hidden", hidden_size)),
    )
    out = Path(output_dir).expanduser()
    out.mkdir(parents = True, exist_ok = True)
    state = {
        name: np.ascontiguousarray(tensor.detach().cpu().float().numpy())
        for name, tensor in layer.state_dict().items()
        if name.startswith(("router", "gate"))
        or ".in_proj." in name
        or ".out_proj." in name
    }
    save_file(state, str(out / "router.safetensors"))
    manifest = {
        "format": "unsloth-moe-v1",
        "status": "router_initialized",
        "hidden_size": hidden_size,
        "top_k": int(plan.get("top_k", 2)),
        "temperature": float(plan.get("router_temperature", 1.0)),
        "router_hidden": int(plan.get("router_hidden", hidden_size)),
        "expert_widths": widths,
        "experts": experts,
        "router_state": "router.safetensors",
        "shared_vocab": plan.get("shared_vocab"),
        "warnings": list(plan.get("warnings") or []),
        "notes": list(plan.get("notes") or [])
        + [
            "Router and projections are initialized; train them on your task data.",
            "Expert weights remain in their source checkpoints and are loaded by the MMoE runtime.",
        ],
    }
    (out / "moe_config.json").write_text(
        json.dumps(manifest, indent = 2), encoding = "utf-8"
    )
    return {
        "status": "completed",
        "output_dir": str(out),
        "router_file": str(out / "router.safetensors"),
        "config_file": str(out / "moe_config.json"),
        "manifest": manifest,
    }
