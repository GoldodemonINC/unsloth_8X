# SPDX-License-Identifier: NRC-1
# Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-1.md

"""A budget estimate for a training run, before it is started.

This is deliberately an estimate, not a promise. It reads the checkpoint's
``config.json`` for parameter count, width and depth, then models the four
things that actually occupy memory: weights, optimizer state, adapter state and
activations. Every coefficient is named below so a wrong number can be argued
with instead of being hidden.

The output is a verdict (fits / tight / over / unknown) and the concrete levers
that would bring an over-budget run under it. It never loads a weight.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

GB = 1024**3
_OVERHEAD_GB = 1.0
_FIT_HEADROOM = 0.85


def _config(source: str) -> dict[str, Any]:
    from core.advanced import merge, scaling

    _, config = merge.read_model_config(source)
    base = scaling._base_params(config)
    if not base:
        raise ValueError(
            "The checkpoint's config.json does not name a parameter count, so a fit "
            "estimate cannot be built."
        )
    return {
        "config": config,
        "params": int(base),
        "hidden": int(config.get("hidden_size") or config.get("n_embd") or 0),
        "layers": int(config.get("num_hidden_layers") or config.get("n_layer") or 0),
        "vocab": int(config.get("vocab_size") or 0),
        "intermediate": int(
            config.get("intermediate_size")
            or config.get("ffn_dim")
            or (4 * int(config.get("hidden_size") or 0))
        ),
    }


def _probe_gpu_gb() -> Optional[float]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        properties = torch.cuda.get_device_properties(0)
        return float(properties.total_memory) / GB
    except Exception:  # noqa: BLE001 - no torch is a normal state for a planning host
        return None


def estimate(spec: Mapping[str, Any]) -> dict[str, Any]:
    source = str(spec.get("model") or "").strip()
    if not source:
        raise ValueError("A model is required for a fit estimate.")
    method = str(spec.get("method") or "lora").strip().lower()
    if method not in ("lora", "qlora", "full", "cpt"):
        raise ValueError("method must be lora, qlora, full or cpt.")

    info = _config(source)
    params = info["params"]
    hidden = max(1, info["hidden"])
    layers = max(1, info["layers"])
    vocab = max(1, info["vocab"])
    intermediate = max(1, info["intermediate"])

    try:
        batch = max(1, int(spec.get("batch_size") or 1))
        sequence = max(128, int(spec.get("max_seq_length") or 2048))
        accumulation = max(1, int(spec.get("gradient_accumulation") or 1))
        rank = max(1, int(spec.get("lora_rank") or 16))
    except (TypeError, ValueError) as error:
        raise ValueError("batch_size, max_seq_length, gradient_accumulation and lora_rank must be numbers.") from error

    load_in_4bit = bool(spec.get("load_in_4bit")) or method == "qlora"
    gradient_checkpointing = bool(spec.get("gradient_checkpointing", True))
    budget_gb = spec.get("gpu_memory_gb")
    if budget_gb in (None, ""):
        budget_gb = _probe_gpu_gb()
    try:
        budget = float(budget_gb) if budget_gb not in (None, "") else None
    except (TypeError, ValueError) as error:
        raise ValueError("gpu_memory_gb must be a number when it is given.") from error

    # Weights. A 4-bit load rounds to roughly 0.55 bytes per parameter once the
    # quantized blocks and their scales are counted; fp16 is the plain 2 bytes.
    bytes_per_param = 0.55 if load_in_4bit else 2.0
    weights = params * bytes_per_param

    # Adapter state. A rank-r adapter on q/k/v/o and the MLPs is roughly
    # r * hidden * 4 tensors per layer for attention plus the MLP pair. The
    # optimizer holds fp32 m and v, so 8 bytes per adapter parameter.
    if method in ("lora", "qlora"):
        adapter_params = rank * hidden * 4 * layers + rank * intermediate * 2 * layers
        adapter = adapter_params * (2.0 + 8.0)
        optimizer = 0.0
        trainable = adapter_params
    else:
        # Full / CPT: gradients in the compute dtype and Adam's two fp32 states.
        adapter = 0.0
        optimizer = params * (2.0 + 8.0)
        trainable = params

    # Activations. About ten hidden-state-sized tensors per layer are live in a
    # forward/backward pair; gradient checkpointing trades recompute for a much
    # smaller peak. The logits tensor is batch * sequence * vocab in the compute
    # dtype and is not optional.
    activation_factor = 20.0 if gradient_checkpointing else 60.0
    activations = batch * sequence * hidden * layers * activation_factor
    logits = batch * sequence * vocab * 2.0
    overhead = _OVERHEAD_GB * GB
    required = weights + adapter + optimizer + activations + logits + overhead

    requirements = {
        "weights_gb": round(weights / GB, 2),
        "adapter_gb": round(adapter / GB, 2),
        "optimizer_gb": round(optimizer / GB, 2),
        "activations_gb": round(activations / GB, 2),
        "logits_gb": round(logits / GB, 2),
        "overhead_gb": round(overhead / GB, 2),
        "total_gb": round(required / GB, 2),
    }

    verdict = "unknown"
    headroom = None
    if budget:
        headroom = round(budget - required / GB, 2)
        if required / GB <= budget * _FIT_HEADROOM:
            verdict = "fits"
        elif required / GB <= budget:
            verdict = "tight"
        else:
            verdict = "over"

    suggestions: list[str] = []
    if verdict in ("over", "tight"):
        if batch > 1:
            suggestions.append(
                f"Drop the batch size from {batch} to 1; activations scale linearly with it."
            )
        if sequence > 1024:
            suggestions.append(
                f"Cut the sequence length from {sequence} to 1024 or 2048; activations scale with it."
            )
        if not gradient_checkpointing:
            suggestions.append("Turn on gradient checkpointing to shrink the activation peak.")
        if not load_in_4bit and method != "full":
            suggestions.append("Load in 4-bit (QLoRA) to cut the weight footprint by about 3.6x.")
        if method in ("lora", "qlora") and rank > 16:
            suggestions.append(f"Lower the LoRA rank from {rank} to 16; the adapter state shrinks with it.")
        if method in ("full", "cpt") and verdict == "over":
            suggestions.append(
                "Full training keeps fp32 optimizer state for every parameter. Use LoRA/QLoRA, "
                "or freeze the base and train only new blocks (CMT)."
            )
        if accumulation > 1:
            suggestions.append(
                f"Keep gradient accumulation at {accumulation}; it raises the effective batch "
                "without raising the activation peak."
            )
    if verdict == "unknown":
        suggestions.append(
            "The host did not report a GPU memory size, so only the requirement could be estimated."
        )

    return {
        "model": source,
        "method": method,
        "params": params,
        "params_label": (
            f"{params / 1e9:.2f}B"
            if params >= 1e9
            else f"{params / 1e6:.1f}M"
        ),
        "trainable_params": int(trainable),
        "config": {
            "hidden_size": hidden,
            "num_layers": layers,
            "vocab_size": vocab,
            "intermediate_size": intermediate,
            "batch_size": batch,
            "gradient_accumulation": accumulation,
            "max_seq_length": sequence,
            "lora_rank": rank,
            "load_in_4bit": load_in_4bit,
            "gradient_checkpointing": gradient_checkpointing,
        },
        "budget_gb": round(budget, 2) if budget else None,
        "required_gb": round(required / GB, 2),
        "headroom_gb": headroom,
        "verdict": verdict,
        "breakdown": requirements,
        "suggestions": suggestions,
        "assumptions": [
            "0.55 bytes/parameter for a 4-bit load, 2 bytes for fp16.",
            "Adam keeps two fp32 moments (8 bytes) per trainable parameter.",
            "Activations are modelled as 20 hidden-sized tensors per layer with checkpointing, 60 without.",
            "A 1 GB runtime overhead is included.",
            "This is a budget estimate, not a guarantee; allocator fragmentation and the model's "
            "attention implementation can move the real peak by a few GB.",
        ],
    }
