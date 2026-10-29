# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping, Optional

GB = 1024**3
FREE_RESERVE_GB = 2.0
TIGHT_RATIO = 0.85


def _round_gb(value: float) -> float:
    return round(value / GB, 2)


def _output_path(output_dir: Optional[str]) -> Optional[Path]:
    if output_dir and str(output_dir).strip():
        return Path(output_dir).expanduser()
    return None


def free_bytes(output_dir: Optional[str]) -> Optional[int]:
    path = _output_path(output_dir)
    if path is None:
        return None
    probe = path
    try:
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        return int(shutil.disk_usage(probe).free)
    except (OSError, RuntimeError, ValueError):
        return None


def estimate(spec: Mapping[str, Any]) -> dict[str, Any]:
    method = str(spec.get("method") or "lora").strip().lower()
    if method not in ("lora", "qlora", "full", "cpt"):
        method = "lora"
    try:
        params = int(spec.get("params") or 0)
        batch = max(1, int(spec.get("batch_size") or 1))
        sequence = max(128, int(spec.get("max_seq_length") or 2048))
        rank = max(1, int(spec.get("lora_rank") or 16))
        max_steps = max(0, int(spec.get("max_steps") or 0))
        save_steps = max(0, int(spec.get("save_steps") or 0))
    except (TypeError, ValueError):
        return {
            "status": "unknown",
            "reason": "The run's shape could not be read, so its disk use cannot be estimated.",
            "free_gb": None,
            "required_gb": None,
            "headroom_gb": None,
            "breakdown": {},
            "checkpoints": {"count": 0, "size_gb": 0.0, "known": False},
            "assumptions": [],
            "blockers": [],
        }
    if params <= 0:
        return {
            "status": "unknown",
            "reason": "The model's parameter count is unknown, so its disk use cannot be estimated.",
            "free_gb": None,
            "required_gb": None,
            "headroom_gb": None,
            "breakdown": {},
            "checkpoints": {"count": 0, "size_gb": 0.0, "known": False},
            "assumptions": [],
            "blockers": [],
        }

    output_dir = spec.get("output_dir")
    free = spec.get("free_bytes")
    free = int(free) if free is not None else free_bytes(output_dir)
    free_gb = _round_gb(free) if free is not None else None

    hidden = max(1, int(spec.get("hidden_size") or 0))
    intermediate = max(1, int(spec.get("intermediate_size") or 0))
    layers = max(1, int(spec.get("num_layers") or 0))
    if not hidden:
        hidden = max(1, int((params * 12) ** 0.5 / 4))
    if not layers:
        layers = max(1, int(params / (12 * hidden * hidden)))

    load_in_4bit = bool(spec.get("load_in_4bit")) or method == "qlora"
    model_cached = bool(spec.get("model_known_cached")) or bool(
        spec.get("model_local_path")
    )
    base_download = 0.0 if model_cached else params * 2.0
    if method in ("lora", "qlora"):
        trainable = rank * hidden * 4 * layers + rank * intermediate * 2 * layers
    else:
        trainable = params

    checkpoint_known = max_steps > 0 and save_steps > 0
    checkpoint_count = (
        max_steps // save_steps if checkpoint_known else 0
    )
    if method in ("lora", "qlora"):
        checkpoint_weights = trainable * 2.0 + params * (0.15 if load_in_4bit else 0.0)
        optimizer_state = trainable * 8.0
        final_save = trainable * 2.0
    else:
        checkpoint_weights = params * 2.0
        optimizer_state = params * 8.0
        final_save = params * 2.0

    optimizer_total = optimizer_state * checkpoint_count
    checkpoints = checkpoint_count * checkpoint_weights + optimizer_total
    reserve = FREE_RESERVE_GB * GB
    required = base_download + checkpoints + final_save + reserve
    required_gb = _round_gb(required)
    headroom_gb = round(free_gb - required_gb, 2) if free_gb is not None else None

    status = "unknown"
    reason: Optional[str] = None
    blockers: list[str] = []
    if free_gb is None:
        reason = "Free disk space could not be read for the outputs folder."
    if free_gb is not None:
        if required <= free * TIGHT_RATIO:
            status = "ok"
        elif required <= free:
            status = "tight"
        else:
            status = "over"
            blockers.append(
                f"The run needs about {required_gb:.2f} GB but only {free_gb:.2f} GB "
                f"is free on {output_dir or 'the outputs folder'}, so it would stop mid-training."
            )

    suggestions: list[str] = []
    if status in ("over", "tight"):
        if model_cached:
            suggestions.append(
                "The base model is already on disk, so no download space is counted; "
                "freeing space is the only lever left for this run."
            )
        if checkpoint_count:
            suggestions.append(
                f"Turn off periodic checkpoints or raise save steps from {save_steps}; "
                f"{checkpoint_count} checkpoints are budgeted."
            )
        if method in ("full", "cpt"):
            suggestions.append(
                "A full fine-tune saves weights and fp32 optimizer state. LoRA/QLoRA "
                "writes a fraction of that per checkpoint."
            )

    assumptions = [
        "Base weights are counted at 2 bytes per parameter only when the model is not already cached.",
        "Each checkpoint keeps weights (or adapter) plus fp32 optimizer state.",
        "A 2 GB reserve is kept free for the filesystem and the host.",
        "Periodic checkpoints are counted only when max_steps and save_steps are both known.",
        "This is an estimate; the allocator, merge step and temporary files can move the real total.",
    ]
    if not checkpoint_known:
        assumptions.append(
            "The number of periodic checkpoints is unknown because max_steps or save_steps is unset."
        )

    return {
        "status": status,
        "reason": reason,
        "path": str(_output_path(output_dir)) if _output_path(output_dir) else None,
        "free_gb": free_gb,
        "required_gb": required_gb,
        "headroom_gb": headroom_gb,
        "reserve_gb": _round_gb(reserve),
        "breakdown": {
            "base_weights_gb": _round_gb(base_download),
            "checkpoints_gb": _round_gb(checkpoints),
            "optimizer_gb": _round_gb(optimizer_total),
            "final_save_gb": _round_gb(final_save),
            "reserve_gb": _round_gb(reserve),
            "total_gb": required_gb,
        },
        "checkpoints": {
            "count": checkpoint_count,
            "size_gb": _round_gb(checkpoints),
            "known": checkpoint_known,
            "save_steps": save_steps,
            "max_steps": max_steps,
        },
        "params": params,
        "batch_size": batch,
        "max_seq_length": sequence,
        "method": method,
        "load_in_4bit": load_in_4bit,
        "suggestions": suggestions,
        "assumptions": assumptions,
        "blockers": blockers,
    }
