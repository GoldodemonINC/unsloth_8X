# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Advanced training, merging, and export mechanics for Unsloth Studio.

Every module here is import-light: torch, safetensors, and transformers are
imported inside the functions that need them so the FastAPI service can import
this package on a CPU-only host (and in tests) without touching the GPU stack.

Modules
-------
qat          Quantization-aware training: INT4 / NVFP4 fake-quantization + STE hooks.
merge        Cross-architecture / cross-scale weight merging, SLERP / TIES / DARE.
moe          Merge-of-experts router block for MMoE merges.
rl           DPO / GRPO objective math and configuration validation.
curriculum   Alternating curriculum co-training scheduler.
layer_lr     Layer-wise learning-rate zone resolution and param groups.
blending     Dynamic multi-dataset blending and ratio scheduling.
eval_harness Post-run evaluation plans, deltas, and reports.
splitter     VRAM/RAM-aware safetensors checkpoint splitting.
"""

from __future__ import annotations

__all__ = [
    "qat",
    "merge",
    "moe",
    "rl",
    "curriculum",
    "layer_lr",
    "blending",
    "eval_harness",
    "splitter",
]
