# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Quantization-aware training (QAT) engine.

QAT keeps the *simulation* of low-bit weights inside the forward graph. Each
weight is fake-quantized (round + clamp to a low-bit grid) before the matmul,
while the gradient is passed straight through to the full-precision weight with
a straight-through estimator (STE)::

    w_ste = w + (fake_quant(w) - w).detach()

The optimizer therefore updates the high-precision master weight, but the loss
it descends already contains the quantization noise the exported checkpoint
will have. That is what lets the run learn weights that compensate for it,
instead of dequantizing to BF16 for the whole run and quantizing once at the end.

Supported grids
---------------
int4   Group-wise symmetric integer grid, q in [-8, 7], one absmax scale per group.
nf4    4-bit NormalFloat quantiles (the QLoRA grid), same group-wise scaling.
nvfp4  NVIDIA FP4: E2M1 values, one FP8 (E4M3) scale per 16-element block, and an
       FP32 per-tensor scale. This mirrors the NVFP4 checkpoint format.

All helpers accept numpy arrays (used by planning/tests) and torch tensors
(used inside the training hooks).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

QAT_FORMATS = ("int4", "nf4", "nvfp4")
_GROUP_SIZES = (16, 32, 64, 128, 256)


@dataclass(frozen=True)
class QatConfig:
    enabled: bool = False
    format: str = "int4"
    group_size: int = 128
    # Per-tensor scale for nvfp4; ignored by the integer grids.
    tensor_scale: float | None = None
    # Layer-name fragments that keep full precision (embeddings, norms, lm_head).
    skip: tuple[str, ...] = ("lm_head", "embed_tokens", "wte", "norm")
    min_numel: int = 4096
    # Straight-through is the point of QAT; False is for diagnostics only.
    straight_through: bool = True


def validate_qat_config(config: dict[str, Any] | None) -> QatConfig:
    """Coerce an API payload into a QatConfig, rejecting unsafe values."""
    config = config or {}
    fmt = str(config.get("format", "int4")).lower()
    if fmt not in QAT_FORMATS:
        raise ValueError(
            f"Unsupported QAT format {fmt!r}; expected one of {', '.join(QAT_FORMATS)}."
        )
    group_size = int(config.get("group_size", 128))
    if group_size not in _GROUP_SIZES:
        raise ValueError(
            f"QAT group_size must be one of {_GROUP_SIZES}; got {group_size}."
        )
    skip = config.get("skip")
    if skip is None:
        skip = QatConfig.skip
    if not isinstance(skip, (list, tuple)) or not all(isinstance(s, str) for s in skip):
        raise ValueError("QAT skip must be a list of layer-name fragments.")
    tensor_scale = config.get("tensor_scale")
    if tensor_scale is not None:
        tensor_scale = float(tensor_scale)
        if not math.isfinite(tensor_scale) or tensor_scale <= 0:
            raise ValueError("QAT tensor_scale must be a positive finite number.")
    min_numel = int(config.get("min_numel", 4096))
    if min_numel < 0:
        raise ValueError("QAT min_numel must be non-negative.")
    return QatConfig(
        enabled=bool(config.get("enabled", False)),
        format=fmt,
        group_size=group_size,
        tensor_scale=tensor_scale,
        skip=tuple(skip),
        min_numel=min_numel,
        straight_through=bool(config.get("straight_through", True)),
    )


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------

# E2M1: 4-bit float with 1 sign, 2 exponent, 1 mantissa bit. The representable
# magnitudes are these eight values; 6 is the largest.
NVFP4_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

# NF4 (Dettmers et al.): 16 quantiles of a unit normal, symmetric around zero.
NF4_LEVELS = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.09105003625154495,
    0.18477343022823334,
    0.28444138169288635,
    0.39491748809814453,
    0.5250730514526367,
    0.6961928009986877,
    1.0,
)


def _padded_groups(values, group_size: int):
    """Reshape a flat sequence into [n_groups, group_size] with zero padding."""
    import numpy as np

    total = int(np.prod(values.shape))
    remainder = total % group_size
    flat = values.reshape(-1)
    if remainder:
        pad = np.zeros((group_size - remainder,), dtype=flat.dtype)
        flat = np.concatenate([flat, pad])
    return flat.reshape(-1, group_size), total, remainder


def _quantize_numpy(values, fmt: str, group_size: int, tensor_scale: float | None = None):
    import numpy as np

    x = np.asarray(values, dtype=np.float32)
    groups, total, remainder = _padded_groups(x, group_size)
    absmax = np.max(np.abs(groups), axis=1, keepdims=True)
    eps = np.finfo(np.float32).tiny

    if fmt == "int4":
        scale = np.maximum(absmax / 7.0, eps)
        q = np.clip(np.rint(groups / scale), -8.0, 7.0)
        deq = q * scale
        scale_out = scale[:, 0]
    elif fmt == "nf4":
        # NF4's grid lives in [-1, 1]; absmax maps onto the outer quantile.
        scale = np.maximum(absmax, eps)
        normalized = groups / scale
        levels = np.asarray(NF4_LEVELS, dtype=np.float32)
        idx = np.abs(normalized[..., None] - levels).argmin(axis=-1)
        deq = levels[idx] * scale
        scale_out = scale[:, 0]
    elif fmt == "nvfp4":
        # Two levels of scaling, exactly as the NVFP4 format stores them: an
        # FP32 per-tensor scale and an FP8 E4M3 scale per 16-element block.
        global_absmax = float(np.max(np.abs(x))) if x.size else 0.0
        tensor_scale = (
            float(tensor_scale)
            if tensor_scale is not None
            else max(global_absmax / 6.0, eps)
        )
        block_scale = np.maximum(absmax / (6.0 * tensor_scale), eps)
        block_scale = _round_e4m3_numpy(block_scale)
        effective_scale = np.maximum(block_scale * tensor_scale, eps)
        normalized = groups / effective_scale
        levels = np.asarray(NVFP4_LEVELS, dtype=np.float32)
        magnitudes = np.abs(normalized)
        signs = np.sign(normalized)
        mag_idx = np.abs(magnitudes[..., None] - levels).argmin(axis=-1)
        deq = signs * levels[mag_idx] * effective_scale
        scale_out = block_scale[:, 0]
    else:  # pragma: no cover - validate_qat_config rejects this
        raise ValueError(f"Unsupported QAT format {fmt!r}")

    flat = deq.reshape(-1)
    if remainder:
        flat = flat[:total]
    return flat.reshape(x.shape), scale_out


def _round_e4m3_numpy(values):
    """Round to FP8 E4M3 (3 mantissa bits, exponent range [-6, 8]) in numpy."""
    import numpy as np

    x = np.asarray(values, dtype=np.float32)
    sign = np.sign(x)
    magnitudes = np.abs(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        exponent = np.floor(np.log2(np.maximum(magnitudes, 1e-30)))
    exponent = np.clip(exponent, -6.0, 8.0)
    step = np.power(2.0, exponent - 3.0)
    rounded = np.rint(magnitudes / step) * step
    # E4M3's largest finite value is 448.
    return np.where(magnitudes > 448.0, 448.0, sign * rounded).astype(np.float32)


def fake_quantize(values, config: QatConfig | dict[str, Any] | None = None, *, fmt: str | None = None):
    """Fake-quantize a numpy array or torch tensor and return the same type."""
    cfg = config if isinstance(config, QatConfig) else validate_qat_config(config)
    chosen = fmt or cfg.format
    if hasattr(values, "detach") and hasattr(values, "dtype"):
        return _fake_quantize_torch(values, chosen, cfg.group_size, cfg.tensor_scale)
    return _quantize_numpy(values, chosen, cfg.group_size, cfg.tensor_scale)[0]


def _fake_quantize_torch(tensor, fmt: str, group_size: int, tensor_scale):
    import torch

    x = tensor.float()
    shape = x.shape
    flat = x.reshape(-1)
    total = flat.numel()
    remainder = total % group_size
    if remainder:
        flat = torch.cat([flat, flat.new_zeros(group_size - remainder)])
    groups = flat.reshape(-1, group_size)
    absmax = groups.abs().amax(dim=1, keepdim=True)
    eps = torch.finfo(torch.float32).tiny

    if fmt == "int4":
        scale = (absmax / 7.0).clamp_min(eps)
        q = torch.clamp(torch.round(groups / scale), -8.0, 7.0)
        deq = q * scale
    elif fmt == "nf4":
        scale = absmax.clamp_min(eps)
        levels = torch.tensor(NF4_LEVELS, dtype=torch.float32, device=x.device)
        normalized = groups / scale
        idx = (normalized.unsqueeze(-1) - levels).abs().argmin(dim=-1)
        deq = levels[idx] * scale
    elif fmt == "nvfp4":
        global_absmax = float(x.abs().max()) if x.numel() else 0.0
        effective_tensor_scale = (
            float(tensor_scale)
            if tensor_scale is not None
            else max(global_absmax / 6.0, eps)
        )
        scale = (absmax / (6.0 * effective_tensor_scale)).clamp_min(eps)
        scale = _round_e4m3_torch(scale)
        effective_scale = (scale * effective_tensor_scale).clamp_min(eps)
        levels = torch.tensor(NVFP4_LEVELS, dtype=torch.float32, device=x.device)
        normalized = groups / effective_scale
        magnitudes = normalized.abs()
        signs = torch.sign(normalized)
        idx = (magnitudes.unsqueeze(-1) - levels).abs().argmin(dim=-1)
        deq = signs * levels[idx] * effective_scale
    else:  # pragma: no cover
        raise ValueError(f"Unsupported QAT format {fmt!r}")

    flat = deq.reshape(-1)
    if remainder:
        flat = flat[:total]
    return flat.reshape(shape).to(tensor.dtype)


def _round_e4m3_torch(values):
    import torch

    try:
        return values.to(torch.float8_e4m3fn).to(torch.float32)
    except Exception:  # pragma: no cover - CPU builds without fp8 casts
        return values.clamp(-448.0, 448.0)


def straight_through_quantize(weight, config: QatConfig):
    """Fake-quantize with a detached residual so gradients reach `weight`."""
    quantized = _fake_quantize_torch(weight, config.format, config.group_size, config.tensor_scale)
    if not config.straight_through:
        return quantized
    return weight + (quantized - weight).detach()


def quantization_error(values, config: QatConfig | dict[str, Any] | None = None) -> dict[str, float]:
    """Mean absolute / relative error of a fake-quantized copy, for diagnostics."""
    import numpy as np

    cfg = config if isinstance(config, QatConfig) else validate_qat_config(config)
    original = np.asarray(values, dtype=np.float32)
    quantized = np.asarray(fake_quantize(original, cfg), dtype=np.float32)
    abs_err = float(np.mean(np.abs(original - quantized)))
    denom = float(np.mean(np.abs(original))) or 1.0
    return {
        "mean_abs_error": abs_err,
        "mean_rel_error": abs_err / denom,
        "max_abs_error": float(np.max(np.abs(original - quantized))),
    }


# ---------------------------------------------------------------------------
# Torch module hooks
# ---------------------------------------------------------------------------


@dataclass
class QatSession:
    """Installed fake-quant hooks; call `remove()` to restore the model."""

    config: QatConfig
    modules: list[Any] = field(default_factory=list)
    _originals: list[tuple[Any, Any]] = field(default_factory=list)

    def remove(self) -> None:
        for module, original in self._originals:
            module.forward = original
        self._originals.clear()
        self.modules.clear()

    def __enter__(self):  # pragma: no cover - convenience
        return self

    def __exit__(self, *_exc):  # pragma: no cover - convenience
        self.remove()


def _should_quantize(name: str, module, cfg: QatConfig) -> bool:
    import torch

    if not isinstance(module, torch.nn.Linear):
        return False
    weight = getattr(module, "weight", None)
    if weight is None or weight.ndim != 2:
        return False
    if weight.numel() < cfg.min_numel:
        return False
    lowered = name.lower()
    return not any(fragment.lower() in lowered for fragment in cfg.skip)


def apply_qat(model, config: QatConfig | dict[str, Any] | None = None) -> QatSession:
    """Install fake-quantization on every eligible Linear in `model`.

    The hook replaces ``module.forward`` rather than mutating the weight, so the
    optimizer keeps updating full-precision parameters and the checkpoint is
    saved un-quantized (ready for the normal export/quantize step).
    """
    import torch.nn.functional as F

    cfg = config if isinstance(config, QatConfig) else validate_qat_config(config)
    if not cfg.enabled:
        return QatSession(config=cfg)

    session = QatSession(config=cfg)
    for name, module in model.named_modules():
        if not _should_quantize(name, module, cfg):
            continue
        session.modules.append(module)
        original = module.forward

        def _forward(x, *args, _module=module, _cfg=cfg, **kwargs):
            weight = straight_through_quantize(_module.weight, _cfg)
            return F.linear(x, weight, _module.bias)

        session._originals.append((module, original))
        module.forward = _forward
    return session


def qat_summary(session: QatSession) -> dict[str, Any]:
    return {
        "enabled": bool(session.modules),
        "format": session.config.format,
        "group_size": session.config.group_size,
        "layers": len(session.modules),
    }
