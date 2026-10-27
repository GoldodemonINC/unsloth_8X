# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Framework-aware safetensors access for the merge and split pipelines.

``safetensors``' numpy backend builds a numpy dtype from the stored dtype name,
and numpy has no bfloat16, so reading any bf16 checkpoint raised
``TypeError: data type 'bfloat16' not understood`` before a single tensor was
blended -- which is most released weights (DeepSeek-R1-Distill, VibeThinker,
every post-2024 fine-tune). Torch is already a hard dependency of Studio, so
tensors are read and written through the torch backend whenever it imports,
keeping bf16 and fp16 intact end to end. The numpy path stays as the fallback
for environments without torch.
"""

from __future__ import annotations

from typing import Any

_TORCH: Any = None
_TORCH_PROBED = False

_DTYPE_ITEMSIZES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}

def torch_module():
    """The imported torch module, or None when it is not installed."""
    global _TORCH, _TORCH_PROBED
    if not _TORCH_PROBED:
        _TORCH_PROBED = True
        try:
            import torch  # noqa: PLC0415 - optional at import time
        except ImportError:
            _TORCH = None
        else:
            _TORCH = torch
    return _TORCH

def read_framework() -> str:
    """The safetensors backend to read through: torch when available, else numpy."""
    return "pt" if torch_module() is not None else "np"

def open_file(path: str | Any):
    """Open a safetensors file in the framework that can represent every dtype."""
    from safetensors import safe_open  # noqa: PLC0415 - lazy, mirrors the rest of this module

    return safe_open(str(path), framework=read_framework())

def load_tensor(handle, key: str):
    """Read one tensor, naming the bf16 cause when only numpy is available."""
    try:
        return handle.get_tensor(key)
    except TypeError as error:
        if "bfloat16" in str(error):
            raise ValueError(
                f"Cannot read {key!r}: it is stored as bfloat16 and this environment "
                "has no torch, so safetensors can only expose it through numpy. "
                "Install torch in the Studio environment to merge or split bf16 checkpoints."
            ) from error
        raise

def dtype_name(handle, key: str) -> str:
    """The stored dtype of one tensor, read from the header without loading it."""
    try:
        return str(handle.get_slice(key).get_dtype()).upper()
    except Exception:  # noqa: BLE001 - unknown headers fall back below
        return ""

_FLOAT_DTYPES = frozenset({"F8_E4M3", "F8_E5M2", "F16", "BF16", "F32", "F64"})

def is_float_dtype_name(name: str) -> bool:
    """Whether a stored dtype name carries floating-point weights."""
    return str(name).upper() in _FLOAT_DTYPES

def is_float_dtype(handle, key: str) -> bool:
    """Whether one stored tensor carries floating-point weights."""
    return is_float_dtype_name(dtype_name(handle, key))

def shape_of(handle, key: str) -> tuple[int, ...]:
    """The stored shape of one tensor, from the header."""
    return tuple(int(dim) for dim in handle.get_slice(key).get_shape())

def empty_like_shape(shape, reference):
    """An unwritten tensor of *shape* in the reference tensor's storage dtype.

    Streaming blends fill this row by row, so the whole output tensor is the
    only thing resident besides the current chunk -- not one float32 copy per
    parent.
    """
    import numpy as np  # noqa: PLC0415

    torch = torch_module()
    if torch is not None and isinstance(reference, torch.Tensor):
        dtype = reference.dtype

        if getattr(dtype, "itemsize", 4) < 2:
            dtype = torch.bfloat16
        return torch.empty(tuple(int(dim) for dim in shape), dtype = dtype)
    return np.empty(tuple(int(dim) for dim in shape), dtype = reference.dtype)

def dtype_itemsize(handle, key: str) -> int:
    """Bytes per element of one tensor, read from the header without loading it."""
    try:
        name = str(handle.get_slice(key).get_dtype())
        return _DTYPE_ITEMSIZES.get(name.upper(), 4)
    except Exception:  # noqa: BLE001 - unknown dtypes fall back to fp32 sizing
        try:
            sample = handle.get_slice(key)[0]
            return int(getattr(sample, "itemsize", 4))
        except Exception:  # noqa: BLE001
            return 4

def to_float32(value):
    """A float32 numpy view of a tensor, whatever framework it came from."""
    import numpy as np  # noqa: PLC0415

    torch = torch_module()
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().to("cpu", torch.float32).numpy()
    return np.asarray(value, dtype=np.float32)

def is_floating(value) -> bool:
    """Whether a tensor carries floating-point weights."""
    import numpy as np  # noqa: PLC0415

    torch = torch_module()
    if torch is not None and isinstance(value, torch.Tensor):
        return bool(value.is_floating_point())
    return bool(np.issubdtype(np.asarray(value).dtype, np.floating))

def cast_like(value, reference):
    """Cast a float32 blend back to the source dtype, the way the pair was stored.

    Sub-byte float formats (fp8) are promoted to bf16 instead: rounding a blend
    back into 4 bits of mantissa would throw the merge away.
    """
    import numpy as np  # noqa: PLC0415

    torch = torch_module()
    if torch is not None and isinstance(reference, torch.Tensor):
        target = reference.dtype
        if getattr(target, "itemsize", 4) < 2:
            target = torch.bfloat16
        return torch.from_numpy(np.ascontiguousarray(value)).to(target)
    return np.asarray(value).astype(reference.dtype, copy=False)

def nbytes(value) -> int:
    """Byte size of a tensor in either framework."""
    torch = torch_module()
    if torch is not None and isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    return int(getattr(value, "nbytes", 0))

def save_tensors(payload: dict[str, Any], path: str) -> None:
    """Write a shard with the backend that matches the tensors it holds."""
    torch = torch_module()
    if torch is not None and payload and all(
        isinstance(value, torch.Tensor) for value in payload.values()
    ):
        from safetensors.torch import save_file  # noqa: PLC0415

        save_file({key: value.contiguous() for key, value in payload.items()}, str(path))
        return
    from safetensors.numpy import save_file  # noqa: PLC0415

    save_file(payload, str(path))
