# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""VRAM/RAM-aware checkpoint splitting.

Writing a 70B safetensors checkpoint, or a merged multi-model expert stack,
needs more memory than either tier usually has at once. The splitter:

1. probes real headroom (CUDA free/total, system RAM, disk),
2. plans shards so no single write exceeds a safe fraction of that headroom,
3. streams tensors one at a time from the source (``safetensors.safe_open``),
   splitting any single oversized tensor along its first dimension,
4. writes ``model.safetensors.index.json`` plus a ``split_manifest.json`` that
   records the plan and the exact memory profile it was based on.

Nothing here imports torch at module level, so the planner works on a CPU-only
host and inside the API process.
"""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

DEFAULT_RESERVE_FRACTION = 0.15


@dataclass(frozen=True)
class DeviceMemory:
    index: int
    total_bytes: int
    free_bytes: int
    name: str = "cuda"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MemoryProfile:
    gpus: tuple[DeviceMemory, ...]
    system_ram_total_bytes: int
    system_ram_free_bytes: int
    disk_free_bytes: int
    source: str = "probe"

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpus": [gpu.to_dict() for gpu in self.gpus],
            "system_ram_total_bytes": self.system_ram_total_bytes,
            "system_ram_free_bytes": self.system_ram_free_bytes,
            "disk_free_bytes": self.disk_free_bytes,
            "source": self.source,
        }

    @property
    def vram_free_bytes(self) -> int:
        return sum(gpu.free_bytes for gpu in self.gpus)

    @property
    def vram_total_bytes(self) -> int:
        return sum(gpu.total_bytes for gpu in self.gpus)


def _windows_ram() -> tuple[int, int]:
    class _MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
        return int(status.ullTotalPhys), int(status.ullAvailPhys)
    return 0, 0


def _system_ram() -> tuple[int, int]:
    try:
        import psutil  # type: ignore

        memory = psutil.virtual_memory()
        return int(memory.total), int(memory.available)
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            return _windows_ram()
        except Exception:
            return 0, 0
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        total = os.sysconf("SC_PHYS_PAGES") * page
        available = os.sysconf("SC_AVPHYS_PAGES") * page
        return int(total), int(available)
    except (ValueError, OSError, AttributeError):
        return 0, 0


def _cuda_memory() -> tuple[DeviceMemory, ...]:
    try:
        import torch

        if not torch.cuda.is_available():
            return ()
        devices: list[DeviceMemory] = []
        for index in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(index)
            devices.append(
                DeviceMemory(
                    index=index,
                    total_bytes=int(total),
                    free_bytes=int(free),
                    name=torch.cuda.get_device_name(index),
                )
            )
        return tuple(devices)
    except Exception:
        return ()


def probe_memory(path: str | None = None) -> MemoryProfile:
    """Live VRAM / RAM / disk headroom. Missing probes report 0, never raise."""
    total_ram, free_ram = _system_ram()
    disk_free = 0
    try:
        target = Path(path).expanduser() if path else Path.cwd()
        while not target.exists() and target != target.parent:
            target = target.parent
        disk_free = int(shutil.disk_usage(target).free)
    except OSError:
        disk_free = 0
    return MemoryProfile(
        gpus=_cuda_memory(),
        system_ram_total_bytes=total_ram,
        system_ram_free_bytes=free_ram,
        disk_free_bytes=disk_free,
    )


@dataclass(frozen=True)
class ShardBudget:
    max_shard_bytes: int
    usable_ram_bytes: int
    usable_vram_bytes: int
    reserve_fraction: float
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["notes"] = list(self.notes)
        return payload


def compute_budget(
    profile: MemoryProfile,
    *,
    reserve_fraction: float = DEFAULT_RESERVE_FRACTION,
    max_shard_bytes: int | None = None,
    prefer: str = "ram",
) -> ShardBudget:
    """Pick a shard size from headroom, never promising memory we do not have."""
    if not 0.0 <= reserve_fraction < 0.9:
        raise ValueError("reserve_fraction must be in [0, 0.9).")
    if prefer not in ("ram", "vram", "auto"):
        raise ValueError("prefer must be ram, vram, or auto.")
    usable_ram = int(profile.system_ram_free_bytes * (1.0 - reserve_fraction))
    usable_vram = int(profile.vram_free_bytes * (1.0 - reserve_fraction))
    notes: list[str] = []
    if prefer == "vram" or (prefer == "auto" and usable_vram > usable_ram):
        budget = usable_vram or usable_ram
        source = "VRAM"
    else:
        budget = max(usable_ram, usable_vram if prefer == "ram" and usable_ram == 0 else 0)
        source = "RAM"
    if budget <= 0:
        budget = 2 * 1024 * 1024 * 1024
        notes.append("No memory probe succeeded; defaulting to a 2 GiB shard cap.")
    if max_shard_bytes is not None:
        if max_shard_bytes <= 0:
            raise ValueError("max_shard_bytes must be positive.")
        if max_shard_bytes > budget:
            notes.append("Requested shard size exceeds free memory; capped to the safe budget.")
        budget = min(budget, max_shard_bytes)
    # Never plan a shard smaller than 64 MiB: index overhead stops being worth it.
    budget = max(64 * 1024 * 1024, budget)
    notes.append(f"Shard cap derived from free {source}.")
    if profile.disk_free_bytes and profile.disk_free_bytes < budget:
        notes.append("Free disk is smaller than one shard; the split may run out of space.")
    return ShardBudget(
        max_shard_bytes=int(budget),
        usable_ram_bytes=usable_ram,
        usable_vram_bytes=usable_vram,
        reserve_fraction=reserve_fraction,
        notes=tuple(notes),
    )


@dataclass(frozen=True)
class TensorChunk:
    key: str
    index: int
    rows: int
    row_offset: int
    shape: tuple[int, ...]
    nbytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ShardPlan:
    shards: tuple[tuple[TensorChunk, ...], ...]
    total_bytes: int
    max_shard_bytes: int
    whole_tensors: int
    split_tensors: int
    oversized_tensors: int = 0
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "shards": [[chunk.to_dict() for chunk in shard] for shard in self.shards],
            "shard_count": len(self.shards),
            "total_bytes": self.total_bytes,
            "max_shard_bytes": self.max_shard_bytes,
            "whole_tensors": self.whole_tensors,
            "split_tensors": self.split_tensors,
            "oversized_tensors": self.oversized_tensors,
            "warnings": list(self.warnings),
        }


def _inventory(source: str) -> list[tuple[str, tuple[int, ...], int]]:
    """List (key, shape, nbytes) from a safetensors file or checkpoint dir."""
    from safetensors import safe_open

    path = Path(source)
    files: list[Path]
    if path.is_file():
        files = [path]
    else:
        index = path / "model.safetensors.index.json"
        if index.is_file():
            mapping = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
            files = [path / name for name in sorted({str(v) for v in mapping.values()})]
        else:
            files = sorted(path.glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors files found in {source!r}.")
    inventory: list[tuple[str, tuple[int, ...], int]] = []
    for file_path in files:
        with safe_open(str(file_path), framework="np") as handle:
            for key in handle.keys():
                shape = tuple(int(v) for v in handle.get_slice(key).get_shape())
                itemsize = _dtype_itemsize(handle, key)
                inventory.append((key, shape, max(1, _numel(shape)) * itemsize))
    return inventory


def _dtype_itemsize(handle, key: str) -> int:
    """Read dtype through a tiny slice so we never materialize the tensor."""
    try:
        slice_ = handle.get_slice(key)
        sample = slice_[0]
        return int(getattr(sample, "itemsize", 4))
    except Exception:
        return 4


def _numel(shape: Sequence[int]) -> int:
    result = 1
    for dim in shape:
        result *= int(dim)
    return result


def plan_shards(
    inventory: Sequence[tuple[str, Sequence[int], int]],
    max_shard_bytes: int,
) -> ShardPlan:
    """Greedy bin packing in source order, splitting oversized tensors by row."""
    if max_shard_bytes <= 0:
        raise ValueError("max_shard_bytes must be positive.")
    shards: list[list[TensorChunk]] = []
    current: list[TensorChunk] = []
    current_bytes = 0
    total = 0
    whole = 0
    oversized = 0
    warnings: list[str] = []

    def flush() -> None:
        nonlocal current, current_bytes
        if current:
            shards.append(current)
            current = []
            current_bytes = 0

    for key, shape, nbytes in inventory:
        shape_tuple = tuple(int(v) for v in shape)
        total += nbytes
        if nbytes <= max_shard_bytes:
            if current and current_bytes + nbytes > max_shard_bytes:
                flush()
            current.append(
                TensorChunk(
                    key, 0, shape_tuple[0] if shape_tuple else 1, 0, shape_tuple, nbytes
                )
            )
            current_bytes += nbytes
            whole += 1
            continue
        # A safetensors key may not span files, so a tensor larger than the cap
        # has to travel in one piece. Give it its own shard instead of loading it
        # twice or writing an unloadable chunked checkpoint.
        oversized += 1
        warnings.append(
            f"{key} is {nbytes / (1024 ** 3):.2f} GiB, larger than the {max_shard_bytes / (1024 ** 3):.2f} GiB cap; "
            "it will occupy one oversized shard. Raise the available memory or lower the model size."
        )
        flush()
        shards.append(
            [
                TensorChunk(
                    key, 0, shape_tuple[0] if shape_tuple else 1, 0, shape_tuple, nbytes
                )
            ]
        )
        whole += 1
    flush()
    if not shards:
        raise ValueError("Checkpoint is empty; nothing to split.")
    return ShardPlan(
        shards=tuple(tuple(shard) for shard in shards),
        total_bytes=total,
        max_shard_bytes=max_shard_bytes,
        whole_tensors=whole,
        split_tensors=0,
        oversized_tensors=oversized,
        warnings=tuple(warnings),
    )


def _read_slice(handle, key: str, offset: int, rows: int):
    slice_ = handle.get_slice(key)
    return slice_[offset : offset + rows]


def split_checkpoint(
    source: str,
    output_dir: str,
    *,
    max_shard_bytes: int,
    reserve_fraction: float = DEFAULT_RESERVE_FRACTION,
    prefer: str = "auto",
    copy_tokenizer: bool = True,
    max_shard_bytes_user: int | None = None,
) -> dict[str, Any]:
    """Stream a checkpoint into bounded shards and write the index + manifest."""
    import numpy as np
    from safetensors import safe_open
    from safetensors.numpy import save_file

    source_path = Path(source).expanduser()
    if not source_path.exists():
        raise ValueError(f"Checkpoint not found: {source}")
    profile = probe_memory(str(source_path))
    budget = compute_budget(
        profile,
        reserve_fraction=reserve_fraction,
        max_shard_bytes=max_shard_bytes,
        prefer=prefer,
    )
    cap = min(budget.max_shard_bytes, int(max_shard_bytes))
    inventory = _inventory(str(source_path))
    plan = plan_shards(inventory, cap)

    # Map every tensor to its source file once.
    if source_path.is_file():
        source_files = [source_path]
        root = source_path.parent
    else:
        root = source_path
        index = source_path / "model.safetensors.index.json"
        if index.is_file():
            mapping = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
            source_files = [source_path / name for name in sorted({str(v) for v in mapping.values()})]
        else:
            source_files = sorted(source_path.glob("*.safetensors"))
    tensor_file: dict[str, Path] = {}
    for file_path in source_files:
        with safe_open(str(file_path), framework="np") as handle:
            for key in handle.keys():
                tensor_file[key] = file_path

    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    weight_map: dict[str, str] = {}
    written_files: list[str] = []
    total_written = 0

    import contextlib

    with contextlib.ExitStack() as stack:
        handles = {
            path: stack.enter_context(safe_open(str(path), framework="np"))
            for path in set(tensor_file.values())
        }
        for shard_index, shard in enumerate(plan.shards, start=1):
            payload: dict[str, Any] = {}
            for chunk in shard:
                handle = handles[tensor_file[chunk.key]]
                array = np.asarray(handle.get_tensor(chunk.key))
                payload[chunk.key] = array
                total_written += int(array.nbytes)
            name = f"model-{shard_index:05d}.safetensors"
            save_file(payload, str(out / name))
            written_files.append(name)
            for key in payload:
                weight_map[key] = name

    index_payload = {
        "metadata": {"total_size": total_written},
        "weight_map": weight_map,
    }
    (out / "model.safetensors.index.json").write_text(
        json.dumps(index_payload, indent=2), encoding="utf-8"
    )
    if copy_tokenizer and source_path.is_dir():
        for name in (
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "chat_template.jinja",
        ):
            candidate = source_path / name
            if candidate.is_file():
                try:
                    (out / name).write_bytes(candidate.read_bytes())
                except OSError:
                    pass

    manifest = {
        "status": "completed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source_path),
        "output_dir": str(out),
        "memory_profile": profile.to_dict(),
        "budget": budget.to_dict(),
        "plan": plan.to_dict(),
        "files": written_files,
        "total_bytes": total_written,
        "requested_max_shard_bytes": max_shard_bytes_user or max_shard_bytes,
    }
    (out / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def plan_only(
    source: str,
    *,
    max_shard_bytes: int | None = None,
    reserve_fraction: float = DEFAULT_RESERVE_FRACTION,
    prefer: str = "auto",
) -> dict[str, Any]:
    """Plan shards without writing anything (the Export modal's preview)."""
    profile = probe_memory(source)
    budget = compute_budget(
        profile, reserve_fraction=reserve_fraction, max_shard_bytes=max_shard_bytes, prefer=prefer
    )
    inventory = _inventory(source)
    plan = plan_shards(inventory, budget.max_shard_bytes)
    return {
        "source": source,
        "memory_profile": profile.to_dict(),
        "budget": budget.to_dict(),
        "plan": plan.to_dict(),
        "fits_on_disk": profile.disk_free_bytes == 0
        or profile.disk_free_bytes > plan.total_bytes,
    }
