# SPDX-License-Identifier: NRC-3
# Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-3.md

"""The Artifacts studio: what this Studio wrote, and what each folder allows next.

A finished run is a directory, not an object. This module turns the checkpoint
folders under ``outputs/`` back into records with a provenance line (the stage
that wrote them and the ``run_manifest.json`` beside them) and a capability
list. The capability list is the point: expanding a LoRA adapter into a dense
checkpoint is not a thing the Scaler can do, and a menu that offers it is worse
than one that explains why not.

Nothing here imports torch. Classification reads ``config.json``, the file
names and the manifest; sizes come from ``stat``. That is what lets the page
open on a machine that cannot load any of the checkpoints it lists.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from shared.scaling import (
    CAP_ADDED_PARAMS,
    GROWTH_FLOOR,
    MOE_CONFIG_KEYS,
    SUPPORTED_MODEL_TYPES,
    _base_params,
)

MANIFEST_NAME = "run_manifest.json"
_MAX_CANDIDATES = 400


# ---------------------------------------------------------------------------
# Reading a folder
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _read_config(directory: Path) -> Optional[dict[str, Any]]:
    return _read_json(directory / "config.json")


def _weights_bytes(directory: Path) -> int:
    total = 0
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if not entry.is_file():
                    continue
                if not entry.name.endswith((".safetensors", ".bin", ".gguf")):
                    continue
                try:
                    total += entry.stat().st_size
                except OSError:
                    continue
    except OSError:
        return 0
    return total


def _manifest_summary(directory: Path) -> tuple[Optional[dict[str, Any]], bool]:
    path = directory / MANIFEST_NAME
    if not path.is_file():
        return None, False
    manifest = _read_json(path)
    if manifest is None:
        return None, True
    summary = {
        "stage": manifest.get("stage"),
        "run_id": manifest.get("run_id"),
        "created_at": manifest.get("created_at"),
        "studio_version": manifest.get("studio_version"),
    }
    return summary, True


def _modified(directory: Path) -> str:
    try:
        stamp = directory.stat().st_mtime
    except OSError:
        stamp = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stamp))


# ---------------------------------------------------------------------------
# Classification and capabilities
# ---------------------------------------------------------------------------


def _action(action_id: str, enabled: bool, reason: Optional[str] = None) -> dict[str, Any]:
    return {"id": action_id, "enabled": bool(enabled), "reason": None if enabled else reason}


def _actions(
    *,
    adapter: bool,
    quantized: bool,
    moe: bool,
    supported: bool,
    params: Optional[int],
    model_type: Optional[str],
) -> list[dict[str, Any]]:
    """Every next step, with the reason it is unavailable when it is."""
    family = model_type or "this model type"
    dense_reason = None
    if adapter:
        dense_reason = (
            "This is a LoRA adapter. Merge it into its base first, then scale the merged model."
        )
    elif quantized:
        dense_reason = (
            "This checkpoint is quantized. Expansion and merging need the fp16/bf16 original."
        )
    elif moe:
        dense_reason = (
            "Mixture-of-Experts checkpoints are not supported by expansion or merging yet."
        )
    elif not supported:
        dense_reason = f"Expansion is not implemented for {family}."
    elif not params:
        dense_reason = "config.json does not name a parameter count, so no target can be planned."

    scale_enabled = dense_reason is None
    if scale_enabled and params and params * GROWTH_FLOOR > params + CAP_ADDED_PARAMS:
        scale_enabled = False
        dense_reason = (
            f"This checkpoint is already past the {CAP_ADDED_PARAMS / 1e9:.0f}B growth ceiling; "
            "2x would exceed it."
        )

    return [
        _action("continue-training", True, None),
        _action("scale", scale_enabled, dense_reason),
        _action("merge", dense_reason is None, dense_reason),
        _action(
            "quantize",
            not adapter,
            "An adapter is not a standalone model; merge it into its base before exporting."
            if adapter
            else None,
        ),
        _action(
            "benchmark",
            supported and not adapter,
            "Benchmarks score a merged checkpoint; merge this adapter into its base first."
            if adapter
            else None,
        ),
    ]


def describe(directory: Path) -> dict[str, Any]:
    """One artifact record, computed from config and file names only."""
    config = _read_config(directory) or {}
    adapter = (directory / "adapter_config.json").is_file()
    quantized = config.get("quantization_config") is not None
    model_type = config.get("model_type") or config.get("architecture")
    model_type = str(model_type) if model_type else None
    moe = any(config.get(key) is not None for key in MOE_CONFIG_KEYS) or bool(
        model_type and "moe" in model_type.lower()
    )
    supported = bool(model_type) and str(model_type) in SUPPORTED_MODEL_TYPES
    params = _base_params(config) if config else None
    manifest, manifest_present = _manifest_summary(directory)
    size = _weights_bytes(directory)
    return {
        "path": str(directory),
        "name": directory.name,
        "modified": _modified(directory),
        "model_type": model_type,
        "params": params,
        "weight_bytes": size,
        "adapter": adapter,
        "quantized": quantized,
        "moe": moe,
        "supported": supported,
        "manifest": manifest,
        "manifest_path": str(directory / MANIFEST_NAME) if manifest_present else None,
        "actions": _actions(
            adapter=adapter,
            quantized=quantized,
            moe=moe,
            supported=supported,
            params=params,
            model_type=model_type,
        ),
    }


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def root() -> Path:
    from shared.paths import outputs_root

    return outputs_root()


def _within(base: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(base.resolve())
    except (OSError, ValueError):
        return False
    return True


def _is_checkpoint(directory: Path) -> bool:
    return (directory / "config.json").is_file()


def candidates(base: Path) -> list[Path]:
    """Checkpoint folders under outputs/, two levels deep, newest first."""
    found: list[Path] = []
    try:
        children = sorted(base.iterdir(), key=lambda item: item.name)
    except OSError:
        return found
    for child in children:
        if len(found) >= _MAX_CANDIDATES:
            break
        if not child.is_dir() or child.name.startswith("."):
            continue
        if child.name == "labs-pipelines":
            continue
        if _is_checkpoint(child):
            found.append(child)
            continue
        try:
            grandchildren = sorted(child.iterdir(), key=lambda item: item.name)
        except OSError:
            continue
        for grandchild in grandchildren:
            if len(found) >= _MAX_CANDIDATES:
                break
            if grandchild.is_dir() and not grandchild.name.startswith("."):
                if _is_checkpoint(grandchild):
                    found.append(grandchild)
    found.sort(key=lambda item: _mtime(item), reverse=True)
    return found


def _mtime(directory: Path) -> float:
    try:
        return directory.stat().st_mtime
    except OSError:
        return 0.0


def scan(limit: int = 50) -> list[dict[str, Any]]:
    """Artifacts under outputs/, newest first. Unreadable folders are skipped."""
    base = root()
    records: list[dict[str, Any]] = []
    for directory in candidates(base):
        if len(records) >= max(1, min(int(limit), 200)):
            break
        try:
            records.append(describe(directory))
        except Exception:  # noqa: BLE001 - one broken folder must not hide the rest
            continue
    return records


def find(path: str) -> dict[str, Any]:
    """Describe one artifact by path. Only folders this Studio wrote are allowed."""
    base = root()
    directory = Path(path).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(f"No such checkpoint folder: {path}")
    if not _within(base, directory):
        raise PermissionError(f"{path} is outside the Studio output folder.")
    if not _is_checkpoint(directory):
        raise FileNotFoundError(f"{path} has no config.json, so it is not a checkpoint folder.")
    return describe(directory)


def read_manifest(path: str) -> dict[str, Any]:
    """The raw run manifest beside one artifact."""
    base = root()
    directory = Path(path).expanduser()
    if not directory.is_dir() or not _within(base, directory):
        raise PermissionError(f"{path} is outside the Studio output folder.")
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No {MANIFEST_NAME} in {path}.")
    manifest = _read_json(manifest_path)
    if manifest is None:
        raise ValueError(f"{manifest_path} is not valid JSON.")
    return {"path": str(manifest_path), "manifest": manifest}
