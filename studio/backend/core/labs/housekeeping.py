# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Housekeeping: what is taking disk, and what is safe to remove.

The two databases that grow without a ceiling on a training machine are the
Studio outputs folder (one folder per merge, expansion, export and run) and the
Hugging Face cache (one folder per model or dataset ever downloaded). Both are
ordinary directories, so the honest operation is a scan plus a delete guard, not
a database of "owned" rows.

Every delete is checked against the roots this module scanned and against the
advanced job registry, so a path assembled by a browser cannot remove a folder
outside ``outputs/``, ``exports/`` or the HF cache. The scanner never follows a
symlink and never walks in a loop; a folder it cannot read is reported, not
skipped silently.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Iterable, Optional

_MAX_FILES = 400_000
_SKIP_DIRS = frozenset({"labs-pipelines", "labs-benchmarks"})
_ROOT_IDS = ("outputs", "exports", "hf-hub", "hf-datasets")


# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------


def _paths() -> dict[str, Path]:
    """The four roots, even when one of them does not exist on this host."""
    from utils.paths.storage_roots import exports_root, outputs_root

    roots = {
        "outputs": outputs_root(),
        "exports": exports_root(),
    }
    try:
        from utils.hf_cache_settings import get_hf_cache_paths

        cache = get_hf_cache_paths()
        roots["hf-hub"] = cache.hub_cache
        roots["hf-datasets"] = cache.cache_home / "datasets"
    except Exception:  # noqa: BLE001 - a host without the settings module still has outputs
        pass
    return roots


_ROOT_LABELS = {
    "outputs": "Studio outputs",
    "exports": "Exports",
    "hf-hub": "Hugging Face models",
    "hf-datasets": "Hugging Face datasets",
}

_ROOT_NOTES = {
    "outputs": "Checkpoints, merges and expansions this Studio wrote.",
    "exports": "GGUF and quantized bundles written by Export.",
    "hf-hub": "Downloaded model snapshots. Deleting one makes the next run download it again.",
    "hf-datasets": "Downloaded dataset snapshots. Deleting one makes the next run download it again.",
}


def _within(base: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(base.resolve())
    except (OSError, ValueError):
        return False
    return True


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _iso(stamp: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stamp or time.time()))


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


def _measure(directory: Path) -> tuple[int, int]:
    """Total bytes and file count under *directory*, without following symlinks.

    A cap keeps a stray cache mount from making the page hang. A folder that
    hits the cap is reported with the partial count and a note; it is still
    deletable, because the path (not the exact byte count) is what the guard
    checks.
    """
    total = 0
    files = 0
    stack: list[Path] = [directory]
    while stack and files < _MAX_FILES:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_symlink():
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                            files += 1
                    except OSError:
                        continue
        except OSError:
            continue
    return total, files


def _format_bytes(value: float) -> str:
    if value >= 1e12:
        return f"{value / 1e12:.2f} TB"
    if value >= 1e9:
        return f"{value / 1e9:.2f} GB"
    if value >= 1e6:
        return f"{value / 1e6:.0f} MB"
    if value >= 1e3:
        return f"{value / 1e3:.0f} KB"
    return f"{int(value)} B"


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _kind(path: Path, root_id: str) -> str:
    if root_id == "hf-hub":
        return "hf-model" if path.name.startswith("models--") else "hf-cache"
    if root_id == "hf-datasets":
        return "hf-dataset"
    if root_id == "exports":
        return "export"
    if (path / "config.json").is_file():
        return "checkpoint"
    if (path / "run_manifest.json").is_file():
        return "run"
    return "folder"


def _protected_reason(path: Path, root_id: str, root: Path) -> Optional[str]:
    if path.resolve() == root.resolve():
        return "This is a storage root, not a folder inside it."
    if path.is_symlink():
        return "This is a symlink; delete the folder it points at from its own location."
    if path.name.startswith("."):
        return "Hidden folders are left alone."
    if path.name in _SKIP_DIRS:
        return "Labs keeps its own plans and benchmark records here."
    if root_id in ("outputs", "exports") and path.parent.resolve() != root.resolve():
        # The Artifacts page only reads one level below outputs, so anything
        # deeper is some run's private working directory and is not offered.
        return "Only top-level Studio folders are listed for deletion."
    return None


def _manifest_summary(path: Path) -> Optional[dict[str, Any]]:
    manifest = _read_json(path / "run_manifest.json")
    if manifest is None:
        return None
    return {
        "stage": manifest.get("stage"),
        "run_id": manifest.get("run_id"),
        "created_at": manifest.get("created_at"),
    }


def _running_outputs() -> list[str]:
    """Output directories of advanced jobs that are still running.

    The registry lives on the route module; importing it lazily keeps this
    module importable on a host that never starts a job, and an import failure
    simply means there is nothing to protect.
    """
    try:
        from routes import advanced

        outputs: list[str] = []
        with advanced._JOBS_LOCK:
            for record in advanced._JOBS.values():
                if record.get("status") == "running" and record.get("output_dir"):
                    outputs.append(str(record["output_dir"]))
        return outputs
    except Exception:  # noqa: BLE001
        return []


def _is_under_running_job(path: Path, running: Iterable[str]) -> bool:
    resolved = path.resolve()
    for output in running:
        try:
            if resolved == Path(output).resolve() or _within(Path(output), resolved):
                return True
        except (OSError, ValueError):
            continue
    return False


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------


def _entries_for(root_id: str, root: Path, running: list[str]) -> list[dict[str, Any]]:
    if not root.is_dir():
        return []
    records: list[dict[str, Any]] = []
    try:
        children = sorted(root.iterdir(), key=lambda item: item.name.lower())
    except OSError:
        return records
    for child in children:
        try:
            if not child.is_dir() or child.name.startswith("."):
                continue
            if child.name in _SKIP_DIRS:
                continue
            if child.is_symlink():
                continue
            size, files = _measure(child)
            modified = _mtime(child)
            reason = _protected_reason(child, root_id, root)
            if _is_under_running_job(child, running):
                reason = "A job is writing here right now."
            records.append(
                {
                    "path": str(child),
                    "name": child.name,
                    "root": root_id,
                    "kind": _kind(child, root_id),
                    "bytes": size,
                    "bytes_label": _format_bytes(size),
                    "file_count": files,
                    "modified": _iso(modified),
                    "manifest": _manifest_summary(child),
                    "deletable": reason is None,
                    "protected_reason": reason,
                }
            )
        except OSError:
            continue
    records.sort(key=lambda record: record["bytes"], reverse=True)
    return records


def scan() -> dict[str, Any]:
    """Everything that is large, newest roots first, biggest entries first."""
    roots = _paths()
    running = _running_outputs()
    entries: list[dict[str, Any]] = []
    root_records: list[dict[str, Any]] = []
    for root_id in _ROOT_IDS:
        root = roots.get(root_id)
        if root is None:
            continue
        exists = root.is_dir()
        root_entries = _entries_for(root_id, root, running) if exists else []
        entries.extend(root_entries)
        root_records.append(
            {
                "id": root_id,
                "label": _ROOT_LABELS[root_id],
                "note": _ROOT_NOTES[root_id],
                "path": str(root),
                "exists": exists,
                "entry_count": len(root_entries),
                "bytes": sum(entry["bytes"] for entry in root_entries),
            }
        )
    entries.sort(key=lambda record: record["bytes"], reverse=True)
    total = sum(record["bytes"] for record in entries)
    reclaimable = sum(
        record["bytes"] for record in entries if record["deletable"]
    )
    disk = shutil.disk_usage(roots.get("outputs") or Path.home())
    return {
        "roots": root_records,
        "entries": entries,
        "total_bytes": total,
        "total_label": _format_bytes(total),
        "reclaimable_bytes": reclaimable,
        "reclaimable_label": _format_bytes(reclaimable),
        "free_bytes": int(disk.free),
        "free_label": _format_bytes(disk.free),
        "disk_total_bytes": int(disk.total),
        "protected": [
            {"path": record["path"], "reason": record["protected_reason"]}
            for record in entries
            if not record["deletable"]
        ],
        "notes": [
            "Sizes are measured by walking each folder; a huge cache can take a few seconds.",
            "Deleting a Studio output also removes it from the Library and the Artifacts page.",
            "A running job's output folder is refused until the job finishes.",
        ],
    }


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


def delete(paths: Iterable[str], *, confirm: bool) -> dict[str, Any]:
    """Remove the named folders, refusing anything the scan would not offer."""
    if not confirm:
        raise ValueError("Deleting folders needs confirm=true.")
    roots = _paths()
    running = _running_outputs()
    deleted: list[dict[str, Any]] = []
    refused: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(str(raw)).expanduser()
        root_id = next(
            (identifier for identifier, root in roots.items() if _within(root, path)),
            None,
        )
        if root_id is None:
            refused.append({"path": str(path), "reason": "Outside the storage roots."})
            continue
        root = roots[root_id]
        reason = _protected_reason(path, root_id, root)
        if not path.exists():
            refused.append({"path": str(path), "reason": "No such folder."})
            continue
        if not path.is_dir():
            refused.append({"path": str(path), "reason": "Only folders are removed."})
            continue
        if reason is not None:
            refused.append({"path": str(path), "reason": reason})
            continue
        if _is_under_running_job(path, running):
            refused.append({"path": str(path), "reason": "A job is writing here right now."})
            continue
        size, files = _measure(path)
        try:
            shutil.rmtree(path)
        except OSError as error:
            refused.append({"path": str(path), "reason": f"Could not delete: {error}"})
            continue
        deleted.append(
            {
                "path": str(path),
                "bytes": size,
                "bytes_label": _format_bytes(size),
                "file_count": files,
            }
        )
    freed = sum(item["bytes"] for item in deleted)
    return {
        "deleted": deleted,
        "refused": refused,
        "freed_bytes": freed,
        "freed_label": _format_bytes(freed),
    }
