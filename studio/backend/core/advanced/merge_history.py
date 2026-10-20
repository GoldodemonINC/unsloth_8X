# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Recent merges, read back from the manifests the runs wrote.

The Merge page can start a run, but once it finishes the settings that produced it
live only in ``run_manifest.json`` beside the output. Rebuilding that setup by hand
from a folder name is guesswork, so this lists what the manifests recorded and hands
the same parent list, method and weights back to the page.

Only folders directly under the outputs root are read: those are the ones the
Library lists as models, and scanning deeper would walk every shard of every
checkpoint for no gain.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .run_manifest import MANIFEST_NAME, read_manifest

# Stages whose manifest describes a merge. ``moe`` writes a router bundle rather
# than a loadable checkpoint, which the page shows differently.
MERGE_STAGES = ("merge", "moe")


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(entry) for entry in value if isinstance(entry, (str, Path))]


def _parents(manifest: dict[str, Any]) -> list[str]:
    """The parent list, under whichever key the manifest version wrote it."""
    for source in (manifest.get("config"), manifest.get("inputs")):
        if not isinstance(source, dict):
            continue
        for key in ("models", "sources", "parents", "model_a_b"):
            parents = _string_list(source.get(key))
            if len(parents) >= 2:
                return parents
        model_a = source.get("model_a")
        model_b = source.get("model_b")
        if isinstance(model_a, str) and isinstance(model_b, str):
            return [model_a, model_b]
    return []


def _weights(manifest: dict[str, Any]) -> list[float] | None:
    config = manifest.get("config")
    if not isinstance(config, dict):
        return None
    raw = config.get("weights")
    if not isinstance(raw, (list, tuple)):
        return None
    try:
        return [float(value) for value in raw]
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_manifest(directory: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """One history row. Unknown fields stay null rather than being guessed."""
    config = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    policy = manifest.get("policy") if isinstance(manifest.get("policy"), dict) else {}
    total_bytes = _float_or_none(
        artifacts.get("total_bytes") or artifacts.get("bytes") or config.get("total_bytes")
    )
    created = _float_or_none(manifest.get("created_at"))
    # An ISO stamp is what the page renders and sorts on; the epoch stays for ordering.
    created_iso = manifest.get("created_at_iso")
    if not isinstance(created_iso, str) and created is not None:
        import time

        created_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(created))
    return {
        "dir": str(directory),
        "name": directory.name,
        "stage": manifest.get("stage"),
        "created_at": created_iso,
        "created_epoch": created,
        "run_id": manifest.get("run_id"),
        "models": _parents(manifest),
        "method": config.get("method"),
        "weights": _weights(manifest),
        "density": _float_or_none(config.get("density")),
        "salt": _float_or_none(config.get("salt") or manifest.get("seeds", {}).get("salt")),
        "top_k": _float_or_none(config.get("top_k")),
        "temperature": _float_or_none(config.get("temperature")),
        "total_bytes": int(total_bytes) if total_bytes and total_bytes > 0 else None,
        "tokenizer_override": bool(
            policy.get("allow_tokenizer_conflict") or config.get("allow_tokenizer_conflict")
        ),
    }


def _sort_key(entry: dict[str, Any]) -> tuple[float, str]:
    epoch = entry.get("created_epoch")
    # Newest first; a manifest without a timestamp sorts by name behind the dated ones.
    return (epoch if isinstance(epoch, float) else float("-inf"), str(entry.get("name") or ""))


def list_merge_runs(root: str | Path, *, limit: int = 20) -> list[dict[str, Any]]:
    """Recent merge and MMoE runs under *root*, newest first.

    A folder with no manifest, or one this build cannot parse, is skipped: history
    is a convenience, and one corrupt file must not empty the list.
    """
    if limit <= 0:
        return []
    root_path = Path(root)
    try:
        children = sorted(root_path.iterdir())
    except (OSError, ValueError):
        return []
    entries: list[dict[str, Any]] = []
    for child in children:
        if not child.is_dir() or not (child / MANIFEST_NAME).is_file():
            continue
        try:
            manifest = read_manifest(child)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if not manifest or manifest.get("stage") not in MERGE_STAGES:
            continue
        entry = summarize_manifest(child, manifest)
        if len(entry["models"]) < 2:
            # Without the parents there is nothing to reload, so the row would only
            # be a folder name.
            continue
        entries.append(entry)
    entries.sort(key = _sort_key, reverse = True)
    return entries[:limit]
