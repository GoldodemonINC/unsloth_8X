# SPDX-License-Identifier: NRC-2
# Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-2.md

"""Run lineage: the family tree every stage already records in its manifest.

A merge manifest names its parents, a training manifest names the checkpoint it
started from, and a scale manifest names the model it grew. Those are absolute
paths, so the tree is built by matching them against the folders that hold a
``run_manifest.json`` -- never by guessing from a folder name. An input that is
a Hub id stays off the graph: no local run produced it, and drawing an edge
into a folder that only shares a substring would be a lie in a provenance view.

Only JSON is read. A machine that cannot load the checkpoints can still draw
their history.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterator, Optional

from core.advanced import run_manifest

MAX_MANIFESTS = 300


def _normalized(value: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.path.expanduser(str(value))))


def _created_iso(manifest: dict[str, Any]) -> str:
    iso = manifest.get("created_at_iso")
    if isinstance(iso, str) and iso:
        return iso
    created = manifest.get("created_at")
    if isinstance(created, (int, float)):
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(created))
    return ""


def _read(path: Path) -> Optional[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _input_values(inputs: Any) -> Iterator[tuple[str, str]]:
    """(label, reference) for every input that names a folder or a model.

    ``checkpoint_digest`` stores the path under ``source``; ``dataset_identity``
    stores it under ``value``. Plain strings are accepted because benchmark
    manifests record their baseline that way.
    """
    if not isinstance(inputs, dict):
        return
    for key, value in inputs.items():
        if isinstance(value, str) and value.strip():
            yield str(key), value
            continue
        if not isinstance(value, dict):
            continue
        for field in ("source", "value", "path"):
            candidate = value.get(field)
            if isinstance(candidate, str) and candidate.strip():
                yield str(key), candidate
                break


def scan(root: str | Path | None = None, *, limit: int = MAX_MANIFESTS) -> dict[str, Any]:
    """Build the graph under *root* (default: the Studio outputs folder)."""
    if root is None:
        from utils.paths.storage_roots import outputs_root

        root = outputs_root()
    base = Path(root).expanduser()
    nodes: list[dict[str, Any]] = []
    by_path: dict[str, dict[str, Any]] = {}
    seen_runs: set[str] = set()
    truncated = False

    if base.is_dir():
        for manifest_path in sorted(base.rglob(run_manifest.MANIFEST_NAME)):
            if len(nodes) >= limit:
                truncated = True
                break
            manifest = _read(manifest_path)
            if manifest is None:
                continue
            directory = manifest_path.parent
            run_id = str(manifest.get("run_id") or "").strip()
            if not run_id or run_id in seen_runs:
                # A copied output folder carries the same run id. Keep the first
                # one and let the copy stay unlinked rather than merging them.
                continue
            seen_runs.add(run_id)
            problems = run_manifest.validate_manifest(manifest)
            node = {
                "id": run_id,
                "label": directory.name,
                "path": str(directory),
                "stage": str(manifest.get("stage") or "unknown"),
                "created_at": _created_iso(manifest),
                "studio_version": manifest.get("studio_version"),
                "valid": not problems,
                "problems": problems[:6],
            }
            nodes.append(node)
            by_path[_normalized(directory)] = node

    edges: list[dict[str, str]] = []
    edge_keys: set[tuple[str, str, str]] = set()
    for node in nodes:
        manifest = _read(Path(node["path"]) / run_manifest.MANIFEST_NAME)
        if manifest is None:
            continue
        for label, reference in _input_values(manifest.get("inputs")):
            parent = by_path.get(_normalized(reference))
            if parent is None or parent["id"] == node["id"]:
                continue
            key = (parent["id"], node["id"], label)
            if key in edge_keys:
                continue
            edge_keys.add(key)
            edges.append({"from": key[0], "to": key[1], "label": label})

    parent_count: dict[str, int] = {node["id"]: 0 for node in nodes}
    for edge in edges:
        parent_count[edge["to"]] = parent_count.get(edge["to"], 0) + 1

    nodes.sort(key=lambda node: (str(node["created_at"]), node["label"]), reverse=True)
    return {
        "root": str(base),
        "nodes": nodes,
        "edges": edges,
        "roots": [node["id"] for node in nodes if parent_count.get(node["id"], 0) == 0],
        "truncated": truncated,
    }
