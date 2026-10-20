# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Recent merges, read back from the manifests the runs wrote.

The Merge page reloads a past run from here, so these pin what a row means: the
parents, the engine settings, and the promise that a folder this build cannot read
is skipped rather than crashing the list.
"""

from __future__ import annotations

import json
from pathlib import Path

from core.advanced.merge_history import list_merge_runs, summarize_manifest
from core.advanced.run_manifest import MANIFEST_NAME, build_manifest


def _write_merge(
    root: Path,
    name: str,
    *,
    stage: str = "merge",
    created_at: float = 1000.0,
    models: list[str] | None = None,
    config: dict | None = None,
    policy: dict | None = None,
    artifacts: dict | None = None,
) -> Path:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        stage,
        config = config if config is not None else {"method": "ties", "density": 0.5},
        inputs = {"models": models if models is not None else ["org/a", "org/b"]},
        artifacts = artifacts or {},
        policy = policy or {},
    )
    manifest["created_at"] = created_at
    (directory / MANIFEST_NAME).write_text(json.dumps(manifest), encoding = "utf-8")
    return directory


def test_merges_are_listed_newest_first(tmp_path):
    _write_merge(tmp_path, "older", created_at = 1000.0)
    _write_merge(tmp_path, "newer", created_at = 2000.0)
    entries = list_merge_runs(tmp_path)
    assert [entry["name"] for entry in entries] == ["newer", "older"]


def test_a_row_carries_what_the_page_needs_to_reload_it(tmp_path):
    _write_merge(
        tmp_path,
        "merge-a-b",
        models = ["org/a", "org/b", "org/c"],
        config = {"method": "dare", "density": 0.3, "weights": [2, 1, 1], "top_k": 2},
        policy = {"allow_tokenizer_conflict": True},
        artifacts = {"total_bytes": 4096},
    )
    (entry,) = list_merge_runs(tmp_path)
    assert entry["models"] == ["org/a", "org/b", "org/c"]
    assert entry["method"] == "dare"
    assert entry["density"] == 0.3
    assert entry["weights"] == [2, 1, 1]
    assert entry["total_bytes"] == 4096
    assert entry["tokenizer_override"] is True
    assert entry["created_at"].endswith("Z")


def test_an_moe_bundle_is_kept_apart_from_a_weight_merge(tmp_path):
    _write_merge(tmp_path, "bundle", stage = "moe", config = {"top_k": 2, "temperature": 1.5})
    (entry,) = list_merge_runs(tmp_path)
    assert entry["stage"] == "moe"
    assert entry["top_k"] == 2
    assert entry["temperature"] == 1.5


def test_other_stages_and_manifestless_folders_are_skipped(tmp_path):
    _write_merge(tmp_path, "split-out", stage = "split")
    (tmp_path / "no-manifest").mkdir()
    _write_merge(tmp_path, "a-merge")
    entries = list_merge_runs(tmp_path)
    assert [entry["name"] for entry in entries] == ["a-merge"]


def test_a_manifest_without_parents_is_skipped(tmp_path):
    directory = _write_merge(tmp_path, "orphan")
    manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding = "utf-8"))
    manifest["inputs"] = {}
    manifest["config"] = {"method": "linear"}
    (directory / MANIFEST_NAME).write_text(json.dumps(manifest), encoding = "utf-8")
    assert list_merge_runs(tmp_path) == []


def test_a_corrupt_manifest_does_not_empty_the_list(tmp_path):
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / MANIFEST_NAME).write_text("{ not json", encoding = "utf-8")
    _write_merge(tmp_path, "fine")
    assert [entry["name"] for entry in list_merge_runs(tmp_path)] == ["fine"]


def test_the_limit_bounds_the_list_and_a_missing_root_is_empty(tmp_path):
    for index in range(5):
        _write_merge(tmp_path, f"merge-{index}", created_at = float(index))
    assert len(list_merge_runs(tmp_path, limit = 2)) == 2
    assert list_merge_runs(tmp_path / "absent") == []
    assert list_merge_runs(tmp_path, limit = 0) == []


def test_the_pair_shorthand_is_read_when_models_are_named_separately(tmp_path):
    directory = tmp_path / "pair"
    directory.mkdir()
    manifest = build_manifest(
        "merge",
        config = {"method": "linear", "weight": 0.4},
        inputs = {"model_a": "org/a", "model_b": "org/b"},
    )
    (directory / MANIFEST_NAME).write_text(json.dumps(manifest), encoding = "utf-8")
    (entry,) = list_merge_runs(tmp_path)
    assert entry["models"] == ["org/a", "org/b"]


def test_summarize_keeps_unknown_fields_null(tmp_path):
    manifest = build_manifest("merge", config = {}, inputs = {"models": ["a/b", "c/d"]})
    entry = summarize_manifest(tmp_path, manifest)
    assert entry["method"] is None
    assert entry["weights"] is None
    assert entry["total_bytes"] is None
