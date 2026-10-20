# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Run manifests (F6 / A3).

Every stage writes ``run_manifest.json``; a manifest that is missing a field, or
whose inputs are not identified, is not reproducible and fails validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.advanced import run_manifest


def _checkpoint(directory: Path, *, vocab: int = 16, seed: int = 0) -> Path:
    np = pytest.importorskip("numpy")
    pytest.importorskip("safetensors")
    from safetensors.numpy import save_file

    directory.mkdir(parents = True, exist_ok = True)
    (directory / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "hidden_size": 8, "vocab_size": vocab}),
        encoding = "utf-8",
    )
    (directory / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "Qwen2Tokenizer"}), encoding = "utf-8"
    )
    rng = np.random.default_rng(seed)
    save_file(
        {"model.embed_tokens.weight": rng.normal(size = (vocab, 8)).astype(np.float32)},
        str(directory / "model.safetensors"),
    )
    return directory


def test_merge_manifest_is_complete_and_reproducible(tmp_path: Path):
    pytest.importorskip("numpy")
    pytest.importorskip("safetensors")
    from core.advanced import merge

    a = _checkpoint(tmp_path / "a", seed = 1)
    b = _checkpoint(tmp_path / "b", seed = 2)
    out = tmp_path / "merged"
    merge.merge_checkpoints(str(a), str(b), str(out), method = "linear", seed = 7)

    manifest = run_manifest.read_manifest(out)
    assert manifest is not None
    assert run_manifest.validate_manifest(manifest) == []
    assert manifest["stage"] == "merge"
    assert manifest["seeds"]["seed"] == 7
    # Inputs are identified by digest, not just by path.
    assert manifest["inputs"]["model_a"]["sha256"]
    assert manifest["inputs"]["model_b"]["sha256"] != manifest["inputs"]["model_a"]["sha256"]
    assert manifest["environment"]["python"]
    # The tokenizer check is an event, with its verdict.
    assert manifest["events"][0]["kind"] == "tokenizer_check"
    assert manifest["events"][0]["status"] in {"ok", "warning"}
    assert manifest["labels"]["output_tokenizer"] == "ok"


def test_manifest_validation_names_missing_fields():
    partial = {"schema_version": 2, "stage": "merge", "run_id": "x"}
    problems = run_manifest.validate_manifest(partial)
    assert any("config" in problem for problem in problems)
    assert any("environment" in problem for problem in problems)
    assert run_manifest.validate_manifest(None) == ["No manifest found."]


def test_checkpoint_digest_changes_when_the_weights_change(tmp_path: Path):
    # Same shapes and file names, different values: the digest must notice.
    first = _checkpoint(tmp_path / "first", seed = 10)
    second = _checkpoint(tmp_path / "second", seed = 11)
    assert run_manifest.checkpoint_digest(first)["sha256"] != run_manifest.checkpoint_digest(second)["sha256"]
    # ...and is stable for the same folder.
    assert run_manifest.checkpoint_digest(first)["sha256"] == run_manifest.checkpoint_digest(first)["sha256"]


def test_dataset_identity_pins_local_content_and_flags_remote_references(tmp_path: Path):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"text": "hello"}\n', encoding = "utf-8")
    local = run_manifest.dataset_identity(dataset)
    assert local["kind"] == "file" and local["sha256"]

    remote = run_manifest.dataset_identity("org/dataset-name")
    assert remote["kind"] == "reference"
    assert remote["sha256"] is None
    assert "not pinned" in remote["note"]


def test_zone_definition_hash_is_stable_and_config_sensitive():
    first = {"zone_multipliers": {"early": 0.25, "late": 0.5}}
    same = {"zone_multipliers": {"late": 0.5, "early": 0.25}}
    different = {"zone_multipliers": {"early": 0.5, "late": 0.5}}
    assert run_manifest.zone_definition_hash(first) == run_manifest.zone_definition_hash(same)
    assert run_manifest.zone_definition_hash(first) != run_manifest.zone_definition_hash(different)
    assert run_manifest.zone_definition_hash(None) is None


def test_update_manifest_merges_into_the_existing_document(tmp_path: Path):
    y = _checkpoint(tmp_path / "y")
    manifest = run_manifest.build_manifest("split", config = {"a": 1}, inputs = {"source": str(y)})
    run_manifest.write_manifest(y, manifest)
    updated = run_manifest.update_manifest(y, artifacts = {"output_dir": str(y)}, stage = "split")
    assert updated is not None
    assert updated["artifacts"]["output_dir"] == str(y)
    assert updated["config"] == {"a": 1}
    assert "updated_at" in updated
