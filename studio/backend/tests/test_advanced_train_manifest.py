# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Training runs write a run manifest (F6 / A3).

Merge and split already recorded a report; a trained folder recorded nothing but
the adapter. This is the training half of the same document.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.advanced import run_manifest


@pytest.fixture()
def worker():
    pytest.importorskip("datasets")
    from core.training import worker as worker_module

    return worker_module


def _config() -> dict:
    return {
        "max_steps": 100,
        "save_steps": 50,
        "batch_size": 2,
        "seed": 11,
        "learning_rate": 2e-4,
        "num_epochs": 1,
        "advanced": {
            "blend": {
                "enabled": True,
                "slots": [{"name": "reasoning", "ratio": 60}, {"name": "code", "ratio": 40}],
            },
            "curriculum": {"enabled": True, "cadence": "reasoning:1"},
            "layer_lr": {"zone_multipliers": {"early": 0.1, "late": 0.5}},
            "qat": {"enabled": True, "format": "int4", "group_size": 128},
        },
    }


def test_training_writes_a_manifest_with_the_advanced_block(worker, tmp_path: Path):
    worker._write_run_manifest(
        str(tmp_path), "unsloth/Qwen3-1.7B", _config(), learning_rate = 2e-4
    )
    manifest = run_manifest.read_manifest(tmp_path)
    assert manifest is not None
    assert run_manifest.validate_manifest(manifest) == []
    assert manifest["stage"] == "train"
    assert manifest["seeds"]["seed"] == 11
    # The blend schedule that was in force, not just "blending was on".
    assert manifest["blend"]["base_ratios"]["reasoning"] == pytest.approx(0.6)
    assert manifest["blend"]["event_count"] == 0
    # Curriculum is labelled experimental wherever it is recorded.
    assert manifest["curriculum"]["label"] == "experimental"
    # The QAT claim is scoped to the grid the run actually trained.
    assert manifest["qat"]["format"] == "int4"
    assert "forward pass" in manifest["qat"]["claim"]
    # The zone definition is fingerprinted, so a resume can tell it changed.
    assert manifest["versions"]["zone_definition"]


def test_training_manifest_failure_never_raises(worker, tmp_path: Path, monkeypatch):
    from core.advanced import policy

    def explode(*args, **kwargs):
        raise OSError("disk is gone")

    monkeypatch.setattr(policy, "atomic_write_json", explode)
    # Must not raise: the run has already been set up by this point.
    worker._write_run_manifest(str(tmp_path), "model", _config(), learning_rate = 1e-4)


def test_a_plain_run_still_validates(worker, tmp_path: Path):
    worker._write_run_manifest(str(tmp_path), "model", {"seed": 1}, learning_rate = 1e-4)
    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding = "utf-8"))
    assert "blend" not in manifest and "curriculum" not in manifest
    assert run_manifest.validate_manifest(manifest) == []
