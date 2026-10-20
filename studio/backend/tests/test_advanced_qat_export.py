# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""QAT quality at the export quant (F1 / A6).

The claim is scoped to the grid the forward pass actually uses, and
``qat/validate`` reports a measured delta at that grid rather than validating the
config and calling it quality.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.advanced import qat


def _checkpoint(
    directory: Path, *, seed: int, scale: float = 1.0, shape: tuple[int, int] = (64, 128)
) -> Path:
    np = pytest.importorskip("numpy")
    pytest.importorskip("safetensors")
    from safetensors.numpy import save_file

    directory.mkdir(parents = True, exist_ok = True)
    rng = np.random.default_rng(seed)
    payload = {
        "model.layers.0.mlp.down_proj.weight": (rng.normal(size = shape) * scale).astype(np.float32),
        "model.layers.1.mlp.down_proj.weight": (rng.normal(size = shape) * scale).astype(np.float32),
        "model.layers.0.input_layernorm.weight": np.ones((shape[1],), dtype = np.float32),
    }
    save_file(payload, str(directory / "model.safetensors"))
    (directory / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "hidden_size": shape[0]}), encoding = "utf-8"
    )
    return directory


def test_identical_checkpoints_are_within_noise(tmp_path: Path):
    parent = _checkpoint(tmp_path / "parent", seed = 1)
    candidate = _checkpoint(tmp_path / "candidate", seed = 1)
    report = qat.validate_checkpoints_at_export(str(parent), str(candidate), config = {"format": "int4"})
    assert report["status"] == "measured"
    assert report["format"] == "int4"
    assert report["tensors_compared"] == 2  # the norm vector is not a matrix
    assert report["delta_points"] == pytest.approx(0.0, abs = 1e-6)
    assert report["verdict"] == "within-noise"


def test_measurement_is_at_the_export_grid_not_bf16(tmp_path: Path):
    parent = _checkpoint(tmp_path / "parent", seed = 2)
    # A heavily scaled candidate keeps its bf16 similarity but changes how much
    # survives 4-bit rounding, which is exactly what the report measures.
    candidate = _checkpoint(tmp_path / "candidate", seed = 3, scale = 3.0)
    report = qat.validate_checkpoints_at_export(
        str(parent), str(candidate), config = {"format": "nf4"}
    )
    assert report["tensors_compared"] == 2
    assert report["format"] == "nf4"
    assert report["noise_floor_points"] == qat.NOISE_POINTS
    assert report["verdict"] in {"pareto-positive", "within-noise", "regressed"}
    assert isinstance(report["delta_points"], float)
    assert "not a benchmark" in report["methodology"]


def test_qat_claim_is_scoped_to_the_trained_grid():
    session = qat.QatSession(config = qat.validate_qat_config({"enabled": True, "format": "nvfp4"}))
    summary = qat.qat_summary(session)
    assert summary["format"] == "nvfp4"


def test_compare_at_export_reports_inconclusive_without_shared_matrices():
    report = qat.compare_at_export({}, {}, config = {"format": "int4"})
    assert report["status"] == "inconclusive"
    assert report["verdict"] == "inconclusive"
    assert report["delta_points"] is None
