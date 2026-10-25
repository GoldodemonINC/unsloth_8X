# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
from pathlib import Path

import pytest

from core.training import fit


def _model(tmp_path: Path) -> Path:
    directory = tmp_path / "model"
    directory.mkdir()
    (directory / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "hidden_size": 512,
                "intermediate_size": 2048,
                "num_hidden_layers": 8,
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "vocab_size": 32000,
            }
        ),
        encoding = "utf-8",
    )
    return directory


def test_lora_fit_reports_a_breakdown_and_headroom(tmp_path):
    report = fit.estimate(
        {
            "model": str(_model(tmp_path)),
            "method": "lora",
            "batch_size": 1,
            "max_seq_length": 1024,
            "gradient_accumulation": 8,
            "lora_rank": 16,
            "gpu_memory_gb": 24,
        }
    )
    assert report["verdict"] == "fits"
    assert report["params"] > 0
    assert report["breakdown"]["total_gb"] > 0
    assert report["headroom_gb"] > 0
    assert report["suggestions"] == []


def test_full_training_is_heavier_than_qlora_and_gets_levers(tmp_path):
    model = str(_model(tmp_path))
    common = {
        "model": model,
        "batch_size": 4,
        "max_seq_length": 8192,
        "lora_rank": 64,
        "gpu_memory_gb": 4,
    }
    qlora = fit.estimate({**common, "method": "qlora"})
    full = fit.estimate({**common, "method": "full"})
    assert full["required_gb"] > qlora["required_gb"]
    assert full["verdict"] == "over"
    assert any("batch size" in item for item in full["suggestions"])
    assert any("sequence length" in item for item in full["suggestions"])
    assert any("batch size" in item for item in qlora["suggestions"])


def test_no_gpu_size_is_reported_as_unknown_not_as_fits(tmp_path):
    report = fit.estimate({"model": str(_model(tmp_path)), "method": "lora"})
    assert report["budget_gb"] is None
    assert report["verdict"] == "unknown"
    assert any("did not report" in item for item in report["suggestions"])


def test_fit_rejects_a_model_without_a_parameter_count(tmp_path):
    directory = tmp_path / "broken"
    directory.mkdir()
    (directory / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        fit.estimate({"model": str(directory), "method": "lora"})
    with pytest.raises(ValueError):
        fit.estimate({"model": "", "method": "lora"})
