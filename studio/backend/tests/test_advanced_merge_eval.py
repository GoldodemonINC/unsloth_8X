# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Structural merge-eval, license audit and eval-noise honesty (Phase 3 P0)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.advanced import eval_harness, licenses, merge_eval


def _checkpoint(directory: Path, *, tokenizer: bool = True, nan: bool = False) -> Path:
    np = pytest.importorskip("numpy")
    pytest.importorskip("safetensors")
    from safetensors.numpy import save_file

    directory.mkdir(parents = True, exist_ok = True)
    (directory / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "hidden_size": 8, "vocab_size": 32}),
        encoding = "utf-8",
    )
    weights = np.ones((32, 8), dtype = np.float32) * 0.02
    if nan:
        weights[0, 0] = float("nan")
    save_file({"model.embed_tokens.weight": weights}, str(directory / "model.safetensors"))
    (directory / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": weights.nbytes},
                "weight_map": {"model.embed_tokens.weight": "model.safetensors"},
            }
        ),
        encoding = "utf-8",
    )
    if tokenizer:
        (directory / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "Qwen2Tokenizer"}), encoding = "utf-8"
        )
    return directory


def test_a_valid_merge_is_structurally_valid(tmp_path: Path):
    result = merge_eval.structural_eval(_checkpoint(tmp_path / "merged"))
    assert result["verdict"] == "structurally-valid"
    assert result["problems"] == []
    assert result["tier"] == "structural"
    assert "nothing about task quality" in result["scope"].lower()
    # The benchmark vocabulary is advertised but never produced by this tier.
    assert result["benchmark_verdicts"] == ["pareto-positive", "within-noise", "regressed"]


def test_nan_weights_are_broken_not_merely_warned(tmp_path: Path):
    result = merge_eval.structural_eval(_checkpoint(tmp_path / "nan", nan = True))
    assert result["verdict"] == "structurally-broken"
    assert any("NaN" in problem for problem in result["problems"])


def test_missing_tokenizer_is_broken(tmp_path: Path):
    result = merge_eval.structural_eval(_checkpoint(tmp_path / "bare", tokenizer = False))
    assert result["verdict"] == "structurally-broken"
    assert any("tokenizer" in problem.lower() for problem in result["problems"])


def test_index_pointing_at_a_missing_shard_is_broken(tmp_path: Path):
    source = _checkpoint(tmp_path / "partial")
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.embed_tokens.weight": "model-00002.safetensors"}}),
        encoding = "utf-8",
    )
    result = merge_eval.structural_eval(source)
    assert result["verdict"] == "structurally-broken"
    assert any("shards that do not exist" in problem for problem in result["problems"])


def test_a_missing_folder_is_unloadable(tmp_path: Path):
    result = merge_eval.structural_eval(tmp_path / "nope")
    assert result["verdict"] == "unloadable"


def test_verdict_label_uses_the_noise_floor():
    assert merge_eval.verdict_label(0.004) == "within-noise"
    assert merge_eval.verdict_label(0.02) == "pareto-positive"
    assert merge_eval.verdict_label(-0.02) == "regressed"
    assert merge_eval.verdict_label(None) == "within-noise"


def test_eval_deltas_label_small_moves_as_within_noise():
    comparison = eval_harness.compute_deltas(
        {"mmlu": 0.600, "gsm8k": 0.500},
        {"mmlu": 0.605, "gsm8k": 0.502},
    )
    assert comparison["honest_verdict"] == "within-noise"
    assert comparison["noise_floor"] == pytest.approx(0.015)
    assert set(comparison["within_noise"]) == {"mmlu", "gsm8k"}
    entry = comparison["entries"][0]
    assert entry["within_noise"] is True
    assert "within noise" in entry["noise_note"]
    markdown = eval_harness.render_markdown_report({"comparison": comparison})
    assert "within noise" in markdown


def test_eval_deltas_outside_the_noise_floor_are_reported_as_wins():
    comparison = eval_harness.compute_deltas({"mmlu": 0.60}, {"mmlu": 0.65})
    assert comparison["honest_verdict"] == "pareto-positive"
    assert comparison["within_noise"] == []


def test_license_audit_reads_config_and_model_card(tmp_path: Path):
    first = _checkpoint(tmp_path / "a")
    (first / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "license": "apache-2.0"}), encoding = "utf-8"
    )
    second = _checkpoint(tmp_path / "b")
    (second / "README.md").write_text(
        "---\nlicense: mit\nbase_model: org/base\n---\n\n# Model\n", encoding = "utf-8"
    )
    report = licenses.audit([first, second])
    assert report["effective"] == "permissive"
    assert report["conflict"] is False
    assert set(report["declared"]) == {"apache-2.0", "mit"}
    assert [entry["license"] for entry in report["parents"]] == ["apache-2.0", "mit"]


def test_a_non_commercial_parent_makes_the_merge_non_commercial(tmp_path: Path):
    first = _checkpoint(tmp_path / "a")
    (first / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "license": "cc-by-nc-4.0"}), encoding = "utf-8"
    )
    second = _checkpoint(tmp_path / "b")
    (second / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "license": "apache-2.0"}), encoding = "utf-8"
    )
    report = licenses.audit([first, second])
    assert report["effective"] == "non-commercial"
    assert report["conflict"] is True
    assert any("non-commercial" in note for note in report["notes"])


def test_an_undeclared_licence_is_said_plainly(tmp_path: Path):
    report = licenses.audit([_checkpoint(tmp_path / "a"), _checkpoint(tmp_path / "b")])
    assert report["effective"] == "undeclared"
    assert any("No parent declares a licence" in note for note in report["notes"])
