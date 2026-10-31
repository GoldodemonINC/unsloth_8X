# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Artifacts and pipelines: what a folder allows, and what a chain will cost."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.advanced import scaling
from core.labs import artifacts, pipelines

LLAMA = {
    "model_type": "llama",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "intermediate_size": 128,
    "vocab_size": 256,
    "tie_word_embeddings": True,
}


def _checkpoint(directory: Path, config: dict | None = None, name: str = "model") -> Path:
    directory.mkdir(parents = True, exist_ok = True)
    (directory / "config.json").write_text(json.dumps(config or LLAMA), encoding = "utf-8")
    (directory / "model.safetensors").write_bytes(b"\0" * 64)
    return directory


def _action(record: dict, action_id: str) -> dict:
    return next(action for action in record["actions"] if action["id"] == action_id)


def test_a_dense_supported_checkpoint_can_do_everything(tmp_path, monkeypatch):
    model = _checkpoint(tmp_path / "outputs" / "run-a")
    monkeypatch.setattr(artifacts, "root", lambda: tmp_path / "outputs")
    record = artifacts.describe(model)
    assert record["params"] and record["params"] > 0
    assert record["supported"] is True
    assert record["adapter"] is False and record["quantized"] is False and record["moe"] is False
    for action_id in ("continue-training", "scale", "merge", "quantize", "benchmark"):
        assert _action(record, action_id)["enabled"] is True, action_id
        assert _action(record, action_id)["reason"] is None


def test_an_adapter_refuses_scaling_with_the_reason(tmp_path):
    model = _checkpoint(tmp_path / "adapter")
    (model / "adapter_config.json").write_text("{}", encoding = "utf-8")
    record = artifacts.describe(model)
    assert record["adapter"] is True
    assert _action(record, "scale")["enabled"] is False
    assert "Merge it into its base" in _action(record, "scale")["reason"]
    assert _action(record, "quantize")["enabled"] is False
    # A saved adapter is exactly what a LoRA/CPT resume wants.
    assert _action(record, "continue-training")["enabled"] is True


def test_a_quantized_checkpoint_refuses_expansion(tmp_path):
    config = dict(LLAMA)
    config["quantization_config"] = {"quant_method": "bitsandbytes", "bits": 4}
    record = artifacts.describe(_checkpoint(tmp_path / "q4", config))
    assert record["quantized"] is True
    assert _action(record, "scale")["enabled"] is False
    assert "fp16/bf16 original" in _action(record, "scale")["reason"]


def test_an_moe_checkpoint_is_refused_by_the_registry_not_by_luck(tmp_path):
    config = dict(LLAMA)
    config["num_local_experts"] = 8
    record = artifacts.describe(_checkpoint(tmp_path / "moe", config))
    assert record["moe"] is True
    assert _action(record, "scale")["enabled"] is False


def test_an_unsupported_family_names_the_family(tmp_path):
    config = dict(LLAMA)
    config["model_type"] = "mamba"
    record = artifacts.describe(_checkpoint(tmp_path / "mamba", config))
    assert record["supported"] is False
    assert "mamba" in _action(record, "scale")["reason"]


def test_scan_lists_only_checkpoint_folders_and_skips_saved_plans(tmp_path, monkeypatch):
    root = tmp_path / "outputs"
    _checkpoint(root / "older")
    _checkpoint(root / "run" / "final")
    (root / "labs-pipelines").mkdir(parents = True)
    (root / "notes").mkdir()
    (root / "notes" / "readme.txt").write_text("not a checkpoint", encoding = "utf-8")
    monkeypatch.setattr(artifacts, "root", lambda: root)
    names = {record["name"] for record in artifacts.scan()}
    assert names == {"older", "final"}


def test_find_refuses_a_folder_outside_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "root", lambda: tmp_path / "outputs")
    (tmp_path / "outputs").mkdir(exist_ok = True)
    outside = _checkpoint(tmp_path / "elsewhere")
    with pytest.raises(PermissionError):
        artifacts.find(str(outside))


def test_manifest_read_refuses_a_path_outside_outputs(tmp_path, monkeypatch):
    root = tmp_path / "outputs"
    model = _checkpoint(root / "run")
    (model / "run_manifest.json").write_text('{"stage": "merge"}', encoding = "utf-8")
    monkeypatch.setattr(artifacts, "root", lambda: root)
    assert artifacts.read_manifest(str(model))["manifest"]["stage"] == "merge"

    outside = _checkpoint(tmp_path / "elsewhere")
    (outside / "run_manifest.json").write_text("{}", encoding = "utf-8")
    with pytest.raises(PermissionError):
        artifacts.read_manifest(str(outside))


def test_manifest_absent_is_reported_not_invented(tmp_path, monkeypatch):
    root = tmp_path / "outputs"
    model = _checkpoint(root / "run")
    monkeypatch.setattr(artifacts, "root", lambda: root)
    record = artifacts.describe(model)
    assert record["manifest"] is None
    assert record["manifest_path"] is None
    with pytest.raises(FileNotFoundError):
        artifacts.read_manifest(str(model))


def _spec(source: Path, *stages: dict) -> dict:
    return {"name": "planned", "source": str(source), "stages": list(stages)}


def test_a_growth_chain_is_priced_and_warned_about(tmp_path):
    source = _checkpoint(tmp_path / "base")
    report = pipelines.preflight(
        _spec(
            source,
            {"kind": "scale", "factor": 2},
            {"kind": "train", "steps": 60, "batch_size": 1, "sequence_length": 2048},
            {"kind": "quantize", "level": "q4_k_m"},
            {"kind": "benchmark", "baseline": str(source), "suites": ["mmlu"]},
        )
    )
    assert report["blocked"] is False, report["blocked_reason"]
    assert report["final_params"] > 0
    assert report["total_output_bytes"] > 0
    assert len(report["stages"]) == 4
    assert all(stage["params_after"] for stage in report["stages"])
    # The token budget is 122,880 tokens: a plan that called that a growth verdict
    # would be lying, so the warning must be there.
    assert any("recovery" in warning for warning in report["warnings"])


def test_growth_below_the_measured_range_is_priced_and_marked(tmp_path):
    source = _checkpoint(tmp_path / "base")
    report = pipelines.preflight(_spec(source, {"kind": "scale", "factor": 1.5}))
    assert report["blocked"] is False, report["blocked_reason"]
    assert report["stages"][0]["details"]["target_params"] > 0
    assert any("1.50x" in warning and "measured" in warning for warning in report["warnings"])


def test_growth_that_adds_nothing_is_blocked_with_the_number(tmp_path):
    source = _checkpoint(tmp_path / "base")
    report = pipelines.preflight(_spec(source, {"kind": "scale", "added_params": -1000}))
    assert report["blocked"] is True
    assert "not larger than" in report["blocked_reason"]
    assert "shrink" in report["blocked_reason"]


def test_a_scale_stage_with_no_target_plans_growth_not_a_no_op(tmp_path):
    source = _checkpoint(tmp_path / "base")
    report = pipelines.preflight(_spec(source, {"kind": "scale"}))
    stage = report["stages"][0]
    assert report["blocked"] is False, report["blocked_reason"]
    assert stage["details"]["target_params"] > stage["params_before"]


def test_the_scale_refusal_is_the_scaler_own_policy(tmp_path):
    source = _checkpoint(tmp_path / "base")
    report = pipelines.preflight(_spec(source, {"kind": "scale", "added_params": -1000}))
    stage = report["stages"][0]
    policy = scaling.growth_policy(stage["params_before"], added_params = -1000)

    assert policy["refusal"] is not None
    assert stage["reason"] == policy["refusal"]


def test_the_default_growth_target_is_the_scaler_default(tmp_path):
    source = _checkpoint(tmp_path / "base")
    report = pipelines.preflight(_spec(source, {"kind": "scale"}))
    stage = report["stages"][0]
    policy = scaling.growth_policy(stage["params_before"])

    assert stage["details"]["target_params"] == policy["default_target_params"]


def test_growth_past_the_cap_is_blocked(tmp_path):
    big = dict(LLAMA)
    big["hidden_size"] = 4096
    big["num_hidden_layers"] = 32
    big["intermediate_size"] = 14336
    big["num_attention_heads"] = 32
    big["num_key_value_heads"] = 8
    source = _checkpoint(tmp_path / "big", big)
    report = pipelines.preflight(_spec(source, {"kind": "scale", "factor": 2}))
    assert report["blocked"] is True
    assert "ceiling" in report["blocked_reason"]


def test_quantize_cannot_be_followed_by_a_transform(tmp_path):
    source = _checkpoint(tmp_path / "base")
    with pytest.raises(pipelines.PipelineError):
        pipelines.preflight(
            _spec(source, {"kind": "quantize", "level": "q4_k_m"}, {"kind": "train"})
        )


def test_an_unknown_quantization_is_blocked_not_guessed(tmp_path):
    source = _checkpoint(tmp_path / "base")
    report = pipelines.preflight(_spec(source, {"kind": "quantize", "level": "q3_nope"}))
    assert report["blocked"] is True
    assert "Unknown quantization" in report["blocked_reason"]


def test_merge_parents_must_share_a_tokenizer(tmp_path):
    first = _checkpoint(tmp_path / "a")
    other = dict(LLAMA)
    other["vocab_size"] = 1024
    second = _checkpoint(tmp_path / "b", other)
    report = pipelines.preflight(
        {"source": str(first), "stages": [{"kind": "merge", "parents": [str(first), str(second)]}]}
    )
    assert report["blocked"] is True
    assert "vocab_size" in report["blocked_reason"]


def test_a_merge_stage_needs_the_parents_in_its_own_spec(tmp_path):
    source = _checkpoint(tmp_path / "a")
    report = pipelines.preflight(
        {"source": str(source), "stages": [{"kind": "merge", "parents": [str(source)]}]}
    )
    assert report["blocked"] is True
    assert "two parent" in report["blocked_reason"]


def test_pipelines_save_list_and_delete(tmp_path, monkeypatch):
    source = _checkpoint(tmp_path / "base")
    monkeypatch.setattr(pipelines, "root", lambda: tmp_path / "saved")
    record = pipelines.save_pipeline(
        {"name": "Two steps", "source": str(source), "stages": [{"kind": "scale", "factor": 2}]}
    )
    assert record["name"] == "Two steps"
    assert record["plan"]["stage_kinds"] == ["scale"]
    saved = pipelines.list_pipelines()
    assert [item["id"] for item in saved] == [record["id"]]
    assert pipelines.delete_pipeline(record["id"]) is True
    assert pipelines.list_pipelines() == []
    assert pipelines.delete_pipeline(record["id"]) is False


def test_a_pipeline_id_is_validated_before_it_reaches_the_filesystem(tmp_path, monkeypatch):
    monkeypatch.setattr(pipelines, "root", lambda: tmp_path / "saved")
    with pytest.raises(pipelines.PipelineError):
        pipelines.delete_pipeline("../../studio.db")
