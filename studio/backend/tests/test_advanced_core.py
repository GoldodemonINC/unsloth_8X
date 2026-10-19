# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the advanced training mechanics (no GPU required)."""

from __future__ import annotations

import numpy as np
import pytest

from core.advanced import blending, curriculum, eval_harness, layer_lr, qat


def test_qat_int4_bounds_error_and_shape():
    rng = np.random.default_rng(0)
    weights = rng.normal(0, 0.02, size=(64, 128)).astype(np.float32)
    cfg = qat.validate_qat_config({"format": "int4", "group_size": 64})
    quantized = qat.fake_quantize(weights, cfg)
    assert quantized.shape == weights.shape
    assert quantized.dtype == np.float32
    # Per-group absmax / 7 scaling means no value may exceed ~the group max.
    assert np.abs(quantized).max() <= np.abs(weights).max() + 1e-5
    error = qat.quantization_error(weights, cfg)
    assert 0 < error["mean_rel_error"] < 0.2


def test_qat_nvfp4_uses_e2m1_grid_and_validates():
    values = np.array([[-7.0, -3.1, -0.2, 0.0, 0.24, 2.9, 9.0]], dtype=np.float32)
    cfg = qat.validate_qat_config({"format": "nvfp4", "group_size": 16})
    quantized = qat.fake_quantize(values, cfg)
    # E2M1 magnitudes cap at 6, and a single FP32 tensor scale lets that grid
    # cover the largest weight in the tensor.
    assert np.abs(quantized).max() <= np.abs(values).max() + 1e-4
    assert np.abs(quantized).max() > 6.0  # the tensor scale is doing its job
    assert quantized[0, 3] == 0.0
    with pytest.raises(ValueError):
        qat.validate_qat_config({"format": "int2"})
    with pytest.raises(ValueError):
        qat.validate_qat_config({"format": "int4", "group_size": 100})


def test_qat_torch_ste_passes_gradient():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    model = nn.Sequential(nn.Linear(64, 64, bias=False))
    weights = torch.linspace(-0.03, 0.03, 64 * 64).reshape(64, 64).float()
    with torch.no_grad():
        model[0].weight.copy_(weights)
        model[0].weight.requires_grad_(True)
    cfg = qat.validate_qat_config(
        {"enabled": True, "format": "int4", "group_size": 32, "min_numel": 0, "skip": []}
    )
    session = qat.apply_qat(model, cfg)
    assert qat.qat_summary(session)["layers"] == 1
    output = model(torch.ones(2, 64))
    loss = output.square().mean()
    loss.backward()
    assert model[0].weight.grad is not None
    assert torch.isfinite(model[0].weight.grad).all()
    # A zero gradient would mean the STE detached the whole path.
    assert model[0].weight.grad.abs().sum() > 0
    session.remove()
    assert qat.qat_summary(session)["layers"] == 0


def test_curriculum_schedule_and_sampler():
    plan = curriculum.build_curriculum_plan("reasoning:2,code:1", total_steps=9)
    assert plan.cycle_steps == 3
    assert plan.stream_sequence(6) == [
        "reasoning",
        "reasoning",
        "code",
        "reasoning",
        "reasoning",
        "code",
    ]
    assert plan.stage_for_step(8).name == "code"

    sampler = curriculum.CurriculumSampler(
        plan=plan,
        streams={"reasoning": list(range(5)), "code": list(range(5, 8))},
        batch_size=2,
        seed=7,
    )
    first = sampler.batch_for_step(0)
    assert len(first) == 2
    assert all(0 <= index < 5 for index in first)
    code_batch = sampler.batch_for_step(2)
    assert all(5 <= index < 8 for index in code_batch)
    with pytest.raises(ValueError):
        curriculum.parse_cadence("reasoning:0")


def test_layer_lr_zones_and_groups():
    assert layer_lr.zone_for_layer(0, 12) == "early"
    assert layer_lr.zone_for_layer(6, 12) == "middle"
    assert layer_lr.zone_for_layer(11, 12) == "late"
    assert layer_lr.classify_param_zone("model.lm_head.weight") == "head"
    assert layer_lr.classify_param_zone("model.embed_tokens.weight") == "embedding"
    assert layer_lr.classify_param_zone("base_model.model.layers.4.self_attn.q_proj.lora_A.weight") == "adapter"

    config = layer_lr.validate_layer_lr_config(
        2e-4,
        {"zone_multipliers": {"early": 0.25, "head": 2.0}},
    )

    class Param:
        requires_grad = True

    named = [
        ("model.embed_tokens.weight", Param()),
        ("model.layers.1.self_attn.q_proj.weight", Param()),
        ("model.layers.7.mlp.down_proj.weight", Param()),
        ("model.layers.11.mlp.down_proj.weight", Param()),
        ("model.lm_head.weight", Param()),
    ]
    groups = layer_lr.build_param_groups(named, config, num_layers=12)
    by_zone = {group["zone"]: group for group in groups}
    assert by_zone["embedding"]["lr"] == pytest.approx(2e-4 * 0.25)
    assert by_zone["early"]["lr"] == pytest.approx(2e-4 * 0.25)
    assert by_zone["head"]["lr"] == pytest.approx(2e-4 * 2.0)
    assert sum(group["param_count"] for group in groups) == 5


def test_blending_plan_and_smooth_schedule():
    plan = blending.validate_blend_slots(
        [{"name": "reasoning", "ratio": 60}, {"name": "code", "ratio": 40}],
        batch_size=10,
    )
    assert plan.ratios["reasoning"] == pytest.approx(0.6)
    layout = plan.batch_layout()
    assert layout.count("reasoning") == 6
    assert layout.count("code") == 4
    # Smooth round-robin must not put all of a stream at the front.
    assert layout.index("code") < 3
    assert blending.smooth_round_robin(["a", "b"], [0.9, 0.1], 10).count("b") == 1

    sampler = blending.BlendedSampler(
        plan=plan,
        streams={"reasoning": list(range(20)), "code": list(range(20, 30))},
        seed=3,
    )
    batch = sampler.next_batch()
    assert len(batch) == 10
    assert len({sample for sample in batch}) == 10
    with pytest.raises(ValueError):
        blending.validate_blend_slots([{"name": "a", "ratio": 0}], batch_size=2)


def test_eval_deltas_and_report():
    comparison = eval_harness.compute_deltas(
        {"mmlu": 0.60, "gsm8k": 0.40, "ifeval": 0.5},
        {"mmlu": 0.62, "gsm8k": 0.35, "ifeval": 0.505},
    )
    assert comparison["verdict"] == "regressed"
    assert "gsm8k" in comparison["regressions"]
    by_suite = {entry["suite"]: entry for entry in comparison["entries"]}
    assert by_suite["mmlu"]["status"] == "improved"
    assert by_suite["ifeval"]["status"] == "flat"

    report = {
        "run_name": "demo",
        "comparison": comparison,
        "warnings": ["limited"],
        "errors": [],
    }
    markdown = eval_harness.render_markdown_report(report)
    assert "MMLU" in markdown
    assert "regressed" in markdown

    plan = eval_harness.build_eval_plan(
        ["mmlu", "gsm8k"], baseline_model="base", candidate_model="run"
    )
    assert plan.suites == ("mmlu", "gsm8k")
    with pytest.raises(ValueError):
        eval_harness.build_eval_plan([], baseline_model="a", candidate_model="b")
    with pytest.raises(ValueError):
        eval_harness.build_eval_plan(
            ["custom"], baseline_model="a", candidate_model="b"
        )


def test_eval_command_includes_limit():
    command = eval_harness.build_lm_eval_command("gsm8k", "org/model", limit=10, batch_size=4)
    assert command[:4] == ["lm_eval", "--model", "hf", "--model_args"]
    assert command[4] == "pretrained=org/model"
    assert "--limit" in command and "10" in command
