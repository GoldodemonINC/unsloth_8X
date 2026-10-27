# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the Scaler's planning: range, growth floor, and refusals."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.advanced import scaling

SMOLLM2_135M = {
    "model_type": "llama",
    "hidden_size": 576,
    "num_hidden_layers": 30,
    "num_attention_heads": 9,
    "num_key_value_heads": 3,
    "intermediate_size": 1536,
    "vocab_size": 49152,
    "tie_word_embeddings": True,
}

QWEN3_8B = {
    "model_type": "qwen3",
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 14336,
    "vocab_size": 128256,
    "tie_word_embeddings": False,
}


def _checkpoint(tmp_path: Path, config: dict, name: str = "model") -> Path:
    directory = tmp_path / name
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return directory


def test_parameter_estimate_matches_a_known_checkpoint(tmp_path):
    """SmolLM2-135M is 134.5M parameters; the plan's count must land on it."""
    plan = scaling.plan(str(_checkpoint(tmp_path, SMOLLM2_135M)))

    assert plan["base_params"] == pytest.approx(134_500_000, rel=0.02)
    assert plan["floor_params"] == plan["base_params"]
    assert plan["max_params"] == plan["base_params"] + scaling.CAP_ADDED_PARAMS


def test_default_target_is_ten_percent_above_the_base(tmp_path):
    """The floor is 1x, so defaulting to it would plan nothing; 10% is the default."""
    plan = scaling.plan(str(_checkpoint(tmp_path, SMOLLM2_135M)))

    assert plan["growth_floor"] == 1.0
    assert plan["direction"] == "grow"
    assert plan["default_target_params"] > plan["base_params"]
    assert plan["runnable"] is True
    assert plan["refusals"] == []
    projection = plan["projection"]
    assert projection["growth_factor"] > 1.0
    assert projection["new_num_layers"] > SMOLLM2_135M["num_hidden_layers"]


def test_a_big_checkpoint_can_grow_below_two_x(tmp_path):
    """8B + 3B is 1.375x: allowed, now that the floor is 1x rather than 2x."""
    plan = scaling.plan(str(_checkpoint(tmp_path, QWEN3_8B)))

    assert plan["blocked"] is False
    assert plan["runnable"] is True
    assert plan["growth_floor"] == 1.0
    assert plan["max_params"] == plan["base_params"] + scaling.CAP_ADDED_PARAMS
    assert 1.0 < plan["projection"]["growth_factor"] < 1.4


def test_added_params_sets_the_target_on_top_of_the_base(tmp_path):
    """The slider asks for growth, not a total: target is always base + added."""
    plan = scaling.plan(str(_checkpoint(tmp_path, SMOLLM2_135M)), added_params=300_000_000)

    assert plan["added_params"] == 300_000_000
    assert plan["target_params"] == plan["base_params"] + 300_000_000
    assert plan["min_added_params"] == 0
    assert plan["max_added_params"] == scaling.CAP_ADDED_PARAMS
    assert plan["floor_added_params"] == plan["floor_params"] - plan["base_params"]
    assert plan["projection"]["projected_params"] == pytest.approx(
        plan["target_params"], rel=0.02
    )


def test_target_and_added_together_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        scaling.plan(
            str(_checkpoint(tmp_path, SMOLLM2_135M)),
            target_params=200_000_000,
            added_params=100_000_000,
        )


def test_growth_below_two_x_plans_normally(tmp_path):
    """1.26x used to be refused by the floor; it is a valid target now."""
    source = str(_checkpoint(tmp_path, SMOLLM2_135M))
    plan = scaling.plan(source, target_params=170_000_000)

    assert plan["direction"] == "grow"
    assert plan["runnable"] is True
    assert plan["blocked"] is False
    assert "below_floor" not in [item["code"] for item in plan["refusals"]]
    assert plan["projection"]["growth_factor"] == pytest.approx(1.26, abs=0.06)


def test_a_target_at_the_base_is_refused_as_no_growth(tmp_path):
    plan = scaling.plan(str(_checkpoint(tmp_path, SMOLLM2_135M)), added_params=0)

    assert plan["runnable"] is False
    assert plan["blocked"] is False
    assert "no_growth" in [item["code"] for item in plan["refusals"]]


def test_a_target_below_the_base_plans_a_shrink(tmp_path):
    plan = scaling.plan(str(_checkpoint(tmp_path, SMOLLM2_135M)), target_params=110_000_000)

    assert plan["direction"] == "shrink"
    assert plan["runnable"] is True
    assert plan["shrink_floor"] == 0.5
    projection = plan["projection"]
    assert 1 <= projection["removed_layers"] <= projection["max_removable_layers"]
    assert projection["new_num_layers"] == (
        SMOLLM2_135M["num_hidden_layers"] - projection["removed_layers"]
    )
    assert 0 < projection["shrink_factor"] < 1
    assert projection["projected_params"] < plan["base_params"]
    assert plan["min_params"] < plan["target_params"] < plan["max_params"]


def test_shrinking_past_half_the_depth_is_refused(tmp_path):
    plan = scaling.plan(
        str(_checkpoint(tmp_path, SMOLLM2_135M)), target_params=20_000_000
    )

    assert plan["direction"] == "shrink"
    assert plan["runnable"] is False
    assert "below_shrink_floor" in [item["code"] for item in plan["refusals"]]


def test_a_narrowing_target_plans_a_width_shrink(tmp_path):
    plan = scaling.plan(
        str(_checkpoint(tmp_path, SMOLLM2_135M)),
        target_params=130_000_000,
        method="mlp_width",
    )

    assert plan["direction"] == "shrink"
    assert plan["method"] == "mlp_width"
    assert plan["method_label"] == "MLP width (narrow the feed-forward)"
    assert plan["runnable"] is True
    assert plan["shrink_floor"] == scaling.WIDTH_SHRINK_FLOOR
    projection = plan["projection"]
    assert projection["new_intermediate_size"] < SMOLLM2_135M["intermediate_size"]
    assert projection["removed_channels"] >= 1
    assert projection["params_per_channel"] == 3 * 576 * 30


def test_a_width_shrink_beyond_the_floor_is_refused(tmp_path):
    plan = scaling.plan(
        str(_checkpoint(tmp_path, SMOLLM2_135M)),
        target_params=60_000_000,
        method="mlp_width",
    )

    assert plan["runnable"] is False
    assert "below_shrink_floor" in [item["code"] for item in plan["refusals"]]


def test_a_width_target_above_one_channel_of_saving_is_refused(tmp_path):
    source = str(_checkpoint(tmp_path, SMOLLM2_135M))
    base = scaling.plan(source)["base_params"]

    plan = scaling.plan(
        source, target_params=float(base - 1_000), method="mlp_width"
    )

    assert plan["runnable"] is False
    assert "not_smaller" in [item["code"] for item in plan["refusals"]]


@pytest.mark.parametrize(
    ("config", "code"),
    [
        ({**SMOLLM2_135M, "quantization_config": {"quant_method": "bitsandbytes"}}, "quantized"),
        ({**SMOLLM2_135M, "num_experts": 8}, "moe"),
        ({**SMOLLM2_135M, "model_type": "mamba"}, "unsupported_family"),
    ],
)
def test_refusals_are_reported_before_the_run(tmp_path, config, code):
    plan = scaling.plan(str(_checkpoint(tmp_path, config)))

    assert plan["blocked"] is True
    assert plan["runnable"] is False
    assert code in [refusal["code"] for refusal in plan["refusals"]]
    assert plan["blocked_reason"]


def test_an_adapter_directory_is_refused(tmp_path):
    directory = _checkpoint(tmp_path, SMOLLM2_135M, name="adapter")
    (directory / "adapter_config.json").write_text("{}", encoding="utf-8")

    plan = scaling.plan(str(directory))

    assert "adapter" in [refusal["code"] for refusal in plan["refusals"]]


def test_width_projection_lands_on_the_requested_parameters(tmp_path):
    plan = scaling.plan(
        str(_checkpoint(tmp_path, SMOLLM2_135M)),
        target_params=200_000_000,
        method="mlp_width",
    )
    projection = plan["projection"]

    assert projection["new_num_layers"] == SMOLLM2_135M["num_hidden_layers"]
    channels = (
        projection["new_intermediate_size"] - SMOLLM2_135M["intermediate_size"]
    )
    assert channels % scaling.WIDTH_ALIGNMENT == 0
    assert projection["projected_params"] == pytest.approx(200_000_000, rel=0.02)


def test_depth_and_width_split_the_growth(tmp_path):
    plan = scaling.plan(
        str(_checkpoint(tmp_path, SMOLLM2_135M)),
        target_params=400_000_000,
        method="depth+mlp_width",
    )
    projection = plan["projection"]

    assert projection["new_num_layers"] > SMOLLM2_135M["num_hidden_layers"]
    assert projection["new_intermediate_size"] > SMOLLM2_135M["intermediate_size"]
    assert projection["projected_params"] == pytest.approx(400_000_000, rel=0.03)


def test_a_missing_checkpoint_is_a_plain_error(tmp_path):
    with pytest.raises(ValueError):
        scaling.plan(str(tmp_path / "nowhere"))


def test_supported_families_match_the_engine():
    """The plan's copy must not drift from the engine's guard."""
    engine = scaling.load_engine()

    assert scaling.SUPPORTED_MODEL_TYPES == engine._SUPPORTED_MODEL_TYPES
    assert tuple(scaling.MOE_CONFIG_KEYS) == tuple(engine._MOE_CONFIG_KEYS)
    assert scaling.GROWTH_FLOOR == engine.GROWTH_FLOOR
    assert scaling.SHRINK_FLOOR == engine.SHRINK_FLOOR
    assert scaling.WIDTH_SHRINK_FLOOR == engine.WIDTH_FLOOR
    assert set(scaling.METHODS) == set(engine._METHODS)
    assert scaling.WIDTH_ALIGNMENT == engine._WIDTH_ALIGNMENT
