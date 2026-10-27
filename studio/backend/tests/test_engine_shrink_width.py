# SPDX-License-Identifier: NRC-2
# Copyright 2026-present the X8 authors. See /LICENSE-NRC-2.md

"""Width shrinking at the engine level: real MLP surgery on a tiny causal LM.

The planner tests never load torch. Whether the three projections still agree on
a single channel count, and whether the narrowed model still runs, can only be
answered on a real module.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from core.advanced import scaling


@pytest.fixture(scope = "module")
def engine():
    return scaling.load_engine()


def _tiny_model(layers: int = 4, hidden: int = 64, intermediate: int = 128):
    config = transformers.LlamaConfig(
        vocab_size = 128,
        hidden_size = hidden,
        intermediate_size = intermediate,
        num_hidden_layers = layers,
        num_attention_heads = 4,
        num_key_value_heads = 4,
    )
    return transformers.LlamaForCausalLM(config)


def test_width_shrink_narrows_every_mlp_and_keeps_the_model_runnable(engine):
    model = _tiny_model()
    base = sum(parameter.numel() for parameter in model.parameters())
    ids = torch.randint(0, 128, (1, 12))

    model, report = engine.shrink_model_width(model, intermediate_size = 96)

    assert report["old_intermediate_size"] == 128
    assert report["new_intermediate_size"] == 96
    assert report["function_preserved"] is False
    assert report["final_params"] < report["base_params"]
    assert report["removed_params"] == report["base_params"] - report["final_params"]
    assert model.config.intermediate_size == 96
    for block in model.model.layers:
        assert block.mlp.gate_proj.weight.shape == (96, 64)
        assert block.mlp.up_proj.weight.shape == (96, 64)
        assert block.mlp.down_proj.weight.shape == (64, 96)
        # nn.Linear caches these; PEFT sizes an adapter from them.
        assert block.mlp.gate_proj.out_features == 96
        assert block.mlp.down_proj.in_features == 96

    logits = model(input_ids = ids).logits
    assert logits.shape[:2] == (1, 12)
    assert torch.isfinite(logits).all()


def test_a_parameter_target_becomes_a_channel_count(engine):
    model = _tiny_model()
    base = sum(parameter.numel() for parameter in model.parameters())

    model, report = engine.shrink_model_width(model, target_params = base * 0.97)

    assert report["new_intermediate_size"] < 128
    assert report["new_intermediate_size"] >= 32
    # One channel costs 3 * hidden * layers parameters across the model.
    removed = report["removed_channels"] * report["params_per_channel"]
    assert abs((report["base_params"] - report["final_params"]) - removed) <= report[
        "params_per_channel"
    ]


def test_a_target_that_is_not_smaller_is_refused(engine):
    model = _tiny_model()
    base = sum(parameter.numel() for parameter in model.parameters())

    with pytest.raises(ValueError, match = "not smaller"):
        engine.shrink_model_width(model, target_params = base * 1.2)


def test_a_widening_intermediate_size_is_refused(engine):
    model = _tiny_model()

    with pytest.raises(ValueError, match = "must narrow below"):
        engine.shrink_model_width(model, intermediate_size = 160)


def test_the_quarter_width_floor_is_enforced(engine):
    model = _tiny_model()

    with pytest.raises(ValueError, match = "25%"):
        engine.shrink_model_width(model, intermediate_size = 16)


def test_the_lowest_scoring_channel_is_the_one_dropped(engine):
    model = _tiny_model()
    with torch.no_grad():
        layer = model.model.layers[0].mlp
        # Channel 5 contributes nothing: its input rows and its output column are
        # zero, so no weight proxy can rank it anywhere but last.
        layer.gate_proj.weight[5].zero_()
        layer.up_proj.weight[5].zero_()
        layer.down_proj.weight[:, 5].zero_()

    _, report = engine.shrink_model_width(model, intermediate_size = 127)

    dropped = report["dropped_channels_sample"][0]
    assert dropped["layer"] == 0
    assert 5 in dropped["dropped"]


def test_mismatched_mlp_widths_are_refused(engine):
    model = _tiny_model()
    with torch.no_grad():
        model.model.layers[1].mlp.gate_proj.weight = torch.nn.Parameter(
            torch.zeros(64, 64)
        )

    with pytest.raises(ValueError, match = "uniform"):
        engine.shrink_model_width(model, intermediate_size = 32)
