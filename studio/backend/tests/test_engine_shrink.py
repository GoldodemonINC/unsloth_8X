# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Depth shrinking at the engine level: real surgery on a tiny causal LM.

The planning tests never load torch; these do, because the questions that matter
here - did the blocks actually go, is the model still runnable, does the KV cache
still index correctly - cannot be answered from config.json.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from core.advanced import scaling


@pytest.fixture(scope = "module")
def engine():
    return scaling.load_engine()


def _tiny_model(layers: int = 8):
    config = transformers.LlamaConfig(
        vocab_size = 128,
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = layers,
        num_attention_heads = 4,
        num_key_value_heads = 4,
    )
    return transformers.LlamaForCausalLM(config)


def test_shrink_drops_blocks_and_keeps_the_model_runnable(engine):
    model = _tiny_model()
    base = sum(parameter.numel() for parameter in model.parameters())
    ids = torch.randint(0, 128, (1, 16))

    model, report = engine.shrink_model(
        model, target_params = base * 0.8, calibration_input_ids = ids
    )

    assert report["scored_by"] == "block_influence"
    assert report["new_num_layers"] < report["current_num_layers"]
    assert report["function_preserved"] is False
    assert report["final_params"] < report["base_params"]
    assert report["removed_params"] == report["base_params"] - report["final_params"]

    assert model.config.num_hidden_layers == report["new_num_layers"]
    assert len(model.model.layers) == report["new_num_layers"]
    assert model(input_ids = ids).logits.shape[:2] == (1, 16)
    assert model(input_ids = ids, use_cache = True).past_key_values is not None

    # Attention indexes the KV cache by layer_idx, so a gap in the numbering writes
    # to the wrong slot: the survivors must be numbered 0..n-1.
    assert [block.self_attn.layer_idx for block in model.model.layers] == list(
        range(report["new_num_layers"])
    )


def test_at_least_half_the_depth_survives(engine):
    model = _tiny_model(8)

    with pytest.raises(ValueError, match = "50%"):
        engine.shrink_model(model, num_layers_to_remove = 6)


def test_explicit_removals_cannot_take_an_edge_block(engine):
    model = _tiny_model(8)

    with pytest.raises(ValueError, match = "protected"):
        engine.shrink_model(
            model, num_layers_to_remove = 1, drop_strategy = "explicit", layer_indices = [0]
        )


def test_the_weight_proxy_scores_without_calibration_text(engine):
    model = _tiny_model(8)
    base = sum(parameter.numel() for parameter in model.parameters())

    model, report = engine.shrink_model(model, target_params = base * 0.85)

    assert report["scored_by"] == "output_projection"
    assert report["removed_layers"]


def test_a_target_that_is_not_smaller_is_refused(engine):
    model = _tiny_model(8)
    base = sum(parameter.numel() for parameter in model.parameters())

    with pytest.raises(ValueError, match = "not smaller"):
        engine.shrink_model(model, target_params = base * 1.2)
