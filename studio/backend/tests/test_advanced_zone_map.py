# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Layer-LR zone coverage (F3 / A5).

Every parameter must land in exactly one zone, including MoE experts and routers
and hybrid/Mamba blocks. Before this, ``classify_param_zone`` returned "middle"
for every layer-indexed parameter and "embedding" for terminal ones, so the
zone map could not be trusted as a preview.
"""

from __future__ import annotations

from core.advanced import layer_lr


def _config(**overrides) -> layer_lr.LayerLrConfig:
    return layer_lr.validate_layer_lr_config(2e-4, overrides or None)


def _dense_names(layers: int = 6) -> list[str]:
    names = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
    for index in range(layers):
        names += [
            f"model.layers.{index}.self_attn.q_proj.weight",
            f"model.layers.{index}.mlp.down_proj.weight",
            f"model.layers.{index}.input_layernorm.weight",
        ]
    return names


def test_every_parameter_lands_in_exactly_one_zone():
    names = _dense_names()
    mapped = layer_lr.zone_map(((name, 1024) for name in names), _config(), num_layers = 6)
    coverage = mapped["coverage"]
    assert coverage["exact_one_zone"] is True
    assert coverage["unclassified"] == []
    assert coverage["duplicates"] == []
    assert coverage["params"] == len(names)
    assert coverage["assigned_numel"] == coverage["total_numel"]
    assert sum(zone["params"] for zone in mapped["zones"]) == len(names)


def test_moe_layers_are_zoned_by_depth_not_dumped_in_one_zone():
    names = ["model.embed_tokens.weight"]
    for index in range(9):
        names += [
            f"model.layers.{index}.block_sparse_moe.gate.weight",
            f"model.layers.{index}.block_sparse_moe.experts.3.w1.weight",
            f"model.layers.{index}.block_sparse_moe.experts.7.w2.weight",
        ]
    mapped = layer_lr.zone_map(((name, 512) for name in names), _config(), num_layers = 9)
    by_name = {entry["name"]: entry for entry in mapped["entries"]}
    assert by_name["model.layers.0.block_sparse_moe.experts.3.w1.weight"]["zone"] == "early"
    assert by_name["model.layers.4.block_sparse_moe.experts.3.w1.weight"]["zone"] == "middle"
    assert by_name["model.layers.8.block_sparse_moe.gate.weight"]["zone"] == "late"
    # The router is reported as such, but it is still zoned by depth.
    assert by_name["model.layers.8.block_sparse_moe.gate.weight"]["kind"] in {
        "moe_router",
        "other",
    }
    assert mapped["coverage"]["exact_one_zone"] is True


def test_hybrid_mamba_blocks_are_not_misclassified_as_embeddings():
    names = ["model.embed_tokens.weight", "model.norm.weight"]
    for index in range(6):
        names += [
            f"model.layers.{index}.mamba.in_proj.weight",
            f"model.layers.{index}.mamba.conv1d.weight",
            f"model.layers.{index}.self_attn.q_proj.weight",
        ]
    mapped = layer_lr.zone_map(((name, 256) for name in names), _config(), num_layers = 6)
    entries = {entry["name"]: entry for entry in mapped["entries"]}
    hybrid = entries["model.layers.0.mamba.in_proj.weight"]
    assert hybrid["zone"] == "early"
    assert hybrid["kind"] == "hybrid_block"
    assert entries["model.layers.5.mamba.conv1d.weight"]["zone"] == "late"
    assert mapped["coverage"]["exact_one_zone"] is True


def test_gpt2_style_h_block_names_are_depth_indexed():
    names = ["transformer.wte.weight", "transformer.ln_f.weight"]
    for index in range(6):
        names += [
            f"transformer.h.{index}.attn.c_attn.weight",
            f"transformer.h.{index}.mlp.c_fc.weight",
        ]
    mapped = layer_lr.zone_map(((name, 128) for name in names), _config(), num_layers = 6)
    entries = {entry["name"]: entry for entry in mapped["entries"]}
    assert entries["transformer.h.0.attn.c_attn.weight"]["zone"] == "early"
    assert entries["transformer.h.5.mlp.c_fc.weight"]["zone"] == "late"
    assert entries["transformer.ln_f.weight"]["zone"] == "head"
    assert mapped["coverage"]["exact_one_zone"] is True


def test_final_norm_belongs_to_the_head_zone():
    config = _config()
    assert layer_lr.classify_param_zone("model.norm.weight", 6) == "head"
    assert layer_lr.classify_param_zone("transformer.ln_f.weight", 6) == "head"


def test_expert_indices_are_not_mistaken_for_layer_depth():
    assert layer_lr.layer_index("model.layers.2.mlp.experts.11.gate_proj.weight") == 2
    assert layer_lr.layer_index("model.embed_tokens.weight") is None


def test_frozen_parameters_are_excluded_from_groups_but_still_classified():
    class FakeParam:
        def __init__(self, numel: int, requires_grad: bool = True):
            self._numel = numel
            self.requires_grad = requires_grad

        def numel(self) -> int:
            return self._numel

    mapped = layer_lr.zone_map(
        [
            ("model.layers.0.self_attn.q_proj.weight", FakeParam(100)),
            ("model.layers.0.self_attn.k_proj.weight", FakeParam(100, requires_grad = False)),
        ],
        _config(),
        num_layers = 1,
    )
    assert mapped["coverage"]["params"] == 1
    included = layer_lr.zone_map(
        [
            ("model.layers.0.self_attn.q_proj.weight", FakeParam(100)),
            ("model.layers.0.self_attn.k_proj.weight", FakeParam(100, requires_grad = False)),
        ],
        _config(),
        num_layers = 1,
        include_frozen = True,
    )
    assert included["coverage"]["params"] == 2


def test_off_the_end_layer_indices_are_clamped_not_dropped():
    names = [f"model.layers.{index}.mlp.down_proj.weight" for index in range(4)]
    # A config that undercounts the depth must not produce an unzoned parameter.
    mapped = layer_lr.zone_map(((name, 10) for name in names), _config(), num_layers = 2)
    assert mapped["coverage"]["exact_one_zone"] is True
    assert all(entry["zone"] in layer_lr.ZONES for entry in mapped["entries"])
