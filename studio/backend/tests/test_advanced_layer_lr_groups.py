# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Layer-LR optimizer groups and the module-kind multiplier axis.

``patch_layer_lr`` used to rewrite ``optimizer.param_groups`` into one group per
parameter, which the LR scheduler then stepped once each. These tests pin the
two properties that matter: the groups collapse to one per (zone, kind, lr),
and every parameter still keeps exactly the LR the zone map promised.
"""

from __future__ import annotations

import pytest

from core.advanced import integration, layer_lr

LAYERS = 6


class _Param:
    def __init__(self, numel: int = 4):
        self._numel = numel
        self.requires_grad = True

    def numel(self) -> int:
        return self._numel


class _Config:
    num_hidden_layers = LAYERS


class _Model:
    def __init__(self, names: list[str]):
        self.config = _Config()
        self._named = [(name, _Param()) for name in names]

    def named_parameters(self):
        return list(self._named)


class _Optimizer:
    def __init__(self, groups):
        self.param_groups = groups


class _Trainer:
    def __init__(self, model: _Model):
        self.model = model

    def create_optimizer(self):
        params = [param for _, param in self.model.named_parameters()]
        # Two groups, like a decay / no-decay split.
        return _Optimizer(
            [
                {"lr": 2e-4, "weight_decay": 0.1, "params": params[: len(params) // 2]},
                {"lr": 2e-4, "weight_decay": 0.0, "params": params[len(params) // 2 :]},
            ]
        )


def _names() -> list[str]:
    names = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
    for index in range(LAYERS):
        names += [
            f"model.layers.{index}.self_attn.q_proj.weight",
            f"model.layers.{index}.self_attn.o_proj.weight",
            f"model.layers.{index}.mlp.down_proj.weight",
            f"model.layers.{index}.input_layernorm.weight",
        ]
    return names


def _patched(names=None, **layer_config):
    model = _Model(names or _names())
    trainer = _Trainer(model)
    report = integration.patch_layer_lr(
        trainer,
        model,
        {"layer_lr": {"enabled": True, **layer_config}},
        2e-4,
    )
    assert report is not None
    return trainer, model, trainer.create_optimizer()


def _lr_by_param(optimizer) -> dict[int, float]:
    return {
        id(param): group["lr"]
        for group in optimizer.param_groups
        for param in group["params"]
    }


def test_groups_collapse_to_one_per_zone_kind_and_lr():
    _, model, optimizer = _patched()
    params = [param for _, param in model.named_parameters()]
    # 27 parameters across 6 zones / 3 kinds: far fewer groups than parameters.
    assert len(optimizer.param_groups) < len(params)
    assert len(optimizer.param_groups) <= len(layer_lr.ZONES) * len(layer_lr.PARAM_KINDS)
    assert sum(len(group["params"]) for group in optimizer.param_groups) == len(params)
    keys = {
        (group["_zone"], group["_kind"], round(group["lr"], 12))
        for group in optimizer.param_groups
    }
    assert len(keys) == len(optimizer.param_groups)
    # No duplicate parameter across groups.
    seen = [id(param) for group in optimizer.param_groups for param in group["params"]]
    assert len(seen) == len(set(seen))


def test_per_parameter_lrs_match_the_zone_map():
    config = layer_lr.validate_layer_lr_config(
        2e-4,
        {
            "zone_multipliers": {"early": 0.25, "late": 0.9, "head": 2.0},
            "depth_curve": [0.5, 1.0],
            "kind_multipliers": {"attention": 0.5, "mlp": 1.5},
        },
    )
    _, model, optimizer = _patched(
        zone_multipliers={"early": 0.25, "late": 0.9, "head": 2.0},
        depth_curve=[0.5, 1.0],
        kind_multipliers={"attention": 0.5, "mlp": 1.5},
    )
    lrs = _lr_by_param(optimizer)
    for name, param in model.named_parameters():
        _, _, multiplier = layer_lr.resolve_assignment(name, config, LAYERS)
        assert lrs[id(param)] == pytest.approx(2e-4 * multiplier), name


def test_kind_multipliers_are_neutral_by_default():
    zone_only = layer_lr.validate_layer_lr_config(
        2e-4, {"zone_multipliers": {"early": 0.5}}
    )
    with_kinds = layer_lr.validate_layer_lr_config(
        2e-4,
        {"zone_multipliers": {"early": 0.5}, "kind_multipliers": {}},
    )
    name = "model.layers.0.self_attn.q_proj.weight"
    assert layer_lr.resolve_multiplier(name, zone_only, LAYERS) == (
        layer_lr.resolve_multiplier(name, with_kinds, LAYERS)
    )
    assert layer_lr.resolve_multiplier(name, zone_only, LAYERS)[1] == pytest.approx(0.5)


def test_kind_and_zone_multipliers_compose():
    config = layer_lr.validate_layer_lr_config(
        2e-4,
        {
            "zone_multipliers": {"early": 0.5, "late": 0.5},
            "kind_multipliers": {"attention": 0.5, "mlp": 2.0, "moe_expert": 0.1},
        },
    )
    early_attention = layer_lr.resolve_assignment(
        "model.layers.0.self_attn.q_proj.weight", config, LAYERS
    )
    early_mlp = layer_lr.resolve_assignment(
        "model.layers.0.mlp.down_proj.weight", config, LAYERS
    )
    late_attention = layer_lr.resolve_assignment(
        "model.layers.5.self_attn.q_proj.weight", config, LAYERS
    )
    expert = layer_lr.resolve_assignment(
        "model.layers.0.block_sparse_moe.experts.3.w1.weight", config, LAYERS
    )
    assert early_attention == ("early", "attention", pytest.approx(0.25))
    assert early_mlp == ("early", "mlp", pytest.approx(1.0))
    assert late_attention == ("late", "attention", pytest.approx(0.25))
    assert expert[1] == "moe_expert"
    assert expert[2] == pytest.approx(0.05)
    # Adapters stay on the zone axis rather than being multiplied twice.
    adapter = layer_lr.resolve_assignment(
        "base_model.model.layers.0.self_attn.q_proj.lora_A.weight", config, LAYERS
    )
    assert adapter[:2] == ("adapter", "other")


def test_the_report_names_only_the_kinds_that_move():
    model = _Model(_names())
    trainer = _Trainer(model)
    report = integration.patch_layer_lr(
        trainer,
        model,
        {"layer_lr": {"enabled": True, "kind_multipliers": {"attention": 0.5}}},
        2e-4,
    )
    assert report["kinds"] == {"attention": 0.5}
    zone_only = integration.patch_layer_lr(
        _Trainer(_Model(_names())),
        _Model(_names()),
        {"layer_lr": {"enabled": True}},
        2e-4,
    )
    assert zone_only["kinds"] == {}


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError):
        layer_lr.validate_layer_lr_config(
            2e-4, {"kind_multipliers": {"attentionn": 0.5}}
        )
    with pytest.raises(ValueError):
        layer_lr.validate_layer_lr_config(
            2e-4, {"kind_multipliers": {"attention": 11.0}}
        )


def test_zero_multiplier_kind_still_owns_its_parameters():
    _, model, optimizer = _patched(kind_multipliers={"mlp": 0.0})
    total = sum(len(group["params"]) for group in optimizer.param_groups)
    assert total == len(list(model.named_parameters()))
    zero = [group for group in optimizer.param_groups if group["lr"] == 0.0]
    assert zero and all(group["_kind"] == "mlp" for group in zero)
