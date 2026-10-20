# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Training a MoE model's experts from the Training Studio.

The scope names a module family; the resolver turns it into the leaf list Unsloth's
loader maps onto ``target_parameters``. These pin the mapping, the refusals, and the
config read the page uses to decide whether to offer the choice at all.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("pydantic")

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from core.training.moe_targets import (  # noqa: E402
    ATTENTION_LEAVES,
    EXPERT_LEAVES,
    MOE_EXPERT_SCOPES,
    ROUTER_LEAVES,
    normalize_scope,
    resolve_target_modules,
    scope_selectors,
    scope_trains_experts,
    scope_trains_router,
)
from utils.models.moe_config import moe_info  # noqa: E402


def _training_request_model():
    # By path: models/__init__.py pulls the auth stack in, and this test only needs
    # the request schema.
    spec = importlib.util.spec_from_file_location(
        "_moe_scope_training_models", _BACKEND_DIR / "models" / "training.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.TrainingStartRequest


def _request(**overrides):
    base = {
        "model_name": "Qwen/Qwen3-30B-A3B",
        "training_type": "LoRA/QLoRA",
        "hf_dataset": "unsloth/alpaca-cleaned",
        "format_type": "auto",
        "train_split": "train",
    }
    base.update(overrides)
    return _training_request_model()(**base)


def test_every_scope_maps_to_the_leaves_unsloth_reads():
    assert resolve_target_modules(None) is None
    assert resolve_target_modules("experts_only") == list(EXPERT_LEAVES)
    assert resolve_target_modules("router_only") == list(ROUTER_LEAVES)
    assert resolve_target_modules("experts_and_attention") == [
        *ATTENTION_LEAVES,
        *EXPERT_LEAVES,
    ]
    assert resolve_target_modules("experts_and_router") == [*EXPERT_LEAVES, *ROUTER_LEAVES]
    for scope in MOE_EXPERT_SCOPES:
        leaves = resolve_target_modules(scope)
        assert leaves, scope
        assert len(set(leaves)) == len(leaves)


def test_the_router_leaf_cannot_swallow_gate_proj():
    # PEFT matches suffixes, so "gate" must stay its own leaf: "gate_proj" ends in
    # "proj", not "gate".
    assert "gate" in ROUTER_LEAVES
    assert "gate" not in EXPERT_LEAVES


def test_a_list_target_survives_a_scope_that_adds_to_it():
    resolved = resolve_target_modules("experts_and_attention", ["q_proj"])
    assert resolved == ["q_proj", *ATTENTION_LEAVES[1:], *EXPERT_LEAVES]


def test_only_scopes_refuse_to_replace_an_explicit_list():
    with pytest.raises(ValueError, match = "replaces the target list"):
        resolve_target_modules("experts_only", ["q_proj"])
    with pytest.raises(ValueError, match = "all-linear"):
        resolve_target_modules("router_only", ["all-linear"])


def test_full_finetuning_and_unknown_scopes_are_refused():
    with pytest.raises(ValueError, match = "full finetuning"):
        resolve_target_modules("experts_only", None, training_type = "Full Finetuning")
    with pytest.raises(ValueError, match = "Unknown moe_expert_scope"):
        normalize_scope("experts-ish")


def test_selectors_follow_the_scope():
    assert scope_selectors(None) == {}
    experts = scope_selectors("experts_only")
    assert experts["finetune_attention_modules"] is False
    assert experts["finetune_mlp_modules"] is True
    both = scope_selectors("experts_and_attention")
    assert both["finetune_attention_modules"] is True
    router = scope_selectors("router_only")
    assert router["finetune_attention_modules"] is False


def test_scope_training_predicates():
    assert scope_trains_experts("experts_only") is True
    assert scope_trains_experts("router_only") is False
    assert scope_trains_router("router_only") is True
    assert scope_trains_router("experts_and_router") is True
    assert scope_trains_router("experts_and_attention") is False
    assert scope_trains_experts(None) is False


def test_the_request_carries_the_scope_and_defaults_to_none():
    assert _request().moe_expert_scope is None
    assert _request(moe_expert_scope = "experts_and_router").moe_expert_scope == "experts_and_router"


def test_a_scope_on_full_finetuning_is_refused_before_the_run():
    with pytest.raises(ValueError, match = "full finetuning"):
        _request(training_type = "Full Finetuning", moe_expert_scope = "experts_only")


def test_an_unknown_scope_is_refused_by_the_schema():
    with pytest.raises(ValueError):
        _request(moe_expert_scope = "everything")


def test_a_dense_config_is_not_moe():
    assert moe_info(SimpleNamespace(hidden_size = 2048, num_hidden_layers = 24)) == (
        False,
        None,
        None,
    )
    assert moe_info(None) == (False, None, None)


def test_the_qwen_and_mixtral_spellings_both_read():
    qwen = SimpleNamespace(num_experts = 128, num_experts_per_tok = 8)
    assert moe_info(qwen) == (True, 128, 8)
    mixtral = SimpleNamespace(num_local_experts = 8, top_k = 2)
    assert moe_info(mixtral) == (True, 8, 2)
    deepseek = SimpleNamespace(n_routed_experts = 256, num_experts_per_tok = 8)
    assert moe_info(deepseek) == (True, 256, 8)


def test_a_nested_text_config_is_read_through_the_vision_wrapper():
    wrapper = SimpleNamespace(
        model_type = "qwen3_vl_moe",
        text_config = SimpleNamespace(num_experts = 64, num_experts_per_tok = 4),
    )
    assert moe_info(wrapper) == (True, 64, 4)


def test_zero_and_boolean_counts_are_not_experts():
    assert moe_info(SimpleNamespace(num_experts = 0)) == (False, None, None)
    assert moe_info(SimpleNamespace(num_experts = True)) == (False, None, None)
