# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""The upscale -> CPT -> eval pipeline, on a tiny random Llama built from a config.

No Hub download, no tokenizer files: a stand-in tokenizer keeps the tests hermetic
and fast, and every model is a few hundred thousand parameters, so the whole file
runs on CPU in seconds. The numbers here are structural (shapes, params, logits,
bookkeeping); whether growth actually helps a real model is what the pipeline
reports, not what these tests assert.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

_MODULE_PATH = Path(__file__).resolve().parent.parent / "unsloth" / "models" / "scaling.py"


def _load_scaling():
    """By path: `unsloth.models` pulls the whole CUDA stack in for a pure-torch module."""
    spec = importlib.util.spec_from_file_location("_unsloth_scaling_under_test", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


scaling = _load_scaling()


def _tiny(
    seed: int = 0,
    *,
    layers: int = 4,
    hidden: int = 64,
    intermediate: int = 128,
):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size = 128,
        hidden_size = hidden,
        intermediate_size = intermediate,
        num_hidden_layers = layers,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        max_position_embeddings = 256,
        tie_word_embeddings = False,
        pad_token_id = 0,
    )
    return LlamaForCausalLM(config).eval()


def _params(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _ids(seed: int = 1, length: int = 8):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(1, 100, (1, length), generator = generator)


class _FakeTokenizer:
    """Character-level stand-in: enough surface for the collator, generation and decode."""

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 0
        self.pad_token = "<pad>"

    def __call__(
        self,
        text,
        *,
        padding = False,
        truncation = False,
        max_length = None,
        return_tensors = None,
    ):
        single = isinstance(text, str)
        texts = [text] if single else list(text)
        encoded = [
            [max(1, (ord(character) % 120) + 1) for character in str(item)[:64]] or [1]
            for item in texts
        ]
        if truncation and max_length is not None:
            encoded = [ids[:max_length] for ids in encoded]
        width = max(len(ids) for ids in encoded) if padding or not single else len(encoded[0])
        input_ids, attention = [], []
        for ids in encoded:
            pad = width - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad)
            attention.append([1] * len(ids) + [0] * pad)
        batch = {
            "input_ids": torch.tensor(input_ids, dtype = torch.long),
            "attention_mask": torch.tensor(attention, dtype = torch.long),
        }
        if single:
            batch = {key: value[0] for key, value in batch.items()}
            if return_tensors == "pt":
                batch = {key: value.unsqueeze(0) for key, value in batch.items()}
        return batch

    def decode(
        self,
        ids,
        skip_special_tokens: bool = True,
    ) -> str:
        return "1"


def _scores(suite):
    """A suite that reads a constant off the model, so a delta is exact and instant."""
    return {name: (lambda model, tokenizer, _name = name: model._scores[_name]) for name in suite}


# ---------------------------------------------------------------------------
# expand_model
# ---------------------------------------------------------------------------


def test_depth_expansion_is_function_preserving_and_reports_params():
    base = _tiny()
    expanded = _tiny()  # identical weights, so the logits are comparable

    model, report = scaling.expand_model(expanded, num_new_layers = 2, force_small_growth = True)
    assert report["method_used"] == "depth"
    assert report["new_num_layers"] == 6
    assert model.config.num_hidden_layers == 6
    assert len(model.model.layers) == 6
    assert report["final_params"] == _params(model) > report["base_params"]
    assert report["growth_factor"] == pytest.approx(report["final_params"] / report["base_params"])
    assert report["warning"], "below-floor growth must carry a warning when forced"
    assert scaling.assert_function_preserved(base, model, _ids()) < 1e-4


def test_depth_insertion_strategies_place_blocks_where_they_say():
    # The copies land in the new list, so "after block 1" puts the first copy at index 2.
    for strategy, expected in (("tail", [4, 5]), ("middle", [2, 3]), ("even", [2, 4])):
        model = _tiny()
        _, report = scaling.expand_model(
            model,
            num_new_layers = 2,
            insert_strategy = strategy,
            force_small_growth = True,
        )
        assert report["inserted_at"] == expected, strategy
        assert all(
            getattr(model.model.layers[index], "_unsloth_expanded_layer", False)
            for index in expected
        )


def test_explicit_insertion_rejects_bad_indices():
    model = _tiny()
    with pytest.raises(ValueError, match = "outside the model"):
        scaling.expand_model(
            model,
            num_new_layers = 1,
            insert_strategy = "explicit",
            layer_indices = [9],
            force_small_growth = True,
        )
    with pytest.raises(ValueError, match = "one source index per new block"):
        scaling.expand_model(
            model,
            num_new_layers = 2,
            insert_strategy = "explicit",
            layer_indices = [1],
            force_small_growth = True,
        )


def test_mlp_width_expansion_touches_only_the_mlp():
    base = _tiny()
    model = _tiny()
    before = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}

    _, report = scaling.expand_model(
        model, mlp_width_target = 256, method = "mlp_width", force_small_growth = True
    )
    assert report["new_intermediate_size"] == 256
    assert report["old_intermediate_size"] == 128
    assert model.config.intermediate_size == 256

    changed = {
        name for name, tensor in model.state_dict().items() if tuple(tensor.shape) != before[name]
    }
    assert changed, "the MLP projections must have changed shape"
    assert all(
        name.endswith(("gate_proj.weight", "up_proj.weight", "down_proj.weight"))
        for name in changed
    ), changed
    assert "model.embed_tokens.weight" not in changed
    assert "lm_head.weight" not in changed
    assert scaling.assert_function_preserved(base, model, _ids()) < 1e-4


def test_depth_plus_width_grows_both():
    model = _tiny()
    _, report = scaling.expand_model(
        model,
        target_params = 3 * _params(model),
        method = "depth+mlp_width",
        force_small_growth = True,
    )
    assert report["method_used"] == "depth+mlp_width"
    assert report["new_intermediate_size"] > 128
    assert report["new_num_layers"] > 4
    assert report["growth_factor"] > 2.0


def test_target_params_lands_within_three_percent_at_a_realistic_growth():
    base_params = _params(_tiny())
    model = _tiny()
    _, report = scaling.expand_model(model, target_params = int(3 * base_params))
    assert report["growth_factor"] >= scaling.GROWTH_FLOOR
    assert abs(report["final_params"] - 3 * base_params) / (3 * base_params) < 0.03


def test_growth_floor_refuses_a_pointless_run():
    model = _tiny()
    with pytest.raises(ValueError, match = "below the 2x floor"):
        scaling.expand_model(model, num_new_layers = 1)
    # The refusal happens after the growth, so nothing is left half-expanded in the caller.
    forced = _tiny()
    _, report = scaling.expand_model(forced, num_new_layers = 1, force_small_growth = True)
    assert report["growth_factor"] < scaling.GROWTH_FLOOR


def test_target_params_is_required_for_depth_cli_style_calls():
    with pytest.raises(ValueError, match = "at least one of"):
        scaling.expand_model(_tiny())


def test_quantized_peft_and_moe_checkpoints_are_refused():
    quantized = _tiny()
    quantized.config.quantization_config = {"quant_method": "bitsandbytes"}
    with pytest.raises(ValueError, match = "quantized model"):
        scaling.expand_model(quantized, num_new_layers = 1, force_small_growth = True)

    adapted = _tiny()
    adapted.peft_config = {"default": object()}
    with pytest.raises(ValueError, match = "plain HF model"):
        scaling.expand_model(adapted, num_new_layers = 1, force_small_growth = True)

    moe = types.SimpleNamespace(config = types.SimpleNamespace(model_type = "qwen3_moe", num_experts = 8))
    with pytest.raises(ValueError, match = "MoE expansion not supported"):
        scaling.expand_model(moe, num_new_layers = 1, force_small_growth = True)

    unsupported = types.SimpleNamespace(config = types.SimpleNamespace(model_type = "gpt2"))
    with pytest.raises(ValueError, match = "not implemented for model_type"):
        scaling.expand_model(unsupported, num_new_layers = 1, force_small_growth = True)


def test_a_non_identity_block_layout_is_refused():
    model = _tiny()
    delattr(model.model.layers[0].self_attn, "o_proj")  # nn.Module drops it from _modules too
    with pytest.raises(ValueError, match = "so the copied block returns its input"):
        scaling.expand_model(model, num_new_layers = 1, force_small_growth = True)


# ---------------------------------------------------------------------------
# save + reload
# ---------------------------------------------------------------------------


def test_save_rebuilds_the_shard_index_and_the_reload_matches(tmp_path):
    model = _tiny()
    _, report = scaling.expand_model(
        model, mlp_width_target = 192, method = "mlp_width", force_small_growth = True
    )
    sample = _ids()

    from transformers import AutoModelForCausalLM

    saved = scaling.save_expanded_model(
        model,
        None,
        tmp_path,
        max_shard_size = "1KB",  # force sharding so the index is exercised
        verify = True,
        sample_input_ids = sample,
    )
    assert saved["verified"] is True
    assert saved["max_logit_diff"] < 1e-4
    assert saved["index"] and saved["index"]["tensors"] > 0
    assert saved["index"]["total_size"] > 0
    assert len(saved["index"]["shards"]) > 1

    import json

    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    assert index["metadata"]["total_size"] == saved["index"]["total_size"]
    assert len(index["weight_map"]) == saved["index"]["tensors"]

    reloaded = AutoModelForCausalLM.from_pretrained(str(tmp_path))
    assert scaling.assert_function_preserved(model, reloaded, sample) < 1e-4
    assert reloaded.config.intermediate_size == report["new_intermediate_size"]


def test_captured_logits_let_one_model_serve_both_sides():
    model = _tiny()
    sample = _ids()
    with torch.no_grad():
        before = model(input_ids = sample).logits.float().clone()  # before growth
    scaling.expand_model(model, num_new_layers = 1, force_small_growth = True)
    assert scaling.assert_function_preserved(None, model, sample, base_logits = before) < 1e-4
    with pytest.raises(ValueError, match = "base_model, or base_logits"):
        scaling.assert_function_preserved(None, model, sample)


def test_saving_after_an_adapter_is_injected_is_refused(tmp_path):
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(
        _tiny(),
        LoraConfig(r = 4, lora_alpha = 8, target_modules = "all-linear", task_type = "CAUSAL_LM"),
    )
    with pytest.raises(ValueError, match = "before any adapter"):
        scaling.save_expanded_model(model, None, tmp_path / "out", verify = False)


def test_a_non_preserving_change_fails_verification():
    base = _tiny()
    broken = _tiny()
    with torch.no_grad():
        broken.model.layers[0].mlp.down_proj.weight.add_(0.5)
    with pytest.raises(AssertionError, match = "not function-preserving"):
        scaling.assert_function_preserved(base, broken, _ids())


# ---------------------------------------------------------------------------
# LoRA capacity
# ---------------------------------------------------------------------------


def test_a_widened_mlp_still_takes_a_lora_adapter():
    """nn.Linear caches in/out features; PEFT sizes lora_A/lora_B from them."""
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model

    model = _tiny()
    scaling.expand_model(model, mlp_width_target = 256, method = "mlp_width", force_small_growth = True)
    model = get_peft_model(
        model,
        LoraConfig(r = 4, lora_alpha = 8, target_modules = "all-linear", task_type = "CAUSAL_LM"),
    )
    model.train()
    out = model(input_ids = _ids(), labels = _ids())
    out.loss.backward()  # a stale out_features fails here, not at save time
    assert model.num_parameters(only_trainable = True) > 0


def test_lora_has_strictly_more_trainable_params_after_expansion():
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model

    base = get_peft_model(
        _tiny(seed = 5),
        LoraConfig(r = 8, lora_alpha = 16, target_modules = "all-linear", task_type = "CAUSAL_LM"),
    )
    expanded_plain = _tiny(seed = 5)
    scaling.expand_model(expanded_plain, num_new_layers = 2, force_small_growth = True)
    expanded = get_peft_model(
        expanded_plain,
        LoraConfig(r = 8, lora_alpha = 16, target_modules = "all-linear", task_type = "CAUSAL_LM"),
    )
    assert expanded.num_parameters(only_trainable = True) > base.num_parameters(only_trainable = True)


# ---------------------------------------------------------------------------
# evaluate_pipeline
# ---------------------------------------------------------------------------


def test_verdict_is_vs_base_only_without_a_same_compute_baseline():
    base, candidate = _tiny(), _tiny()
    base._scores = {"a": 0.2, "b": 0.3}
    candidate._scores = {"a": 0.8, "b": 0.6}

    result = scaling.evaluate_pipeline(
        base, candidate, _scores(base._scores), tokenizer = _FakeTokenizer()
    )
    assert result["verdict"] == "better"
    assert result["comparison"] == "vs base only"
    assert result["mean_delta_points"] == pytest.approx((0.7 - 0.25) * 100)

    unchanged = _tiny()
    unchanged._scores = dict(base._scores)
    same = scaling.evaluate_pipeline(
        base, unchanged, _scores(base._scores), tokenizer = _FakeTokenizer()
    )
    assert same["verdict"] == "same"

    worse_candidate = _tiny()
    worse_candidate._scores = {"a": 0.0, "b": 0.0}
    worse = scaling.evaluate_pipeline(
        base, worse_candidate, _scores(base._scores), tokenizer = _FakeTokenizer()
    )
    assert worse["verdict"] == "worse"


def test_a_same_compute_baseline_changes_the_label_and_the_table():
    base, candidate, baseline = _tiny(), _tiny(), _tiny()
    base._scores = {"a": 0.2}
    candidate._scores = {"a": 0.6}
    baseline._scores = {"a": 0.5}
    result = scaling.evaluate_pipeline(
        base,
        candidate,
        _scores(base._scores),
        tokenizer = _FakeTokenizer(),
        same_compute_baseline = baseline,
        gpu_hours_expanded = 1.5,
        gpu_hours_baseline = 1.2,
    )
    assert result["comparison"] == "vs same-compute baseline"
    assert result["same_compute"]["mean_delta_points"] == pytest.approx(10.0)
    assert result["gpu_hours_baseline"] == 1.2
    assert result["gpu_hours_expanded"] == 1.5


def test_a_single_task_suite_still_clears_the_bar():
    base, candidate = _tiny(), _tiny()
    base._scores = {"only": 0.2}
    candidate._scores = {"only": 0.8}
    result = scaling.evaluate_pipeline(
        base, candidate, _scores(base._scores), tokenizer = _FakeTokenizer()
    )
    assert result["verdict"] == "better"


def test_a_one_task_gain_is_not_better():
    base, candidate = _tiny(), _tiny()
    base._scores = {"a": 0.2, "b": 0.3}
    candidate._scores = {"a": 0.9, "b": 0.3}
    result = scaling.evaluate_pipeline(
        base, candidate, _scores(base._scores), tokenizer = _FakeTokenizer()
    )
    assert result["verdict"] == "same"
    assert "bar is 2 task(s)" in result["reason"]


# ---------------------------------------------------------------------------
# cpt_train
# ---------------------------------------------------------------------------


def test_cpt_train_reports_recovery_cost_and_a_best_checkpoint(tmp_path):
    from datasets import Dataset

    model = _tiny()
    scaling.expand_model(model, num_new_layers = 1, force_small_growth = True)

    texts = [
        "the quick brown fox jumps over the lazy dog",
        "pack my box with five dozen liquor jugs",
    ]
    dataset = Dataset.from_dict({"text": texts * 8})
    tokenizer = _FakeTokenizer()
    suite = scaling.heldout_perplexity_suite(texts, max_length = 32)

    result = scaling.cpt_train(
        model,
        tokenizer,
        dataset,
        eval_suite = suite,
        max_steps = 4,
        eval_every = 2,
        patience = 2,
        learning_rate = 5e-2,
        per_device_batch_size = 2,
        gradient_accumulation_steps = 1,
        max_seq_length = 32,
        use_lora = True,
        lora_rank = 8,
        gradient_checkpointing = False,
        output_dir = str(tmp_path / "cpt"),
    )

    assert result["steps_run"] == 4
    assert result["tokens_seen"] > 0
    assert result["gpu_hours"] >= 0.0
    assert result["base_scores_source"] == "step-0-expanded"
    assert len(result["history"]) >= 2
    assert result["best_checkpoint"] and Path(result["best_checkpoint"]).is_dir()
    # One block at the even gap after block 2, so it lands at index 3 of the new list.
    assert result["new_layer_indices"] == [3]
    assert result["new_layer_grad_nonzero"] is True
    # Recovery is recorded exactly when some eval cleared the anchor.
    improved = max(entry["mean"] for entry in result["history"]) > sum(
        result["base_scores"].values()
    ) / len(result["base_scores"])
    assert (result["recovery_step"] is not None) == improved
    assert result["recovery_step"] is None or result["recovery_step"] > 0


def test_cpt_train_uses_a_supplied_anchor_and_rejects_a_bad_suite(tmp_path):
    from datasets import Dataset

    model = _tiny()
    dataset = Dataset.from_dict({"text": ["a b c"] * 4})
    tokenizer = _FakeTokenizer()

    result = scaling.cpt_train(
        model,
        tokenizer,
        dataset,
        eval_suite = {"heldout": lambda model, tokenizer: 0.5},
        base_scores = {"heldout": 0.4},
        max_steps = 1,
        eval_every = 1,
        learning_rate = 1e-2,
        per_device_batch_size = 1,
        gradient_accumulation_steps = 1,
        max_seq_length = 16,
        use_lora = True,
        lora_rank = 4,
        gradient_checkpointing = False,
        output_dir = str(tmp_path / "cpt2"),
    )
    assert result["base_scores_source"] == "provided"
    assert result["recovery_step"] == 1

    with pytest.raises(ValueError, match = "must be a mapping"):
        scaling.cpt_train(
            _tiny(),
            tokenizer,
            dataset,
            eval_suite = lambda model, tokenizer: 1.0,
            max_steps = 1,
            output_dir = str(tmp_path / "cpt3"),
        )
