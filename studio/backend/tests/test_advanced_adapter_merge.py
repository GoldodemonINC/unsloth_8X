# SPDX-License-Identifier: NRC-2
# Copyright 2026-present the X8 authors. See /LICENSE-NRC-2.md

"""Tests for merging LoRA adapters.

The arithmetic that matters is the reconstructed delta: a merged adapter must
apply the blend the user asked for, which is checked by multiplying the written
A and B back together.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from core.advanced import adapter_merge

# Both modules must be at least as wide as the merged rank (2 + 2): a PEFT
# adapter carries one r for the whole model, so a narrow module caps every one.
MODULES = {"model.layers.0.self_attn.q_proj": (6, 6), "model.layers.0.mlp.down_proj": (5, 5)}


def _adapter(root, name, *, rank = 2, alpha = 4, base = "org/base", seed = 0, modules = None):
    modules = modules or MODULES
    rng = np.random.default_rng(seed)
    tensors = {}
    for module, (out_features, in_features) in modules.items():
        tensors[f"{module}.lora_A.weight"] = rng.normal(size = (rank, in_features)).astype(np.float32)
        tensors[f"{module}.lora_B.weight"] = rng.normal(size = (out_features, rank)).astype(np.float32)
    path = root / name
    path.mkdir()
    from safetensors.numpy import save_file

    save_file(tensors, str(path / "adapter_model.safetensors"))
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": rank,
                "lora_alpha": alpha,
                "base_model_name_or_path": base,
                "target_modules": list(modules),
            }
        ),
        encoding = "utf-8",
    )
    return path


def _delta(directory) -> dict[str, np.ndarray]:
    adapter = adapter_merge.read_adapter(directory)
    return adapter_merge.adapter_deltas(adapter)


def test_an_adapter_reads_its_delta(tmp_path):
    a = _adapter(tmp_path, "a")
    b = _adapter(tmp_path, "b")

    deltas = _delta(a)

    assert set(deltas) == set(MODULES)
    assert deltas["model.layers.0.self_attn.q_proj"].shape == (6, 6)


def test_linear_merge_reconstructs_the_weighted_deltas(tmp_path):
    a = _adapter(tmp_path, "a", seed = 1)
    b = _adapter(tmp_path, "b", seed = 2)
    expected = 0.5 * (_delta(a)["model.layers.0.self_attn.q_proj"] + _delta(b)["model.layers.0.self_attn.q_proj"])

    report = adapter_merge.merge_adapters([a, b], tmp_path / "out")

    merged = _delta(tmp_path / "out")["model.layers.0.self_attn.q_proj"]
    assert np.allclose(merged, expected, atol = 1e-5)
    assert report["output_rank"] == 4


def test_a_weight_of_one_keeps_only_that_adapter(tmp_path):
    a = _adapter(tmp_path, "a", seed = 3)
    b = _adapter(tmp_path, "b", seed = 4)
    expected = _delta(a)["model.layers.0.mlp.down_proj"]

    adapter_merge.merge_adapters([a, b], tmp_path / "out", weights = [1.0, 0.0])

    assert np.allclose(
        _delta(tmp_path / "out")["model.layers.0.mlp.down_proj"], expected, atol = 1e-5
    )


@pytest.mark.parametrize("method", ["ties", "dare"])
def test_the_other_methods_write_finite_tensors(tmp_path, method):
    a = _adapter(tmp_path, "a", seed = 5)
    b = _adapter(tmp_path, "b", seed = 6)

    report = adapter_merge.merge_adapters([a, b], tmp_path / "out", method = method, density = 0.5)

    assert report["method"] == method
    for delta in _delta(tmp_path / "out").values():
        assert np.isfinite(delta).all()


def test_the_output_rank_can_be_capped(tmp_path):
    a = _adapter(tmp_path, "a")
    b = _adapter(tmp_path, "b")

    report = adapter_merge.merge_adapters([a, b], tmp_path / "out", rank = 1)

    assert report["output_rank"] == 1
    tensor = _delta(tmp_path / "out")["model.layers.0.self_attn.q_proj"]
    assert tensor.shape == (6, 6)


def test_the_written_config_does_not_double_apply_scaling(tmp_path):
    a = _adapter(tmp_path, "a")
    b = _adapter(tmp_path, "b")

    adapter_merge.merge_adapters([a, b], tmp_path / "out")

    config = json.loads((tmp_path / "out" / "adapter_config.json").read_text(encoding = "utf-8"))
    assert config["r"] == config["lora_alpha"]
    assert config["use_rslora"] is False
    assert (tmp_path / "out" / "x8_adapter_merge.json").is_file()


def test_adapters_for_different_bases_are_refused(tmp_path):
    a = _adapter(tmp_path, "a", base = "org/one")
    b = _adapter(tmp_path, "b", base = "org/two")

    with pytest.raises(adapter_merge.AdapterMergeError) as error:
        adapter_merge.merge_adapters([a, b], tmp_path / "out")

    assert "different base models" in str(error.value)


def test_slerp_is_refused_for_a_delta(tmp_path):
    a = _adapter(tmp_path, "a")
    b = _adapter(tmp_path, "b")

    with pytest.raises(adapter_merge.AdapterMergeError) as error:
        adapter_merge.merge_adapters([a, b], tmp_path / "out", method = "slerp")

    assert "linear, ties, dare" in str(error.value)


def test_a_full_checkpoint_is_refused_with_a_useful_message(tmp_path):
    checkpoint = tmp_path / "full"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding = "utf-8")

    with pytest.raises(adapter_merge.AdapterMergeError) as error:
        adapter_merge.read_adapter(checkpoint)

    assert "Merge studio" in str(error.value)


def test_one_adapter_is_not_a_merge(tmp_path):
    a = _adapter(tmp_path, "a")

    with pytest.raises(adapter_merge.AdapterMergeError):
        adapter_merge.merge_adapters([a], tmp_path / "out")


def test_weights_must_match_the_adapters(tmp_path):
    a = _adapter(tmp_path, "a")
    b = _adapter(tmp_path, "b")

    with pytest.raises(adapter_merge.AdapterMergeError):
        adapter_merge.merge_adapters([a, b], tmp_path / "out", weights = [1.0])


def test_adapters_with_no_shared_module_are_refused(tmp_path):
    a = _adapter(tmp_path, "a", modules = {"model.layers.0.self_attn.q_proj": (6, 6)})
    b = _adapter(tmp_path, "b", modules = {"model.layers.0.mlp.down_proj": (5, 5)})

    with pytest.raises(adapter_merge.AdapterMergeError) as error:
        adapter_merge.merge_adapters([a, b], tmp_path / "out")

    assert "no LoRA module in common" in str(error.value)
