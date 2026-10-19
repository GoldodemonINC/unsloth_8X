# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for merging, MMoE routing, and the checkpoint splitter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.advanced import merge, splitter


def _spec(architecture: str, hidden: int, layers: int, heads: int, ffn: int, vocab: int):
    return merge.ModelSpec(
        source=f"/tmp/{architecture}-{hidden}",
        architecture=architecture,
        hidden_size=hidden,
        num_layers=layers,
        num_heads=heads,
        intermediate_size=ffn,
        vocab_size=vocab,
    )


def test_alignment_modes():
    identical = merge.plan_alignment(
        _spec("llama", 2048, 24, 16, 8192, 32000),
        _spec("llama", 2048, 24, 16, 8192, 32000),
    )
    assert identical.mode == "identical"
    assert identical.hidden_strategy == "keep"

    cross_scale = merge.plan_alignment(
        _spec("qwen2", 1024, 24, 16, 4096, 151000),
        _spec("qwen2", 2048, 28, 16, 8192, 151000),
    )
    assert cross_scale.mode == "cross_scale"
    assert cross_scale.target_hidden == 1024
    assert cross_scale.target_layers == 24
    assert cross_scale.hidden_strategy == "crop"

    cross_arch = merge.plan_alignment(
        _spec("llama", 2048, 24, 16, 8192, 32000),
        _spec("qwen3", 3584, 28, 28, 18944, 151000),
    )
    assert cross_arch.mode == "cross_architecture"
    assert cross_arch.mmoe_required is True
    assert any("MMoE" in warning for warning in cross_arch.warnings)


def test_align_tensor_crop_pad_and_svd():
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    cropped = merge.align_tensor(arr, (2, 3))
    assert cropped.shape == (2, 3)
    padded = merge.align_tensor(arr, (5, 6))
    assert padded.shape == (5, 6)
    assert np.all(padded[3:, :] == 0)
    projected = merge.align_tensor(arr, (2, 2), method="svd")
    assert projected.shape == (2, 2)


def test_slerp_endpoints_and_midpoint():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert np.allclose(merge.blend_tensors([a, b], method="slerp", weights=[1, 0]), a)
    assert np.allclose(merge.blend_tensors([a, b], method="slerp", weights=[0, 1]), b)
    mid = merge.blend_tensors([a, b], method="slerp", weights=[0.5, 0.5])
    assert np.isclose(np.linalg.norm(mid), 1.0, atol=1e-4)


def test_ties_and_dare_are_deterministic():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(8, 8)).astype(np.float32)
    b = a + rng.normal(scale=0.05, size=(8, 8)).astype(np.float32)
    ties_first = merge.blend_tensors([a, b], method="ties", weights=[0.5, 0.5], density=0.5)
    ties_second = merge.blend_tensors([a, b], method="ties", weights=[0.5, 0.5], density=0.5)
    assert np.array_equal(ties_first, ties_second)

    dare_first = merge.blend_tensors([a, b], method="dare", weights=[0.5, 0.5], density=0.3, seed=5)
    dare_second = merge.blend_tensors([a, b], method="dare", weights=[0.5, 0.5], density=0.3, seed=5)
    assert np.array_equal(dare_first, dare_second)
    assert not np.allclose(dare_first, (a + b) / 2, atol=1e-6)


def test_probe_spec_and_inspect(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "intermediate_size": 128,
                "vocab_size": 256,
            }
        ),
        encoding="utf-8",
    )
    spec = merge.probe_model_spec(str(model_dir))
    assert spec.hidden_size == 64
    assert spec.num_layers == 4
    assert spec.head_dim == 16
    inspected = merge.inspect_checkpoint(str(model_dir))
    assert inspected["tensor_count"] == 0
    assert "error" in inspected


def test_merge_state_dicts_copies_and_blends():
    state_a = {"w": np.ones((2, 2), dtype=np.float32), "only_a": np.zeros((1,), dtype=np.float32)}
    state_b = {"w": np.zeros((2, 2), dtype=np.float32), "only_b": np.ones((1,), dtype=np.float32)}
    merged, report = merge.merge_state_dicts([state_a, state_b], method="linear")
    assert np.allclose(merged["w"], 0.5)
    assert report["blended_tensors"] == 1
    assert report["copied_tensors"] == 2


def test_moe_forward_and_balance():
    torch = pytest.importorskip("torch")
    from core.advanced import moe

    layer = moe.build_moe_layer(32, [16, 48], top_k=2, temperature=0.8, router_hidden=24)
    hidden = torch.randn(2, 5, 32, requires_grad=True)
    output = layer(hidden)
    assert output.shape == (2, 5, 32)
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert hidden.grad is not None
    assert float(layer.aux_loss().detach()) >= 0.0
    config = layer.export_config()
    assert config["expert_widths"] == [16, 48]
    assert config["top_k"] == 2


def test_split_plan_and_streaming_split(tmp_path: Path):
    pytest.importorskip("safetensors")
    from safetensors.numpy import save_file

    source = tmp_path / "src"
    source.mkdir()
    tensors = {
        "a": np.zeros((32, 64), dtype=np.float32),
        "b": np.ones((8, 8), dtype=np.float32),
        "c": np.arange(16, dtype=np.float32),
    }
    save_file(tensors, str(source / "model.safetensors"))
    (source / "config.json").write_text("{}", encoding="utf-8")

    inventory = splitter._inventory(str(source))
    assert len(inventory) == 3
    plan = splitter.plan_shards(inventory, max_shard_bytes=5_000)
    assert len(plan.shards) >= 2
    assert sum(len(shard) for shard in plan.shards) == 3

    out = tmp_path / "out"
    manifest = splitter.split_checkpoint(
        str(source),
        str(out),
        max_shard_bytes=5_000,
        reserve_fraction=0.0,
        prefer="ram",
    )
    assert manifest["status"] == "completed"
    assert len(manifest["files"]) >= 2
    index = json.loads((out / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert set(index["weight_map"]) == {"a", "b", "c"}
    assert (out / "split_manifest.json").is_file()


def test_split_oversized_tensor_gets_own_shard():
    plan = splitter.plan_shards(
        [("a", (2, 2), 100), ("huge", (4096, 4096), 64 * 1024 * 1024), ("b", (2, 2), 100)],
        max_shard_bytes=1024,
    )
    assert plan.oversized_tensors == 1
    huge_shard = next(shard for shard in plan.shards if shard[0].key == "huge")
    assert len(huge_shard) == 1
    assert any("larger than" in warning for warning in plan.warnings)


def test_hf_repo_id_accepts_hub_forms(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert merge.hf_repo_id("unsloth/Llama-3.2-1B") == "unsloth/Llama-3.2-1B"
    assert merge.hf_repo_id("hf://unsloth/Llama-3.2-1B") == "unsloth/Llama-3.2-1B"
    assert (
        merge.hf_repo_id("https://huggingface.co/unsloth/Llama-3.2-1B")
        == "unsloth/Llama-3.2-1B"
    )
    # A local path that happens to look like org/name wins over the Hub.
    local = tmp_path / "samples" / "model"
    local.mkdir(parents=True)
    assert merge.hf_repo_id("samples/model") is None
    # Windows paths and absolute paths are never repo ids.
    assert merge.hf_repo_id("C:\models\llama") is None
    assert merge.hf_repo_id(str(tmp_path / "llama")) is None
    assert merge.hf_repo_id("just-a-name") is None


def test_resolve_checkpoint_source_rejects_unknown_path(tmp_path):
    with pytest.raises(ValueError, match="Hugging Face repo id"):
        merge.resolve_checkpoint_source(str(tmp_path / "missing"))


def test_resolve_checkpoint_source_returns_local_path(tmp_path):
    local = tmp_path / "ckpt"
    local.mkdir()
    assert merge.resolve_checkpoint_source(str(local)) == str(local)
