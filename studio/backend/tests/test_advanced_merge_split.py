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


def test_every_parent_in_a_merge_plans_to_the_same_target():
    """3+ parents need one target, or their tensors never share a shape.

    Planning each parent against the first gave parent 2 a 1024-wide target and
    parent 3 a 1536-wide one, so a key owned by all three could not be added up
    and the merge quietly copied a single parent instead.
    """
    specs = [
        _spec("qwen2", 2048, 32, 16, 8192, 151000),
        _spec("qwen2", 1024, 24, 16, 4096, 151000),
        _spec("qwen2", 1536, 28, 16, 6144, 151000),
    ]
    plans = merge.plan_alignment_many(specs)

    assert len(plans) == 3
    # The smallest dimension any parent has, for every parent.
    assert {plan.target_hidden for plan in plans} == {1024}
    assert {plan.target_layers for plan in plans} == {24}
    assert {plan.target_intermediate for plan in plans} == {4096}
    # The largest parent has to move too; leaving it at its own width was the bug.
    assert plans[0].hidden_strategy == "crop"
    assert plans[0].mode == "cross_scale"


def test_a_shared_shape_group_plans_as_identical():
    specs = [_spec("llama", 2048, 24, 16, 8192, 32000) for _ in range(4)]
    plans = merge.plan_alignment_many(specs)
    assert {plan.mode for plan in plans} == {"identical"}
    assert plans[0].target_hidden == 2048


def test_a_cross_architecture_parent_still_recommends_mmoe():
    specs = [
        _spec("llama", 2048, 24, 16, 8192, 32000),
        _spec("llama", 2048, 24, 16, 8192, 32000),
        _spec("qwen3", 2048, 24, 16, 8192, 32000),
    ]
    plans = merge.plan_alignment_many(specs)
    assert [plan.mmoe_required for plan in plans] == [False, False, True]
    assert any("MMoE" in warning for warning in plans[2].warnings)


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


def _write_bf16_checkpoint(directory: Path, tensors: dict, *, vocab: int = 16) -> Path:
    """A bf16 checkpoint on disk, config and all, like a Hub download."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    directory.mkdir(parents = True, exist_ok = True)
    (directory / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "intermediate_size": 8,
                "vocab_size": vocab,
            }
        ),
        encoding="utf-8",
    )
    payload = {
        key: (value.to(torch.bfloat16) if value.is_floating_point() else value).contiguous()
        for key, value in tensors.items()
    }
    save_file(payload, str(directory / "model.safetensors"))
    return directory


def test_merge_bfloat16_checkpoints_keeps_dtype(tmp_path: Path):
    """A bf16 pair used to die on read: numpy has no bfloat16 dtype."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from safetensors import safe_open

    generator = torch.Generator().manual_seed(0)
    a_tensors = {
        "model.embed_tokens.weight": torch.randn(16, 8, generator = generator),
        "model.layers.0.mlp.down_proj.weight": torch.randn(8, 8, generator = generator),
        "model.layers.0.self_attn.rotary_emb.inv_freq": torch.arange(4, dtype = torch.int64),
    }
    b_tensors = {
        "model.embed_tokens.weight": torch.randn(16, 8, generator = generator),
        "model.layers.0.mlp.down_proj.weight": torch.randn(8, 8, generator = generator),
        "model.layers.0.self_attn.rotary_emb.inv_freq": torch.arange(4, dtype = torch.int64),
    }
    model_a = _write_bf16_checkpoint(tmp_path / "a", a_tensors)
    model_b = _write_bf16_checkpoint(tmp_path / "b", b_tensors)

    out = tmp_path / "merged"
    report = merge.merge_checkpoints(
        str(model_a), str(model_b), str(out), method = "slerp", weight = 0.75
    )
    assert report["status"] == "completed"
    assert report["metrics"]["blended_tensors"] == 2
    assert report["metrics"]["skipped_tensors"] == 1
    # bf16 weights are 2 bytes each; the int64 bookkeeping tensor keeps its 8.
    assert report["metrics"]["total_bytes"] == 16 * 8 * 2 + 8 * 8 * 2 + 4 * 8
    assert (out / "merge_report.json").is_file()

    weight_map = json.loads((out / "model.safetensors.index.json").read_text(encoding = "utf-8"))["weight_map"]
    assert set(weight_map) == set(a_tensors)
    with safe_open(str(out / weight_map["model.embed_tokens.weight"]), framework = "pt") as handle:
        merged = handle.get_tensor("model.embed_tokens.weight")
        integers = handle.get_tensor("model.layers.0.self_attn.rotary_emb.inv_freq")
    assert merged.dtype == torch.bfloat16
    assert merged.shape == (16, 8)
    assert integers.dtype == torch.int64
    assert torch.equal(integers, a_tensors["model.layers.0.self_attn.rotary_emb.inv_freq"])
    # SLERP between two close-to-orthogonal draws is not a plain lerp, but the
    # output must still be a unit-ish blend of the two: check it moved off both.
    a_embed = a_tensors["model.embed_tokens.weight"].to(torch.float32)
    b_embed = b_tensors["model.embed_tokens.weight"].to(torch.float32)
    merged_embed = merged.to(torch.float32)
    assert not torch.allclose(merged_embed, a_embed, atol = 1e-2)
    assert not torch.allclose(merged_embed, b_embed, atol = 1e-2)


def test_split_bfloat16_checkpoint_sizes_and_reloads(tmp_path: Path):
    """bf16 tensors are 2 bytes each, and the shards must keep the dtype."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from safetensors import safe_open

    generator = torch.Generator().manual_seed(1)
    tensors = {
        "a": torch.randn(32, 64, generator = generator),
        "b": torch.randn(8, 8, generator = generator),
    }
    source = _write_bf16_checkpoint(tmp_path / "src", tensors)

    inventory = splitter._inventory(str(source))
    by_key = {key: nbytes for key, _shape, nbytes in inventory}
    assert by_key["a"] == 32 * 64 * 2
    assert by_key["b"] == 8 * 8 * 2

    out = tmp_path / "split"
    manifest = splitter.split_checkpoint(
        str(source), str(out), max_shard_bytes = 5_000, reserve_fraction = 0.0, prefer = "ram"
    )
    assert manifest["status"] == "completed"
    assert manifest["total_bytes"] == sum(by_key.values())

    weight_map = json.loads((out / "model.safetensors.index.json").read_text(encoding = "utf-8"))["weight_map"]
    for key, shard in weight_map.items():
        with safe_open(str(out / shard), framework = "pt") as handle:
            restored = handle.get_tensor(key)
        assert restored.dtype == torch.bfloat16
        assert torch.equal(restored, tensors[key].to(torch.bfloat16))


def test_slerp_keeps_the_tensor_shape():
    """The 1-D SLERP math used to leak out, writing flat tensors into the checkpoint."""
    rng = np.random.default_rng(3)
    a = rng.normal(size = (4, 6)).astype(np.float32)
    b = rng.normal(size = (4, 6)).astype(np.float32)
    blended = merge.blend_tensors([a, b], method = "slerp", weights = [0.25, 0.75])
    assert blended.shape == (4, 6)

    c = rng.normal(size = (4, 6)).astype(np.float32)
    blended_three = merge.blend_tensors([a, b, c], method = "slerp")
    assert blended_three.shape == (4, 6)

    cube = rng.normal(size = (2, 3, 4)).astype(np.float32)
    other = rng.normal(size = (2, 3, 4)).astype(np.float32)
    assert merge.blend_tensors([cube, other], method = "slerp").shape == (2, 3, 4)

    # Endpoints still come back exactly, at the right shape.
    assert np.allclose(merge.blend_tensors([a, b], method = "slerp", weights = [1, 0]), a)


def test_merge_bfloat16_aligns_mismatched_rows(tmp_path: Path):
    """Padding a bf16 tensor goes through float32 and must come back shaped."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from safetensors import safe_open

    generator = torch.Generator().manual_seed(7)
    small = {"model.embed_tokens.weight": torch.randn(12, 8, generator = generator)}
    full = {"model.embed_tokens.weight": torch.randn(16, 8, generator = generator)}
    model_a = _write_bf16_checkpoint(tmp_path / "a", small, vocab = 16)
    model_b = _write_bf16_checkpoint(tmp_path / "b", full, vocab = 16)

    out = tmp_path / "merged"
    report = merge.merge_checkpoints(str(model_a), str(model_b), str(out), method = "linear", weight = 0.5)
    assert report["metrics"]["blended_tensors"] == 1

    weight_map = json.loads((out / "model.safetensors.index.json").read_text(encoding = "utf-8"))["weight_map"]
    with safe_open(str(out / weight_map["model.embed_tokens.weight"]), framework = "pt") as handle:
        merged = handle.get_tensor("model.embed_tokens.weight")
    assert merged.dtype == torch.bfloat16
    assert merged.shape == (16, 8)
    # Rows the small model never had stay at half the full model's value.
    expected_tail = full["model.embed_tokens.weight"][12:].to(torch.float32) * 0.5
    assert torch.allclose(merged[12:].to(torch.float32), expected_tail, atol = 1e-2)


# ---------------------------------------------------------------------------
# N-way merges (2 to 6 parents)
# ---------------------------------------------------------------------------


def _values_checkpoint(tmp_path: Path, name: str, value: float, *, with_probe: bool = True) -> Path:
    torch = pytest.importorskip("torch")
    # Match the config _write_bf16_checkpoint writes (vocab 16, hidden 8), so the
    # alignment plan is the identity and the test measures the blend, not padding.
    tensors = {"model.embed_tokens.weight": torch.full((16, 8), float(value))}
    if with_probe:
        tensors["model.norm.weight"] = torch.full((8,), float(value))
    return _write_bf16_checkpoint(tmp_path / name, tensors)


def _merged_float(out: Path, key: str = "model.embed_tokens.weight"):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from safetensors import safe_open

    weight_map = json.loads(
        (out / "model.safetensors.index.json").read_text(encoding = "utf-8")
    )["weight_map"]
    with safe_open(str(out / weight_map[key]), framework = "pt") as handle:
        tensor = handle.get_tensor(key).to("cpu", torch.float32)
    return tensor


def test_three_way_merge_uses_normalized_weights(tmp_path: Path):
    pytest.importorskip("torch")
    a = _values_checkpoint(tmp_path, "a", 1.0)
    b = _values_checkpoint(tmp_path, "b", 2.0)
    c = _values_checkpoint(tmp_path, "c", 6.0)
    out = tmp_path / "merged"

    report = merge.merge_checkpoints_many(
        [str(a), str(b), str(c)], str(out), method = "linear", weights = [1, 1, 2]
    )

    assert report["status"] == "completed"
    assert report["model_count"] == 3
    assert report["weights"] == [0.25, 0.25, 0.5]
    assert len(report["models"]) == 3
    assert len(report["alignments"]) == 3
    assert report["metrics"]["blended_tensors"] == 2
    # 0.25*1 + 0.25*2 + 0.5*6 = 3.75
    assert _merged_float(out).mean().item() == pytest.approx(3.75, abs = 2e-2)


def test_a_key_missing_from_one_parent_is_reweighted_not_zeroed(tmp_path: Path):
    torch = pytest.importorskip("torch")
    a = _values_checkpoint(tmp_path, "a", 1.0, with_probe = False)
    b = _values_checkpoint(tmp_path, "b", 3.0, with_probe = False)
    # The third parent owns only the norm tensor, so it contributes nothing to
    # the embed blend: [1, 1, 2] becomes [0.5, 0.5] there.
    c = _write_bf16_checkpoint(
        tmp_path / "c", {"model.norm.weight": torch.full((8,), 9.0)}
    )
    out = tmp_path / "merged"

    merge.merge_checkpoints_many(
        [str(a), str(b), str(c)], str(out), method = "linear", weights = [1, 1, 2]
    )

    assert _merged_float(out).mean().item() == pytest.approx(2.0, abs = 2e-2)
    # The norm tensor exists only in the third parent, so it is copied through.
    assert _merged_float(out, "model.norm.weight").mean().item() == pytest.approx(
        9.0, abs = 2e-2
    )


def test_merge_progress_reports_stages_and_tensor_counts(tmp_path: Path):
    pytest.importorskip("torch")
    a = _values_checkpoint(tmp_path, "a", 1.0)
    b = _values_checkpoint(tmp_path, "b", 2.0)
    events: list[dict] = []

    merge.merge_checkpoints_many(
        [str(a), str(b)],
        str(tmp_path / "merged"),
        method = "linear",
        progress = events.append,
    )

    stages = [event["stage"] for event in events]
    assert stages[0] == "resolving"
    assert "merging" in stages
    assert stages[-1] == "completed"
    merging = [event for event in events if event["stage"] == "merging"]
    assert merging[-1]["done"] == merging[-1]["total"]
    assert merging[-1]["blended_tensors"] == 2
    assert merging[-1]["bytes_written"] > 0


def test_merge_rejects_a_bad_parent_count_or_weight_list(tmp_path: Path):
    pytest.importorskip("torch")
    a = _values_checkpoint(tmp_path, "a", 1.0)
    b = _values_checkpoint(tmp_path, "b", 2.0)
    with pytest.raises(ValueError, match = "at least two"):
        merge.merge_checkpoints_many([str(a)], str(tmp_path / "one"))
    with pytest.raises(ValueError, match = "At most"):
        merge.merge_checkpoints_many(
            [str(a)] * 7, str(tmp_path / "seven")
        )
    with pytest.raises(ValueError, match = "Expected 2 weights"):
        merge.merge_checkpoints_many(
            [str(a), str(b)], str(tmp_path / "bad-weights"), weights = [1.0]
        )
