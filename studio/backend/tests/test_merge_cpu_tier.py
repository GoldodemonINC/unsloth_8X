# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The CPU merge tier: three parents, lazy row chunks, RAM-bounded shards."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.advanced import merge, splitter


def _write_fp32_checkpoint(
    directory: Path,
    tensors: dict[str, np.ndarray],
    *,
    vocab: int = 16,
    dtype: type = np.float32,
) -> Path:
    """A float safetensors checkpoint with the config the alignment plan reads."""
    pytest.importorskip("safetensors")
    from safetensors.numpy import save_file

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
        encoding = "utf-8",
    )
    save_file(
        {key: value.astype(dtype) for key, value in tensors.items()},
        str(directory / "model.safetensors"),
    )
    return directory


def _value_checkpoint(tmp_path: Path, name: str, value: float, *, rows: int = 16) -> Path:
    return _write_fp32_checkpoint(
        tmp_path / name,
        {
            "model.embed_tokens.weight": np.full((rows, 8), float(value)),
            "model.norm.weight": np.full((8,), float(value)),
        },
    )


def _merged_array(out: Path, key: str = "model.embed_tokens.weight") -> np.ndarray:
    pytest.importorskip("safetensors")
    from safetensors import safe_open

    weight_map = json.loads(
        (out / "model.safetensors.index.json").read_text(encoding = "utf-8")
    )["weight_map"]
    with safe_open(str(out / weight_map[key]), framework = "np") as handle:
        return np.asarray(handle.get_tensor(key), dtype = np.float32)


def test_cpu_merge_rejects_more_parents_than_ram_allows(tmp_path: Path, monkeypatch):
    """The cap is a RAM budget, and an eight-parent request blows it."""
    monkeypatch.setattr(merge, "host_has_gpu", lambda: False)
    sources = [_value_checkpoint(tmp_path, f"model-{index}", float(index + 1)) for index in range(8)]

    with pytest.raises(ValueError, match = "At most 6 checkpoints"):
        merge.merge_checkpoints_many([str(source) for source in sources], str(tmp_path / "out"))


def test_a_ram_starved_cpu_host_stops_at_two(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(merge, "host_has_gpu", lambda: False)
    monkeypatch.setattr(merge, "free_ram_bytes", lambda: 40 * 1024**2)
    sources = [_value_checkpoint(tmp_path, f"m{index}", float(index + 1)) for index in range(3)]

    with pytest.raises(ValueError, match = "can merge at most 2"):
        merge.merge_checkpoints_many([str(source) for source in sources], str(tmp_path / "out"))


def test_cpu_three_parent_merge_reports_the_cpu_tier(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(merge, "host_has_gpu", lambda: False)
    a = _value_checkpoint(tmp_path, "a", 1.0)
    b = _value_checkpoint(tmp_path, "b", 2.0)
    c = _value_checkpoint(tmp_path, "c", 6.0)
    out = tmp_path / "merged"

    report = merge.merge_checkpoints_many(
        [str(a), str(b), str(c)], str(out), method = "linear", weights = [1, 1, 2]
    )

    assert report["status"] == "completed"
    assert report["tier"] == "cpu"
    assert report["model_count"] == 3
    assert report["parent_cap_reason"] == "ram"
    assert report["max_models"] >= 3
    # Both floating keys take the lazy row-chunk path; nothing is blended whole.
    assert report["metrics"]["blended_tensors"] == 2
    assert report["metrics"]["streamed_tensors"] == 2
    assert report["metrics"]["shard_buffer_bytes"] > 0
    # 0.25*1 + 0.25*2 + 0.5*6 = 3.75
    assert _merged_array(out).mean() == pytest.approx(3.75, abs = 2e-2)


def test_gpu_host_still_accepts_four_parents(tmp_path: Path, monkeypatch):
    """The three-parent tier must not leak into a host that has a GPU."""
    monkeypatch.setattr(merge, "host_has_gpu", lambda: True)
    sources = [_value_checkpoint(tmp_path, f"m{index}", float(index + 1)) for index in range(4)]
    out = tmp_path / "merged"

    report = merge.merge_checkpoints_many(
        [str(source) for source in sources], str(out), method = "linear"
    )

    assert report["model_count"] == 4
    assert report["tier"] == "gpu"
    assert report["max_models"] == merge.MAX_MERGE_MODELS


def test_row_chunking_matches_a_whole_tensor_blend(tmp_path: Path, monkeypatch):
    """Chunk size is a memory knob only: the blended values must not move."""
    monkeypatch.setattr(merge, "host_has_gpu", lambda: False)
    rng = np.random.default_rng(0)
    # 16 rows is the checkpoint's declared vocab, so alignment is the identity
    # and the key is eligible for the row-chunk path.
    left = rng.normal(size = (16, 8)).astype(np.float32)
    right = rng.normal(size = (16, 8)).astype(np.float32)
    a = _write_fp32_checkpoint(tmp_path / "a", {"model.embed_tokens.weight": left})
    b = _write_fp32_checkpoint(tmp_path / "b", {"model.embed_tokens.weight": right})

    whole = tmp_path / "whole"
    chunked = tmp_path / "chunked"
    monkeypatch.setattr(merge, "chunk_rows", lambda shape, parents: 16)
    merge.merge_checkpoints(str(a), str(b), str(whole), method = "linear", weight = 0.25)
    monkeypatch.setattr(merge, "chunk_rows", lambda shape, parents: 3)
    report = merge.merge_checkpoints(
        str(a), str(b), str(chunked), method = "linear", weight = 0.25
    )

    assert report["metrics"]["streamed_tensors"] == 1
    expected = 0.75 * left + 0.25 * right
    assert np.allclose(_merged_array(chunked), expected, atol = 1e-6)
    assert np.allclose(_merged_array(whole), _merged_array(chunked), atol = 1e-6)


def test_streamed_blend_keeps_the_stored_dtype(tmp_path: Path, monkeypatch):
    """A float16 pair must stay float16; the row chunks are float32 scratch only."""
    monkeypatch.setattr(merge, "host_has_gpu", lambda: False)
    monkeypatch.setattr(merge, "chunk_rows", lambda shape, parents: 3)
    left = np.full((16, 8), 1.0)
    right = np.full((16, 8), 3.0)
    a = _write_fp32_checkpoint(
        tmp_path / "a", {"model.embed_tokens.weight": left}, dtype = np.float16
    )
    b = _write_fp32_checkpoint(
        tmp_path / "b", {"model.embed_tokens.weight": right}, dtype = np.float16
    )
    out = tmp_path / "merged"

    report = merge.merge_checkpoints(
        str(a), str(b), str(out), method = "linear", weight = 0.5
    )

    assert report["metrics"]["streamed_tensors"] == 1
    from safetensors import safe_open

    weight_map = json.loads(
        (out / "model.safetensors.index.json").read_text(encoding = "utf-8")
    )["weight_map"]
    with safe_open(str(out / weight_map["model.embed_tokens.weight"]), framework = "np") as handle:
        merged = handle.get_tensor("model.embed_tokens.weight")
    assert merged.dtype == np.float16
    assert np.allclose(np.asarray(merged, dtype = np.float32), 2.0, atol = 2e-2)


def test_shard_buffer_never_outgrows_free_ram(monkeypatch):
    """A 4 GiB disk shard does not mean a 4 GiB RAM buffer on an 8 GB host."""

    class _Profile:
        system_ram_free_bytes = 1024**3  # 1 GiB free

    monkeypatch.setattr(splitter, "probe_memory", lambda *args, **kwargs: _Profile())

    assert merge.shard_budget_bytes(4 * 1024**3) == 256 * 1024**2
    # A tiny request keeps the floor, so flushing does not degenerate per tensor.
    assert merge.shard_budget_bytes(4 * 1024**2) == 64 * 1024**2


def test_the_parent_cap_follows_free_ram_not_the_missing_gpu(tmp_path: Path, monkeypatch):
    """A small checkpoint streams six parents on a CPU host; a large one does not."""
    small = _value_checkpoint(tmp_path, "small", 1.0, rows = 16)
    big = _value_checkpoint(tmp_path, "big", 1.0, rows = 4096)

    generous = merge.merge_model_limit([str(small)], has_gpu = False, free_bytes = 16 * 1024**3)
    assert generous == merge.MAX_MERGE_MODELS

    starved = merge.merge_model_limit([str(small)], has_gpu = False, free_bytes = 40 * 1024**2)
    assert starved == 2

    assert merge.merge_model_limit(None, has_gpu = False, free_bytes = 16 * 1024**3) == 3
    assert merge.merge_model_limit([str(big)], has_gpu = True) == merge.MAX_MERGE_MODELS


class _FakeTensor:
    pass


class _FakeTorch:
    Tensor = _FakeTensor

    def __init__(self, states: dict[str, dict]):
        self.states = states
        self.mmapped: list[str] = []

    def load(self, path, **kwargs):
        self.mmapped.append(str(path))
        return self.states[str(path)]


def test_pickle_checkpoints_stream_through_a_memory_map(tmp_path: Path, monkeypatch):
    """A .bin pair merges with the same lazy row path, gated by the policy flag."""
    pytest.importorskip("safetensors")
    rng = np.random.default_rng(3)
    left = rng.normal(size = (16, 8)).astype(np.float32)
    right = rng.normal(size = (16, 8)).astype(np.float32)
    sources = []
    for name, array in (("pickle-a", left), ("pickle-b", right)):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen2",
                    "hidden_size": 8,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "intermediate_size": 8,
                    "vocab_size": 16,
                }
            ),
            encoding = "utf-8",
        )
        (directory / "pytorch_model.bin").write_bytes(b"not a real pickle")
        sources.append(directory)

    from core.advanced import policy as policy_module

    fake = _FakeTorch(
        {
            str((sources[0] / "pytorch_model.bin").resolve()): {
                "model.embed_tokens.weight": left
            },
            str((sources[1] / "pytorch_model.bin").resolve()): {
                "model.embed_tokens.weight": right
            },
        }
    )
    monkeypatch.setattr(merge.tensor_io, "torch_module", lambda: fake)
    monkeypatch.setattr(merge, "chunk_rows", lambda shape, parents: 3)

    out = tmp_path / "pickle-merged"
    report = merge.merge_checkpoints_many(
        [str(source) for source in sources],
        str(out),
        method = "linear",
        policy = policy_module.LoadingPolicy(allow_non_safetensors = True),
    )

    assert report["metrics"]["streamed_tensors"] == 1
    assert np.allclose(_merged_array(out), (left + right) / 2, atol = 1e-6)
    merge.release_pickle_states()


def test_pickle_checkpoints_stay_refused_by_default(tmp_path: Path):
    directory = tmp_path / "pickle-only"
    directory.mkdir()
    (directory / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "hidden_size": 8, "num_hidden_layers": 1}),
        encoding = "utf-8",
    )
    (directory / "pytorch_model.bin").write_bytes(b"not a real pickle")

    from core.advanced.policy import PolicyViolation

    with pytest.raises(PolicyViolation, match = "pickle"):
        merge.merge_checkpoints(str(directory), str(directory), str(tmp_path / "out"))


def test_shard_buffer_falls_back_when_the_probe_fails(monkeypatch):
    def _broken(*args, **kwargs):
        raise OSError("no probe")

    monkeypatch.setattr(splitter, "probe_memory", _broken)

    assert merge.shard_budget_bytes(4 * 1024**3) == 4 * 1024**3
