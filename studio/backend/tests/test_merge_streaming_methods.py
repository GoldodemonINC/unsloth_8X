# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.advanced import merge


def _checkpoint(
    directory: Path, tensors: dict[str, np.ndarray], *, vocab: int = 16
) -> Path:
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
    save_file(tensors, str(directory / "model.safetensors"))
    return directory


def _merged(out: Path, key: str = "model.embed_tokens.weight") -> np.ndarray:
    from safetensors import safe_open

    weight_map = json.loads(
        (out / "model.safetensors.index.json").read_text(encoding = "utf-8")
    )["weight_map"]
    with safe_open(str(out / weight_map[key]), framework = "np") as handle:
        return np.asarray(handle.get_tensor(key), dtype = np.float32)


@pytest.fixture()
def parents(tmp_path: Path):
    rng = np.random.default_rng(7)
    arrays = [rng.normal(size = (16, 8)).astype(np.float32) for _ in range(3)]
    paths = [
        _checkpoint(tmp_path / f"m{index}", {"model.embed_tokens.weight": array})
        for index, array in enumerate(arrays)
    ]
    return arrays, paths


@pytest.mark.parametrize("method", ["linear", "slerp", "ties", "dare"])
def test_row_chunks_reproduce_the_whole_tensor_blend(
    parents, tmp_path: Path, monkeypatch, method: str
):
    arrays, paths = parents
    weights = [0.75, 0.25]
    monkeypatch.setattr(merge, "chunk_rows", lambda shape, count: 3)
    out = tmp_path / f"out-{method}"

    report = merge.merge_checkpoints(
        str(paths[0]),
        str(paths[1]),
        str(out),
        method = method,
        weight = 0.25,
        density = 0.5,
        seed = 11,
    )

    expected = merge.blend_tensors(
        [arrays[0], arrays[1]], method = method, weights = weights, density = 0.5, seed = 11
    )
    assert report["metrics"]["streamed_tensors"] == 1
    assert report["metrics"]["blended_tensors"] == 1
    assert np.allclose(_merged(out), expected, atol = 1e-5)


@pytest.mark.parametrize("method", ["linear", "slerp", "ties", "dare"])
def test_chunk_size_does_not_change_the_result(
    parents, tmp_path: Path, monkeypatch, method: str
):
    _arrays, paths = parents
    one_chunk = tmp_path / f"one-{method}"
    many_chunks = tmp_path / f"many-{method}"

    monkeypatch.setattr(merge, "chunk_rows", lambda shape, count: 16)
    merge.merge_checkpoints(
        str(paths[0]), str(paths[1]), str(one_chunk), method = method, weight = 0.25
    )
    monkeypatch.setattr(merge, "chunk_rows", lambda shape, count: 2)
    merge.merge_checkpoints(
        str(paths[0]), str(paths[1]), str(many_chunks), method = method, weight = 0.25
    )

    assert np.allclose(_merged(one_chunk), _merged(many_chunks), atol = 1e-5)


def test_n_way_slerp_falls_back_to_the_whole_tensor_fold(parents, tmp_path: Path, monkeypatch):
    arrays, paths = parents
    monkeypatch.setattr(merge, "chunk_rows", lambda shape, count: 2)
    out = tmp_path / "slerp-three"

    report = merge.merge_checkpoints_many(
        [str(path) for path in paths], str(out), method = "slerp"
    )

    expected = merge.blend_tensors(arrays, method = "slerp")
    assert report["metrics"]["streamed_tensors"] == 0
    assert report["metrics"]["blended_tensors"] == 1
    assert np.allclose(_merged(out), expected, atol = 1e-5)


def test_dare_mask_is_addressable_by_flat_index():
    first = merge.dare_mask(0, 32, seed = 3407, density = 0.4, salt = 1)
    window = merge.dare_mask(8, 8, seed = 3407, density = 0.4, salt = 1)

    assert np.array_equal(first[8:16], window)
    assert not np.array_equal(
        merge.dare_mask(0, 32, seed = 3407, density = 0.4, salt = 2), first
    )


def test_dare_density_of_one_matches_the_linear_blend(parents):
    arrays, _paths = parents
    density_one = merge.blend_tensors(
        arrays[:2], method = "dare", weights = [0.25, 0.75], density = 1.0
    )
    linear = merge.blend_tensors(arrays[:2], method = "linear", weights = [0.25, 0.75])

    assert np.allclose(density_one, linear, atol = 1e-5)


def test_ties_threshold_matches_the_exact_partition(parents):
    arrays, _paths = parents
    for density in (0.2, 0.5, 0.9):
        whole = merge.blend_tensors(
            arrays[:2], method = "ties", weights = [0.3, 0.7], density = density
        )
        stacked = np.stack(
            [np.asarray(arrays[0], dtype = np.float32), np.asarray(arrays[1], dtype = np.float32)]
        )
        reference = stacked.mean(axis = 0)
        deltas = stacked - reference
        flat = deltas.reshape(2, -1)
        count = flat.shape[1]
        keep = max(1, int(round(count * density)))
        threshold = np.partition(np.abs(flat), count - keep, axis = 1)[:, count - keep]
        trimmed = np.where(np.abs(flat) >= threshold[:, None], flat, 0.0)
        elected = np.sign(trimmed.sum(axis = 0))
        signed = np.where(np.sign(flat) == elected[None, :], flat, 0.0)
        norm = np.asarray([0.3, 0.7], dtype = np.float32)
        mass = np.zeros(count, dtype = np.float32)
        for row, weight in zip(signed, norm):
            mass += (row != 0).astype(np.float32) * weight
        expected = reference + (
            (signed * norm[:, None]).sum(axis = 0) / np.maximum(mass, 1e-8)
        ).reshape(reference.shape)

        assert np.allclose(whole, expected, atol = 1e-5)
