# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
from pathlib import Path

import pytest

from core.labs import cmt


def _model(tmp_path: Path) -> Path:
    directory = tmp_path / "tiny"
    directory.mkdir()
    (directory / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "hidden_size": 512,
                "intermediate_size": 2048,
                "num_hidden_layers": 8,
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "vocab_size": 32000,
            }
        ),
        encoding = "utf-8",
    )
    (directory / "model.safetensors").write_bytes(b"not a real checkpoint")
    return directory


def _jsonl(tmp_path: Path, name: str, rows: int, text: str = "hello world") -> Path:
    path = tmp_path / name
    with path.open("w", encoding = "utf-8") as stream:
        for index in range(rows):
            stream.write(json.dumps({"text": f"{text} {index}"}) + "\n")
    return path


def test_sizing_rule_matches_the_documented_formula():
    assert cmt._added_from_tokens(20_000_000) == 4_600_000
    assert cmt._added_from_tokens(1_000_000_000) == 171_200_000
    assert cmt._added_from_tokens(5_000_000_000) == cmt.MAX_ADDED_PARAMS
    assert cmt._added_from_tokens(0) == 1_200_000
    assert cmt._added_from_rows(10_000) == 1_000_000
    assert cmt._added_from_rows(0) == cmt.MIN_ADDED_PARAMS


def test_dataset_inspection_counts_rows_tokens_and_its_own_split(tmp_path):
    path = _jsonl(tmp_path, "a.jsonl", 100)
    entry = cmt._inspect(
        {"path": str(path), "name": "alpha", "text_column": None, "eval_fraction": 0.2}
    )
    assert entry["name"] == "alpha"
    assert entry["rows"] == 100
    assert entry["eval_rows"] == 20
    assert entry["train_rows"] == 80
    assert entry["tokens_estimate"] > 0
    assert entry["bytes"] > 0


def test_plan_sizes_from_tokens_and_warns_that_growth_is_whole_blocks(tmp_path):
    model = _model(tmp_path)
    first = _jsonl(tmp_path, "one.jsonl", 1200, "a sentence about training")
    second = _jsonl(tmp_path, "two.jsonl", 800, "another held out example")
    report = cmt.plan(
        {
            "source": str(model),
            "datasets": [
                {"path": str(first), "name": "one", "eval_fraction": 0.1},
                {"path": str(second), "name": "two", "eval_fraction": 0.25},
            ],
        }
    )
    assert report["totals"]["datasets"] == 2
    assert report["totals"]["rows"] == 2000
    assert report["totals"]["eval_rows"] == 320
    assert report["added_source"] == "tokens"
    assert cmt.MIN_ADDED_PARAMS <= report["sizing"]["added_params"] <= cmt.MAX_ADDED_PARAMS
    assert report["expansion"]["base_params"] > 0
    assert report["expansion"]["added_params"] > 0
    assert report["expansion"]["floor_bypassed"] is True
    assert report["training"]["method"] == "new-blocks-only"
    assert any("whole blocks" in warning for warning in report["warnings"])


def test_plan_accepts_an_override_and_requires_two_distinct_datasets(tmp_path):
    model = _model(tmp_path)
    one = _jsonl(tmp_path, "one.jsonl", 10)
    two = _jsonl(tmp_path, "two.jsonl", 10)

    report = cmt.plan(
        {
            "source": str(model),
            "added_params": 2_000_000,
            "datasets": [{"path": str(one)}, {"path": str(two)}],
        }
    )
    assert report["added_source"] == "override"
    assert report["sizing"]["added_params"] == 2_000_000

    with pytest.raises(ValueError):
        cmt.plan({"source": str(model), "datasets": [{"path": str(one)}]})
    with pytest.raises(ValueError):
        cmt.plan(
            {
                "source": str(model),
                "datasets": [{"path": str(one)}, {"path": str(one)}],
            }
        )
    with pytest.raises(ValueError):
        cmt.plan(
            {
                "source": str(model),
                "datasets": [
                    {"path": str(one)},
                    {"path": str(two), "eval_fraction": 0.9},
                ],
            }
        )


def test_plan_rejects_a_dataset_type_it_cannot_read(tmp_path):
    model = _model(tmp_path)
    binary = tmp_path / "rows.bin"
    binary.write_bytes(b"\x00\x01")
    other = _jsonl(tmp_path, "other.jsonl", 10)
    with pytest.raises(ValueError):
        cmt.plan(
            {
                "source": str(model),
                "datasets": [{"path": str(binary)}, {"path": str(other)}],
            }
        )
