# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The merge tokenizer gate (F2 / A1).

A merge whose parents disagree about bos/eos/pad must refuse to write anything:
the result would load, run, and quietly mis-handle every prompt.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.advanced import merge, tokenizer_check


def _write_checkpoint(
    directory: Path,
    *,
    bos: str = "<s>",
    bos_id: int = 1,
    eos: str = "</s>",
    eos_id: int = 2,
    pad: str | None = None,
    pad_id: int | None = None,
    vocab: int = 32,
    template: str | None = None,
) -> Path:
    """A tiny but complete checkpoint: config, tokenizer files and one shard."""
    np = pytest.importorskip("numpy")
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
                "bos_token_id": bos_id,
                "eos_token_id": eos_id,
            }
        ),
        encoding = "utf-8",
    )
    special: dict[str, object] = {"bos_token": bos, "eos_token": eos}
    config_tokens: dict[str, object] = {"bos_token_id": bos_id, "eos_token_id": eos_id}
    if pad is not None:
        special["pad_token"] = pad
        config_tokens["pad_token_id"] = pad_id
    if template is not None:
        config_tokens["chat_template"] = template
    (directory / "special_tokens_map.json").write_text(json.dumps(special), encoding = "utf-8")
    (directory / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "Qwen2Tokenizer", **config_tokens}), encoding = "utf-8"
    )
    save_file(
        {"model.embed_tokens.weight": np.zeros((vocab, 8), dtype = np.float32)},
        str(directory / "model.safetensors"),
    )
    return directory


def test_conflicting_eos_refuses_the_merge_and_writes_nothing(tmp_path: Path):
    pytest.importorskip("numpy")
    pytest.importorskip("safetensors")
    a = _write_checkpoint(tmp_path / "a", eos = "</s>", eos_id = 2)
    b = _write_checkpoint(tmp_path / "b", eos = "<|end|>", eos_id = 5)
    out = tmp_path / "merged"

    with pytest.raises(tokenizer_check.TokenizerConflictError) as error:
        merge.merge_checkpoints(str(a), str(b), str(out), method = "linear")

    message = str(error.value)
    assert "eos_token" in message
    assert "<|end|>" in message
    assert "allow_tokenizer_conflict" in message
    # The whole point: the gate runs before the output folder is created.
    assert not out.exists()


def test_matching_tokenizers_merge_and_report_ok(tmp_path: Path):
    pytest.importorskip("numpy")
    pytest.importorskip("safetensors")
    a = _write_checkpoint(tmp_path / "a", pad = "<pad>", pad_id = 0)
    b = _write_checkpoint(tmp_path / "b", pad = "<pad>", pad_id = 0)
    out = tmp_path / "merged"

    report = merge.merge_checkpoints(str(a), str(b), str(out), method = "linear")

    assert report["tokenizer"]["status"] == "ok"
    assert report["tokenizer"]["output"]["status"] == "ok"
    assert (out / "tokenizer_config.json").is_file()
    assert (out / "run_manifest.json").is_file()


def test_pad_present_on_one_parent_is_a_warning_not_a_conflict(tmp_path: Path):
    pytest.importorskip("numpy")
    pytest.importorskip("safetensors")
    a = _write_checkpoint(tmp_path / "a", pad = "<pad>", pad_id = 0)
    b = _write_checkpoint(tmp_path / "b")

    report = tokenizer_check.compare_tokenizers([a, b])
    assert report.status == "warning"
    assert any("pad_token" in warning for warning in report.warnings)
    assert not report.conflicts


def test_different_bos_ids_conflict(tmp_path: Path):
    a = _write_checkpoint(tmp_path / "a", bos = "<s>", bos_id = 1)
    b = _write_checkpoint(tmp_path / "b", bos = "<|start|>", bos_id = 3)
    report = tokenizer_check.compare_tokenizers([a, b])
    assert report.status == "conflict"
    assert any("bos_token" in conflict for conflict in report.conflicts)


def test_override_allows_a_known_conflict_and_records_it(tmp_path: Path):
    pytest.importorskip("numpy")
    pytest.importorskip("safetensors")
    a = _write_checkpoint(tmp_path / "a", eos_id = 2)
    b = _write_checkpoint(tmp_path / "b", eos_id = 7)
    out = tmp_path / "merged"

    report = merge.merge_checkpoints(
        str(a), str(b), str(out), method = "linear", allow_tokenizer_conflict = True
    )
    assert report["tokenizer"]["status"] == "conflict"
    assert report["tokenizer"]["override_used"] is True
    manifest = json.loads((out / "run_manifest.json").read_text(encoding = "utf-8"))
    assert manifest["tokenizer"]["override_used"] is True


def test_a_full_disk_mid_merge_leaves_no_partial_shard(tmp_path: Path, monkeypatch):
    """The streaming writer is atomic: no half-shard survives a full disk."""
    pytest.importorskip("numpy")
    pytest.importorskip("safetensors")
    import errno

    from core.advanced import policy, tensor_io

    a = _write_checkpoint(tmp_path / "a")
    b = _write_checkpoint(tmp_path / "b")
    out = tmp_path / "merged"

    def explode(payload, path):
        # The policy hands the writer a .partial path; fill it, then hit ENOSPC.
        Path(path).write_bytes(b"half a shard")
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(tensor_io, "save_tensors", explode)
    with pytest.raises(policy.DiskFullError):
        merge.merge_checkpoints(str(a), str(b), str(out), method = "linear")
    assert list(out.glob("*.safetensors")) == []
    assert list(out.glob("*.partial")) == []


def test_differing_chat_templates_warn(tmp_path: Path):
    a = _write_checkpoint(tmp_path / "a", template = "{{ messages }}")
    b = _write_checkpoint(tmp_path / "b", template = "{% for m in messages %}{{ m }}{% endfor %}")
    report = tokenizer_check.compare_tokenizers([a, b])
    assert report.status == "warning"
    assert any("chat template" in warning.lower() for warning in report.warnings)


def test_missing_output_tokenizer_is_reported_as_broken(tmp_path: Path):
    source = _write_checkpoint(tmp_path / "a")
    empty = tmp_path / "empty"
    empty.mkdir()
    expected = tokenizer_check.fingerprint_tokenizer(source)
    result = tokenizer_check.verify_output(empty, expected)
    assert result["status"] == "broken"
    assert result["problems"]


def test_merge_above_one_child_reads_tokenizer_from_a_hub_cache_layout(tmp_path: Path):
    """A HF cache folder keeps the real files one level down under snapshots/."""
    snapshot = _write_checkpoint(tmp_path / "cache" / "snapshots" / "abc123")
    fingerprint = tokenizer_check.fingerprint_tokenizer(tmp_path / "cache")
    assert fingerprint.source == str(snapshot)
    assert fingerprint.special["bos_token"].token_id == 1
