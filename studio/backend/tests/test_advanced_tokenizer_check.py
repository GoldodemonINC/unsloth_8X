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
    bos_id: int | None = 1,
    eos: str = "</s>",
    eos_id: int | None = 2,
    pad: str | None = None,
    pad_id: int | None = None,
    vocab: int = 32,
    template: str | None = None,
    include_ids: bool = True,
    added: dict[str, int] | None = None,
    decoder: dict[str, int] | None = None,
    tokenizer_json: dict[str, int] | None = None,
    added_token_objects: bool = False,
) -> Path:
    """A tiny but complete checkpoint: config, tokenizer files and one shard."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("safetensors")
    from safetensors.numpy import save_file

    directory.mkdir(parents = True, exist_ok = True)
    model_config: dict[str, object] = {
        "model_type": "qwen2",
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "intermediate_size": 8,
        "vocab_size": vocab,
    }
    if include_ids:
        model_config["bos_token_id"] = bos_id
        model_config["eos_token_id"] = eos_id
    (directory / "config.json").write_text(json.dumps(model_config), encoding = "utf-8")
    special: dict[str, object] = {"bos_token": bos, "eos_token": eos}
    config_tokens: dict[str, object] = {"tokenizer_class": "Qwen2Tokenizer"}
    if include_ids:
        config_tokens["bos_token_id"] = bos_id
        config_tokens["eos_token_id"] = eos_id
    if pad is not None:
        special["pad_token"] = pad
        if include_ids:
            config_tokens["pad_token_id"] = pad_id
    if template is not None:
        config_tokens["chat_template"] = template
    if added_token_objects:
        # How DeepSeek-R1-Distill-Qwen saves them: an ``AddedToken`` with a
        # content string and no id anywhere in this file.
        special = {
            key: {"__type": "AddedToken", "content": str(value), "lstrip": False, "rstrip": False}
            for key, value in special.items()
            if value is not None
        }
    if decoder is not None:
        # ``{"151643": {"content": "<|endoftext|>", "special": true}}``, as Qwen ships it.
        config_tokens["added_tokens_decoder"] = {
            str(index): {"id": token_id, "content": token, "special": True}
            for index, (token, token_id) in enumerate(decoder.items())
        }
    (directory / "special_tokens_map.json").write_text(json.dumps(special), encoding = "utf-8")
    (directory / "tokenizer_config.json").write_text(
        json.dumps(config_tokens), encoding = "utf-8"
    )
    if added is not None:
        (directory / "added_tokens.json").write_text(json.dumps(added), encoding = "utf-8")
    if tokenizer_json is not None:
        (directory / "tokenizer.json").write_text(
            json.dumps(
                {
                    "model": {"vocab": dict(tokenizer_json)},
                    "added_tokens": [
                        {"id": token_id, "content": token, "special": True}
                        for token, token_id in tokenizer_json.items()
                    ],
                }
            ),
            encoding = "utf-8",
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


def test_the_same_id_under_different_labels_does_not_conflict(tmp_path: Path):
    """Qwen checkpoints label id 151643 both ``<|endoftext|>`` and ``<|end_of_text|>``.

    The vocabulary index is what the model emits, so the labels disagreeing is a
    note, not a reason to refuse a merge.
    """
    a = _write_checkpoint(tmp_path / "a", eos = "<|endoftext|>", eos_id = 151643)
    b = _write_checkpoint(tmp_path / "b", eos = "<|end_of_text|>", eos_id = 151643)
    report = tokenizer_check.compare_tokenizers([a, b])
    assert report.status == "ok"
    assert report.conflicts == ()
    assert any("different labels" in note for note in report.notes)


def test_added_tokens_prove_two_labels_are_the_same_token(tmp_path: Path):
    """With no id in either config, ``added_tokens.json`` resolves the label to an id."""
    payload = {"<|endoftext|>": 151643, "<|end_of_text|>": 151643}
    a = _write_checkpoint(
        tmp_path / "a",
        eos = "<|endoftext|>",
        eos_id = None,
        include_ids = False,
        added = payload,
    )
    b = _write_checkpoint(
        tmp_path / "b",
        eos = "<|end_of_text|>",
        eos_id = None,
        include_ids = False,
        added = payload,
    )
    report = tokenizer_check.compare_tokenizers([a, b])
    assert report.status == "ok"
    assert report.fingerprints[0].special["eos_token"].token_id == 151643


def test_an_unresolvable_label_difference_still_conflicts(tmp_path: Path):
    a = _write_checkpoint(
        tmp_path / "a", eos = "<eos-a>", eos_id = None, include_ids = False
    )
    b = _write_checkpoint(
        tmp_path / "b", eos = "<eos-b>", eos_id = None, include_ids = False
    )
    report = tokenizer_check.compare_tokenizers([a, b])
    assert report.status == "conflict"
    assert any("cannot be proven equal" in conflict for conflict in report.conflicts)


def test_different_eos_ids_explain_why_it_matters(tmp_path: Path):
    a = _write_checkpoint(tmp_path / "a", eos = "<|endoftext|>", eos_id = 151643)
    b = _write_checkpoint(tmp_path / "b", eos = "<|endoftext|>", eos_id = 151645)
    report = tokenizer_check.compare_tokenizers([a, b])
    assert report.status == "conflict"
    assert any("Generation would stop" in conflict for conflict in report.conflicts)


def test_added_tokens_decoder_resolves_ids(tmp_path: Path):
    """Qwen ships ``added_tokens_decoder`` in tokenizer_config.json, not added_tokens.json."""
    payload = {"<|endoftext|>": 151643, "<|end_of_text|>": 151643}
    a = _write_checkpoint(
        tmp_path / "a", eos = "<|endoftext|>", eos_id = None, include_ids = False, decoder = payload
    )
    b = _write_checkpoint(
        tmp_path / "b", eos = "<|end_of_text|>", eos_id = None, include_ids = False, decoder = payload
    )
    # A pad token named without an id must resolve through the same table.
    c = _write_checkpoint(
        tmp_path / "c",
        eos = "<|endoftext|>",
        pad = "<|endoftext|>",
        eos_id = None,
        include_ids = False,
        decoder = payload,
    )
    assert tokenizer_check.compare_tokenizers([a, b]).status == "ok"
    report = tokenizer_check.compare_tokenizers([a, c])
    # Only one parent declares a pad token, so that is a warning -- but the id
    # itself must resolve, which is what keeps it out of the conflict list.
    assert report.status == "warning", report.conflicts
    assert not report.conflicts
    assert report.fingerprints[1].special["pad_token"].token_id == 151643


def test_tokenizer_json_resolves_ids_as_a_last_resort(tmp_path: Path):
    """With no id anywhere smaller, the fast tokeniser's vocabulary settles it."""
    payload = {"<|endoftext|>": 151643, "<|end_of_text|>": 151643}
    a = _write_checkpoint(
        tmp_path / "a",
        eos = "<|endoftext|>",
        eos_id = None,
        include_ids = False,
        tokenizer_json = payload,
    )
    b = _write_checkpoint(
        tmp_path / "b",
        eos = "<|end_of_text|>",
        eos_id = None,
        include_ids = False,
        tokenizer_json = payload,
    )
    report = tokenizer_check.compare_tokenizers([a, b])
    assert report.status == "ok", report.conflicts
    assert report.fingerprints[0].special["eos_token"].token_id == 151643


def test_added_token_objects_resolve_through_tokenizer_json(tmp_path: Path):
    """The real DeepSeek-R1-Distill-Qwen shape: AddedToken objects, ids only in tokenizer.json."""
    a = _write_checkpoint(
        tmp_path / "a",
        eos = "<｜end▁of▁sentence｜>",
        pad = "<｜end▁of▁sentence｜>",
        eos_id = None,
        pad_id = None,
        include_ids = False,
        added_token_objects = True,
        tokenizer_json = {"<｜end▁of▁sentence｜>": 151643, "<|endoftext|>": 151643},
    )
    b = _write_checkpoint(
        tmp_path / "b",
        eos = "<|endoftext|>",
        pad = "<|endoftext|>",
        eos_id = None,
        pad_id = None,
        include_ids = False,
        added_token_objects = True,
        tokenizer_json = {"<｜end▁of▁sentence｜>": 151643, "<|endoftext|>": 151643},
    )
    report = tokenizer_check.check_before_merge([a, b])
    assert report.status == "ok", report.conflicts
    assert report.fingerprints[0].special["pad_token"].token_id == 151643
    assert report.fingerprints[1].special["eos_token"].token_id == 151643
    assert any("different labels" in note for note in report.notes)


def test_a_corrupt_tokenizer_json_does_not_crash_the_gate(tmp_path: Path):
    a = _write_checkpoint(tmp_path / "a", eos = "<eos-a>", eos_id = None, include_ids = False)
    b = _write_checkpoint(tmp_path / "b", eos = "<eos-b>", eos_id = None, include_ids = False)
    (a / "tokenizer.json").write_text("{not json", encoding = "utf-8")
    report = tokenizer_check.compare_tokenizers([a, b])
    assert report.status == "conflict"


def test_the_same_label_at_different_ids_explains_the_vocab_shift(tmp_path: Path):
    """Two builds of the same tokenizer can place <|im_end|> at different indices.

    The labels match, so "id A vs id B" reads like a typo; what it means is that
    the embedding rows would be blended across different tokens.
    """
    a = _write_checkpoint(tmp_path / "a", eos = "<|im_end|>", eos_id = 248044)
    b = _write_checkpoint(tmp_path / "b", eos = "<|im_end|>", eos_id = 248046)
    report = tokenizer_check.compare_tokenizers([a, b])
    assert report.status == "conflict"
    joined = " ".join(report.conflicts)
    assert "same token sits at id 248044" in joined
    assert "248046" in joined


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


def test_hub_parents_are_compared_from_metadata_without_weights(
    tmp_path: Path, monkeypatch
):
    """The refusal has to be visible on Inspect, before the snapshot is downloaded.

    This is the reported Qwen2 pair: identical shapes and vocabulary, but one
    parent declares ``<|endoftext|>`` as eos/pad and the other ``<|im_end|>``.
    If the preview could not read a Hub parent, the UI had no conflict to show,
    so the override its own error message recommends was unreachable.
    """
    pytest.importorskip("numpy")
    pytest.importorskip("safetensors")
    a = _write_checkpoint(
        tmp_path / "a",
        eos = "<|endoftext|>",
        eos_id = 151643,
        pad = "<|endoftext|>",
        pad_id = 151643,
    )
    b = _write_checkpoint(
        tmp_path / "b",
        eos = "<|im_end|>",
        eos_id = 151645,
        pad = "<|im_end|>",
        pad_id = 151645,
    )
    roots = {"org/a": a, "org/b": b}
    requested: list[str] = []

    def fake_download(repo_id: str, filename: str, token: str | None) -> str:
        requested.append(f"{repo_id}/{filename}")
        target = roots[repo_id] / filename
        if not target.is_file():
            # What hf_hub_download does for a file a repo does not ship.
            raise FileNotFoundError(filename)
        return str(target)

    monkeypatch.setattr(tokenizer_check, "_hub_download", fake_download)

    report = tokenizer_check.compare_tokenizers(
        ["hf://org/a", "hf://org/b"], token = "hf_test"
    )

    assert report.status == "conflict"
    assert any(
        "eos_token" in conflict and "151643" in conflict and "151645" in conflict
        for conflict in report.conflicts
    )
    # The result carries one parent's tokenizer; say which, since that decides
    # what the merged model will stop on.
    assert any("first parent" in note for note in report.notes)
    # Reported by repo id, not by the cache folder it was read from.
    assert report.fingerprints[0].source == "hf://org/a"
    # Metadata only: no weight file was ever requested.
    assert requested
    assert not any(name.endswith(".safetensors") for name in requested)


def test_unreadable_hub_metadata_explains_that_a_token_is_needed(monkeypatch):
    def deny(repo_id: str, filename: str, token: str | None) -> str:
        raise RuntimeError("401 Client Error: Invalid username or password")

    monkeypatch.setattr(tokenizer_check, "_hub_download", deny)

    with pytest.raises(ValueError) as error:
        tokenizer_check.fingerprint_source("hf://gated/private-model")
    message = str(error.value)
    assert "gated/private-model" in message
    assert "Hugging Face token" in message
