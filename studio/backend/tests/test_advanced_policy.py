# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Loading/writing policy: pickles, remote code, traversal, atomic writes, disk full.

These are the guards that run before a merge touches a tensor or a shard.
"""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path

import pytest

from core.advanced import policy, tensor_io


def _checkpoint(directory: Path, *, weights: tuple[str, ...] = ("model.safetensors",)) -> Path:
    directory.mkdir(parents = True, exist_ok = True)
    (directory / "config.json").write_text(json.dumps({"model_type": "qwen2"}), encoding = "utf-8")
    for name in weights:
        (directory / name).write_bytes(b"\0")
    return directory


def test_pickle_weights_are_refused_by_default(tmp_path: Path):
    source = _checkpoint(tmp_path / "pickled", weights = ("pytorch_model.bin",))
    with pytest.raises(policy.PolicyViolation) as error:
        policy.enforce_source_policy(source, label = "Model A")
    message = str(error.value)
    assert "Model A" in message
    assert "pytorch_model.bin" in message
    assert "allow_non_safetensors" in message


def test_pickle_weights_allowed_when_explicitly_opted_in(tmp_path: Path):
    source = _checkpoint(tmp_path / "pickled", weights = ("pytorch_model.bin",))
    report = policy.enforce_source_policy(
        source, policy.LoadingPolicy(allow_non_safetensors = True)
    )
    assert report["allowed"] is True
    assert report["pickles"] == ["pytorch_model.bin"]


def test_remote_code_is_refused_by_default(tmp_path: Path):
    source = _checkpoint(tmp_path / "remote")
    (source / "config.json").write_text(
        json.dumps({"auto_map": {"AutoModelForCausalLM": "modeling_custom.Custom"}}),
        encoding = "utf-8",
    )
    (source / "modeling_custom.py").write_text("raise SystemExit(1)\n", encoding = "utf-8")
    with pytest.raises(policy.RemoteCodeRejected) as error:
        policy.enforce_source_policy(source)
    assert "allow_remote_code" in str(error.value)
    assert "auto_map" in str(error.value)


def test_remote_code_opt_in_records_a_warning(tmp_path: Path):
    source = _checkpoint(tmp_path / "remote")
    (source / "config.json").write_text(
        json.dumps({"auto_map": {"AutoConfig": "configuration_custom.Custom"}}),
        encoding = "utf-8",
    )
    report = policy.enforce_source_policy(
        source, policy.LoadingPolicy(allow_remote_code = True)
    )
    assert report["requires_remote_code"] is True
    warning = policy.remote_code_warning(source)
    assert warning is not None and "explicitly allowed" in warning


def test_safe_child_blocks_traversal(tmp_path: Path):
    root = tmp_path / "run"
    root.mkdir()
    assert policy.safe_child(root, "model-00001.safetensors").name == "model-00001.safetensors"
    for bad in ("../escape", "..\\escape", "/absolute/path", "C:/windows/system32"):
        with pytest.raises(policy.PolicyViolation):
            policy.safe_child(root, bad)


def test_ensure_free_space_raises_for_an_impossible_write(tmp_path: Path):
    with pytest.raises(policy.DiskFullError) as error:
        policy.ensure_free_space(tmp_path, 10**18)
    assert error.value.needed_bytes == 10**18


def test_atomic_write_leaves_nothing_behind_when_the_rename_fails(tmp_path, monkeypatch):
    target = tmp_path / "run_manifest.json"

    def explode(src, dst):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(os, "replace", explode)
    with pytest.raises(policy.DiskFullError):
        policy.atomic_write_bytes(target, b"{}" * 10)
    assert not target.exists()
    assert list(tmp_path.glob("*.partial")) == []


def test_disk_full_while_writing_a_shard_removes_the_partial(tmp_path, monkeypatch):
    target = tmp_path / "model-00001.safetensors"
    payload = {"w": [1.0, 2.0]}

    def fail(payload_arg, path_arg):
        Path(path_arg).write_bytes(b"half a shard")
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(tensor_io, "save_tensors", fail)
    with pytest.raises(policy.DiskFullError) as error:
        policy.atomic_save_tensors(payload, target)
    assert "partial shard was removed" in str(error.value)
    assert not target.exists()
    assert list(tmp_path.glob("*.partial")) == []


def test_atomic_save_writes_the_shard_on_success(tmp_path):
    pytest.importorskip("safetensors")
    import numpy as np

    target = tmp_path / "model-00001.safetensors"
    policy.atomic_save_tensors({"w": np.zeros((2, 2), dtype = np.float32)}, target)
    assert target.is_file()
    assert list(tmp_path.glob("*.partial")) == []
