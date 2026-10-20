# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Resource broker (F5 / A2).

Two jobs that individually fit must not both start. The broker grants, queues
with a reason, or denies when the request could never fit -- and CPU stages are
accounted in RAM/disk, never as a CUDA error.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.advanced import resources

GB = 1024**3


def _profile(*, vram: float = 24, ram: float = 64, disk: float = 500) -> dict:
    return {
        "vram_free_bytes": int(vram * GB),
        "vram_total_bytes": int(vram * GB),
        "ram_free_bytes": int(ram * GB),
        "ram_total_bytes": int(ram * GB),
        "disk_free_bytes": int(disk * GB),
        "gpus": [{"index": 0, "name": "fake", "total_bytes": int(vram * GB), "free_bytes": int(vram * GB)}]
        if vram
        else [],
    }


def _checkpoint(directory: Path, *, tensors: int = 4, size: int = 8 * 1024 * 1024) -> Path:
    np = pytest.importorskip("numpy")
    pytest.importorskip("safetensors")
    from safetensors.numpy import save_file

    directory.mkdir(parents = True, exist_ok = True)
    payload = {}
    rows = max(1, size // (64 * 4))
    for index in range(tensors):
        payload[f"model.layers.0.w{index}"] = np.zeros((rows, 64), dtype = np.float32)
    save_file(payload, str(directory / "model.safetensors"))
    (directory / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": tensors * rows * 64 * 4},
                "weight_map": {f"model.layers.0.w{i}": "model.safetensors" for i in range(tensors)},
            }
        ),
        encoding = "utf-8",
    )
    return directory


def test_merge_estimate_is_streaming_not_whole_model(tmp_path: Path):
    source = _checkpoint(tmp_path / "model")
    estimate = resources.estimate("merge", sources = [source], memory = _profile())
    need = estimate["need"]
    total = estimate["model_bytes"]
    assert total > 0
    # Peak RAM is a tensor pair, well under the full checkpoint.
    assert 0 < need["ram_bytes"] < total
    assert need["vram_bytes"] == 0
    assert any("Streaming" in note for note in estimate["notes"])


def test_second_job_is_queued_with_a_reason_not_started():
    broker = resources.ResourceBroker(probe = lambda: _profile())
    first = broker.acquire(
        "job-1", "eval", {"vram_bytes": int(16 * GB), "ram_bytes": 0, "disk_bytes": 0}
    )
    assert first["status"] == "granted"

    second = broker.acquire(
        "job-2", "eval", {"vram_bytes": int(16 * GB), "ram_bytes": 0, "disk_bytes": 0}
    )
    assert second["status"] == "queued"
    assert "waiting for" in second["reason"]
    assert "GPU memory" in second["reason"]
    assert "eval" in second["reason"]
    assert broker.leases()

    broker.release("job-1")
    third = broker.acquire(
        "job-2", "eval", {"vram_bytes": int(16 * GB), "ram_bytes": 0, "disk_bytes": 0}
    )
    assert third["status"] == "granted"


def test_impossible_request_is_denied_not_queued():
    broker = resources.ResourceBroker(probe = lambda: _profile(vram = 8))
    result = broker.acquire(
        "job-1", "eval", {"vram_bytes": int(40 * GB), "ram_bytes": 0, "disk_bytes": 0}
    )
    assert result["status"] == "denied"
    assert "offers at most" in result["reason"]
    assert "Colab" in result["reason"]


def test_cpu_stage_reports_ram_and_disk_shortfalls():
    broker = resources.ResourceBroker(probe = lambda: _profile(vram = 0, ram = 8, disk = 10))
    result = broker.acquire(
        "job-1",
        "merge",
        {"vram_bytes": 0, "ram_bytes": int(20 * GB), "disk_bytes": int(50 * GB)},
    )
    assert result["status"] == "denied"
    assert "system RAM" in result["reason"]
    assert "disk space" in result["reason"]
    assert "CUDA" not in result["reason"]


def test_snapshot_reports_limits_reservations_and_leases():
    broker = resources.ResourceBroker(probe = lambda: _profile())
    broker.acquire("job-1", "split", {"vram_bytes": 0, "ram_bytes": GB, "disk_bytes": 0})
    snapshot = broker.snapshot()
    assert snapshot["limits"]["ram_bytes"] == int(64 * GB * resources.RAM_HEADROOM)
    assert snapshot["reserved"]["ram_bytes"] == GB
    assert [lease["job_id"] for lease in snapshot["leases"]] == ["job-1"]


def test_wait_returns_a_clear_reason_instead_of_blocking_forever():
    broker = resources.ResourceBroker(probe = lambda: _profile())
    broker.acquire("job-1", "eval", {"vram_bytes": int(18 * GB), "ram_bytes": 0, "disk_bytes": 0})
    result = broker.acquire(
        "job-2",
        "eval",
        {"vram_bytes": int(8 * GB), "ram_bytes": 0, "disk_bytes": 0},
        wait_seconds = 0.05,
    )
    assert result["status"] == "queued"
    assert "queued" in result["reason"]
