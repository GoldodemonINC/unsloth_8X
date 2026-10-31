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

CPU_HOST_GRANTS = ("autoscale", "merge", "merge_eval", "moe", "scale", "split")
GPU_ONLY = ("cmt", "eval", "qat")
CPU_HOST_NOTED = ("autoscale", "scale")


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


def test_every_cost_model_row_states_its_cpu_capability():
    for kind, model in resources._COST_MODEL.items():
        assert isinstance(model.get("cpu"), bool), kind


def test_the_cpu_contract_covers_every_row_exactly_once():
    assert set(CPU_HOST_GRANTS) | set(GPU_ONLY) == set(resources._COST_MODEL)
    assert set(CPU_HOST_GRANTS).isdisjoint(GPU_ONLY)
    for kind in resources._COST_MODEL:
        assert resources._COST_MODEL[kind]["cpu"] == (kind in CPU_HOST_GRANTS), kind


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


@pytest.mark.parametrize("kind", CPU_HOST_GRANTS)
def test_a_cpu_capable_stage_is_leased_on_a_host_with_no_gpu(tmp_path: Path, kind: str):
    source = _checkpoint(tmp_path / "model")
    memory = _profile(vram = 0)
    payload = resources.estimate(kind, sources = [source], memory = memory)
    assert payload["need"]["vram_bytes"] == 0
    assert not any("needs a GPU" in note for note in payload["notes"])
    broker = resources.ResourceBroker(probe = lambda: memory)
    result = broker.acquire("job-1", kind, resources.Need(**payload["need"]))
    assert result["status"] == "granted"


@pytest.mark.parametrize("kind", CPU_HOST_NOTED)
def test_a_stage_that_will_run_on_cpu_says_so(tmp_path: Path, kind: str):
    source = _checkpoint(tmp_path / "model")
    payload = resources.estimate(kind, sources = [source], memory = _profile(vram = 0))
    assert any("runs on the CPU" in note for note in payload["notes"])


@pytest.mark.parametrize("kind", GPU_ONLY)
def test_a_gpu_only_stage_is_still_refused_on_a_host_with_no_gpu(tmp_path: Path, kind: str):
    source = _checkpoint(tmp_path / "model")
    memory = _profile(vram = 0)
    payload = resources.estimate(kind, sources = [source], memory = memory)
    assert payload["need"]["vram_bytes"] > 0
    assert any("can be planned but not run here" in note for note in payload["notes"])
    broker = resources.ResourceBroker(probe = lambda: memory)
    result = broker.acquire("job-1", kind, resources.Need(**payload["need"]))
    assert result["status"] == "denied"
    assert "GPU memory" in result["reason"]
    assert "this machine offers at most" in result["reason"]


def test_a_scale_stage_reserves_the_peak_of_the_checkpoint_it_writes(tmp_path: Path):
    source = _checkpoint(tmp_path / "model")
    total = resources.checkpoint_bytes(source)
    payload = resources.estimate("scale", sources = [source], memory = _profile(vram = 0))
    assert payload["need"]["ram_bytes"] == 2 * total


def test_a_grown_target_reserves_ram_for_the_checkpoint_it_will_produce(tmp_path: Path):
    source = _checkpoint(tmp_path / "model")
    plain = resources.estimate("scale", sources = [source], memory = _profile(vram = 0))
    grown = resources.estimate(
        "scale",
        sources = [source],
        memory = _profile(vram = 0),
        growth_ratio = 3.0,
    )
    assert grown["need"]["ram_bytes"] == 3 * plain["need"]["ram_bytes"]
    assert grown["need"]["ram_bytes"] > plain["need"]["ram_bytes"]


def test_a_growth_ratio_below_one_does_not_shrink_the_reservation(tmp_path: Path):
    source = _checkpoint(tmp_path / "model")
    plain = resources.estimate("scale", sources = [source], memory = _profile(vram = 0))
    smaller = resources.estimate(
        "scale",
        sources = [source],
        memory = _profile(vram = 0),
        growth_ratio = 0.5,
    )
    assert smaller["need"]["ram_bytes"] == plain["need"]["ram_bytes"]
