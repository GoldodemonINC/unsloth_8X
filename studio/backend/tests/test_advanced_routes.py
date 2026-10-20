# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Route-level checks for the advanced API.

These exercise the request models and the source validation without starting a
training run or touching the network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from routes import advanced  # noqa: E402


@pytest.fixture()
def client() -> TestClient:
    app = fastapi.FastAPI()
    app.include_router(advanced.router, prefix = "/api/advanced")
    return TestClient(app)


@pytest.fixture()
def started_jobs(monkeypatch) -> list[dict]:
    """Capture what a run route hands the job registry instead of really running it."""
    jobs: list[dict] = []

    def fake_start_job(kind, work, *, output_dir = None):
        jobs.append({"kind": kind, "output_dir": output_dir})
        return {
            "job_id": "job-1",
            "kind": kind,
            "status": "running",
            "created_at": 0.0,
            "finished_at": None,
            "output_dir": output_dir,
            "result": None,
            "error": None,
        }

    monkeypatch.setattr(advanced, "_start_job", fake_start_job)
    return jobs


def test_capabilities_exposes_merge_and_split_surfaces(client: TestClient):
    response = client.get("/api/advanced/capabilities")
    assert response.status_code == 200
    body = response.json()
    assert set(body["merge_methods"]) == {"linear", "slerp", "ties", "dare"}
    assert "memory" in body and "qat_formats" in body


def test_merge_plan_rejects_unknown_sources(client: TestClient, tmp_path):
    response = client.post(
        "/api/advanced/merge/plan",
        json={
            "model_a": str(tmp_path / "missing-a"),
            "model_b": str(tmp_path / "missing-b"),
            "top_k": 2,
            "temperature": 1.0,
        },
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "Hugging Face repo id" in detail
    assert "missing-a" in detail


def test_merge_run_rejects_unknown_sources_before_starting(client: TestClient, tmp_path):
    response = client.post(
        "/api/advanced/merge/run",
        json={
            "model_a": str(tmp_path / "missing-a"),
            "model_b": str(tmp_path / "missing-b"),
            "output_dir": str(tmp_path / "out"),
            "method": "linear",
            "weight": 0.5,
            "density": 0.5,
            "max_shard_gb": 4,
        },
    )
    assert response.status_code == 400
    assert "Model A" in response.json()["detail"]


def test_splitter_stays_local_only(client: TestClient):
    # The splitter streams existing safetensors; a Hub id is not a valid source.
    response = client.post(
        "/api/advanced/split/plan",
        json={"source": "unsloth/Llama-3.2-1B", "prefer": "auto"},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "existing local checkpoint path" in detail


def test_merge_plan_accepts_repo_ids_structurally(client: TestClient, monkeypatch):
    """A repo id passes validation and reaches the probe (stubbed, no network)."""
    from core.advanced import merge as merge_module

    seen: list[tuple[str, str | None]] = []

    def fake_probe(source: str, *, token: str | None = None):
        seen.append((source, token))
        return merge_module.ModelSpec(
            source = f"hf://{source}",
            architecture = "llama",
            hidden_size = 2048,
            num_layers = 16,
            num_heads = 8,
            intermediate_size = 8192,
            vocab_size = 128256,
        )

    monkeypatch.setattr(merge_module, "probe_model_spec", fake_probe)

    response = client.post(
        "/api/advanced/merge/plan",
        json={
            "model_a": "unsloth/Llama-3.2-1B",
            "model_b": "unsloth/Llama-3.2-1B-Instruct",
            "top_k": 2,
            "temperature": 1.0,
            "hf_token": "hf_test",
        },
    )
    assert response.status_code == 200, response.text
    assert seen == [
        ("unsloth/Llama-3.2-1B", "hf_test"),
        ("unsloth/Llama-3.2-1B-Instruct", "hf_test"),
    ]
    body = response.json()
    assert body["recommended_mode"] == "standard"
    assert body["models"][0]["hidden_size"] == 2048


# ---------------------------------------------------------------------------
# Output folders
#
# The UI prefills a bare name, not a machine-specific path, so the same request has to work
# on Windows and on a Colab runtime where ``C:\\models`` does not exist.
# ---------------------------------------------------------------------------


def _two_checkpoints(tmp_path: Path) -> tuple[str, str]:
    sources = []
    for name in ("model-a", "model-b"):
        path = tmp_path / name
        path.mkdir()
        sources.append(str(path))
    return sources[0], sources[1]


def test_a_bare_merge_output_name_lands_under_the_outputs_root(
    client: TestClient, tmp_path, started_jobs
):
    from utils.paths import resolve_output_dir

    model_a, model_b = _two_checkpoints(tmp_path)
    response = client.post(
        "/api/advanced/merge/run",
        json={
            "model_a": model_a,
            "model_b": model_b,
            "output_dir": "merged-test",
            "method": "linear",
            "weight": 0.5,
            "density": 0.5,
            "max_shard_gb": 4,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    resolved = str(resolve_output_dir("merged-test"))
    assert body["output_dir"] == resolved
    assert Path(body["output_dir"]).is_absolute()
    assert Path(body["output_dir"]).name == "merged-test"
    assert started_jobs == [{"kind": "merge", "output_dir": resolved}]


def test_a_blank_merge_output_dir_still_writes_somewhere(client: TestClient, tmp_path, started_jobs):
    from utils.paths import resolve_output_dir

    model_a, model_b = _two_checkpoints(tmp_path)
    response = client.post(
        "/api/advanced/merge/run",
        json={"model_a": model_a, "model_b": model_b, "output_dir": "  "},
    )
    assert response.status_code == 200, response.text
    assert response.json()["output_dir"] == str(resolve_output_dir("merged"))


def test_an_absolute_merge_output_dir_is_used_as_given(client: TestClient, tmp_path, started_jobs):
    model_a, model_b = _two_checkpoints(tmp_path)
    target = tmp_path / "elsewhere" / "merged"
    response = client.post(
        "/api/advanced/merge/run",
        json={"model_a": model_a, "model_b": model_b, "output_dir": str(target)},
    )
    assert response.status_code == 200, response.text
    assert response.json()["output_dir"] == str(target)


def test_a_windows_drive_path_is_used_as_given_on_any_runtime(client: TestClient, tmp_path, started_jobs):
    # ``C:/models`` is not absolute under POSIX rules; it must still never be joined onto the
    # outputs root, or a Windows path would silently land inside the Colab/Linux folder.
    model_a, model_b = _two_checkpoints(tmp_path)
    response = client.post(
        "/api/advanced/merge/run",
        json={"model_a": model_a, "model_b": model_b, "output_dir": "C:/models/merged"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["output_dir"] == str(Path("C:/models/merged").expanduser())


def test_a_drive_relative_name_stays_inside_the_outputs_root(client: TestClient, tmp_path, started_jobs):
    # ``C:merged`` is drive-relative, not absolute: it must be contained, never left to the
    # process working directory.
    from utils.paths import resolve_output_dir

    model_a, model_b = _two_checkpoints(tmp_path)
    response = client.post(
        "/api/advanced/merge/run",
        json={"model_a": model_a, "model_b": model_b, "output_dir": "C:merged"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["output_dir"] == str(resolve_output_dir("C:merged"))


def test_an_output_dir_that_escapes_the_outputs_root_is_rejected(client: TestClient, tmp_path):
    source = tmp_path / "checkpoint"
    source.mkdir()
    response = client.post(
        "/api/advanced/split/run",
        json={"source": str(source), "output_dir": "../../escape", "prefer": "auto"},
    )
    assert response.status_code == 400
    assert ".." in response.json()["detail"]


def test_a_bare_split_output_name_lands_under_the_outputs_root(
    client: TestClient, tmp_path, started_jobs
):
    from utils.paths import resolve_output_dir

    source = tmp_path / "checkpoint"
    source.mkdir()
    response = client.post(
        "/api/advanced/split/run",
        json={"source": str(source), "output_dir": "sharded-test", "prefer": "auto"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["output_dir"] == str(resolve_output_dir("sharded-test"))
