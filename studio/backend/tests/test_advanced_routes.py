# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Route-level checks for the advanced API.

These exercise the request models and the source validation without starting a
training run or touching the network.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from routes import advanced  # noqa: E402


@pytest.fixture()
def client() -> TestClient:
    app = fastapi.FastAPI()
    app.include_router(advanced.router, prefix = "/api/advanced")
    return TestClient(app)


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
