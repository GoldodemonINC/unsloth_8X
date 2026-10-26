# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Scaler's HTTP surface: planning is wired, and a refused plan never starts a job."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from routes.advanced import router

CONFIG = {
    "model_type": "llama",
    "hidden_size": 576,
    "num_hidden_layers": 30,
    "num_attention_heads": 9,
    "num_key_value_heads": 3,
    "intermediate_size": 1536,
    "vocab_size": 49152,
    "tie_word_embeddings": True,
}


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix = "/api/advanced")
    return TestClient(app)


def _checkpoint(tmp_path: Path, config: dict) -> str:
    directory = tmp_path / "model"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(directory)


def test_plan_endpoint_returns_the_slider_range(client, tmp_path):
    response = client.post("/api/advanced/scale/plan", json = {"source": _checkpoint(tmp_path, CONFIG)})

    assert response.status_code == 200
    body = response.json()
    assert body["min_params"] == body["base_params"]
    assert body["max_params"] == body["base_params"] + body["cap_added_params"]
    assert body["floor_params"] == body["base_params"]
    assert body["default_target_params"] > body["base_params"]
    assert body["runnable"] is True
    assert body["methods"][0]["id"] == "depth"


def test_plan_endpoint_accepts_added_params(client, tmp_path):
    response = client.post(
        "/api/advanced/scale/plan",
        json={"source": _checkpoint(tmp_path, CONFIG), "added_params": 300_000_000},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["added_params"] == 300_000_000
    assert body["target_params"] == body["base_params"] + 300_000_000


def test_plan_endpoint_reports_a_rotten_source_as_400(client, tmp_path):
    response = client.post(
        "/api/advanced/scale/plan", json = {"source": str(tmp_path / "missing")}
    )

    assert response.status_code == 400


def test_capabilities_advertise_the_scaler(client):
    response = client.get("/api/advanced/capabilities")

    assert response.status_code == 200
    scaler = response.json()["scaler"]
    assert scaler["cap_added_params"] == 3_000_000_000
    assert scaler["growth_floor"] == 1.0
    assert scaler["shrink_floor"] == 0.5
    assert {"depth", "mlp_width", "depth+mlp_width"} == {
        item["id"] for item in scaler["methods"]
    }
    assert [item["id"] for item in scaler["shrink_methods"]] == ["depth"]


def test_a_shrink_run_past_the_floor_is_refused_before_a_job_starts(client, tmp_path):
    """The plan owns the refusals; the run endpoint must not start a job anyway."""
    source = _checkpoint(tmp_path, CONFIG)
    response = client.post(
        "/api/advanced/scale/run",
        json = {
            "source": source,
            "target_params": 20_000_000,
            "direction": "shrink",
            "output_dir": str(tmp_path / "out"),
        },
    )

    assert response.status_code == 400
    assert "at least 50%" in response.json()["detail"].lower()


def test_plan_endpoint_plans_a_shrink_below_the_base(client, tmp_path):
    response = client.post(
        "/api/advanced/scale/plan",
        json={"source": _checkpoint(tmp_path, CONFIG), "target_params": 110_000_000},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["direction"] == "shrink"
    assert body["runnable"] is True
    assert body["projection"]["removed_layers"] >= 1
    assert body["methods"] == [{"id": "depth", "label": "Depth (drop whole transformer blocks)"}]
