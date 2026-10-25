# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Auto-scale: the growth comes from the data, and the gate is on the server."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from core import labs
from core.labs import autoscale, cmt
from routes import labs as labs_routes


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


def _jsonl(tmp_path: Path, name: str, rows: int, text: str = "a sentence about training") -> Path:
    path = tmp_path / name
    with path.open("w", encoding = "utf-8") as stream:
        for index in range(rows):
            stream.write(json.dumps({"text": f"{text} {index}"}) + "\n")
    return path


def test_the_shared_sizing_rule_prefers_tokens_then_rows_then_the_floor():
    from_tokens = cmt.sizing(tokens = 20_000_000, rows = 10)
    assert from_tokens["rule"] == "tokens"
    assert from_tokens["added_params"] == 4_600_000

    from_rows = cmt.sizing(rows = 20_000)
    assert from_rows["rule"] == "rows"
    assert from_rows["added_params"] == 2_000_000

    empty = cmt.sizing()
    assert empty["rule"] == "floor"
    assert empty["added_params"] == cmt.MIN_ADDED_PARAMS
    assert cmt.MIN_ADDED_PARAMS <= empty["added_params"] <= cmt.MAX_ADDED_PARAMS


def test_plan_sizes_from_the_dataset_it_was_given(tmp_path):
    model = _model(tmp_path)
    dataset = _jsonl(tmp_path, "rows.jsonl", 2000)
    report = autoscale.plan({"model": str(model), "datasets": [{"path": str(dataset)}]})

    assert report["sizing"]["rule"] == "tokens"
    assert report["sizing"]["rows"] == 2000
    assert report["sizing"]["tokens"] > 0
    assert cmt.MIN_ADDED_PARAMS <= report["sizing"]["added_params"] <= cmt.MAX_ADDED_PARAMS
    assert report["expansion"]["base_params"] > 0
    assert report["expansion"]["target_params"] > report["expansion"]["base_params"]
    assert report["expansion"]["added_params"] > 0
    assert report["expansion"]["floor_bypassed"] is True
    assert report["datasets"][0]["rows"] == 2000
    assert "training is a separate run" in " ".join(report["notes"])


def test_plan_without_a_dataset_uses_the_floor_and_says_why(tmp_path):
    model = _model(tmp_path)
    report = autoscale.plan({"model": str(model)})
    assert report["sizing"]["rule"] == "floor"
    assert report["sizing"]["added_params"] == cmt.MIN_ADDED_PARAMS
    assert any("1M floor" in warning for warning in report["warnings"])


def test_plan_refuses_a_size_outside_the_band_and_a_repeated_dataset(tmp_path):
    model = _model(tmp_path)
    dataset = _jsonl(tmp_path, "rows.jsonl", 50)
    path = model  # any string; the bounds are checked before the model is read
    with pytest.raises(ValueError):
        autoscale.plan({"model": str(path), "added_params": 500_000})
    with pytest.raises(ValueError):
        autoscale.plan({"model": str(path), "added_params": 800_000_000})
    with pytest.raises(ValueError):
        autoscale.plan({"model": str(model), "datasets": [{"path": str(dataset)}, {"path": str(dataset)}]})
    with pytest.raises(ValueError):
        autoscale.plan({"model": str(model), "datasets": [{"path": str(tmp_path / "missing.jsonl")}]})
    with pytest.raises(ValueError):
        autoscale.plan({"model": ""})


def _client(monkeypatch, features: dict) -> TestClient:
    monkeypatch.setattr("storage.studio_db.get_app_setting", lambda key, default = None: {"enabled": True, "features": features})
    monkeypatch.setattr("storage.studio_db.upsert_app_settings", lambda values: values)
    app = FastAPI()
    app.dependency_overrides[labs_routes._require_owner] = lambda: None
    app.include_router(labs_routes.router, prefix = "/api/labs")
    return TestClient(app)


def test_auto_scale_routes_close_until_their_feature_is_on(monkeypatch):
    client = _client(monkeypatch, {"auto-scale": False})
    response = client.post("/api/labs/auto-scale/plan", json = {"spec": {"model": "/tmp/none"}})
    assert response.status_code == 403
    assert "Scale before training" in response.json()["detail"]

    client = _client(monkeypatch, {})
    monkeypatch.setattr("storage.studio_db.get_app_setting", lambda key, default = None: {"enabled": False, "features": {}})
    closed = client.post("/api/labs/auto-scale/plan", json = {"spec": {"model": "/tmp/none"}})
    assert closed.status_code == 403
    assert "Labs" in closed.json()["detail"]


def test_plan_route_returns_the_sized_growth(monkeypatch, tmp_path):
    client = _client(monkeypatch, {"auto-scale": True})
    model = _model(tmp_path)
    dataset = _jsonl(tmp_path, "rows.jsonl", 500)
    response = client.post(
        "/api/labs/auto-scale/plan",
        json = {"spec": {"model": str(model), "datasets": [{"path": str(dataset)}]}},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["sizing"]["rows"] == 500
    assert body["expansion"]["target_params"] > body["expansion"]["base_params"]


def test_run_route_refuses_a_blocked_plan(monkeypatch, tmp_path):
    client = _client(monkeypatch, {"auto-scale": True})
    blocked = {
        "runnable": False,
        "blocked_reason": "The expansion engine is not installed on this host.",
    }
    monkeypatch.setattr(autoscale, "plan", lambda spec, token = None: blocked)
    response = client.post(
        "/api/labs/auto-scale/run",
        json = {"spec": {"model": str(_model(tmp_path))}},
    )
    assert response.status_code == 400
    assert "not installed" in response.json()["detail"]


def test_every_registered_feature_is_described_for_the_settings_tab(monkeypatch):
    monkeypatch.setattr("storage.studio_db.get_app_setting", lambda key, default = None: None)
    state = labs.describe_state()
    ids = [feature["id"] for feature in state["features"]]
    assert "auto-scale" in ids
    assert ids.index("auto-scale") == labs.FEATURE_IDS.index("auto-scale")
    entry = next(feature for feature in state["features"] if feature["id"] == "auto-scale")
    assert entry["status"] in ("alpha", "beta")
    assert entry["studio"] is False
    assert "1M-700M" in entry["description"]
