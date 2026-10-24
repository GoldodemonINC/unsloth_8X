# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Labs: one stored switch, one gate, and endpoints that close with it."""

from __future__ import annotations

import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from core import labs
from routes import labs as labs_routes


def _store(monkeypatch, state: dict | None = None) -> dict:
    store = {"value": state}

    def get(key, default = None):
        assert key == labs.SETTING_KEY
        return store["value"] if store["value"] is not None else default

    def upsert(values):
        store["value"] = values[labs.SETTING_KEY]
        return values

    monkeypatch.setattr("storage.studio_db.get_app_setting", get)
    monkeypatch.setattr("storage.studio_db.upsert_app_settings", upsert)
    return store


def _client(monkeypatch, state: dict | None = None) -> tuple[TestClient, dict]:
    store = _store(monkeypatch, state)
    app = FastAPI()
    app.dependency_overrides[labs_routes._require_owner] = lambda: None
    app.include_router(labs_routes.router, prefix = "/api/labs")
    return TestClient(app), store


def test_labs_defaults_off_and_every_feature_on(monkeypatch):
    _store(monkeypatch)
    state = labs.read_state()
    assert state["enabled"] is False
    assert state["features"] == {feature_id: True for feature_id in labs.FEATURE_IDS}
    assert labs.is_enabled() is False
    assert labs.feature_enabled("pipelines") is False


def test_turning_labs_on_does_not_override_an_explicit_off(monkeypatch):
    _store(monkeypatch, {"enabled": False, "features": {"pipelines": False}})
    labs.set_state(enabled = True)
    assert labs.is_enabled() is True
    assert labs.feature_enabled("pipelines") is False
    assert labs.feature_enabled("artifacts") is True


def test_unknown_feature_is_refused_and_never_stored(monkeypatch):
    store = _store(monkeypatch)
    with pytest.raises(ValueError):
        labs.set_state(features = {"not-a-feature": True})
    assert store["value"] is None
    with pytest.raises(ValueError):
        labs.feature_enabled("not-a-feature")


def test_a_disabled_feature_says_which_switch_to_turn_on(monkeypatch):
    _store(monkeypatch, {"enabled": False, "features": {}})
    with pytest.raises(labs.LabsDisabled) as master:
        labs.require("pipelines")
    assert "Settings → Labs" in str(master.value)

    _store(monkeypatch, {"enabled": True, "features": {"pipelines": False}})
    with pytest.raises(labs.LabsDisabled) as feature:
        labs.require("pipelines")
    assert "Pipelines" in str(feature.value)


def test_state_route_is_readable_and_put_saves(monkeypatch):
    client, store = _client(monkeypatch)
    response = client.get("/api/labs")
    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is False
    assert {feature["id"] for feature in body["features"]} == set(labs.FEATURE_IDS)
    assert all(feature["enabled"] for feature in body["features"])

    saved = client.put("/api/labs", json = {"enabled": True, "features": {"pipelines": False}})
    assert saved.status_code == 200
    assert saved.json()["enabled"] is True
    assert store["value"]["enabled"] is True
    assert store["value"]["features"]["pipelines"] is False


def test_put_rejects_an_unknown_feature(monkeypatch):
    client, store = _client(monkeypatch)
    response = client.put("/api/labs", json = {"features": {"teleport": True}})
    assert response.status_code == 400
    assert store["value"] is None


def test_every_studio_route_is_closed_while_labs_is_off(monkeypatch):
    client, _ = _client(monkeypatch)
    for path in ("/api/labs/artifacts", "/api/labs/pipelines", "/api/labs/artifacts/manifest?path=/tmp"):
        response = client.get(path)
        assert response.status_code == 403, path
        assert "Labs" in response.json()["detail"]

    planned = client.post(
        "/api/labs/pipelines/plan",
        json = {"spec": {"source": "/tmp/none", "stages": [{"kind": "benchmark"}]}},
    )
    assert planned.status_code == 403


def test_a_feature_that_is_off_closes_only_its_route(monkeypatch, tmp_path):
    client, _ = _client(
        monkeypatch, {"enabled": True, "features": {"pipelines": False, "artifacts": True}}
    )
    monkeypatch.setattr("core.labs.artifacts.root", lambda: tmp_path)
    assert client.get("/api/labs/artifacts").status_code == 200
    assert client.get("/api/labs/pipelines").status_code == 403


def test_manifest_route_needs_the_manifest_feature(monkeypatch, tmp_path):
    model = tmp_path / "trained"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding = "utf-8")
    (model / "run_manifest.json").write_text('{"stage": "train"}', encoding = "utf-8")
    client, _ = _client(
        monkeypatch,
        {"enabled": True, "features": {"artifacts": True, "manifest-viewer": False}},
    )
    monkeypatch.setattr("core.labs.artifacts.root", lambda: tmp_path)
    response = client.get(f"/api/labs/artifacts/manifest?path={model}")
    assert response.status_code == 403

    client, _ = _client(
        monkeypatch,
        {"enabled": True, "features": {"artifacts": True, "manifest-viewer": True}},
    )
    monkeypatch.setattr("core.labs.artifacts.root", lambda: tmp_path)
    allowed = client.get(f"/api/labs/artifacts/manifest?path={model}")
    assert allowed.status_code == 200
    assert allowed.json()["manifest"]["stage"] == "train"


def test_artifact_detail_attaches_the_raw_manifest_only_when_its_feature_is_on(
    monkeypatch, tmp_path
):
    model = tmp_path / "trained"
    model.mkdir()
    (model / "config.json").write_text(
        '{"model_type": "llama", "hidden_size": 64, "num_hidden_layers": 2,'
        ' "intermediate_size": 128, "vocab_size": 256,'
        ' "num_attention_heads": 2, "num_key_value_heads": 2}',
        encoding = "utf-8",
    )
    (model / "run_manifest.json").write_text('{"stage": "train"}', encoding = "utf-8")
    base = {"enabled": True, "features": {"artifacts": True, "manifest-viewer": False}}
    client, store = _client(monkeypatch, dict(base))
    monkeypatch.setattr("core.labs.artifacts.root", lambda: tmp_path)
    without = client.get(f"/api/labs/artifacts/detail?path={model}").json()
    assert "raw_manifest" not in without
    assert without["manifest"]["stage"] == "train"

    store["value"]["features"]["manifest-viewer"] = True
    with_manifest = client.get(f"/api/labs/artifacts/detail?path={model}").json()
    assert with_manifest["raw_manifest"]["stage"] == "train"


def test_describe_state_lists_the_registry_without_duplicating_it(monkeypatch):
    _store(monkeypatch, {"enabled": True, "features": {}})
    state = labs.describe_state()
    assert [feature["id"] for feature in state["features"]] == list(labs.FEATURE_IDS)
    assert all(feature["enabled"] for feature in state["features"])
    assert {feature["studio"] for feature in state["features"]} == {True, False}
