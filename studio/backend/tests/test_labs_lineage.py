# SPDX-License-Identifier: NRC-2
# Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-2.md

"""Lineage is drawn from recorded paths, never from folder-name resemblance."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from core import labs
from core.labs import lineage
from routes import labs as labs_routes


def _manifest(directory: Path, *, run_id: str, stage: str, inputs: dict, created: float) -> Path:
    directory.mkdir(parents = True, exist_ok = True)
    payload = {
        "schema_version": 2,
        "run_id": run_id,
        "stage": stage,
        "created_at": created,
        "created_at_iso": f"2026-01-0{int(created)}T00:00:00Z",
        "studio_version": "2026.6.0",
        "environment": {"python": "3.12"},
        "config": {},
        "seeds": {},
        "inputs": inputs,
        "artifacts": {},
        "events": [],
    }
    path = directory / "run_manifest.json"
    path.write_text(json.dumps(payload), encoding = "utf-8")
    return path


def test_parent_paths_become_edges_and_unrecorded_paths_do_not(tmp_path):
    base = tmp_path / "base-run"
    _manifest(base, run_id = "train-base", stage = "train", inputs = {}, created = 1)
    other = tmp_path / "other"
    _manifest(other, run_id = "merge-other", stage = "merge", inputs = {}, created = 1.5)
    merged = tmp_path / "merged"
    _manifest(
        merged,
        run_id = "merge-a-b",
        stage = "merge",
        created = 2,
        inputs = {
            "model_a": {"source": str(base), "kind": "directory"},
            "model_b": {"source": str(other), "kind": "directory"},
            "model_c": {"source": str(tmp_path / "not-a-run"), "kind": "directory"},
        },
    )
    scaled = tmp_path / "scaled"
    _manifest(
        scaled,
        run_id = "scale-merged",
        stage = "scale",
        created = 3,
        inputs = {"source": str(merged), "kind": "directory"},
    )

    report = lineage.scan(tmp_path)

    assert {node["id"] for node in report["nodes"]} == {
        "train-base",
        "merge-other",
        "merge-a-b",
        "scale-merged",
    }
    assert {
        (edge["from"], edge["to"], edge["label"]) for edge in report["edges"]
    } == {
        ("train-base", "merge-a-b", "model_a"),
        ("merge-other", "merge-a-b", "model_b"),
        ("merge-a-b", "scale-merged", "source"),
    }
    assert set(report["roots"]) == {"train-base", "merge-other"}
    assert report["truncated"] is False


def test_a_hub_reference_never_links_to_a_folder_that_merely_exists(tmp_path):
    folder = tmp_path / "looks-like-a-repo"
    _manifest(folder, run_id = "train-one", stage = "train", inputs = {}, created = 1)
    trained = tmp_path / "trained"
    _manifest(
        trained,
        run_id = "train-two",
        stage = "train",
        created = 2,
        inputs = {"base_model": {"kind": "reference", "value": "owner/model"}},
    )

    report = lineage.scan(tmp_path)
    assert report["edges"] == []
    assert set(report["roots"]) == {"train-one", "train-two"}


def test_a_copied_run_folder_is_not_counted_twice(tmp_path):
    first = tmp_path / "a-first"
    _manifest(first, run_id = "same-run", stage = "train", inputs = {}, created = 1)
    copy = tmp_path / "b-copy"
    _manifest(copy, run_id = "same-run", stage = "train", inputs = {}, created = 1)

    report = lineage.scan(tmp_path)
    assert len(report["nodes"]) == 1
    assert report["nodes"][0]["path"] == str(first)


def test_an_incomplete_manifest_is_shown_with_its_problems(tmp_path):
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "run_manifest.json").write_text(
        json.dumps({"run_id": "broken-run", "stage": "", "created_at": 1}),
        encoding = "utf-8",
    )

    report = lineage.scan(tmp_path)
    node = report["nodes"][0]
    assert node["valid"] is False
    assert any("Missing required field" in problem for problem in node["problems"])


def test_the_route_is_closed_until_the_experiment_is_on(monkeypatch, tmp_path):
    lineage_dir = tmp_path / "outputs"
    lineage_dir.mkdir()
    _manifest(lineage_dir / "run", run_id = "run-1", stage = "train", inputs = {}, created = 1)

    monkeypatch.setattr(
        "storage.studio_db.get_app_setting",
        lambda key, default = None: {"enabled": True, "features": {"lineage": True}},
    )
    monkeypatch.setattr(
        "storage.studio_db.upsert_app_settings",
        lambda values: values,
    )
    monkeypatch.setattr(lineage, "scan", lambda root = None: {"nodes": [], "edges": []})
    app = FastAPI()
    app.dependency_overrides[labs_routes._require_owner] = lambda: None
    app.include_router(labs_routes.router, prefix = "/api/labs")
    client = TestClient(app)

    allowed = client.get("/api/labs/lineage")
    assert allowed.status_code == 200

    monkeypatch.setattr(
        "storage.studio_db.get_app_setting",
        lambda key, default = None: {"enabled": True, "features": {"lineage": False}},
    )
    refused = client.get("/api/labs/lineage")
    assert refused.status_code == 403
    assert "Run lineage" in refused.json()["detail"]


def test_lineage_is_registered_for_the_settings_tab(monkeypatch):
    monkeypatch.setattr("storage.studio_db.get_app_setting", lambda key, default = None: None)
    state = labs.describe_state()
    entry = next(feature for feature in state["features"] if feature["id"] == "lineage")
    assert entry["status"] in ("alpha", "beta")
    assert entry["studio"] is True
    assert "manifest" in entry["description"]
