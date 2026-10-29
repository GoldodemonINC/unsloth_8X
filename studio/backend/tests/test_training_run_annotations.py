# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json

import pytest


def _db(monkeypatch, tmp_path):
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    studio_db.create_run(
        id = "run-a",
        model_name = "unsloth/alpha",
        dataset_name = "alpaca",
        config_json = "{}",
        started_at = "2026-01-02T00:00:00Z",
        total_steps = 10,
    )
    studio_db.create_run(
        id = "run-b",
        model_name = "unsloth/beta",
        dataset_name = "sharegpt",
        config_json = "{}",
        started_at = "2026-01-01T00:00:00Z",
        total_steps = 20,
    )
    return studio_db


def test_tags_and_notes_round_trip(monkeypatch, tmp_path):
    db = _db(monkeypatch, tmp_path)

    db.update_run_metadata(
        "run-a",
        {"tags": json.dumps(["baseline", "lr-test"]), "notes": "fixed seed"},
    )

    run = db.get_run("run-a")

    assert run["tags"] == ["baseline", "lr-test"]
    assert run["notes"] == "fixed seed"


def test_search_matches_names_notes_and_tags(monkeypatch, tmp_path):
    db = _db(monkeypatch, tmp_path)
    db.update_run_metadata(
        "run-a",
        {"tags": json.dumps(["baseline"]), "notes": "warm restart"},
    )

    assert [run["id"] for run in db.list_runs(query = "warm")["runs"]] == ["run-a"]
    assert [run["id"] for run in db.list_runs(query = "sharegpt")["runs"]] == ["run-b"]
    assert [run["id"] for run in db.list_runs(tag = "baseline")["runs"]] == ["run-a"]
    assert db.list_runs(query = "warm")["total"] == 1


def test_a_run_without_tags_answers_an_empty_list(monkeypatch, tmp_path):
    db = _db(monkeypatch, tmp_path)

    assert db.get_run("run-a")["tags"] == []
    assert db.list_runs()["runs"][0]["tags"] == []


def test_tag_filter_ignores_a_malformed_stored_value(monkeypatch, tmp_path):
    db = _db(monkeypatch, tmp_path)
    conn = db.get_connection()
    conn.execute(
        "UPDATE training_runs SET tags = ? WHERE id = ?",
        ("not json", "run-a"),
    )
    conn.commit()
    conn.close()

    assert db.list_runs(tag = "baseline")["runs"] == []
    assert db.get_run("run-a")["tags"] == []


def test_the_update_request_trims_and_deduplicates_tags():
    from models.training import TrainingRunUpdateRequest

    payload = TrainingRunUpdateRequest.model_validate(
        {"tags": ["  baseline ", "Baseline", "lr test", ""]}
    )

    assert payload.tags == ["baseline", "lr test"]


def test_the_update_request_refuses_more_than_twelve_tags():
    from models.training import TrainingRunUpdateRequest

    with pytest.raises(ValueError):
        TrainingRunUpdateRequest.model_validate(
            {"tags": [f"tag-{index}" for index in range(13)]}
        )
