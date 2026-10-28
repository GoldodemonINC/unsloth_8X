# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The pre-flight check reads the rows the trainer would, and never rewrites them."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.training import dataset_preflight
from routes import training as training_routes


def _write_jsonl(path: Path, rows: list[str]) -> Path:
    path.write_text("\n".join(rows) + "\n", encoding = "utf-8")
    return path


def test_jsonl_counts_broken_missing_duplicate_and_long_rows(tmp_path):
    dataset = _write_jsonl(
        tmp_path / "train.jsonl",
        [
            json.dumps({"text": "a normal row"}),
            json.dumps({"instruction": "do a thing", "output": "done"}),
            '{"text": "the closing brace is missing"',
            json.dumps({"label": "no trainable text here"}),
            json.dumps({"text": "a normal row"}),
            json.dumps({"text": "x" * 40}),
        ],
    )

    report = dataset_preflight.scan(dataset, long_row_chars = 20)

    assert report["rows"] == 5, "the broken line is counted as an issue, not a row"
    assert report["checked_rows"] == 5
    assert report["invalid_rows"] == 1
    assert report["missing_text_rows"] == 1
    assert report["duplicate_rows"] == 1
    assert report["long_rows"] == 1
    assert report["blocking_rows"] == 2
    assert report["ok"] is False
    assert report["text_field"] == "text"
    assert report["samples"]["invalid"][0]["line"] == 3
    assert report["samples"]["missing_text"][0]["keys"] == ["label"]
    assert dataset.read_text(encoding = "utf-8").endswith("\n"), "the scan must not rewrite the file"


def test_json_array_and_wrapped_object_are_both_read(tmp_path):
    array = tmp_path / "array.json"
    array.write_text(json.dumps([{"text": "one"}, {"text": "two"}]), encoding = "utf-8")
    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(json.dumps({"data": [{"content": "three"}]}), encoding = "utf-8")

    assert dataset_preflight.scan(array)["rows"] == 2
    report = dataset_preflight.scan(wrapped)
    assert report["rows"] == 1
    assert report["text_field"] == "content"


def test_malformed_json_is_reported_instead_of_raising(tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text('{"data": [', encoding = "utf-8")
    report = dataset_preflight.scan(broken)
    assert report["invalid_rows"] == 1
    assert report["ok"] is False
    assert "not valid JSON" in report["samples"]["invalid"][0]["error"]


def test_csv_and_folders_are_supported(tmp_path):
    csv_path = tmp_path / "rows.csv"
    csv_path.write_text("text,label\nhello,1\n,2\n", encoding = "utf-8")
    report = dataset_preflight.scan(csv_path)
    assert report["rows"] == 2
    assert report["missing_text_rows"] == 1
    assert report["text_field"] == "text"

    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "a.jsonl").write_text('{"text": "one"}\n', encoding = "utf-8")
    (folder / "b.jsonl").write_text('{"text": "two"}\n', encoding = "utf-8")
    (folder / "notes.txt").write_text("ignored", encoding = "utf-8")
    folder_report = dataset_preflight.scan(folder)
    assert folder_report["files"] == 2
    assert folder_report["rows"] == 2
    assert folder_report["format"] == "folder"


def test_unknown_and_missing_sources_are_refused_with_a_reason(tmp_path):
    with pytest.raises(FileNotFoundError):
        dataset_preflight.scan(tmp_path / "missing.jsonl")
    with pytest.raises(ValueError, match = "JSONL, JSON, CSV or Parquet"):
        dataset_preflight.scan(_write_jsonl(tmp_path / "train.txt", ["hello"]))


def test_scan_stops_at_the_row_limit_and_says_so(tmp_path):
    dataset = _write_jsonl(
        tmp_path / "train.jsonl",
        [json.dumps({"text": f"row {index}"}) for index in range(10)],
    )
    report = dataset_preflight.scan(dataset, max_rows = 4)
    assert report["checked_rows"] == 4
    assert report["truncated"] is True
    assert any("Raise the row limit" in warning for warning in report["warnings"])


def test_the_route_returns_the_report_and_refuses_a_missing_path(tmp_path, monkeypatch):
    # The real resolver only admits paths under a dataset root; this test's file
    # is in pytest's temp folder, so keep that policy out of the way.
    monkeypatch.setattr(training_routes, "resolve_dataset_path", lambda value: Path(value))
    app = FastAPI()
    app.dependency_overrides[training_routes.get_current_subject] = lambda: "test-user"
    app.include_router(training_routes.router, prefix = "/api/train")
    client = TestClient(app)

    dataset = _write_jsonl(tmp_path / "train.jsonl", [json.dumps({"text": "hello"})])
    response = client.post("/api/train/dataset/preflight", json = {"path": str(dataset)})
    assert response.status_code == 200
    assert response.json()["rows"] == 1
    assert response.json()["ok"] is True

    missing = client.post(
        "/api/train/dataset/preflight",
        json = {"path": str(tmp_path / "missing.jsonl")},
    )
    assert missing.status_code == 400
    assert "not found" in missing.json()["detail"].lower()
