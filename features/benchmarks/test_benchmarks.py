# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from pathlib import Path

import pytest

from features.benchmarks import benchmarks


def _report() -> dict:
    return {
        "baseline_scores": {"mmlu": 0.40, "gsm8k": 0.12},
        "candidate_scores": {"mmlu": 0.44, "gsm8k": 0.11, "extra": 0.3},
        "comparison": {
            "entries": [
                {"suite": "mmlu", "baseline": 0.40, "candidate": 0.44, "delta": 0.04}
            ],
            "aggregate_delta": 0.015,
            "verdict": "pareto-positive",
            "improved": ["mmlu"],
            "regressions": ["gsm8k"],
            "flat": [],
        },
        "verdict": "pareto-positive",
        "errors": ["one suite timed out"],
        "warnings": [],
    }


def test_save_and_list_a_record(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmarks, "outputs", lambda: tmp_path)
    record = benchmarks.save_record(
        job_id = "run-1",
        artifact = str(tmp_path / "outputs" / "merged"),
        baseline = "org/base",
        report = _report(),
        suites = ["mmlu", "gsm8k"],
        limit = 50,
        batch_size = 4,
        duration_seconds = 12.5,
        output_dir = str(tmp_path / "runs" / "run-1"),
    )
    assert len(record["id"]) == 32
    assert record["candidate_scores"]["mmlu"] == pytest.approx(0.44)
    assert record["verdict"] == "pareto-positive"
    assert record["errors"] == ["one suite timed out"]

    listed = benchmarks.list_records()
    assert [item["id"] for item in listed] == [record["id"]]
    assert benchmarks.find_record(record["id"])["baseline_name"] == "base"

    assert benchmarks.list_records(artifact = "elsewhere") == []
    assert benchmarks.delete_record(record["id"]) is True
    assert benchmarks.list_records() == []
    assert benchmarks.delete_record(record["id"]) is False


def test_save_rejects_a_report_without_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmarks, "outputs", lambda: tmp_path)
    with pytest.raises(ValueError):
        benchmarks.save_record(
            job_id = "run-1",
            artifact = str(tmp_path / "a"),
            baseline = "b",
            report = {"errors": ["lm-eval failed"]},
            suites = ["mmlu"],
            limit = None,
            batch_size = 8,
            duration_seconds = 1.0,
            output_dir = str(tmp_path / "run"),
        )


def test_record_ids_are_checked_before_a_path_is_built(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmarks, "outputs", lambda: tmp_path)
    with pytest.raises(ValueError):
        benchmarks.delete_record("../../escape")
    with pytest.raises(ValueError):
        benchmarks.find_record("not-hex!")


def test_create_run_dir_stays_under_the_benchmark_root(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmarks, "outputs", lambda: tmp_path)
    first = benchmarks.create_run_dir()
    second = benchmarks.create_run_dir()
    assert first.is_dir() and second.is_dir()
    assert first != second
    assert Path(first).parent == benchmarks.runs_root()
