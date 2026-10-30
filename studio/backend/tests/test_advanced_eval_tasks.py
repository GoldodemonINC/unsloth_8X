# SPDX-License-Identifier: NRC-3
# Copyright 2026-present the X8 authors. See /LICENSE-NRC-3.md

"""Tests for file-backed custom evaluation tasks.

These cover the parts that can be wrong without a GPU: discovery, validation,
answer normalisation, the plan handshake, and the YAML lm-eval is handed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.advanced import eval_harness, eval_tasks


def _jsonl(path: Path, *records: dict) -> Path:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding = "utf-8"
    )
    return path


def _generative(path: Path) -> Path:
    return _jsonl(
        path,
        {"prompt": "Is a refund possible after 30 days?", "answer": "no"},
        {"prompt": "Can I change my plan?", "answer": "yes"},
    )


def test_a_jsonl_file_becomes_a_generative_task(tmp_path, monkeypatch):
    monkeypatch.setenv(eval_tasks.DIRECTORY_ENV, str(tmp_path))
    _generative(tmp_path / "support-triage.jsonl")

    tasks, problems = eval_tasks.load_tasks()

    assert problems == []
    task = tasks["support-triage"]
    assert task.suite_id == "custom:support-triage"
    assert task.lm_eval_name == "x8_local_support-triage"
    assert task.kind == "generative"
    assert task.metric == "exact_match"
    assert len(task.samples) == 2


def test_a_choices_file_becomes_a_multiple_choice_task(tmp_path):
    path = tmp_path / "severity.json"
    path.write_text(
        json.dumps(
            {
                "name": "Ticket severity",
                "description": "Pick the queue a ticket belongs to.",
                "samples": [
                    {"prompt": "Site is down.", "choices": ["low", "high"], "answer": "high"},
                    {"prompt": "Typo in docs.", "choices": ["low", "high"], "answer": 0},
                ],
            }
        ),
        encoding = "utf-8",
    )

    task = eval_tasks.load_task(path)

    assert task.kind == "multiple_choice"
    assert task.metric == "acc"
    # Text answers are normalised to the index lm-eval compares against.
    assert [sample["answer"] for sample in task.samples] == [1, 0]


def test_a_metadata_line_renames_a_jsonl_task(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text(
        json.dumps({"__task__": {"name": "Billing QA", "label": "Billing", "kind": "generative"}})
        + "\n"
        + json.dumps({"prompt": "Refund?", "answer": "yes"})
        + "\n",
        encoding = "utf-8",
    )

    task = eval_tasks.load_task(path)

    assert task.slug == "billing-qa"
    assert task.label == "Billing"


def test_one_broken_file_does_not_hide_the_others(tmp_path):
    _generative(tmp_path / "good.jsonl")
    (tmp_path / "broken.json").write_text("{ not json", encoding = "utf-8")

    tasks, problems = eval_tasks.load_tasks(tmp_path)

    assert list(tasks) == ["good"]
    assert len(problems) == 1
    assert "broken.json" in problems[0]


def test_an_unknown_choice_answer_is_rejected_with_the_sample_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    _jsonl(path, {"prompt": "A?", "choices": ["x", "y"], "answer": "z"})

    with pytest.raises(eval_tasks.EvalTaskError) as error:
        eval_tasks.load_task(path)

    assert "sample 1" in str(error.value)


def test_a_metric_that_does_not_match_the_kind_is_rejected():
    with pytest.raises(eval_tasks.EvalTaskError) as error:
        eval_tasks._build_task(
            name = "bad",
            raw_samples = [{"prompt": "A?", "answer": "b"}],
            source = "bad.jsonl",
            kind = "generative",
            metric = "acc",
        )

    assert "multiple_choice" in str(error.value)


def test_a_duplicate_slug_is_reported_not_overwritten(tmp_path):
    (tmp_path / "a.json").write_text(
        json.dumps({"name": "same", "samples": [{"prompt": "q", "answer": "a"}]}),
        encoding = "utf-8",
    )
    (tmp_path / "b.json").write_text(
        json.dumps({"name": "same", "samples": [{"prompt": "r", "answer": "b"}]}),
        encoding = "utf-8",
    )

    tasks, problems = eval_tasks.load_tasks(tmp_path)

    assert list(tasks) == ["same"]
    assert tasks["same"].samples[0]["prompt"] == "q"
    assert any("already used" in problem for problem in problems)


def test_the_plan_accepts_a_custom_suite_and_still_rejects_a_missing_one(tmp_path):
    _generative(tmp_path / "triage.jsonl")

    plan = eval_harness.build_eval_plan(
        ["mmlu", "custom:triage"],
        baseline_model = "org/base",
        candidate_model = "org/tuned",
        tasks_dir = tmp_path,
    )

    assert plan.suites == ("mmlu", "custom:triage")
    with pytest.raises(ValueError):
        eval_harness.build_eval_plan(
            ["custom:absent"],
            baseline_model = "org/base",
            candidate_model = "org/tuned",
            tasks_dir = tmp_path,
        )


def test_the_command_runs_the_generated_task_by_name(tmp_path):
    command = eval_harness.build_lm_eval_command(
        "custom:triage",
        "org/m",
        limit = 5,
        batch_size = 2,
        task_name = "x8_local_triage",
        include_path = str(tmp_path),
    )

    assert "x8_local_triage" in command
    assert "custom:triage" not in command
    assert command[command.index("--include_path") + 1] == str(tmp_path)


def test_a_builtin_suite_is_not_a_custom_task():
    assert eval_tasks.resolve_suite("mmlu") is None


def test_written_task_files_carry_the_normalised_samples(tmp_path):
    source = tmp_path / "triage.json"
    source.write_text(
        json.dumps(
            {
                "name": "Triage",
                "samples": [
                    {"prompt": "Down?", "choices": ["no", "yes"], "answer": "yes"},
                ],
            }
        ),
        encoding = "utf-8",
    )
    task = eval_tasks.load_task(source)

    written = eval_tasks.write_task_files(task, tmp_path / "include")

    config = Path(written["config"]).read_text(encoding = "utf-8")
    assert "task: x8_local_triage" in config
    assert "output_type: multiple_choice" in config
    assert f"test: {Path(written['data']).as_posix()}" in config
    rows = Path(written["data"]).read_text(encoding = "utf-8").splitlines()
    assert json.loads(rows[0]) == {"prompt": "Down?", "choices": ["no", "yes"], "answer": 1}


def test_a_generative_task_renders_generation_settings(tmp_path):
    source = _generative(tmp_path / "triage.jsonl")
    task = eval_tasks.load_task(source)

    written = eval_tasks.write_task_files(task, tmp_path / "include")

    config = Path(written["config"]).read_text(encoding = "utf-8")
    assert "output_type: generate_until" in config
    assert "metric: exact_match" in config
    assert "do_sample: false" in config
