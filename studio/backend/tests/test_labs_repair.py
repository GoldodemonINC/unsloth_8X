# SPDX-License-Identifier: NRC-2
# Copyright 2026-present the X8 authors. See /LICENSE-NRC-2.md

"""Tests for the shrink repair preset.

The point of the preset is the order of its stages and the tolerance the gate
enforces, so those are what is pinned here.
"""

from __future__ import annotations

import pytest

from core.advanced import repair
from core.labs import pipelines


def _report(aggregate: float, suites: tuple[str, ...] = ("mmlu",)) -> dict:
    return {
        "comparison": {
            "entries": [{"suite": suite, "delta": aggregate} for suite in suites],
            "aggregate_delta": aggregate,
        }
    }


def test_the_spec_measures_before_and_after_training(tmp_path):
    spec = repair.build_repair_spec(
        str(tmp_path / "shrunken"), str(tmp_path / "base"), suites = ["mmlu", "gsm8k"]
    )

    kinds = [stage["kind"] for stage in spec["stages"]]
    assert kinds == ["benchmark", "train", "benchmark"]
    assert spec["stages"][0]["baseline"] == str(tmp_path / "base")
    assert spec["stages"][2]["baseline"] == str(tmp_path / "base")
    assert spec["stages"][1]["steps"] == 120
    assert spec["gate"]["minimum"] == pytest.approx(-0.02)
    assert spec["gate"]["suites"] == ["mmlu", "gsm8k"]


def test_the_same_folder_cannot_be_its_own_baseline(tmp_path):
    with pytest.raises(repair.RepairError):
        repair.build_repair_spec(str(tmp_path), str(tmp_path))


def test_an_unknown_suite_is_rejected(tmp_path):
    with pytest.raises(repair.RepairError):
        repair.build_repair_spec(
            str(tmp_path / "a"), str(tmp_path / "b"), suites = ["not-a-suite"]
        )


def test_the_tolerance_must_be_a_fraction(tmp_path):
    with pytest.raises(repair.RepairError):
        repair.build_repair_spec(str(tmp_path / "a"), str(tmp_path / "b"), tolerance = 2)


def test_the_preset_plans_through_the_pipeline_engine(tmp_path):
    spec = pipelines.build_preset(
        repair.PRESET_ID, str(tmp_path / "shrunken"), str(tmp_path / "base")
    )

    plan = pipelines.preflight(spec)

    assert plan["blocked"] is False, plan["blocked_reason"]
    assert plan["preset"] == repair.PRESET_ID
    assert plan["gate"]["minimum"] == pytest.approx(-0.02)
    assert len(plan["stages"]) == 3


def test_an_unknown_preset_id_is_refused(tmp_path):
    with pytest.raises(pipelines.PipelineError):
        pipelines.build_preset("does-not-exist", str(tmp_path / "a"), str(tmp_path / "b"))


def test_a_gate_without_a_minimum_is_refused(tmp_path):
    spec = {
        "source": str(tmp_path),
        "stages": [{"kind": "benchmark"}],
        "gate": {"metric": "aggregate_delta"},
    }

    with pytest.raises(pipelines.PipelineError):
        pipelines.preflight(spec)


def test_a_repair_that_clears_the_tolerance_passes():
    verdict = repair.judge_repair(_report(-0.05), _report(-0.01))

    assert verdict["passed"] is True
    assert verdict["verdict"] == "recovered"
    assert verdict["gain"] == pytest.approx(0.04)


def test_a_repair_that_stays_below_the_tolerance_fails_with_the_number():
    verdict = repair.judge_repair(_report(-0.05), _report(-0.06))

    assert verdict["passed"] is False
    assert verdict["verdict"] == "still-regressed"
    assert any("6.0 points" in note for note in verdict["notes"])


def test_a_passing_repair_that_lost_ground_is_called_out():
    # Both clear the two-point gate, but the repair ended worse than the shrink.
    verdict = repair.judge_repair(_report(-0.005), _report(-0.019))

    assert verdict["passed"] is True
    assert any("cost more than it returned" in note for note in verdict["notes"])


def test_a_report_without_scores_is_inconclusive_not_a_failure():
    verdict = repair.judge_repair(_report(-0.05), {"comparison": {"entries": []}})

    assert verdict["verdict"] == "inconclusive"
    assert verdict["passed"] is False
    assert verdict["after"] is None


def test_the_preset_describes_itself_for_the_ui():
    described = repair.describe_preset()

    assert described["id"] == "shrink-repair"
    assert described["defaults"]["tolerance"] == pytest.approx(0.02)
    assert len(described["stages"]) == 3
