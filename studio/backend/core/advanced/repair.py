# SPDX-License-Identifier: NRC-2
# Copyright 2026-present the X8 authors. See /LICENSE-NRC-2.md

"""The shrink repair preset.

Shrinking a checkpoint removes transformer blocks, and a removed block is a
removed residual contribution: the output drifts from the base and the Scaler
says so. Until now that warning was the end of the story -- the user was left to
work out a recovery run by hand, and had no agreed definition of "recovered".

This module supplies both halves:

* :func:`build_repair_spec` returns the chain that measures the drift, trains the
  shrunken model back toward the base, and measures again. It is a normal
  pipeline spec, so ``pipelines.preflight`` prices it before anything starts.
* :func:`judge_repair` takes the two benchmark reports and decides whether the
  repair actually closed the gap, against a tolerance the caller states up
  front.

The gate is deliberately stated as a tolerance rather than "must match the
base": a shrunken model has fewer parameters, and demanding parity would fail
every honest repair. Default is two points of aggregate benchmark score, the
same order as the harness noise floor.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from . import eval_harness

PRESET_ID = "shrink-repair"
PRESET_LABEL = "Shrink repair"
DEFAULT_TOLERANCE = 0.02
DEFAULT_SUITES = ("mmlu",)


class RepairError(ValueError):
    """The preset was asked for something it cannot plan."""


def _suites(suites: Sequence[str] | None) -> list[str]:
    selected = list(suites or DEFAULT_SUITES)
    if not selected:
        raise RepairError("Pick at least one benchmark suite for the repair gate.")
    unknown = [suite for suite in selected if suite not in eval_harness.SUITES]
    if unknown:
        # Custom tasks are allowed too: they are the ones a domain fine-tune
        # should be judged on, and they validate through the same plan.
        from . import eval_tasks

        if not eval_tasks.resolve_suite(unknown[0]):
            raise RepairError(f"Unknown benchmark suite: {', '.join(unknown)}.")
    return selected


def build_repair_spec(
    source: str,
    base: str,
    *,
    suites: Sequence[str] | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    steps: int = 120,
    batch_size: int = 1,
    sequence_length: int = 2048,
    learning_rate: float = 5e-5,
    name: str | None = None,
) -> dict[str, Any]:
    """The recovery chain for a shrunken *source*, judged against *base*.

    Stage order is the point of the preset: measure before training, train, then
    measure with the same baseline and suites. A single post-training score would
    not say whether the repair helped, only where it ended up.
    """
    if not str(source or "").strip():
        raise RepairError("A shrunken checkpoint is required.")
    if not str(base or "").strip():
        raise RepairError("The base checkpoint is required: it is the scoring baseline.")
    if str(source).strip() == str(base).strip():
        raise RepairError("The shrunken checkpoint and the base are the same folder.")
    if not 0 <= float(tolerance) <= 1:
        raise RepairError("tolerance is a fraction between 0 and 1, e.g. 0.02 for two points.")
    if int(steps) < 1:
        raise RepairError("steps must be at least 1.")
    if int(batch_size) < 1:
        raise RepairError("batch_size must be at least 1.")
    if int(sequence_length) < 128:
        raise RepairError("sequence_length must be at least 128.")

    selected = _suites(suites)
    benchmark = {"kind": "benchmark", "baseline": base, "suites": selected}
    return {
        "name": name or f"{PRESET_LABEL}: {base} -> {source}",
        "preset": PRESET_ID,
        "source": source,
        "baseline": base,
        "stages": [
            dict(benchmark, label = "Measure the gap", phase = "before"),
            {
                "kind": "train",
                "label": "Recovery training",
                "phase": "repair",
                "steps": int(steps),
                "batch_size": int(batch_size),
                "sequence_length": int(sequence_length),
                "learning_rate": float(learning_rate),
                "notes": (
                    "Train the shrunken model on the same data as the original "
                    "fine-tune. The base is the reference, not a teacher: no "
                    "distillation hook is required for this chain to run."
                ),
            },
            dict(benchmark, label = "Measure the repair", phase = "after"),
        ],
        "gate": {
            "metric": "aggregate_delta",
            "minimum": -abs(float(tolerance)),
            "reference": "baseline",
            "suites": selected,
        },
        "notes": [
            "The before stage is what makes the gate meaningful: it records the drift "
            "the shrink introduced, so 'recovered' is measured against the shrunken "
            "model, not only against the base.",
            "A repair can fail. Fewer blocks hold less capacity, and no amount of "
            "training adds it back; the gate reports that rather than hiding it.",
        ],
    }


def describe_preset() -> dict[str, Any]:
    """What the UI needs to offer the preset without hard-coding it."""
    return {
        "id": PRESET_ID,
        "label": PRESET_LABEL,
        "description": (
            "Measure the drift a shrink introduced, train the shrunken model back "
            "toward the base, and judge the result against a stated tolerance."
        ),
        "defaults": {
            "suites": list(DEFAULT_SUITES),
            "tolerance": DEFAULT_TOLERANCE,
            "steps": 120,
            "batch_size": 1,
            "sequence_length": 2048,
            "learning_rate": 5e-5,
        },
        "stages": ["benchmark (before)", "train (recovery)", "benchmark (after)"],
        "gate": "aggregate benchmark delta must stay within -tolerance of the base",
    }


def _aggregate(report: Mapping[str, Any] | None) -> float | None:
    if not isinstance(report, Mapping):
        return None
    comparison = report.get("comparison")
    if not isinstance(comparison, Mapping):
        return None
    entries = comparison.get("entries")
    if not entries:
        return None
    value = comparison.get("aggregate_delta")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def judge_repair(
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any] | None,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """Decide whether *after* cleared the gate that *before* established.

    ``before`` and ``after`` are the harness reports for the shrunken model and
    the repaired model, both compared against the same base.
    """
    if not 0 <= float(tolerance) <= 1:
        raise RepairError("tolerance is a fraction between 0 and 1, e.g. 0.02 for two points.")
    floor = -abs(float(tolerance))
    before_delta = _aggregate(before)
    after_delta = _aggregate(after)
    if after_delta is None:
        return {
            "verdict": "inconclusive",
            "passed": False,
            "tolerance": abs(float(tolerance)),
            "minimum": floor,
            "before": before_delta,
            "after": None,
            "gain": None,
            "notes": [
                "The post-repair report has no comparable suite scores. Check the "
                "report's errors before treating this as a failed repair."
            ],
        }
    gain = after_delta - before_delta if before_delta is not None else None
    passed = after_delta >= floor
    notes: list[str] = []
    if before_delta is None:
        notes.append(
            "No pre-repair report was supplied, so the gain is unknown; the verdict "
            "judges the repaired model against the base only."
        )
    if passed and gain is not None and gain < 0:
        notes.append(
            "The repaired model cleared the gate but scored below the shrunken model: "
            "recovery training cost more than it returned."
        )
    if not passed:
        notes.append(
            f"The repaired model is {abs(after_delta) * 100:.1f} points below the base, "
            f"past the {abs(float(tolerance)) * 100:.1f}-point tolerance. Fewer blocks "
            "hold less capacity; a second pass rarely beats starting from a lighter "
            "shrink target."
        )
    return {
        "verdict": "recovered" if passed else "still-regressed",
        "passed": passed,
        "tolerance": abs(float(tolerance)),
        "minimum": floor,
        "before": before_delta,
        "after": after_delta,
        "gain": gain,
        "notes": notes,
    }
