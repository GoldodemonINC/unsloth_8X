# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One loss-anomaly contract, two languages.

The chart callout reads the curve in TypeScript; the webhook alert reads it in Python. If a
threshold moves on one side only, the UI can flag a run the alert never fires for (or the
reverse). ``tests/fixtures/loss_anomaly_cases.json`` is the shared contract: this module runs
the Python mirror over every case, and ``studio/frontend/tests/loss-anomaly-parity.test.ts``
runs the shipped TypeScript detector over the same file. A divergence names the case.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from core.training.loss_anomaly import detect_loss_anomaly

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "loss_anomaly_cases.json"


def _cases() -> list[dict]:
    return json.loads(_FIXTURE.read_text(encoding = "utf-8"))["cases"]


def _points(case: dict) -> list[tuple[float, float]]:
    raw = case.get("points")
    if raw is None:
        raw = [[index + 1, value] for index, value in enumerate(case["values"])]
    return [(step, math.nan if value is None else float(value)) for step, value in raw]


def _verdict(anomaly: dict | None) -> dict | None:
    if anomaly is None:
        return None
    value = anomaly["value"]
    return {
        "kind": anomaly["kind"],
        "severity": anomaly["severity"],
        "step": anomaly["step"],
        "value": value if value is not None and math.isfinite(value) else None,
    }


@pytest.mark.parametrize("case", _cases(), ids = lambda case: case["name"])
def test_python_mirror_matches_the_shared_case_table(case: dict) -> None:
    actual = _verdict(detect_loss_anomaly(_points(case)))
    assert actual == case["expected"], (
        f"{case['name']}: {case['why']}.\n"
        f"  shared table says: {case['expected']}\n"
        f"  Python mirror says: {actual}\n"
        "The webhook alert and the chart callout have drifted. Fix the Python mirror to "
        "match the shipped UI unless the table itself is wrong."
    )
