// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  type LossAnomaly,
  type LossPoint,
  detectLossAnomaly,
} from "../src/features/training/lib/loss-anomaly.ts";

type Verdict = {
  kind: string;
  severity: string;
  step: number;
  value: number | null;
};

type Case = {
  name: string;
  why: string;
  values?: (number | null)[];
  points?: [number, number | null][];
  expected: Verdict | null;
};

const fixture = JSON.parse(
  readFileSync(
    new URL(
      "../../backend/tests/fixtures/loss_anomaly_cases.json",
      import.meta.url,
    ),
    "utf8",
  ),
) as { cases: Case[] };

function points(case_: Case): LossPoint[] {
  const raw =
    case_.points ??
    (case_.values ?? []).map(
      (value, index) => [index + 1, value] as [number, number | null],
    );
  return raw.map(([step, value]) => ({
    step,
    value: value === null ? Number.NaN : value,
  }));
}

function verdict(anomaly: LossAnomaly | null): Verdict | null {
  if (!anomaly) {
    return null;
  }
  return {
    kind: anomaly.kind,
    severity: anomaly.severity,
    step: anomaly.step,
    value: Number.isFinite(anomaly.value) ? anomaly.value : null,
  };
}

for (const case_ of fixture.cases) {
  test(`loss-anomaly parity: ${case_.name}`, () => {
    assert.deepEqual(
      verdict(detectLossAnomaly(points(case_))),
      case_.expected,
      `${case_.name}: ${case_.why}. The chart callout and the Python webhook mirror have drifted.`,
    );
  });
}
