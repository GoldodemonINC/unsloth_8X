// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  type LossAnomaly,
  detectLossAnomaly,
} from "../src/features/training/lib/loss-anomaly.ts";

type Segment = {
  count?: number;
  value?: number;
  from?: number;
  to?: number;
  values?: number[];
};

type ChartVerdict = {
  kind: string;
  severity: string;
  step: number;
};

type Case = {
  name: string;
  why: string;
  segments: Segment[];
  chart: ChartVerdict | null;
};

const fixture = JSON.parse(
  readFileSync(
    new URL(
      "../../backend/tests/fixtures/loss_anomaly_window_cases.json",
      import.meta.url,
    ),
    "utf8",
  ),
) as { cases: Case[] };

function expand(segments: Segment[]): number[] {
  const values: number[] = [];
  for (const segment of segments) {
    if (segment.values) {
      values.push(...segment.values);
      continue;
    }
    const count = segment.count ?? 0;
    if (segment.value !== undefined) {
      for (let index = 0; index < count; index += 1) {
        values.push(segment.value);
      }
      continue;
    }
    const from = segment.from ?? 0;
    const to = segment.to ?? from;
    for (let index = 0; index < count; index += 1) {
      values.push(from + ((to - from) * index) / (count - 1));
    }
  }
  return values;
}

function verdict(anomaly: LossAnomaly | null): ChartVerdict | null {
  if (!anomaly) {
    return null;
  }
  return {
    kind: anomaly.kind,
    severity: anomaly.severity,
    step: anomaly.step,
  };
}

for (const case_ of fixture.cases) {
  test(`loss-anomaly window chart: ${case_.name}`, () => {
    const points = expand(case_.segments).map((value, index) => ({
      step: index + 1,
      value,
    }));
    assert.deepEqual(
      verdict(detectLossAnomaly(points)),
      case_.chart,
      `${case_.name}: ${case_.why}. The chart callout and the webhook alert table disagree.`,
    );
  });
}
