// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { detectLossAnomaly } from "../src/features/training/lib/loss-anomaly.ts";

const series = (values: number[]) =>
  values.map((value, index) => ({ step: index + 1, value }));

test("a healthy decreasing curve raises nothing", () => {
  const values = Array.from({ length: 60 }, (_, index) => 2.5 - index * 0.02);
  assert.equal(detectLossAnomaly(series(values)), null);
});

test("a converged flat curve is not a plateau", () => {
  const values = Array.from({ length: 60 }, () => 0.03);
  assert.equal(detectLossAnomaly(series(values)), null);
});

test("a one-step jump above the recent median is a spike", () => {
  const values = [...Array.from({ length: 20 }, () => 2), 6];
  const anomaly = detectLossAnomaly(series(values));
  assert.equal(anomaly?.kind, "spike");
  assert.equal(anomaly?.step, 21);
  assert.equal(anomaly?.severity, "warning");
});

test("a curve that turns upward in its last third is divergence", () => {
  const fall = Array.from({ length: 30 }, (_, index) => 2.5 - index * 0.05);
  const rise = Array.from({ length: 30 }, (_, index) => 1 + index * 0.1);
  const anomaly = detectLossAnomaly(series([...fall, ...rise]));
  assert.equal(anomaly?.kind, "divergence");
  assert.equal(anomaly?.severity, "critical");
});

test("a flat, still-high curve is a plateau", () => {
  const values = Array.from({ length: 60 }, () => 0.8);
  const anomaly = detectLossAnomaly(series(values));
  assert.equal(anomaly?.kind, "plateau");
  assert.equal(anomaly?.step, 60);
});

test("a non-finite loss is critical and points at its step", () => {
  const values = [...Array.from({ length: 20 }, () => 1), Number.NaN];
  const anomaly = detectLossAnomaly(series(values));
  assert.equal(anomaly?.kind, "nan");
  assert.equal(anomaly?.severity, "critical");
  assert.equal(anomaly?.step, 21);
});

test("fewer than eight points is not enough to judge", () => {
  assert.equal(detectLossAnomaly(series([2, 1, 3, 5, 2, 1, 4])), null);
});
