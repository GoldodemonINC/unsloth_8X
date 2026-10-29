import assert from "node:assert/strict";
import { test } from "node:test";
import {
  buildLossChartEvents,
  isLrChange,
  resumedAtStep,
} from "../src/features/studio/sections/charts/loss-chart-events.ts";

function series(points: Array<[number, number]>) {
  return points.map(([step, value]) => ({ step, value }));
}

test("checkpoint markers are sorted, deduplicated and capped", () => {
  const events = buildLossChartEvents({
    checkpointSteps: [30, 10, 10, 20],
  });
  assert.deepEqual(
    events.map((event) => [event.kind, event.step]),
    [
      ["checkpoint", 10],
      ["checkpoint", 20],
      ["checkpoint", 30],
    ],
  );

  const many = buildLossChartEvents({
    checkpointSteps: Array.from({ length: 40 }, (_, index) => index + 1),
  });
  assert.equal(many.length, 12);
  assert.equal(many[0].step, 1);
  assert.equal(many[many.length - 1].step, 40);
});

test("a discrete learning-rate change is marked once", () => {
  assert.equal(isLrChange(0.0001, 0.0001), false);
  assert.equal(isLrChange(0.0001, 0.00009), false);
  assert.equal(isLrChange(0.0001, 0.0003), true);

  const events = buildLossChartEvents({
    lrHistory: series([
      [1, 0.0001],
      [2, 0.0001],
      [3, 0.0003],
      [4, 0.0003],
    ]),
  });
  assert.deepEqual(
    events.map((event) => [event.kind, event.step]),
    [["lrChange", 3]],
  );
});

test("resume is only marked with recorded lineage and a resumed first step", () => {
  assert.equal(
    resumedAtStep({ resume_from_checkpoint: "/outputs/run" }, series([[400, 1.2]])),
    400,
  );
  assert.equal(
    resumedAtStep({ resume_from_checkpoint: "/outputs/run" }, series([[1, 1.2]])),
    null,
  );
  assert.equal(resumedAtStep({}, series([[400, 1.2]])), null);
  assert.equal(resumedAtStep(null, series([[400, 1.2]])), null);
});

test("anomaly and resume share the axis without duplicating a marker", () => {
  const events = buildLossChartEvents({
    anomalyStep: 120,
    resumeStep: 120,
  });
  assert.deepEqual(
    events.map((event) => event.kind),
    ["resume", "anomaly"],
  );
});
