import assert from "node:assert/strict";
import { test } from "node:test";
import {
  checkpointStepFromPath,
  checkpointStepsForOutputDir,
} from "../src/features/studio/sections/charts/run-checkpoint-steps.ts";

function inventory(paths: string[]) {
  return [
    {
      name: "run",
      checkpoints: paths.map((path) => ({ display_name: path, path })),
    },
  ];
}

test("a checkpoint step is read from the final path segment", () => {
  assert.equal(checkpointStepFromPath("/outputs/run/checkpoint-120"), 120);
  assert.equal(checkpointStepFromPath("C:\\outputs\\run\\checkpoint-40\\"), 40);
  assert.equal(checkpointStepFromPath("/outputs/run"), null);
  assert.equal(checkpointStepFromPath("/outputs/run/checkpoint-final"), null);
});

test("only checkpoints inside the run directory are kept, sorted", () => {
  const models = inventory([
    "/outputs/run/checkpoint-60",
    "/outputs/run/checkpoint-20",
    "/outputs/run/checkpoint-60",
    "/outputs/other-run/checkpoint-999",
    "/outputs/run/nested/checkpoint-5",
  ]);
  assert.deepEqual(checkpointStepsForOutputDir(models, "/outputs/run"), [20, 60]);
});

test("windows paths and trailing separators still match", () => {
  const models = inventory(["C:\\outputs\\run\\checkpoint-30"]);
  assert.deepEqual(checkpointStepsForOutputDir(models, "C:\\outputs\\run\\"), [30]);
});

test("a missing or foreign output directory yields no markers", () => {
  const models = inventory(["/outputs/run/checkpoint-10"]);
  assert.deepEqual(checkpointStepsForOutputDir(models, null), []);
  assert.deepEqual(checkpointStepsForOutputDir(models, ""), []);
  assert.deepEqual(checkpointStepsForOutputDir(models, "/outputs/renamed"), []);
});
