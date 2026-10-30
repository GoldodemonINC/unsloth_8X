// SPDX-License-Identifier: NRC-3
// Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-3.md

import assert from "node:assert/strict";
import test from "node:test";
import {
  type RetentionEntry,
  type RetentionRun,
  normalizeRetentionPath,
  planRetention,
} from "../src/features/labs/lib/retention-policy.ts";

function entry(path: string, deletable = true): RetentionEntry {
  return { path, bytes: 100, deletable };
}

function run(
  id: string,
  output_dir: string,
  project_name: string | null,
  final_loss: number | null,
): RetentionRun {
  return {
    id,
    output_dir,
    project_name,
    final_loss,
    final_step: 100,
    started_at: "2026-01-01T00:00:00Z",
  };
}

test("keeps the lowest loss per project and lists the rest", () => {
  const entries = [
    entry("/out/best"),
    entry("/out/worse"),
    entry("/out/other"),
  ];
  const runs = [
    run("a", "/out/best", "alpha", 0.4),
    run("b", "/out/worse", "alpha", 0.9),
    run("c", "/out/other", "beta", 1.2),
  ];
  const plan = planRetention(entries, runs, 1);
  assert.deepEqual(
    plan.candidates.map((item) => item.path),
    ["/out/worse"],
  );
  assert.equal(plan.keptRuns, 2);
});

test("a null loss ranks after every finite loss", () => {
  const entries = [entry("/out/known"), entry("/out/unknown")];
  const runs = [
    run("a", "/out/unknown", "alpha", null),
    run("b", "/out/known", "alpha", 2),
  ];
  const plan = planRetention(entries, runs, 1);
  assert.deepEqual(
    plan.candidates.map((item) => item.path),
    ["/out/unknown"],
  );
});

test("protected entries are never candidates", () => {
  const entries = [entry("/out/keep"), entry("/out/protected", false)];
  const runs = [
    run("a", "/out/keep", "alpha", 0.2),
    run("b", "/out/protected", "alpha", 0.9),
  ];
  const plan = planRetention(entries, runs, 1);
  assert.deepEqual(plan.candidates, []);
});

test("shared output paths survive when a kept run uses them", () => {
  const entries = [entry("/out/shared"), entry("/out/old")];
  const runs = [
    run("kept", "/out/shared/", "alpha", 0.1),
    run("child", "/out/shared", "alpha", 0.5),
    run("old", "/out/old", "alpha", 0.9),
  ];
  const plan = planRetention(entries, runs, 2);
  assert.deepEqual(
    plan.candidates.map((item) => item.path),
    ["/out/old"],
  );
});

test("normalizeRetentionPath ignores separators and trailing slashes", () => {
  assert.equal(normalizeRetentionPath("C:\\out\\run\\"), "C:/out/run");
});
