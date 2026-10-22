// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  MAX_MERGE_MODELS,
  MAX_MERGE_MODELS_CPU,
  appendWeight,
  blendShares,
  maxMergeModels,
  parentLabel,
  removeWeight,
} from "../src/features/advanced/merge-slots.ts";

test("two parents keep the A/B names, more are numbered", () => {
  assert.equal(parentLabel(0, 2), "Model A");
  assert.equal(parentLabel(1, 2), "Model B");
  assert.equal(parentLabel(0, 3), "Model 1");
  assert.equal(parentLabel(5, MAX_MERGE_MODELS), "Model 6");
});

test("the sixth parent is allowed and a seventh is not a slot at all", () => {
  let weights = [0.5, 0.5];
  for (let count = 2; count < MAX_MERGE_MODELS; count += 1) {
    weights = appendWeight(weights);
  }
  assert.equal(weights.length, MAX_MERGE_MODELS);
  // Every parent keeps an equal say, so the newcomer does not outvote the rest.
  for (const value of weights) assert.ok(Math.abs(value - 0.5) < 1e-12);
  assert.deepEqual(
    blendShares(weights, MAX_MERGE_MODELS).map((share) =>
      Math.round(share * 100),
    ),
    [17, 17, 17, 17, 17, 17],
  );
});

test("a CPU host merges two or three parents; a GPU host up to six", () => {
  assert.equal(MAX_MERGE_MODELS_CPU, 3);
  assert.equal(maxMergeModels(false), MAX_MERGE_MODELS_CPU);
  assert.equal(maxMergeModels(true), MAX_MERGE_MODELS);
  // Until capabilities answer, keep the wider window: a failed probe must not
  // silently narrow what a GPU host can merge.
  assert.equal(maxMergeModels(null), MAX_MERGE_MODELS);
  assert.equal(maxMergeModels(undefined), MAX_MERGE_MODELS);
});

test("the plan's RAM-derived cap wins over the flat CPU tier", () => {
  // The backend measures the actual checkpoints, so a CPU host with room for
  // five small parents must be allowed five rather than trimmed back to three.
  assert.equal(maxMergeModels(false, 2), 2);
  assert.equal(maxMergeModels(false, 5), 5);
  assert.equal(maxMergeModels(false, 9), MAX_MERGE_MODELS);
  assert.equal(maxMergeModels(true, 4), 4);
  // No usable plan yet: fall back to the capability tier.
  assert.equal(maxMergeModels(false, null), MAX_MERGE_MODELS_CPU);
  assert.equal(maxMergeModels(false, 1), MAX_MERGE_MODELS_CPU);
  assert.equal(maxMergeModels(false, Number.NaN), MAX_MERGE_MODELS_CPU);
});

test("adding a parent to a weighted blend keeps the existing ratio", () => {
  // Two parents at 1:3 add a third at their average (2), i.e. 1:3:2.
  const next = appendWeight([1, 3]);
  assert.deepEqual(next, [1, 3, 2]);
  assert.deepEqual(blendShares(next, 3), [1 / 6, 3 / 6, 2 / 6]);
});

test("removing a parent drops its weight and leaves the rest in order", () => {
  assert.deepEqual(removeWeight([1, 2, 3], 1), [1, 3]);
  assert.deepEqual(removeWeight([1, 2, 3], 0), [2, 3]);
});

test("weights that no longer line up with the slots still produce shares", () => {
  // A weight list short by one must not yield NaN% in the slot labels.
  const shares = blendShares([1], 3);
  assert.deepEqual(shares, [1, 0, 0]);
  assert.deepEqual(blendShares([0, 0, 0], 3), [1 / 3, 1 / 3, 1 / 3]);
  assert.deepEqual(blendShares([Number.NaN, 2], 2), [0, 1]);
});
