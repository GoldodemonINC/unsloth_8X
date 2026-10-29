// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  type TrainingQueueItem,
  nextReadyQueueItem,
  isScheduledQueueItem,
  parseSweepNumbers,
  sanitizeQueueConfig,
  sweepCombinationCount,
} from "../src/features/training/lib/training-queue-plan.ts";
import type { TrainingStartRequest } from "../src/features/training/types/api.ts";

function item(
  id: string,
  overrides: Partial<TrainingQueueItem> = {},
): TrainingQueueItem {
  return {
    id,
    label: id,
    groupId: null,
    config: {} as TrainingStartRequest,
    notBefore: null,
    status: "queued",
    error: null,
    createdAt: "2026-01-01T00:00:00.000Z",
    startedAt: null,
    finishedAt: null,
    ...overrides,
  };
}

test("nextReadyQueueItem skips running and failed entries", () => {
  const items = [
    item("done", { status: "done" }),
    item("failed", { status: "failed" }),
    item("next"),
  ];
  assert.equal(nextReadyQueueItem(items, Date.now())?.id, "next");
});

test("nextReadyQueueItem respects a future start time", () => {
  const now = Date.parse("2026-01-01T02:00:00.000Z");
  const items = [
    item("later", { notBefore: "2026-01-01T03:00:00.000Z" }),
    item("ready", { notBefore: "2026-01-01T01:00:00.000Z" }),
  ];
  assert.equal(nextReadyQueueItem(items, now)?.id, "ready");
  assert.equal(isScheduledQueueItem(items[0], now), true);
});

test("a null or unparseable start time is ready", () => {
  const now = Date.now();
  assert.equal(nextReadyQueueItem([item("plain")], now)?.id, "plain");
  assert.equal(
    nextReadyQueueItem([item("broken", { notBefore: "not-a-date" })], now)?.id,
    "broken",
  );
});

test("sweep sizes multiply and an empty axis means one value", () => {
  assert.equal(
    sweepCombinationCount({
      learningRates: [1e-4, 2e-4],
      loraRanks: [16, 32],
      epochs: [3],
      batchSizes: [],
    }),
    4,
  );
});

test("parseSweepNumbers drops duplicates and non-positive values", () => {
  assert.deepEqual(parseSweepNumbers("2e-4, 1e-4 2e-4, 0, -1, x"), [2e-4, 1e-4]);
});

test("sanitizeQueueConfig never stores Hub or W&B credentials", () => {
  const config = sanitizeQueueConfig({
    hf_token: "hf_secret",
    wandb_token: "wandb_secret",
    model_name: "org/model",
  } as TrainingStartRequest);
  assert.equal(config.hf_token, null);
  assert.equal(config.wandb_token, null);
  assert.equal(config.model_name, "org/model");
});
