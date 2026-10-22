// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { resolveStartTrainingButtonLabelKey } from "../src/features/studio/wizard/start-training-cta-state.ts";

const readyState = {
  stopRequested: false,
  startBlocked: false,
  isLoadingModel: false,
  isCheckingDataset: false,
  hasModel: true,
  hasDataset: true,
};

test("the stop label takes priority over every other start state", () => {
  assert.equal(
    resolveStartTrainingButtonLabelKey({
      ...readyState,
      stopRequested: true,
      startBlocked: true,
      isLoadingModel: true,
      isCheckingDataset: true,
    }),
    "studio.training.stopping",
  );
});

test("start button labels cover pending work and incomplete selections", () => {
  const cases = [
    [{ ...readyState, startBlocked: true }, "studio.training.starting"],
    [{ ...readyState, isLoadingModel: true }, "studio.training.loadingModel"],
    [
      { ...readyState, isCheckingDataset: true },
      "studio.training.checkingDataset",
    ],
    [
      { ...readyState, hasModel: false, hasDataset: false },
      "studio.training.chooseModelAndDataset",
    ],
    [{ ...readyState, hasModel: false }, "studio.training.chooseModel"],
    [{ ...readyState, hasDataset: false }, "studio.training.chooseDataset"],
    [readyState, "studio.training.startTraining"],
  ] as const;

  for (const [state, expected] of cases) {
    assert.equal(resolveStartTrainingButtonLabelKey(state), expected);
  }
});

test("a simulated run needs no model or dataset to offer its own start label", () => {
  assert.equal(
    resolveStartTrainingButtonLabelKey({
      ...readyState,
      hasModel: false,
      hasDataset: false,
      isCheckingDataset: true,
      simulated: true,
    }),
    "studio.training.startSimulated",
  );
  // Pending work still wins: the fake run is being started or stopped already.
  assert.equal(
    resolveStartTrainingButtonLabelKey({
      ...readyState,
      simulated: true,
      startBlocked: true,
    }),
    "studio.training.starting",
  );
  assert.equal(
    resolveStartTrainingButtonLabelKey({
      ...readyState,
      simulated: true,
      stopRequested: true,
    }),
    "studio.training.stopping",
  );
});
