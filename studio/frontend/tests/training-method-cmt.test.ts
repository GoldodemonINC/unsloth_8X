// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { isAdapterMethod, isTrainingMethod } = await import(
  "../src/types/training.ts"
);
const {
  getTrainingMethodLabel,
  isTrainingMethodSupportedOnDevice,
  toBackendTrainingType,
} = await import("../src/features/training/lib/training-methods.ts");
const { TRAINING_METHOD_META, visibleTrainingMethods } = await import(
  "../src/features/training/lib/training-method-meta.ts"
);

test("CMT is a known training method with its own picker row", () => {
  assert.equal(isTrainingMethod("cmt"), true);
  assert.equal(getTrainingMethodLabel("cmt"), "CMT");
  assert.equal(TRAINING_METHOD_META.cmt.labelKey, "labs.cmt.methodLabel");
});

test("CMT appears in the picker only while the Labs experiment is on", () => {
  const off = visibleTrainingMethods({
    cmtEnabled: false,
    isDecision: false,
    decisionTakesQlora: false,
  });
  assert.equal(off.includes("cmt"), false);

  const on = visibleTrainingMethods({
    cmtEnabled: true,
    isDecision: false,
    decisionTakesQlora: false,
  });
  assert.equal(on.includes("cmt"), true);
});

test("a decision model never offers CMT", () => {
  for (const decisionTakesQlora of [true, false]) {
    const rows = visibleTrainingMethods({
      cmtEnabled: true,
      isDecision: true,
      decisionTakesQlora,
    });
    assert.equal(rows.includes("cmt"), false);
  }
});

test("CMT needs an accelerator and is never the adapter path", () => {
  assert.equal(isTrainingMethodSupportedOnDevice("cmt", "cuda"), true);
  assert.equal(
    isTrainingMethodSupportedOnDevice("cmt", "cuda", true),
    false,
  );
  assert.equal(isTrainingMethodSupportedOnDevice("cmt", "mac"), false);
  assert.equal(isTrainingMethodSupportedOnDevice("cmt", "cpu"), false);
  assert.equal(isAdapterMethod("cmt"), false);
});

test("CMT is refused before a training start is built", () => {
  assert.throws(
    () => toBackendTrainingType("cmt"),
    /its own engine/,
  );
  // The trainer still builds for every method it owns.
  assert.equal(toBackendTrainingType("cpt"), "Continued Pretraining");
});
