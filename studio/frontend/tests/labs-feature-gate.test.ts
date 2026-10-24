// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { labsFeatureOn } from "../src/features/labs/lib/feature-gate.ts";

const artifacts = { id: "artifacts", enabled: true };
const pipelinesOff = { id: "pipelines", enabled: false };

test("the master switch alone decides nothing", () => {
  assert.equal(
    labsFeatureOn({ enabled: false, features: [artifacts] }, "artifacts"),
    false,
  );
});

test("with Labs on, an experiment that is on opens", () => {
  assert.equal(
    labsFeatureOn({ enabled: true, features: [artifacts] }, "artifacts"),
    true,
  );
});

test("a switched-off experiment stays closed with Labs on", () => {
  assert.equal(
    labsFeatureOn(
      { enabled: true, features: [artifacts, pipelinesOff] },
      "pipelines",
    ),
    false,
  );
});

test("a feature the server does not list is closed, not open by omission", () => {
  assert.equal(
    labsFeatureOn({ enabled: true, features: [artifacts] }, "manifest-viewer"),
    false,
  );
});
