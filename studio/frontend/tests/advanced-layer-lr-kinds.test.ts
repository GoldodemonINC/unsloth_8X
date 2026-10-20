// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  DEFAULT_ADVANCED_TRAINING,
  DEFAULT_LAYER_KIND_MULTIPLIERS,
  advancedStateFromRunConfig,
  buildAdvancedRequest,
} from "../src/features/advanced/types.ts";

const OPTIONS = { batchSize: 4, isPolicyAlignment: false };

test("a zone-only layer LR run posts no kind multipliers", () => {
  const request = buildAdvancedRequest(
    { ...DEFAULT_ADVANCED_TRAINING, layerLrEnabled: true },
    OPTIONS,
  );
  const layerLr: Partial<NonNullable<typeof request>["layer_lr"]> =
    request?.layer_lr ?? {};
  assert.equal(layerLr.enabled, true);
  assert.equal(
    "kind_multipliers" in layerLr,
    false,
    "neutral kinds must not change the payload a zone-only run already sent",
  );
});

test("only non-neutral module groups are posted", () => {
  const request = buildAdvancedRequest(
    {
      ...DEFAULT_ADVANCED_TRAINING,
      layerLrEnabled: true,
      layerLrKinds: {
        ...DEFAULT_LAYER_KIND_MULTIPLIERS,
        attention: 0.5,
        mlp: 1.5,
      },
    },
    OPTIONS,
  );
  assert.deepEqual(request?.layer_lr?.kind_multipliers, {
    attention: 0.5,
    mlp: 1.5,
  });
});

test("a state persisted before module groups sends no kind multipliers", () => {
  const stale = { ...DEFAULT_ADVANCED_TRAINING, layerLrEnabled: true } as Record<
    string,
    unknown
  >;
  delete stale.layerLrKinds;
  const request = buildAdvancedRequest(stale as never, OPTIONS);
  assert.equal(
    "kind_multipliers" in (request?.layer_lr ?? {}),
    false,
    "missing kind state must behave like a neutral table, not crash the payload",
  );
});

test("restoring a run brings back the module groups and resets the rest", () => {
  const restored = advancedStateFromRunConfig(
    {
      layer_lr: {
        enabled: true,
        zone_multipliers: { early: 0.1 },
        kind_multipliers: { attention: 0.25, norm: 0 },
        depth_curve: [0.5, 1],
      },
    },
    {
      ...DEFAULT_ADVANCED_TRAINING,
      layerLrKinds: { ...DEFAULT_LAYER_KIND_MULTIPLIERS, moe_expert: 3 },
    },
  );
  assert.equal(restored.layerLrEnabled, true);
  assert.equal(restored.layerLrZones.early, 0.1);
  assert.equal(restored.layerLrKinds.attention, 0.25);
  assert.equal(restored.layerLrKinds.norm, 0);
  // A kind absent from the run config falls back to neutral, not to stale state.
  assert.equal(restored.layerLrKinds.moe_expert, 1);
  assert.equal(restored.layerLrDepthCurveEnabled, true);
});

test("restoring a zone-only run leaves every module group neutral", () => {
  const restored = advancedStateFromRunConfig(
    { layer_lr: { enabled: true, zone_multipliers: { late: 0.5 } } },
    DEFAULT_ADVANCED_TRAINING,
  );
  assert.deepEqual(restored.layerLrKinds, DEFAULT_LAYER_KIND_MULTIPLIERS);
});
