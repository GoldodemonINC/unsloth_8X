// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  BUILTIN_PRESETS,
  parseStoredPresets,
  sanitizePresetValues,
} = await import("../src/features/training/lib/training-preset-data.ts");

test("every built-in preset has a complete, valid value set", () => {
  assert.ok(BUILTIN_PRESETS.length >= 5);
  const ids = new Set<string>();
  for (const preset of BUILTIN_PRESETS) {
    assert.equal(ids.has(preset.id), false, `duplicate preset id ${preset.id}`);
    ids.add(preset.id);
    assert.ok(preset.name.length > 0);
    assert.deepEqual(sanitizePresetValues(preset.values), preset.values);
    assert.ok(preset.values.loraRank >= 1);
    assert.ok(preset.values.learningRate > 0);
  }
});

test("sanitizePresetValues keeps only known training fields", () => {
  const values = sanitizePresetValues({
    trainingMethod: "qlora",
    loraRank: 16,
    loraAlpha: 32,
    loraDropout: 0,
    learningRate: 0.0002,
    epochs: 1,
    batchSize: 2,
    gradientAccumulation: 4,
    contextLength: 2048,
    gradientCheckpointing: "unsloth",
    packing: true,
    trainOnCompletions: false,
    dataset: "org/should-not-be-applied",
  });
  assert.ok(values);
  assert.equal(
    (values as unknown as Record<string, unknown>).dataset,
    undefined,
  );
  assert.equal(values?.batchSize, 2);
});

test("an incomplete or malformed preset is rejected instead of half-applied", () => {
  assert.equal(sanitizePresetValues(null), null);
  assert.equal(sanitizePresetValues({ trainingMethod: "qlora" }), null);
  assert.equal(
    sanitizePresetValues({
      trainingMethod: "not-a-method",
      loraRank: 16,
      loraAlpha: 32,
      loraDropout: 0,
      learningRate: 0.0002,
      epochs: 1,
      batchSize: 1,
      gradientAccumulation: 8,
      contextLength: 2048,
      gradientCheckpointing: "unsloth",
      packing: false,
      trainOnCompletions: true,
    }),
    null,
  );
});

test("parseStoredPresets drops broken rows and keeps the valid ones", () => {
  const now = new Date().toISOString();
  const raw = JSON.stringify([
    {
      id: "a",
      name: "Mine",
      description: "",
      createdAt: now,
      values: BUILTIN_PRESETS[0]?.values,
    },
    { id: "b", name: "Missing values", values: null },
    "not an object",
  ]);
  const parsed = parseStoredPresets(raw);
  assert.equal(parsed.length, 1);
  assert.equal(parsed[0]?.name, "Mine");
  assert.deepEqual(parsed[0]?.values, BUILTIN_PRESETS[0]?.values);

  assert.deepEqual(parseStoredPresets("not json"), []);
  assert.deepEqual(parseStoredPresets(null), []);
});
