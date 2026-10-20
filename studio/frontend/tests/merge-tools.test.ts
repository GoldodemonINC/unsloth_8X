// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  BUILT_IN_MERGE_PRESETS,
  MERGE_METHOD_NOTES,
  applyPreset,
  applyWeightTool,
  budgetVerdict,
  comparisonRows,
  etaSeconds,
  fitWeights,
  folderCollisionNotice,
  methodUsesDensity,
  parseRecipe,
  progressTimeline,
  reviewItems,
  serializeRecipe,
  structuralChecks,
  throughputBytesPerSecond,
  type MergeSetup,
} from "../src/features/advanced/lib/merge-tools.ts";
import {
  MERGE_PRESET_STORAGE_KEY,
  loadSavedPresets,
  removeSavedPreset,
  upsertSavedPreset,
  writeSavedPresets,
} from "../src/features/advanced/lib/merge-preset-store.ts";

function setup(overrides: Partial<MergeSetup> = {}): MergeSetup {
  return {
    models: ["org/a", "org/b", "org/c"],
    weights: [1, 1, 1],
    method: "linear",
    density: 0.5,
    mode: "standard",
    topK: 2,
    temperature: 1,
    outputDir: "",
    ...overrides,
  };
}

function memoryStorage(initial: Record<string, string> = {}) {
  const data = new Map(Object.entries(initial));
  return {
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => {
      data.set(key, value);
    },
    dump: () => Object.fromEntries(data),
  };
}

test("presets change the engine and leave the parents alone", () => {
  const ties = BUILT_IN_MERGE_PRESETS.find(
    (preset) => preset.id === "ties-half",
  );
  assert.ok(ties);
  const next = applyPreset(setup({ weights: [1, 2, 3] }), ties);
  assert.deepEqual(next.models, ["org/a", "org/b", "org/c"]);
  assert.equal(next.method, "ties");
  assert.equal(next.density, 0.5);
  // Every preset sets its own weights policy; ties-half evens them.
  assert.deepEqual(next.weights, [1, 1, 1]);
});

test("the lean-on-first preset doubles the first parent only", () => {
  const focus = BUILT_IN_MERGE_PRESETS.find(
    (preset) => preset.id === "focus-first",
  );
  assert.ok(focus);
  assert.deepEqual(applyPreset(setup(), focus).weights, [2, 1, 1]);
});

test("the MMoE preset never touches weights, since they do not apply there", () => {
  const mmoe = BUILT_IN_MERGE_PRESETS.find(
    (preset) => preset.id === "mmoe-top2",
  );
  assert.ok(mmoe);
  const next = applyPreset(setup({ weights: [0.7, 0.2, 0.1] }), mmoe);
  assert.equal(next.mode, "mmoe");
  assert.deepEqual(next.weights, [0.7, 0.2, 0.1]);
});

test("density is only offered for the methods that read it", () => {
  assert.equal(methodUsesDensity("ties"), true);
  assert.equal(methodUsesDensity("dare"), true);
  assert.equal(methodUsesDensity("linear"), false);
  assert.equal(methodUsesDensity("slerp"), false);
  assert.ok(MERGE_METHOD_NOTES.slerp.summary.length > 20);
});

test("weight tools keep the list lined up with the slots", () => {
  assert.deepEqual(applyWeightTool([0.2, 0.8], "equal"), [1, 1]);
  assert.deepEqual(applyWeightTool([2, 6], "normalize"), [0.25, 0.75]);
  // Inversion is around the average, so an even list does not collapse to zero.
  assert.deepEqual(applyWeightTool([0.2, 0.8], "invert"), [0.8, 0.2]);
  assert.deepEqual(applyWeightTool([1, 1], "invert"), [1, 1]);
  assert.deepEqual(applyWeightTool([], "equal"), []);
  // A zero total has no shares to normalise, so the even split stands in.
  assert.deepEqual(applyWeightTool([0, 0], "normalize"), [1, 1]);
  for (const tool of ["equal", "normalize", "invert"] as const) {
    assert.equal(applyWeightTool([1, 2, 3], tool).length, 3);
  }
});

test("fitWeights pads a short list and drops a stale tail", () => {
  assert.deepEqual(fitWeights([2], 3), [2, 1, 1]);
  assert.deepEqual(fitWeights([2, 3, 4], 2), [2, 3]);
  assert.deepEqual(fitWeights([-1, Number.NaN], 2), [1, 1]);
});

test("a recipe round-trips and refuses what it cannot read", () => {
  const text = serializeRecipe(
    setup({ weights: [2, 1, 1], density: 0.35 }),
    "my blend",
  );
  const parsed = parseRecipe(text);
  assert.equal(parsed.ok, true);
  if (!parsed.ok) return;
  assert.equal(parsed.recipe.name, "my blend");
  assert.deepEqual(parsed.recipe.setup.models, ["org/a", "org/b", "org/c"]);
  assert.equal(parsed.recipe.setup.density, 0.35);
  assert.deepEqual(parsed.recipe.setup.weights, [2, 1, 1]);
});

test("a recipe with one parent, an unknown method or bad JSON is refused", () => {
  assert.equal(parseRecipe("not json").ok, false);
  const oneParent = parseRecipe(
    JSON.stringify({
      version: 1,
      setup: { models: ["org/a"], method: "linear" },
    }),
  );
  assert.equal(oneParent.ok, false);
  if (!oneParent.ok) assert.match(oneParent.error, /two parents/);
  const badMethod = parseRecipe(
    JSON.stringify({
      version: 1,
      setup: { models: ["a/b", "c/d"], method: "voodoo" },
    }),
  );
  assert.equal(badMethod.ok, false);
  if (!badMethod.ok) assert.match(badMethod.error, /merge method/);
  const future = parseRecipe(
    JSON.stringify({
      version: 99,
      setup: { models: ["a/b", "c/d"], method: "linear" },
    }),
  );
  assert.equal(future.ok, false);
  if (!future.ok) assert.match(future.error, /newer/);
});

test("a recipe without weights gets an even split rather than zeros", () => {
  const parsed = parseRecipe(
    JSON.stringify({ setup: { models: ["a/b", "c/d"], method: "slerp" } }),
  );
  assert.equal(parsed.ok, true);
  if (!parsed.ok) return;
  assert.deepEqual(parsed.recipe.setup.weights, [1, 1]);
  assert.equal(parsed.recipe.setup.mode, "standard");
});

test("saved presets round-trip through storage and replace by name", () => {
  const storage = memoryStorage();
  const first = upsertSavedPreset([], "My blend", setup(), 1000);
  assert.equal(writeSavedPresets(storage, first), true);
  assert.deepEqual(
    loadSavedPresets(storage).map((preset) => preset.name),
    ["My blend"],
  );
  const replaced = upsertSavedPreset(
    first,
    "my blend",
    setup({ method: "ties" }),
    2000,
  );
  assert.equal(replaced.length, 1);
  assert.equal(replaced[0]?.setup.method, "ties");
  assert.deepEqual(removeSavedPreset(replaced, replaced[0]!.id), []);
});

test("a corrupt or hostile preset entry is dropped, not rendered", () => {
  const storage = memoryStorage({
    [MERGE_PRESET_STORAGE_KEY]: JSON.stringify([
      { id: "a", name: "broken", savedAt: 1, setup: { models: ["only/one"] } },
      { id: "b", name: "fine", savedAt: 2, setup: setup() },
      "not an object",
    ]),
  });
  const loaded = loadSavedPresets(storage);
  assert.deepEqual(
    loaded.map((preset) => preset.name),
    ["fine"],
  );
  assert.deepEqual(loadSavedPresets(null), []);
  assert.equal(writeSavedPresets(null, []), false);
});

test("a plan's parents compare against the first by shape", () => {
  const rows = comparisonRows({
    models: [
      {
        source: "a",
        architecture: "LlamaForCausalLM",
        hidden_size: 2048,
        num_layers: 16,
        num_heads: 8,
        intermediate_size: 8192,
        vocab_size: 32000,
        parameter_count: 1_100_000_000,
      },
      {
        source: "b",
        architecture: "LlamaForCausalLM",
        hidden_size: 2048,
        num_layers: 16,
        num_heads: 8,
        intermediate_size: 8192,
        vocab_size: 32000,
        parameter_count: 1_100_000_000,
      },
    ],
    alignment: {
      mode: "identical",
      hidden_strategy: "keep",
      layer_strategy: "keep",
      vocab_strategy: "keep",
      warnings: [],
      mmoe_required: false,
    },
  });
  assert.deepEqual(
    rows.map((row) => row.label),
    ["Model A", "Model B"],
  );
  assert.equal(rows[1]?.sameShape, true);
});

test("the budget verdict flags a shortfall only for what the machine reported", () => {
  const plan = {
    models: [],
    alignment: {
      mode: "identical",
      hidden_strategy: "",
      layer_strategy: "",
      vocab_strategy: "",
      warnings: [],
      mmoe_required: false,
    },
    estimates: {
      need: { vram_bytes: 0, ram_bytes: 40e9, disk_bytes: 30e9 },
      notes: [],
      streaming: true,
      model_bytes: 30e9,
    },
  };
  assert.deepEqual(
    budgetVerdict(plan, { ramBytes: 0, diskBytes: 0 }).shortfalls,
    [],
  );
  assert.deepEqual(
    budgetVerdict(plan, { ramBytes: 16e9, diskBytes: 100e9 }).shortfalls,
    ["ram"],
  );
  assert.deepEqual(
    budgetVerdict(plan, { ramBytes: 64e9, diskBytes: 10e9 }).shortfalls,
    ["disk"],
  );
  assert.equal(
    budgetVerdict(
      {
        models: [],
        alignment: {
          mode: "identical",
          hidden_strategy: "",
          layer_strategy: "",
          vocab_strategy: "",
          warnings: [],
          mmoe_required: false,
        },
      },
      { ramBytes: 1, diskBytes: 1 },
    ).fits,
    null,
  );
});

test("review items put blockers first and always say something", () => {
  const blocked = reviewItems({
    setup: { models: ["a/b", "c/d"], weights: [1, 1], mode: "standard" },
    plan: null,
    validation: {
      blocked: true,
      blocked_reason: "Model B ships pickle weights.",
      sources: [{ label: "Model B", pickles: ["pytorch_model.bin"] }],
      tokenizer: { status: "conflict", reason: "Different <|endoftext|> id." },
    },
  });
  assert.equal(blocked[0]?.level, "block");
  assert.ok(blocked.some((item) => item.text.includes("pickle")));
  const clear = reviewItems({
    setup: { models: ["a/b", "c/d"], weights: [1, 1], mode: "standard" },
    plan: null,
    validation: null,
  });
  assert.equal(clear.length, 1);
  assert.equal(clear[0]?.level, "info");
});

test("zero weights and a lone parent are blocking", () => {
  const items = reviewItems({
    setup: { models: ["a/b"], weights: [0, 0], mode: "standard" },
    plan: null,
    validation: null,
  });
  assert.deepEqual(
    items.filter((item) => item.level === "block").map((item) => item.id),
    ["parents", "weights"],
  );
  // In MMoE mode weights are not read, so all-zero is fine.
  const mmoe = reviewItems({
    setup: { models: ["a/b", "b/c"], weights: [0, 0], mode: "mmoe" },
    plan: null,
    validation: null,
  });
  assert.equal(
    mmoe.some((item) => item.id === "weights"),
    false,
  );
});

test("the timeline marks the stage that failed and nothing after it", () => {
  const job = {
    status: "failed" as const,
    progress: { stage: "merging", done: 2, total: 10, blended_tensors: 40 },
  };
  const steps = progressTimeline(job);
  assert.deepEqual(
    steps.map((step) => step.state),
    ["done", "done", "done", "failed", "pending", "pending"],
  );
  assert.match(steps[3]!.detail, /2 of 10 shards/);
  assert.equal(progressTimeline(null).length, 0);
  const finished = progressTimeline({
    status: "completed",
    progress: { stage: "completed" },
  });
  assert.ok(finished.every((step) => step.state === "done"));
});

test("throughput and ETA read the job's own counters", () => {
  const job = {
    status: "running" as const,
    created_at: 1000,
    progress: { stage: "merging", done: 2, total: 10, bytes_written: 400 },
  };
  assert.equal(throughputBytesPerSecond(job, 1_010_000), 400 / 10);
  // 4 seconds for 2 of 10 shards leaves 8 shards at 2 s each.
  assert.equal(etaSeconds(job, 1_004_000), 16);
  assert.equal(
    etaSeconds({ ...job, status: "completed" as const }, 1_004_000),
    null,
  );
  assert.equal(
    etaSeconds(
      { status: "running", progress: { stage: "merging" } },
      1_004_000,
    ),
    null,
  );
});

test("the structural report becomes per-check rows", () => {
  const rows = structuralChecks({
    verdict: "structurally-broken",
    structurally_valid: false,
    problems: ["Tensor model.layers.3.mlp.down_proj.weight has NaN values."],
    warnings: ["tokenizer_config.json was not copied"],
    tier: "structural",
    scope: "merge",
  });
  assert.equal(rows[0]?.ok, false);
  assert.equal(rows[1]?.ok, false);
  assert.equal(rows[2]?.ok, true);
  assert.equal(rows[3]?.ok, true);
});

test("a taken folder name is called out, an unused one is not", () => {
  assert.match(
    folderCollisionNotice("qwen-llama", ["Qwen-Llama"]) ?? "",
    /already exists/,
  );
  assert.equal(folderCollisionNotice("fresh-name", ["qwen-llama"]), null);
  assert.equal(folderCollisionNotice(null, ["qwen-llama"]), null);
});
