// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

import type { TrainingConfigState } from "../src/features/training/types/config.ts";

registerBundlerResolver();
const { buildTrainingStartPayload } = await import(
  "../src/features/training/api/mappers.ts"
);
const { initialTrainingConfigState } = await import(
  "../src/features/training/stores/training-config-policy.ts"
);
const { partializeTrainingConfig } = await import(
  "../src/features/training/stores/training-config-persistence.ts"
);
const { hasSeparateStreamingEvalSplit, streamingCompatiblePatch } =
  await import("../src/features/training/stores/training-config-policy.ts");
const { runConfigDraftSelections } = await import(
  "../src/features/training/lib/run-config-draft.ts"
);

const TRAIN_DATASET = "unsloth/alpaca-cleaned";

function config(patch: Partial<TrainingConfigState> = {}): TrainingConfigState {
  return {
    ...initialTrainingConfigState,
    selectedModel: "unsloth/Qwen3-1.7B",
    modelType: "text",
    datasetSource: "huggingface",
    dataset: TRAIN_DATASET,
    datasetSplit: "train",
    datasetFormat: "auto",
    evalSteps: 0.1,
    ...patch,
  };
}

test("a second dataset replaces the training dataset's eval split", () => {
  const payload = buildTrainingStartPayload(
    config({
      datasetEvalSplit: "validation",
      evalDataset: "tatsu-lab/alpaca_eval",
      evalDatasetSubset: "default",
      evalDatasetSplit: "test",
    }),
    null,
  );

  assert.equal(payload.eval_hf_dataset, "tatsu-lab/alpaca_eval");
  assert.equal(payload.eval_dataset_subset, "default");
  // The eval split belongs to the second dataset, not to the training repo.
  assert.equal(payload.eval_split, "test");
});

test("a second dataset without a split reads its own train split", () => {
  const payload = buildTrainingStartPayload(
    config({ evalDataset: "tatsu-lab/alpaca_eval" }),
    null,
  );
  assert.equal(payload.eval_split, "train");
  assert.equal(payload.eval_dataset_subset, null);
});

test("no second dataset keeps the old eval split payload byte for byte", () => {
  const payload = buildTrainingStartPayload(
    config({ datasetEvalSplit: "validation" }),
    null,
  );
  assert.equal(payload.eval_hf_dataset, null);
  assert.equal(payload.eval_dataset_subset, null);
  assert.equal(payload.eval_split, "validation");
});

test("an uploaded training dataset never carries a Hub eval repo", () => {
  const payload = buildTrainingStartPayload(
    config({
      datasetSource: "upload",
      dataset: null,
      uploadedFile: "/tmp/train.jsonl",
      uploadedEvalFile: "/tmp/eval.jsonl",
      evalDataset: "tatsu-lab/alpaca_eval",
    }),
    null,
  );
  assert.equal(payload.eval_hf_dataset, null);
  assert.deepEqual(payload.local_eval_datasets, ["/tmp/eval.jsonl"]);
});

test("streaming is allowed with a separate eval dataset and no eval split", () => {
  const state = config({ datasetStreaming: true, evalDataset: "org/eval" });
  assert.equal(hasSeparateStreamingEvalSplit(state), true);
  // The old rule required datasetEvalSplit !== datasetSplit; a second dataset
  // must not push streaming off (which would zero evalSteps here).
  assert.equal(streamingCompatiblePatch(state).evalSteps, undefined);
});

test("streaming still needs an eval source when no second dataset is picked", () => {
  const state = config({ datasetStreaming: true, datasetEvalSplit: null });
  assert.equal(hasSeparateStreamingEvalSplit(state), false);
  assert.equal(streamingCompatiblePatch(state).evalSteps, 0);
});

test("restoring a run keeps eval_split with the dataset that owns it", () => {
  const draft = runConfigDraftSelections({
    model_name: "unsloth/Qwen3-1.7B",
    hf_dataset: TRAIN_DATASET,
    train_split: "train",
    eval_hf_dataset: "tatsu-lab/alpaca_eval",
    eval_dataset_subset: null,
    eval_split: "test",
  });

  assert.equal(draft.evalDataset, "tatsu-lab/alpaca_eval");
  assert.equal(draft.evalDatasetSplit, "test");
  // Not also the training dataset's eval split: that split is not in this repo.
  assert.equal(draft.datasetEvalSplit, null);
});

test("the eval selection persists, and older state starts with none", () => {
  const persisted = partializeTrainingConfig(
    config({ evalDataset: "org/eval", evalDatasetSplit: "test" }) as never,
  );
  assert.equal(persisted.evalDataset, "org/eval");
  assert.equal(persisted.evalDatasetSplit, "test");

  // A blob written before the feature has none of these keys; the rehydrate merge
  // spreads it over the defaults, so the selection starts empty rather than stale.
  const older: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(persisted)) {
    if (key.startsWith("evalDataset")) continue;
    older[key] = value;
  }
  assert.equal("evalDataset" in older, false);
  const merged = {
    ...initialTrainingConfigState,
    ...(older as Partial<TrainingConfigState>),
  };
  assert.equal(merged.evalDataset ?? null, null);
  assert.equal(merged.evalDatasetSubset ?? null, null);
  assert.equal(merged.evalDatasetSplit ?? null, null);
});
