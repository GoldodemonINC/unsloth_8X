// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  CORA_DATASET_MAX_BYTES,
  CORA_DATASET_MIN_BYTES,
  CORA_HYPERPARAMS,
  coraDatasetSizeIssue,
} = await import("../src/config/training.ts");
const { buildTrainingMethodPatch } = await import(
  "../src/features/training/stores/training-method-transition.ts"
);
const {
  getTrainingMethodLabel,
  isTrainingMethodSupportedOnDevice,
  toBackendTrainingType,
} = await import("../src/features/training/lib/training-methods.ts");
const { DEFAULT_ADVANCED_TRAINING } = await import(
  "../src/features/advanced/types.ts"
);

function state(method: string) {
  return {
    trainingMethod: method,
    trainingMethodProvenance: {
      learningRateManuallySet: false,
      modelAdapterLearningRate: null,
      datasetFormatBeforeCpt: null,
      targetModulesBeforeCpt: null,
      loraRankBeforeCpt: null,
      loraAlphaBeforeCpt: null,
      loraVariantBeforeCpt: null,
      trainOnCompletionsBeforeCpt: null,
    },
    datasetFormat: "auto",
    targetModules: ["q_proj", "k_proj"],
    loraRank: 16,
    loraAlpha: 32,
    loraVariant: "lora",
    trainOnCompletions: false,
    datasetStreaming: false,
    selectedModel: "unsloth/Qwen3-0.6B",
    modelDefaultsAppliedFor: null,
    modelType: "text",
    isEmbeddingModel: false,
    isVisionModel: false,
    isAudioModel: false,
    isDatasetImage: false,
    isDatasetAudio: false,
    learningRate: 2e-4,
    optimizerType: "adamw_8bit",
    lrSchedulerType: "linear",
    batchSize: 4,
    gradientAccumulation: 8,
    contextLength: 2048,
    advanced: { ...DEFAULT_ADVANCED_TRAINING },
  } as const;
}

test("selecting CoRA applies the CPU recipe and turns on its two extras", () => {
  const patch = buildTrainingMethodPatch(state("qlora") as never, "cora");

  assert.equal(patch.trainingMethod, "cora");
  assert.equal(patch.loraRank, CORA_HYPERPARAMS.loraRank);
  assert.equal(patch.loraAlpha, CORA_HYPERPARAMS.loraAlpha);
  assert.equal(patch.loraVariant, "loftq");
  assert.equal(patch.optimizerType, "paged_adamw_8bit");
  assert.equal(patch.lrSchedulerType, "cosine");
  assert.equal(patch.batchSize, 1);
  assert.equal(patch.gradientAccumulation, 3);
  assert.equal(patch.contextLength, 1024);
  assert.equal(patch.learningRate, CORA_HYPERPARAMS.learningRate);
  assert.equal(patch.advanced?.qatEnabled, true);
  assert.equal(patch.advanced?.layerLrEnabled, true);
  assert.equal(patch.advanced?.curriculumEnabled, false);
  assert.equal(patch.advanced?.blendEnabled, false);
  assert.equal(patch.advanced?.autoEvalEnabled, false);
});

test("leaving CoRA restores the defaults only for untouched knobs", () => {
  const coraState = {
    ...state("cora"),
    ...CORA_HYPERPARAMS,
    advanced: {
      ...DEFAULT_ADVANCED_TRAINING,
      qatEnabled: true,
      layerLrEnabled: true,
    },
  };
  const untouched = buildTrainingMethodPatch(coraState as never, "qlora");
  assert.equal(untouched.loraRank, 16);
  assert.equal(untouched.batchSize, 4);
  assert.equal(untouched.optimizerType, "adamw_8bit");
  assert.equal(untouched.lrSchedulerType, "linear");
  assert.equal(untouched.advanced?.qatEnabled, false);
  assert.equal(untouched.advanced?.layerLrEnabled, false);

  const edited = buildTrainingMethodPatch(
    { ...coraState, loraRank: 32, batchSize: 2 } as never,
    "qlora",
  );
  assert.equal(edited.loraRank, undefined);
  assert.equal(edited.batchSize, undefined);
  assert.equal(edited.lrSchedulerType, "linear");
});

test("CoRA records as LoRA/QLoRA and labels itself", () => {
  assert.equal(toBackendTrainingType("cora"), "LoRA/QLoRA");
  assert.equal(getTrainingMethodLabel("cora"), "CoRA");
});

test("CoRA is offered on CPU hosts but not on Apple MLX", () => {
  assert.equal(isTrainingMethodSupportedOnDevice("cora", "cuda"), true);
  assert.equal(isTrainingMethodSupportedOnDevice("cora", "cpu"), true);
  assert.equal(isTrainingMethodSupportedOnDevice("cora", "mac"), false);
});

test("the CoRA dataset window is 10 MB to 0.8 GB", () => {
  assert.equal(coraDatasetSizeIssue(null), null);
  assert.equal(coraDatasetSizeIssue(0), null);
  assert.match(coraDatasetSizeIssue(CORA_DATASET_MIN_BYTES / 2) ?? "", /10 MB/);
  assert.equal(coraDatasetSizeIssue(CORA_DATASET_MIN_BYTES * 2), null);
  assert.equal(coraDatasetSizeIssue(CORA_DATASET_MAX_BYTES), null);
  assert.match(
    coraDatasetSizeIssue(CORA_DATASET_MAX_BYTES * 2) ?? "",
    /0.8 GB/,
  );
});
