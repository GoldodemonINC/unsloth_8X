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
const { initialTrainingConfigState, moeModelPatch } = await import(
  "../src/features/training/stores/training-config-policy.ts"
);
const { partializeTrainingConfig } = await import(
  "../src/features/training/stores/training-config-persistence.ts"
);
const { runConfigDraftSelections } = await import(
  "../src/features/training/lib/run-config-draft.ts"
);
const { MOE_EXPERT_SCOPES, moeScopeTargets } = await import(
  "../src/features/advanced/types.ts"
);

function config(patch: Partial<TrainingConfigState> = {}): TrainingConfigState {
  return {
    ...initialTrainingConfigState,
    selectedModel: "Qwen/Qwen3-30B-A3B",
    modelType: "text",
    datasetSource: "huggingface",
    dataset: "unsloth/alpaca-cleaned",
    datasetSplit: "train",
    datasetFormat: "auto",
    trainingMethod: "qlora",
    ...patch,
  };
}

test("a LoRA run posts the chosen MoE scope", () => {
  const payload = buildTrainingStartPayload(
    config({
      isMoeModel: true,
      numExperts: 128,
      moeExpertScope: "experts_only",
    }),
    null,
  );
  assert.equal(payload.moe_expert_scope, "experts_only");
});

test("no scope means the payload keeps its old shape", () => {
  const payload = buildTrainingStartPayload(
    config({ isMoeModel: true, numExperts: 128 }),
    null,
  );
  assert.equal(payload.moe_expert_scope, null);
});

test("full finetuning never carries a scope, since it trains every expert", () => {
  const payload = buildTrainingStartPayload(
    config({ trainingMethod: "full", moeExpertScope: "router_only" }),
    null,
  );
  assert.equal(payload.moe_expert_scope, null);
});

test("a dense model clears the scope the previous model had", () => {
  const patch = moeModelPatch({ is_moe: false });
  assert.equal(patch.isMoeModel, false);
  assert.equal(patch.numExperts, null);
  assert.equal(patch.moeExpertScope, null);
});

test("a MoE model keeps the scope and reports its expert counts", () => {
  const patch = moeModelPatch({
    is_moe: true,
    num_experts: 64,
    num_experts_per_tok: 4,
  });
  assert.equal(patch.isMoeModel, true);
  assert.equal(patch.numExperts, 64);
  assert.equal(patch.numExpertsPerToken, 4);
  assert.equal("moeExpertScope" in patch, false);
});

test("an absent is_moe flag reads as dense, not as unknown", () => {
  const patch = moeModelPatch({});
  assert.equal(patch.isMoeModel, false);
  assert.equal(patch.moeExpertScope, null);
});

test("every scope names what it trains and what it freezes", () => {
  assert.deepEqual(moeScopeTargets("experts_and_attention"), {
    attention: true,
    experts: true,
    router: false,
  });
  assert.deepEqual(moeScopeTargets("experts_only"), {
    attention: false,
    experts: true,
    router: false,
  });
  assert.deepEqual(moeScopeTargets("experts_and_router"), {
    attention: false,
    experts: true,
    router: true,
  });
  assert.deepEqual(moeScopeTargets("router_only"), {
    attention: false,
    experts: false,
    router: true,
  });
  assert.equal(MOE_EXPERT_SCOPES.length, 4);
});

test("the scope persists with the rest of the config", () => {
  const persisted = partializeTrainingConfig(
    config({ moeExpertScope: "experts_and_router" }) as never,
  );
  assert.equal(persisted.moeExpertScope, "experts_and_router");
  // Derived model facts are re-read on load, not restored from a stale blob.
  assert.equal("isMoeModel" in persisted, false);
  assert.equal("numExperts" in persisted, false);
});

test("restoring a run brings the scope back, and junk is dropped", () => {
  const draft = runConfigDraftSelections({
    model_name: "Qwen/Qwen3-30B-A3B",
    hf_dataset: "unsloth/alpaca-cleaned",
    train_split: "train",
    moe_expert_scope: "experts_only",
  });
  assert.equal(draft.moeExpertScope, "experts_only");

  const junk = runConfigDraftSelections({
    model_name: "Qwen/Qwen3-30B-A3B",
    hf_dataset: "unsloth/alpaca-cleaned",
    moe_expert_scope: "everything",
  });
  assert.equal(junk.moeExpertScope, null);
});
