// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

const { filterMergeSearchResults, isMergeableHubRepo } = await import(
  "../src/features/advanced/merge-search-filter.ts"
);

test("merge search hides Unsloth repackagings", () => {
  assert.equal(isMergeableHubRepo("unsloth/DeepSeek-R1-Distill-Qwen-1.5B"), false);
  assert.equal(isMergeableHubRepo("Unsloth/Llama-3.2-1B"), false);
  assert.equal(isMergeableHubRepo("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"), true);
  // Only the org prefix counts, not an unrelated mention in the repo name.
  assert.equal(isMergeableHubRepo("some-org/unsloth-notes"), true);
});

test("merge search hides quantized repackagings", () => {
  assert.equal(isMergeableHubRepo("acme/model-gguf"), false);
  assert.equal(isMergeableHubRepo("acme/model-bnb-4bit"), false);
  assert.equal(isMergeableHubRepo("acme/model-fp8"), false);
  assert.equal(isMergeableHubRepo("acme/model"), true);
});

test("filtering keeps Hub order and drops only unusable parents", () => {
  const rows = [
    { id: "unsloth/Llama-3.2-1B" },
    { id: "meta-llama/Llama-3.2-1B" },
    { id: "acme/model-GGUF" },
    { id: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" },
  ];
  assert.deepEqual(
    filterMergeSearchResults(rows).map((row) => row.id),
    ["meta-llama/Llama-3.2-1B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"],
  );
});
