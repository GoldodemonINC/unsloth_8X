// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Which Hub repos the merge picker offers.
 *
 * Two things are hidden from the list, and both still work when the repo id is
 * typed in full:
 *
 * - **Quantized repackagings.** Weight-space merging needs full-precision
 *   safetensors, so a bnb/GGUF/MLX repo cannot be a parent.
 * - **Unsloth repackagings.** They load with `pad_token = <|vision_pad|>` while
 *   the upstream weights they came from pad with the eos token, so the tokenizer
 *   gate refuses that pairing. Offering them only leads to a refused merge.
 */
const QUANTIZED_RE =
  /(^|[-_.])(bnb|awq|gptq|gguf|mlx|exl2|4bit|8bit|int4|int8|fp8)([-_.]|$)/i;

const UNSLOTH_ORG_RE = /^unsloth\//i;

/** True when a search result is a usable merge parent. */
export function isMergeableHubRepo(id: string): boolean {
  return !QUANTIZED_RE.test(id) && !UNSLOTH_ORG_RE.test(id);
}

/** Keep the order the Hub returned, minus the repos that cannot be parents. */
export function filterMergeSearchResults<T extends { id: string }>(
  rows: readonly T[],
): T[] {
  return rows.filter((row) => isMergeableHubRepo(row.id));
}
