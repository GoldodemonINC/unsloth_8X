// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const MIN_MERGE_MODELS = 2;
export const MAX_MERGE_MODELS = 6;
export const MAX_MERGE_MODELS_CPU = 3;

export function maxMergeModels(
  hasGpu: boolean | null | undefined,
  planMax?: number | null,
): number {
  if (
    typeof planMax === "number" &&
    Number.isFinite(planMax) &&
    planMax >= MIN_MERGE_MODELS
  ) {
    return Math.min(MAX_MERGE_MODELS, Math.round(planMax));
  }
  return hasGpu === false ? MAX_MERGE_MODELS_CPU : MAX_MERGE_MODELS;
}

export function parentLabel(index: number, total: number): string {
  if (total === 2) {
    return index === 0 ? "Model A" : "Model B";
  }
  return `Model ${index + 1}`;
}

export function blendShares(weights: number[], count: number): number[] {
  if (count <= 0) {
    return [];
  }
  const total = weights
    .slice(0, count)
    .reduce((sum, value) => sum + (Number.isFinite(value) ? value : 0), 0);
  if (!(total > 0)) {
    return Array.from({ length: count }, () => 1 / count);
  }
  return Array.from({ length: count }, (_, index) => {
    const value = weights[index];
    return Number.isFinite(value) ? (value as number) / total : 0;
  });
}

export function appendWeight(weights: number[]): number[] {
  if (weights.length === 0) {
    return [1];
  }
  const average =
    weights.reduce((sum, value) => sum + value, 0) / weights.length;
  return [...weights, average > 0 ? average : 1];
}

export function removeWeight(weights: number[], index: number): number[] {
  return weights.filter((_, position) => position !== index);
}
