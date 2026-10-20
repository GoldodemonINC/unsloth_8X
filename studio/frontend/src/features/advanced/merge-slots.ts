// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Slot bookkeeping for a merge of 2 to 6 parents.
 *
 * The page keeps the parent list and the weight list as two pieces of state, so
 * every add and remove has to keep them the same length; getting that wrong is
 * invisible until the weights silently stop lining up with the slots. Keeping
 * the arithmetic here lets it be tested without mounting the page.
 */

export const MIN_MERGE_MODELS = 2;
export const MAX_MERGE_MODELS = 6;

/** Slot names: the familiar A/B pair for two parents, numbered slots beyond it. */
export function parentLabel(index: number, total: number): string {
  if (total === 2) {
    return index === 0 ? "Model A" : "Model B";
  }
  return `Model ${index + 1}`;
}

/**
 * Each parent's share of the blend, in slot order.
 *
 * Shares are relative, so the backend normalizes the same way this does; a
 * weight list that has drifted out of step with the slots (or sums to zero)
 * still yields a usable even split instead of NaN.
 */
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

/**
 * The weight list after adding a parent.
 *
 * A new parent joins at the current average, so an even blend stays even. Giving
 * it a fixed weight instead would make the newcomer outvote everyone (1 against
 * two 0.5s is half the blend), which is not what "add a model" should mean.
 */
export function appendWeight(weights: number[]): number[] {
  if (weights.length === 0) {
    return [1];
  }
  const average =
    weights.reduce((sum, value) => sum + value, 0) / weights.length;
  return [...weights, average > 0 ? average : 1];
}

/** The weight list after removing the parent at *index*, in the same order. */
export function removeWeight(weights: number[], index: number): number[] {
  return weights.filter((_, position) => position !== index);
}
