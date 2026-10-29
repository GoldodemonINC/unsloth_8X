// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { buildTrainingStartPayload } from "../api/mappers";
import type { TrainingConfigState } from "../types/config";
import {
  SWEEP_MAX_ITEMS,
  type SweepAxes,
  type SweepVariant,
  sweepVariantLabel,
} from "./training-queue-plan";

export type {
  SweepAxes,
  SweepVariant,
  TrainingQueueItem,
  TrainingQueueStatus,
} from "./training-queue-plan";
export {
  SWEEP_MAX_ITEMS,
  createQueueItemId,
  isScheduledQueueItem,
  nextReadyQueueItem,
  parseSweepNumbers,
  sanitizeQueueConfig,
  sweepCombinationCount,
  sweepVariantLabel,
} from "./training-queue-plan";

export function expandSweep(
  base: TrainingConfigState,
  axes: SweepAxes,
  hfToken: string | null,
): SweepVariant[] {
  const learningRates =
    axes.learningRates.length > 0 ? axes.learningRates : [base.learningRate];
  const loraRanks =
    axes.loraRanks.length > 0 ? axes.loraRanks : [base.loraRank];
  const epochs = axes.epochs.length > 0 ? axes.epochs : [base.epochs];
  const batchSizes =
    axes.batchSizes.length > 0 ? axes.batchSizes : [base.batchSize];

  const variants: SweepVariant[] = [];
  for (const learningRate of learningRates) {
    for (const loraRank of loraRanks) {
      for (const value of epochs) {
        for (const batchSize of batchSizes) {
          if (variants.length >= SWEEP_MAX_ITEMS) {
            return variants;
          }
          variants.push({
            label: sweepVariantLabel(learningRate, loraRank, value, batchSize),
            config: buildTrainingStartPayload(
              { ...base, learningRate, loraRank, epochs: value, batchSize },
              hfToken,
              null,
            ),
          });
        }
      }
    }
  }
  return variants;
}
