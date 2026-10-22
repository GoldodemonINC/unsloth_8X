// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TranslationKey } from "@/i18n";

export function resolveStartTrainingButtonLabelKey({
  stopRequested,
  startBlocked,
  isLoadingModel,
  isCheckingDataset,
  hasModel,
  hasDataset,
  simulated = false,
}: {
  stopRequested: boolean;
  startBlocked: boolean;
  isLoadingModel: boolean;
  isCheckingDataset: boolean;
  hasModel: boolean;
  hasDataset: boolean;
  /** A simulated run needs no model, dataset or GPU, so none of those gate it. */
  simulated?: boolean;
}): TranslationKey {
  if (stopRequested) {
    return "studio.training.stopping";
  }
  if (startBlocked) {
    return "studio.training.starting";
  }
  if (simulated) {
    return "studio.training.startSimulated";
  }
  if (isLoadingModel) {
    return "studio.training.loadingModel";
  }
  if (isCheckingDataset) {
    return "studio.training.checkingDataset";
  }
  if (!(hasModel || hasDataset)) {
    return "studio.training.chooseModelAndDataset";
  }
  if (!hasModel) {
    return "studio.training.chooseModel";
  }
  return hasDataset
    ? "studio.training.startTraining"
    : "studio.training.chooseDataset";
}
