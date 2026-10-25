// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { useDebugTrainingSimulation } from "@/features/settings/hooks/use-debug-training-simulation";
import {
  type StartValidationResult,
  useTrainingActions,
  useTrainingConfigStore,
  useTrainingReadiness,
} from "@/features/training";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import type { DatasetSource } from "@/types/training";
import { Refresh01Icon, Rocket01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useShallow } from "zustand/react/shallow";
import { resolveStartTrainingButtonLabelKey } from "./start-training-cta-state";

function resolveStartTrainingError(input: {
  t: ReturnType<typeof useT>;
  startError: string | null | undefined;
  modelError: string | null;
  isIncompatible: boolean;
  isAudioModel: boolean;
  isDatasetAudio: boolean | null | undefined;
  datasetUnverified: boolean;
  hasModel: boolean;
  hasDataset: boolean;
  datasetSource: DatasetSource;
  configValidation: StartValidationResult;
}): string | null {
  const {
    t,
    startError,
    modelError,
    isIncompatible,
    isAudioModel,
    isDatasetAudio,
    datasetUnverified,
    hasModel,
    hasDataset,
    datasetSource,
    configValidation,
  } = input;
  if (isIncompatible) {
    return !isAudioModel && isDatasetAudio === true
      ? t("studio.training.audioIncompatible")
      : t("studio.training.visionIncompatible");
  }
  if (!hasModel) {
    return null;
  }
  if (!hasDataset) {
    if (datasetSource === "s3" && !configValidation.ok) {
      return t(configValidation.errorKey);
    }
    return null;
  }
  if (!configValidation.ok) {
    return t(configValidation.errorKey);
  }
  if (startError) {
    return startError;
  }
  if (modelError) {
    return t("studio.training.modelUnverified");
  }
  return datasetUnverified ? t("studio.training.datasetUnverified") : null;
}

export function StartTrainingCta() {
  const t = useT();
  const {
    isAudioModel,
    isDatasetAudio,
    datasetSource,
    ensureModelDefaultsLoaded,
  } = useTrainingConfigStore(
    useShallow((state) => ({
      isAudioModel: state.isAudioModel,
      isDatasetAudio: state.isDatasetAudio,
      datasetSource: state.datasetSource,
      ensureModelDefaultsLoaded: state.ensureModelDefaultsLoaded,
    })),
  );
  const {
    isReady,
    isLoadingModel,
    isCheckingDataset,
    isIncompatible,
    datasetUnverified,
    modelError,
    hasModel,
    hasDataset,
    configValidation,
  } = useTrainingReadiness();
  const { startError, startBlocked, stopRequested, startTrainingRun } =
    useTrainingActions();
  const trainingMethod = useTrainingConfigStore((s) => s.trainingMethod);
  // Settings -> Debugging: a fake trainer runs instead, so no model, dataset
  // or GPU is needed and none of the usual readiness gates apply.
  const simulated = useDebugTrainingSimulation();

  const disabled = startBlocked || (!simulated && !isReady);
  const buttonLabel = t(
    resolveStartTrainingButtonLabelKey({
      stopRequested,
      startBlocked,
      isLoadingModel,
      isCheckingDataset,
      hasModel,
      hasDataset,
      simulated,
    }),
  );
  const errorMessage = simulated
    ? startError
    : resolveStartTrainingError({
        t,
        startError,
        modelError,
        isIncompatible,
        isAudioModel,
        isDatasetAudio,
        datasetUnverified,
        hasModel,
        hasDataset,
        datasetSource,
        configValidation,
      });
  const isWarning =
    !(startError || isIncompatible || !configValidation.ok) &&
    !!(modelError || datasetUnverified);
  const showsModelWarning =
    !!modelError &&
    !startError &&
    !isIncompatible &&
    configValidation.ok &&
    hasModel &&
    hasDataset;

  // CMT never posts a training start: its engine owns the datasets, the expansion and
  // the training loop, and the CMT card above carries the button that runs it.
  if (trainingMethod === "cmt") {
    return (
      <p
        role="status"
        className="rounded-xl border border-border/60 bg-muted/30 px-3 py-2.5 text-ui-11p5 leading-relaxed text-muted-foreground"
      >
        {t("labs.cmt.trainerNote")}
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      <Button
        size="lg"
        className={cn(
          "h-11 w-full justify-center rounded-xl text-ui-13p5 font-semibold tracking-tight",
          "bg-primary text-primary-foreground shadow-sm",
          "hover:bg-primary/90",
          "disabled:bg-[color-mix(in_oklab,var(--foreground)_calc(8%*var(--contrast-wash-gain,1)),transparent)] disabled:text-muted-foreground disabled:shadow-none dark:disabled:bg-[rgb(255_255_255_/_calc(0.06*var(--contrast-wash-gain,1)))]",
          "transition-colors duration-200",
        )}
        onClick={() => {
          startTrainingRun().catch(() => undefined);
        }}
        disabled={disabled}
      >
        <HugeiconsIcon
          icon={Rocket01Icon}
          strokeWidth={1.75}
          className="size-4"
        />
        {buttonLabel}
      </Button>
      {simulated && (
        <output
          data-testid="simulated-training-note"
          className="block text-ui-11p5 leading-relaxed text-muted-foreground"
        >
          {t("studio.training.simulatedNote")}
        </output>
      )}
      {errorMessage && (
        <div
          className={cn(
            "flex items-start justify-between gap-2 text-ui-11p5 leading-relaxed",
            isWarning ? "text-status-warning" : "text-destructive",
          )}
        >
          <p
            role={isWarning ? "status" : "alert"}
            className="min-w-0 break-words"
          >
            {errorMessage}
          </p>
          {showsModelWarning && (
            <button
              type="button"
              onClick={ensureModelDefaultsLoaded}
              className="inline-flex shrink-0 items-center gap-1 rounded-md px-1.5 py-0.5 font-medium text-foreground transition-colors hover:bg-[color-mix(in_oklab,var(--foreground)_calc(6%*var(--contrast-wash-gain,1)),transparent)] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
            >
              <HugeiconsIcon
                icon={Refresh01Icon}
                strokeWidth={1.75}
                className="size-3"
              />
              {t("picker.retry")}
            </button>
          )}
        </div>
      )}
    </div>
  );
}
