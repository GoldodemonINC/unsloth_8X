// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TrainingSeriesPoint } from "@/features/training";
import { detectLossAnomaly } from "@/features/training/lib/loss-anomaly";
import { useT } from "@/i18n";
import { Alert02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useMemo } from "react";

const COPY = {
  nan: {
    title: "trainingAnomaly.nan",
    hint: "trainingAnomaly.nanHint",
  },
  divergence: {
    title: "trainingAnomaly.divergence",
    hint: "trainingAnomaly.divergenceHint",
  },
  spike: {
    title: "trainingAnomaly.spike",
    hint: "trainingAnomaly.spikeHint",
  },
  plateau: {
    title: "trainingAnomaly.plateau",
    hint: "trainingAnomaly.plateauHint",
  },
} as const;

export function LossAnomalyCallout({
  lossHistory,
}: {
  lossHistory: readonly TrainingSeriesPoint[];
}): ReactElement | null {
  const t = useT();
  const anomaly = useMemo(() => detectLossAnomaly(lossHistory), [lossHistory]);
  if (!anomaly) return null;
  const copy = COPY[anomaly.kind];

  return (
    <output
      className={
        anomaly.severity === "critical"
          ? "flex items-start gap-2.5 rounded-xl border border-destructive/30 bg-destructive/5 px-3.5 py-3"
          : "flex items-start gap-2.5 rounded-xl border border-amber-500/30 bg-amber-500/5 px-3.5 py-3"
      }
    >
      <HugeiconsIcon
        icon={Alert02Icon}
        strokeWidth={1.75}
        className={
          anomaly.severity === "critical"
            ? "mt-0.5 size-4 shrink-0 text-destructive"
            : "mt-0.5 size-4 shrink-0 text-amber-600 dark:text-amber-400"
        }
      />
      <div className="min-w-0 flex-1 space-y-0.5">
        <p className="text-ui-12 font-medium text-foreground">
          {t(copy.title)}
        </p>
        <p className="text-ui-11 leading-relaxed text-muted-foreground">
          {t(copy.hint)}
        </p>
      </div>
      <span className="shrink-0 rounded-full border border-border/60 bg-background/60 px-2 py-0.5 font-mono text-ui-10 text-muted-foreground">
        {t("trainingAnomaly.step", { step: anomaly.step })}
      </span>
    </output>
  );
}
