// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Spinner } from "@/components/ui/spinner";
import {
  type DiskPreflightReport,
  checkDiskPreflight,
} from "@/features/training/api/disk-preflight-api";
import { useTrainingConfigStore } from "@/features/training";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { Alert02Icon, CheckmarkCircle02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useEffect, useState } from "react";
import { useShallow } from "zustand/react/shallow";

function tone(status: DiskPreflightReport["status"]): string {
  if (status === "ok") return "text-emerald-600 dark:text-emerald-400";
  if (status === "tight") return "text-amber-600 dark:text-amber-400";
  if (status === "over") return "text-destructive";
  return "text-muted-foreground";
}

export function DiskPreflightCard(): ReactElement | null {
  const t = useT();
  const {
    selectedModel,
    trainingMethod,
    contextLength,
    batchSize,
    gradientAccumulation,
    loraRank,
    maxSteps,
    saveSteps,
    modelKnownCached,
  } = useTrainingConfigStore(
    useShallow((state) => ({
      selectedModel: state.selectedModel,
      trainingMethod: state.trainingMethod,
      contextLength: state.contextLength,
      batchSize: state.batchSize,
      gradientAccumulation: state.gradientAccumulation,
      loraRank: state.loraRank,
      maxSteps: state.maxSteps,
      saveSteps: state.saveSteps,
      modelKnownCached: state.modelKnownCached,
    })),
  );
  const [report, setReport] = useState<DiskPreflightReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);

  useEffect(() => {
    if (!selectedModel) {
      setReport(null);
      setError(null);
      return;
    }
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      setChecking(true);
      setError(null);
      checkDiskPreflight({
        model: selectedModel,
        method: trainingMethod,
        loadIn4Bit: trainingMethod === "qlora",
        maxSeqLength: contextLength,
        batchSize,
        gradientAccumulation,
        loraRank,
        maxSteps,
        saveSteps,
        modelKnownCached,
        signal: controller.signal,
      })
        .then((next) => {
          if (!controller.signal.aborted) setReport(next);
        })
        .catch((failure: unknown) => {
          if (controller.signal.aborted) return;
          setReport(null);
          setError(
            failure instanceof Error
              ? failure.message
              : t("diskPreflight.failed"),
          );
        })
        .finally(() => {
          if (!controller.signal.aborted) setChecking(false);
        });
    }, 300);
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [
    batchSize,
    contextLength,
    gradientAccumulation,
    loraRank,
    maxSteps,
    modelKnownCached,
    saveSteps,
    selectedModel,
    t,
    trainingMethod,
  ]);

  if (!selectedModel) return null;

  return (
    <section
      aria-labelledby="disk-preflight-title"
      className="flex flex-col gap-2 rounded-xl border border-border/60 bg-muted/20 p-3"
    >
      <div className="flex items-center justify-between gap-2">
        <h3
          id="disk-preflight-title"
          className="text-ui-11 font-medium text-foreground"
        >
          {t("diskPreflight.title")}
        </h3>
        {checking ? <Spinner className="size-3.5" /> : null}
      </div>

      {error ? (
        <p role="alert" className="text-ui-10 leading-relaxed text-destructive">
          {error}
        </p>
      ) : report?.status === "unknown" ? (
        <p className="text-ui-10 leading-relaxed text-muted-foreground">
          {report.reason ?? t("diskPreflight.unknown")}
        </p>
      ) : report ? (
        <>
          <div className="flex items-start gap-2">
            <HugeiconsIcon
              icon={
                report.status === "over" ? Alert02Icon : CheckmarkCircle02Icon
              }
              strokeWidth={1.75}
              className={cn("mt-0.5 size-3.5 shrink-0", tone(report.status))}
            />
            <div className="min-w-0 space-y-0.5">
              <p className={cn("text-ui-10p5 font-medium", tone(report.status))}>
                {report.status === "over"
                  ? t("diskPreflight.over")
                  : report.status === "tight"
                    ? t("diskPreflight.tight")
                    : t("diskPreflight.ok")}
              </p>
              <p className="text-ui-10 leading-relaxed text-muted-foreground">
                {t("diskPreflight.summary", {
                  required: (report.required_gb ?? 0).toFixed(2),
                  free: (report.free_gb ?? 0).toFixed(2),
                })}
              </p>
            </div>
          </div>
          <dl className="grid grid-cols-2 gap-x-3 gap-y-1 text-ui-10">
            <div className="flex justify-between gap-2">
              <dt className="text-muted-foreground">
                {t("diskPreflight.baseWeights")}
              </dt>
              <dd className="font-mono text-foreground">
                {report.breakdown.base_weights_gb.toFixed(2)} GB
              </dd>
            </div>
            <div className="flex justify-between gap-2">
              <dt className="text-muted-foreground">
                {t("diskPreflight.checkpoints")}
              </dt>
              <dd className="font-mono text-foreground">
                {report.breakdown.checkpoints_gb.toFixed(2)} GB
                {report.checkpoints.known
                  ? ` · ${report.checkpoints.count}`
                  : ""}
              </dd>
            </div>
            <div className="flex justify-between gap-2">
              <dt className="text-muted-foreground">
                {t("diskPreflight.finalSave")}
              </dt>
              <dd className="font-mono text-foreground">
                {report.breakdown.final_save_gb.toFixed(2)} GB
              </dd>
            </div>
            <div className="flex justify-between gap-2">
              <dt className="text-muted-foreground">
                {t("diskPreflight.reserve")}
              </dt>
              <dd className="font-mono text-foreground">
                {report.breakdown.reserve_gb.toFixed(2)} GB
              </dd>
            </div>
          </dl>
          {report.status === "over" && report.blockers.length > 0 ? (
            <p className="text-ui-10 leading-relaxed text-destructive">
              {report.blockers[0]}
            </p>
          ) : null}
          {!report.checkpoints.known ? (
            <p className="text-ui-10 leading-relaxed text-muted-foreground/80">
              {t("diskPreflight.stepsUnknown")}
            </p>
          ) : null}
        </>
      ) : (
        <p className="text-ui-10 leading-relaxed text-muted-foreground">
          {t("diskPreflight.idle")}
        </p>
      )}
    </section>
  );
}
