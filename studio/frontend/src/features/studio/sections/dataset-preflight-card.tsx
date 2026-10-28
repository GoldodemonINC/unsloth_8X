// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { useTrainingConfigStore } from "@/features/training";
import { checkDatasetPreflight } from "@/features/training/api/datasets-api";
import type { DatasetPreflightReport } from "@/features/training/api/datasets-api";
import { useT } from "@/i18n";
import { Alert02Icon, CheckmarkCircle02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useEffect, useRef, useState } from "react";
import { useShallow } from "zustand/react/shallow";

/**
 * Reads the picked local file before a run starts. The report is advisory: it
 * names the rows the trainer cannot read instead of rewriting the dataset, so
 * fixing one is the user's decision and the manifest keeps matching the file.
 */
export function DatasetPreflightCard(): ReactElement | null {
  const t = useT();
  const { datasetSource, uploadedFile } = useTrainingConfigStore(
    useShallow((state) => ({
      datasetSource: state.datasetSource,
      uploadedFile: state.uploadedFile,
    })),
  );
  const path = datasetSource === "upload" ? uploadedFile : null;
  const [report, setReport] = useState<DatasetPreflightReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);

  // A new selection invalidates the old report; a late response cannot replace it.
  useEffect(() => {
    if (!path) return;
    setReport(null);
    setError(null);
    setRunning(false);
    return () => controllerRef.current?.abort();
  }, [path]);

  if (!path) return null;

  const run = async () => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setRunning(true);
    setError(null);
    try {
      const next = await checkDatasetPreflight(path, controller.signal);
      if (!controller.signal.aborted) setReport(next);
    } catch (failure) {
      if (controller.signal.aborted) return;
      setReport(null);
      setError(
        failure instanceof Error
          ? failure.message
          : t("datasetPreflight.failed"),
      );
    } finally {
      if (!controller.signal.aborted) setRunning(false);
    }
  };

  const counts = report
    ? (
        [
          ["datasetPreflight.invalid", report.invalid_rows],
          ["datasetPreflight.missingText", report.missing_text_rows],
          ["datasetPreflight.duplicates", report.duplicate_rows],
          ["datasetPreflight.longRows", report.long_rows],
        ] as const
      ).filter(([, count]) => count > 0)
    : [];
  const blockingSamples = report
    ? [...report.samples.invalid, ...report.samples.missing_text].slice(0, 8)
    : [];

  return (
    <section className="rounded-2xl border border-border/60 bg-card p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 space-y-1">
          <h3 className="text-ui-12 font-medium text-foreground">
            {t("datasetPreflight.title")}
          </h3>
          <p className="max-w-2xl text-ui-11 leading-relaxed text-muted-foreground">
            {t("datasetPreflight.description")}
          </p>
        </div>
        <Button
          size="sm"
          variant="outline"
          className="shrink-0"
          disabled={running}
          onClick={() => void run()}
        >
          {running ? <Spinner className="size-3.5" /> : null}
          {running ? t("datasetPreflight.running") : t("datasetPreflight.run")}
        </Button>
      </div>

      {error ? (
        <p className="mt-3 text-ui-11 text-destructive">{error}</p>
      ) : report ? (
        <div className="mt-3 space-y-2">
          <div className="flex items-start gap-2">
            <HugeiconsIcon
              icon={report.ok ? CheckmarkCircle02Icon : Alert02Icon}
              strokeWidth={1.75}
              className={
                report.ok
                  ? "mt-0.5 size-4 shrink-0 text-emerald-500"
                  : "mt-0.5 size-4 shrink-0 text-destructive"
              }
            />
            <div className="min-w-0 space-y-1">
              <p className="text-ui-11 font-medium text-foreground">
                {report.ok
                  ? t("datasetPreflight.clean")
                  : t("datasetPreflight.blocked")}
              </p>
              <p className="text-ui-10 text-muted-foreground">
                {t("datasetPreflight.rowsChecked", {
                  count: report.checked_rows.toLocaleString(),
                })}
                {report.text_field
                  ? ` · ${t("datasetPreflight.textField", {
                      field: report.text_field,
                    })}`
                  : null}
              </p>
            </div>
          </div>

          {counts.length > 0 ? (
            <div className="flex flex-wrap gap-1.5">
              {counts.map(([key, count]) => (
                <span
                  key={key}
                  className="rounded-full border border-border/60 bg-muted/40 px-2 py-0.5 text-ui-10 text-muted-foreground"
                >
                  {t(key, { count: count.toLocaleString() })}
                </span>
              ))}
            </div>
          ) : null}

          {report.truncated ? (
            <p className="text-ui-10 text-amber-600 dark:text-amber-400">
              {t("datasetPreflight.truncated", {
                count: report.checked_rows.toLocaleString(),
              })}
            </p>
          ) : null}

          {blockingSamples.length > 0 ? (
            <details className="group text-ui-10 text-muted-foreground">
              <summary className="cursor-pointer select-none text-ui-10 text-muted-foreground hover:text-foreground">
                {t("datasetPreflight.details")}
              </summary>
              <ul className="mt-1.5 space-y-1 font-mono">
                {blockingSamples.map((sample, index) => (
                  <li
                    key={`${sample.file}-${sample.line ?? sample.row ?? index}`}
                    className="truncate"
                  >
                    {sample.file}
                    {sample.line ? `:${sample.line}` : null}
                    {sample.row
                      ? ` · ${t("datasetPreflight.row", { row: sample.row })}`
                      : null}
                    {" — "}
                    {sample.error ??
                      t("datasetPreflight.noTextKeys", {
                        keys: (sample.keys ?? []).join(", ") || "—",
                      })}
                  </li>
                ))}
              </ul>
            </details>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
