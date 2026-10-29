// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  ResumeSafetyComparison,
  ResumeSafetyReport,
} from "@/features/training/types/history";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import {
  Alert02Icon,
  CheckmarkCircle02Icon,
  InformationCircleIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ReactElement } from "react";

function shortSha(value: string | null | undefined): string {
  return value ? value.slice(0, 12) : "—";
}

function ComparisonRow({
  label,
  comparison,
  stored,
  current,
}: {
  label: string;
  comparison: ResumeSafetyComparison;
  stored: string | null | undefined;
  current: string | null | undefined;
}): ReactElement {
  const t = useT();
  const changed = comparison.status === "changed";
  const unavailable = comparison.status === "unavailable";
  return (
    <div className="grid gap-1 rounded-lg border border-border/50 bg-background/40 p-2.5">
      <div className="flex items-start gap-2">
        <HugeiconsIcon
          icon={
            changed
              ? Alert02Icon
              : unavailable
                ? InformationCircleIcon
                : CheckmarkCircle02Icon
          }
          strokeWidth={1.75}
          className={cn(
            "mt-0.5 size-3.5 shrink-0",
            changed
              ? "text-destructive"
              : unavailable
                ? "text-muted-foreground"
                : "text-emerald-600 dark:text-emerald-400",
          )}
        />
        <div className="min-w-0 space-y-0.5">
          <p className="text-ui-11 font-medium text-foreground">
            {label} ·{" "}
            <span
              className={cn(
                changed ? "text-destructive" : "text-muted-foreground",
              )}
            >
              {changed
                ? t("resumeSafety.changed")
                : unavailable
                  ? t("resumeSafety.unavailable")
                  : t("resumeSafety.match")}
            </span>
          </p>
          <p className="break-words text-ui-10 leading-relaxed text-muted-foreground">
            {comparison.message}
          </p>
          {!unavailable ? (
            <p className="font-mono text-ui-10 text-muted-foreground/80">
              {shortSha(stored)} → {shortSha(current)}
            </p>
          ) : null}
        </div>
      </div>
    </div>
  );
}

export function ResumeSafetyCard({
  report,
  error,
}: {
  report: ResumeSafetyReport | null;
  error: string | null;
}): ReactElement | null {
  const t = useT();
  if (error) {
    return (
      <section className="rounded-2xl border border-border/60 bg-card p-4">
        <h3 className="text-ui-11 font-medium text-foreground">
          {t("resumeSafety.title")}
        </h3>
        <p role="alert" className="mt-2 text-ui-10 text-destructive">
          {error}
        </p>
      </section>
    );
  }
  if (!report) return null;

  const adapter = report.checks.checkpoint?.adapter;
  const optimizer = report.checks.checkpoint?.optimizer;
  const dataset = report.dataset;
  const tokenizer = report.tokenizer;

  return (
    <section
      aria-labelledby="resume-safety-title"
      className="rounded-2xl border border-border/60 bg-card p-4"
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 space-y-1">
          <h3
            id="resume-safety-title"
            className="text-ui-11 font-medium text-foreground"
          >
            {t("resumeSafety.title")}
          </h3>
          <p className="max-w-2xl text-ui-10 leading-relaxed text-muted-foreground">
            {report.ok
              ? t("resumeSafety.ready")
              : t("resumeSafety.blocked", {
                  reason: report.blockers[0]?.message ?? "",
                })}
          </p>
        </div>
        <span
          className={cn(
            "rounded-full border px-2 py-0.5 text-ui-10 font-medium",
            report.ok
              ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
              : "border-destructive/30 bg-destructive/10 text-destructive",
          )}
        >
          {report.ok ? t("resumeSafety.pass") : t("resumeSafety.fail")}
        </span>
      </div>

      <div className="mt-3 grid gap-2 sm:grid-cols-2">
        <ComparisonRow
          label={t("resumeSafety.dataset")}
          comparison={dataset}
          stored={dataset.stored_sha256}
          current={dataset.current_sha256}
        />
        <ComparisonRow
          label={t("resumeSafety.tokenizer")}
          comparison={tokenizer}
          stored={report.stored_tokenizer_fingerprint?.sha256}
          current={report.checks.current_tokenizer?.sha256}
        />
        <div className="grid gap-1 rounded-lg border border-border/50 bg-background/40 p-2.5">
          <p className="text-ui-11 font-medium text-foreground">
            {t("resumeSafety.adapter")}
          </p>
          <p className="text-ui-10 text-muted-foreground">
            {adapter
              ? t("resumeSafety.adapterPresent")
              : t("resumeSafety.adapterMissing")}
          </p>
        </div>
        <div className="grid gap-1 rounded-lg border border-border/50 bg-background/40 p-2.5">
          <p className="text-ui-11 font-medium text-foreground">
            {t("resumeSafety.optimizer")}
          </p>
          <p className="text-ui-10 text-muted-foreground">
            {optimizer
              ? t("resumeSafety.optimizerPresent")
              : t("resumeSafety.optimizerMissing")}
          </p>
        </div>
      </div>

      {report.warnings.length > 0 ? (
        <ul className="mt-3 space-y-1">
          {report.warnings.map((warning) => (
            <li
              key={warning.code}
              className="flex items-start gap-1.5 text-ui-10 leading-relaxed text-amber-700 dark:text-amber-300"
            >
              <HugeiconsIcon
                icon={InformationCircleIcon}
                strokeWidth={1.75}
                className="mt-0.5 size-3 shrink-0"
              />
              <span className="min-w-0 break-words">{warning.message}</span>
            </li>
          ))}
        </ul>
      ) : null}
    </section>
  );
}
