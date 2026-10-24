// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { type TranslationKey, useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  Alert02Icon,
  Cancel01Icon,
  Delete02Icon,
  PlusSignIcon,
  Refresh01Icon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type ReactElement,
  type ReactNode,
  useCallback,
  useEffect,
  useState,
} from "react";
import {
  type PipelineKind,
  type PipelinePlan,
  type PipelineStageSpec,
  type SavedPipeline,
  deletePipeline,
  listPipelines,
  planPipeline,
  savePipeline,
} from "../api";

type NumericKey = "factor" | "steps" | "batch_size" | "sequence_length";

const KINDS: PipelineKind[] = [
  "merge",
  "scale",
  "train",
  "quantize",
  "benchmark",
];
const METHODS = ["depth", "mlp_width", "depth+mlp_width"] as const;
const LEVELS = ["f16", "q8_0", "q6_k", "q5_k_m", "q4_k_m", "q2_k"] as const;
const KIND_KEY: Record<PipelineKind, TranslationKey> = {
  merge: "labs.kind.merge",
  scale: "labs.kind.scale",
  train: "labs.kind.train",
  quantize: "labs.kind.quantize",
  benchmark: "labs.kind.benchmark",
};

const DEFAULTS: Record<PipelineKind, PipelineStageSpec> = {
  merge: { kind: "merge", parents: [] },
  scale: { kind: "scale", factor: 2, method: "depth" },
  train: {
    kind: "train",
    steps: 60,
    batch_size: 1,
    sequence_length: 2048,
  },
  quantize: { kind: "quantize", level: "q4_k_m" },
  benchmark: { kind: "benchmark", baseline: "", suites: ["mmlu"] },
};

function formatBytes(value: number): string {
  if (!value) return "0 B";
  if (value >= 1e12) return `${(value / 1e12).toFixed(2)} TB`;
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)} GB`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(0)} MB`;
  return `${Math.round(value)} B`;
}

function formatParams(value: number | null): string {
  if (!value) return "—";
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  return `${Math.round(value)}`;
}

function Field({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}): ReactElement {
  return (
    <div className="flex min-w-0 flex-col gap-1">
      <span className="text-ui-10 text-muted-foreground">{label}</span>
      {children}
    </div>
  );
}

const INPUT_CLASS =
  "h-8 min-w-0 rounded-lg border border-border bg-background px-2 text-sm text-foreground outline-none focus-visible:ring-1 focus-visible:ring-ring";

export function PipelinesPanel(): ReactElement {
  const t = useT();
  const [source, setSource] = useState("");
  const [name, setName] = useState("");
  const [stages, setStages] = useState<PipelineStageSpec[]>([DEFAULTS.scale]);
  const [plan, setPlan] = useState<PipelinePlan | null>(null);
  const [planning, setPlanning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState<SavedPipeline[]>([]);
  const [savedLoading, setSavedLoading] = useState(false);
  const [busy, setBusy] = useState(false);

  const refreshSaved = useCallback(() => {
    setSavedLoading(true);
    listPipelines()
      .then(setSaved)
      .catch(() => setSaved([]))
      .finally(() => setSavedLoading(false));
  }, []);

  useEffect(() => {
    refreshSaved();
  }, [refreshSaved]);

  const updateStage = (index: number, patch: Partial<PipelineStageSpec>) => {
    setStages((previous) =>
      previous.map((stage, position) =>
        position === index ? { ...stage, ...patch } : stage,
      ),
    );
  };

  const changeKind = (index: number, kind: PipelineKind) => {
    setStages((previous) =>
      previous.map((stage, position) =>
        position === index ? DEFAULTS[kind] : stage,
      ),
    );
  };

  const numberField = (index: number, key: NumericKey, raw: string) => {
    const parsed = Number(raw);
    const clean =
      raw.trim() !== "" && Number.isFinite(parsed) && parsed > 0
        ? parsed
        : undefined;
    setStages((previous) =>
      previous.map((stage, position) =>
        position === index ? { ...stage, [key]: clean } : stage,
      ),
    );
  };

  const move = (index: number, offset: number) => {
    setStages((previous) => {
      const next = [...previous];
      const target = index + offset;
      if (target < 0 || target >= next.length) return previous;
      const [item] = next.splice(index, 1);
      next.splice(target, 0, item);
      return next;
    });
  };

  const runPlan = async () => {
    setPlanning(true);
    setError(null);
    try {
      const report = await planPipeline({
        name: name.trim() || undefined,
        source: source.trim() || undefined,
        stages,
      });
      setPlan(report);
    } catch (failure) {
      setPlan(null);
      setError(
        failure instanceof Error
          ? failure.message
          : t("labs.pipelines.planFailed"),
      );
    } finally {
      setPlanning(false);
    }
  };

  const save = async () => {
    setBusy(true);
    setError(null);
    try {
      const record = await savePipeline({
        name: name.trim() || undefined,
        source: source.trim() || undefined,
        stages,
      });
      toast.success(t("labs.pipelines.saved"), {
        description: record.name,
      });
      refreshSaved();
    } catch (failure) {
      setError(
        failure instanceof Error
          ? failure.message
          : t("labs.pipelines.saveFailed"),
      );
    } finally {
      setBusy(false);
    }
  };

  const load = (record: SavedPipeline) => {
    setName(record.spec.name ?? record.name);
    setSource(record.spec.source ?? "");
    setStages(record.spec.stages ?? []);
    setPlan(null);
    setError(null);
  };

  const remove = async (record: SavedPipeline) => {
    setBusy(true);
    try {
      await deletePipeline(record.id);
      refreshSaved();
    } catch (failure) {
      toast.error(t("labs.pipelines.deleteFailed"), {
        description: failure instanceof Error ? failure.message : undefined,
      });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="grid gap-3 sm:grid-cols-2">
        <Field label={t("labs.pipelines.source")}>
          <input
            className={cn(INPUT_CLASS, "font-mono text-ui-11")}
            value={source}
            aria-label={t("labs.pipelines.source")}
            placeholder={t("labs.pipelines.sourcePlaceholder")}
            onChange={(event) => setSource(event.target.value)}
          />
        </Field>
        <Field label={t("labs.pipelines.name")}>
          <input
            className={INPUT_CLASS}
            value={name}
            aria-label={t("labs.pipelines.name")}
            placeholder={t("labs.pipelines.namePlaceholder")}
            onChange={(event) => setName(event.target.value)}
          />
        </Field>
      </div>

      <ul className="flex flex-col gap-2">
        {stages.map((stage, index) => (
          <li
            key={`${stage.kind}-${index}`}
            className="flex flex-col gap-2 rounded-2xl border border-border/60 p-3"
          >
            <div className="flex flex-wrap items-end gap-2">
              <Field label={t("labs.pipelines.stage")}>
                <select
                  className={INPUT_CLASS}
                  value={stage.kind}
                  aria-label={t("labs.pipelines.stage")}
                  onChange={(event) =>
                    changeKind(index, event.target.value as PipelineKind)
                  }
                >
                  {KINDS.map((kind) => (
                    <option key={kind} value={kind}>
                      {t(KIND_KEY[kind])}
                    </option>
                  ))}
                </select>
              </Field>

              {stage.kind === "scale" ? (
                <>
                  <Field label={t("labs.pipelines.factor")}>
                    <input
                      type="number"
                      min="1.1"
                      step="0.1"
                      aria-label={t("labs.pipelines.factor")}
                      className={cn(INPUT_CLASS, "w-24")}
                      value={stage.factor ?? ""}
                      onChange={(event) =>
                        numberField(index, "factor", event.target.value)
                      }
                    />
                  </Field>
                  <Field label={t("labs.pipelines.method")}>
                    <select
                      className={INPUT_CLASS}
                      value={stage.method ?? "depth"}
                      onChange={(event) =>
                        updateStage(index, { method: event.target.value })
                      }
                    >
                      {METHODS.map((method) => (
                        <option key={method} value={method}>
                          {method}
                        </option>
                      ))}
                    </select>
                  </Field>
                </>
              ) : null}

              {stage.kind === "merge" ? (
                <Field label={t("labs.pipelines.parents")}>
                  <input
                    className={cn(INPUT_CLASS, "min-w-64 font-mono text-ui-11")}
                    value={(stage.parents ?? []).join(", ")}
                    placeholder="/path/one, /path/two"
                    onChange={(event) =>
                      updateStage(index, {
                        parents: event.target.value
                          .split(",")
                          .map((item) => item.trim())
                          .filter(Boolean),
                      })
                    }
                  />
                </Field>
              ) : null}

              {stage.kind === "train" ? (
                <>
                  <Field label={t("labs.pipelines.steps")}>
                    <input
                      type="number"
                      min="1"
                      aria-label={t("labs.pipelines.steps")}
                      className={cn(INPUT_CLASS, "w-20")}
                      value={stage.steps ?? ""}
                      onChange={(event) =>
                        numberField(index, "steps", event.target.value)
                      }
                    />
                  </Field>
                  <Field label={t("labs.pipelines.batch")}>
                    <input
                      type="number"
                      min="1"
                      aria-label={t("labs.pipelines.batch")}
                      className={cn(INPUT_CLASS, "w-16")}
                      value={stage.batch_size ?? ""}
                      onChange={(event) =>
                        numberField(index, "batch_size", event.target.value)
                      }
                    />
                  </Field>
                  <Field label={t("labs.pipelines.sequence")}>
                    <input
                      type="number"
                      min="128"
                      step="128"
                      aria-label={t("labs.pipelines.sequence")}
                      className={cn(INPUT_CLASS, "w-24")}
                      value={stage.sequence_length ?? ""}
                      onChange={(event) =>
                        numberField(
                          index,
                          "sequence_length",
                          event.target.value,
                        )
                      }
                    />
                  </Field>
                  <Field label={t("labs.pipelines.dataset")}>
                    <input
                      className={cn(INPUT_CLASS, "font-mono text-ui-11")}
                      value={stage.dataset ?? ""}
                      placeholder="user/dataset"
                      onChange={(event) =>
                        updateStage(index, { dataset: event.target.value })
                      }
                    />
                  </Field>
                </>
              ) : null}

              {stage.kind === "quantize" ? (
                <Field label={t("labs.pipelines.level")}>
                  <select
                    className={INPUT_CLASS}
                    value={stage.level ?? "q4_k_m"}
                    onChange={(event) =>
                      updateStage(index, { level: event.target.value })
                    }
                  >
                    {LEVELS.map((level) => (
                      <option key={level} value={level}>
                        {level}
                      </option>
                    ))}
                  </select>
                </Field>
              ) : null}

              {stage.kind === "benchmark" ? (
                <>
                  <Field label={t("labs.pipelines.baseline")}>
                    <input
                      className={cn(
                        INPUT_CLASS,
                        "min-w-56 font-mono text-ui-11",
                      )}
                      value={stage.baseline ?? ""}
                      placeholder={t("labs.pipelines.baselinePlaceholder")}
                      onChange={(event) =>
                        updateStage(index, { baseline: event.target.value })
                      }
                    />
                  </Field>
                  <Field label={t("labs.pipelines.suites")}>
                    <input
                      className={cn(INPUT_CLASS, "w-40")}
                      value={(stage.suites ?? []).join(", ")}
                      placeholder="mmlu, gsm8k"
                      onChange={(event) =>
                        updateStage(index, {
                          suites: event.target.value
                            .split(",")
                            .map((item) => item.trim())
                            .filter(Boolean),
                        })
                      }
                    />
                  </Field>
                </>
              ) : null}

              <div className="ml-auto flex items-center gap-1">
                <Button
                  size="icon-xs"
                  variant="ghost"
                  aria-label={t("labs.pipelines.moveUp")}
                  disabled={index === 0}
                  onClick={() => move(index, -1)}
                >
                  ↑
                </Button>
                <Button
                  size="icon-xs"
                  variant="ghost"
                  aria-label={t("labs.pipelines.moveDown")}
                  disabled={index === stages.length - 1}
                  onClick={() => move(index, 1)}
                >
                  ↓
                </Button>
                <Button
                  size="icon-xs"
                  variant="ghost"
                  aria-label={t("labs.pipelines.remove")}
                  disabled={stages.length === 1}
                  onClick={() =>
                    setStages((previous) =>
                      previous.filter((_, position) => position !== index),
                    )
                  }
                >
                  <HugeiconsIcon icon={Cancel01Icon} strokeWidth={1.75} />
                </Button>
              </div>
            </div>
          </li>
        ))}
      </ul>

      <div className="flex flex-wrap items-center gap-2">
        <Button
          size="sm"
          variant="outline"
          onClick={() => setStages((previous) => [...previous, DEFAULTS.scale])}
        >
          <HugeiconsIcon icon={PlusSignIcon} strokeWidth={1.75} />
          {t("labs.pipelines.addStage")}
        </Button>
        <Button size="sm" disabled={planning} onClick={() => void runPlan()}>
          {planning ? <Spinner /> : null}
          {planning ? t("labs.pipelines.planning") : t("labs.pipelines.plan")}
        </Button>
        <Button
          size="sm"
          variant="outline"
          disabled={busy || planning}
          onClick={() => void save()}
        >
          {t("labs.pipelines.save")}
        </Button>
        <span className="text-ui-10 text-muted-foreground">
          {t("labs.pipelines.planOnly")}
        </span>
      </div>

      {error ? (
        <p className="flex items-start gap-2 text-ui-11 text-destructive">
          <HugeiconsIcon
            icon={Alert02Icon}
            strokeWidth={1.75}
            className="mt-px size-3.5 shrink-0"
          />
          {error}
        </p>
      ) : null}

      {plan ? (
        <div className="flex flex-col gap-3 rounded-2xl border border-border/60 p-4">
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <div>
              <p className="text-ui-10 text-muted-foreground">
                {t("labs.pipelines.finalParams")}
              </p>
              <p className="font-mono text-sm text-foreground">
                {formatParams(plan.final_params)}
              </p>
            </div>
            <div>
              <p className="text-ui-10 text-muted-foreground">
                {t("labs.pipelines.totalOutput")}
              </p>
              <p className="font-mono text-sm text-foreground">
                {formatBytes(plan.total_output_bytes)}
              </p>
            </div>
            <div>
              <p className="text-ui-10 text-muted-foreground">
                {t("labs.pipelines.peakStage")}
              </p>
              <p className="font-mono text-sm text-foreground">
                {formatBytes(plan.peak_stage_bytes)}
              </p>
            </div>
            <div>
              <p className="text-ui-10 text-muted-foreground">
                {t("labs.pipelines.verdict")}
              </p>
              <p
                className={cn(
                  "text-sm font-medium",
                  plan.blocked
                    ? "text-destructive"
                    : "text-emerald-600 dark:text-emerald-400",
                )}
              >
                {plan.blocked
                  ? t("labs.pipelines.blocked")
                  : t("labs.pipelines.runnable")}
              </p>
            </div>
          </div>

          {plan.blocked_reason ? (
            <p className="rounded-xl border border-destructive/40 bg-destructive/5 p-2.5 text-ui-11 text-foreground">
              {plan.blocked_reason}
            </p>
          ) : null}

          <ul className="flex flex-col gap-2">
            {plan.stages.map((stage) => (
              <li
                key={`${stage.kind}-${stage.index}`}
                className="flex flex-col gap-1 rounded-xl border border-border/50 p-2.5"
              >
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-ui-11 font-medium text-foreground">
                    {stage.index + 1}. {stage.label}
                  </span>
                  <span className="font-mono text-ui-10 text-muted-foreground">
                    {formatParams(stage.params_before)} →{" "}
                    {formatParams(stage.params_after)}
                  </span>
                  <span className="ml-auto font-mono text-ui-10 text-muted-foreground">
                    {formatBytes(stage.output_bytes)}
                  </span>
                </div>
                {stage.blocked ? (
                  <p className="text-ui-10 text-destructive">{stage.reason}</p>
                ) : null}
                {stage.warnings.map((warning) => (
                  <p
                    key={warning}
                    className="flex items-start gap-1.5 text-ui-10 leading-snug text-muted-foreground"
                  >
                    <HugeiconsIcon
                      icon={Alert02Icon}
                      strokeWidth={1.75}
                      className="mt-px size-3 shrink-0 text-amber-600 dark:text-amber-400"
                    />
                    {warning}
                  </p>
                ))}
              </li>
            ))}
          </ul>

          {plan.warnings.length > 0 ? (
            <div className="text-ui-10 leading-relaxed text-muted-foreground">
              {plan.warnings.map((warning) => (
                <p key={warning}>{warning}</p>
              ))}
            </div>
          ) : null}
          {plan.notes.map((note) => (
            <p key={note} className="text-ui-10 text-muted-foreground/80">
              {note}
            </p>
          ))}
        </div>
      ) : null}

      <div className="flex flex-col gap-2 rounded-2xl border border-border/60 p-4">
        <div className="flex items-center gap-2">
          <p className="text-ui-11 font-medium text-foreground">
            {t("labs.pipelines.savedPlans")}
          </p>
          <Button
            size="icon-xs"
            variant="ghost"
            className="ml-auto"
            aria-label={t("labs.refresh")}
            onClick={refreshSaved}
          >
            <HugeiconsIcon
              icon={savedLoading ? Tick02Icon : Refresh01Icon}
              strokeWidth={1.75}
            />
          </Button>
        </div>
        {saved.length === 0 ? (
          <p className="text-ui-11 text-muted-foreground">
            {t("labs.pipelines.noSaved")}
          </p>
        ) : (
          <ul className="flex flex-col gap-1.5">
            {saved.map((record) => (
              <li
                key={record.id}
                className="flex flex-wrap items-center gap-2 text-ui-11"
              >
                <span className="min-w-0 truncate font-medium text-foreground">
                  {record.name}
                </span>
                <span className="font-mono text-ui-10 text-muted-foreground">
                  {record.plan.stage_kinds.join(" → ")}
                </span>
                {record.plan.blocked ? (
                  <Badge
                    variant="outline"
                    className="h-5 border-destructive/40 px-1.5 text-ui-10 text-destructive"
                  >
                    blocked
                  </Badge>
                ) : null}
                <div className="ml-auto flex items-center gap-1">
                  <Button
                    size="xs"
                    variant="ghost"
                    onClick={() => load(record)}
                  >
                    {t("labs.pipelines.load")}
                  </Button>
                  <Button
                    size="icon-xs"
                    variant="ghost"
                    aria-label={t("labs.pipelines.delete")}
                    disabled={busy}
                    onClick={() => void remove(record)}
                  >
                    <HugeiconsIcon icon={Delete02Icon} strokeWidth={1.75} />
                  </Button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
