// SPDX-License-Identifier: NRC-2
// Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-2.md

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { fetchJob } from "@/features/advanced/api/advanced-api";
import { useTrainingConfigStore } from "@/features/training";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  Alert02Icon,
  ArrowDown01Icon,
  Delete02Icon,
  PlusSignIcon,
  PlayIcon,
  Refresh01Icon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type ReactElement,
  useCallback,
  useEffect,
  useState,
} from "react";
import {
  type CmtPlan,
  type CmtSpec,
  type LabsJob,
  planCmt,
  runCmt,
} from "../api";

const INPUT_CLASS =
  "h-8 min-w-0 rounded-lg border border-border bg-background px-2 text-sm text-foreground outline-none focus-visible:ring-1 focus-visible:ring-ring";

interface DatasetRow {
  id: string;
  path: string;
  name: string;
  evalPercent: string;
  textColumn: string;
}

interface CmtResult {
  output_dir?: string;
  manifest_path?: string;
  expanded_dir?: string;
  mode?: string;
  base_params?: number;
  final_params?: number;
  added_params?: number;
  growth_factor?: number;
  new_layers?: number;
  per_dataset?: Record<
    string,
    {
      name: string;
      eval_rows: number;
      base_score: number | null;
      best_score: number | null;
      delta: number | null;
    }
  >;
  training?: {
    steps_run?: number | null;
    tokens_seen?: number | null;
    gpu_hours?: number | null;
    peak_mem_gb?: number | null;
    recovery_step?: number | null;
    new_layer_grad_nonzero?: boolean | null;
    final_checkpoint?: string | null;
    best_checkpoint?: string | null;
  };
  warnings?: string[];
}

function params(value: number | null | undefined): string {
  if (!value) return "—";
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return `${Math.round(value)}`;
}

function newRow(): DatasetRow {
  return {
    id: crypto.randomUUID(),
    path: "",
    name: "",
    evalPercent: "10",
    textColumn: "",
  };
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForCmt(
  job: LabsJob,
  onProgress: (message: string) => void,
): Promise<CmtResult> {
  for (let attempt = 0; attempt < 21_600; attempt += 1) {
    const current = await fetchJob(job.job_id);
    const progress = current.progress;
    onProgress(
      String(
        progress?.message ??
          progress?.stage ??
          (current.status === "running" ? "Running…" : current.status),
      ),
    );
    if (current.status === "completed") {
      return (current.result ?? {}) as CmtResult;
    }
    if (current.status === "failed") {
      throw new Error(current.error || "The CMT job failed.");
    }
    await sleep(2000);
  }
  throw new Error("The CMT job did not finish in twelve hours.");
}

export function CmtPanel(): ReactElement {
  const t = useT();
  const selectedModel = useTrainingConfigStore((state) => state.selectedModel);
  const [source, setSource] = useState(selectedModel ?? "");
  const [rows, setRows] = useState<DatasetRow[]>([newRow(), newRow()]);
  const [mode, setMode] = useState<"cpt" | "finetune">("cpt");
  const [added, setAdded] = useState("");
  const [steps, setSteps] = useState("100");
  const [batch, setBatch] = useState("1");
  const [accum, setAccum] = useState("8");
  const [sequence, setSequence] = useState("2048");
  const [learningRate, setLearningRate] = useState("0.0002");
  const [evalEvery, setEvalEvery] = useState("20");
  const [patience, setPatience] = useState("4");
  const [plan, setPlan] = useState<CmtPlan | null>(null);
  const [planning, setPlanning] = useState(false);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [result, setResult] = useState<CmtResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!source && selectedModel) setSource(selectedModel);
  }, [selectedModel, source]);

  const updateRow = (id: string, patch: Partial<DatasetRow>) => {
    setRows((current) =>
      current.map((row) => (row.id === id ? { ...row, ...patch } : row)),
    );
    setPlan(null);
  };

  const buildSpec = useCallback((): CmtSpec => {
    const datasets = rows
      .filter((row) => row.path.trim() !== "")
      .map((row) => ({
        path: row.path.trim(),
        name: row.name.trim() || undefined,
        text_column: row.textColumn.trim() || null,
        eval_fraction: Math.max(
          0,
          Math.min(0.5, (Number(row.evalPercent) || 0) / 100),
        ),
      }));
    return {
      source: source.trim(),
      datasets,
      mode,
      added_params: added.trim() === "" ? null : Number(added),
      max_steps: Number(steps) || 100,
      batch_size: Number(batch) || 1,
      gradient_accumulation: Number(accum) || 8,
      max_seq_length: Number(sequence) || 2048,
      learning_rate: Number(learningRate) || 2e-4,
      eval_every: Number(evalEvery) || 20,
      patience: Number(patience) || 4,
    };
  }, [
    accum,
    added,
    batch,
    evalEvery,
    learningRate,
    mode,
    patience,
    rows,
    sequence,
    source,
    steps,
  ]);

  const makePlan = async () => {
    setPlanning(true);
    setError(null);
    setPlan(null);
    try {
      const spec = buildSpec();
      if (!spec.source) throw new Error(t("labs.cmt.needModel"));
      if (spec.datasets.length < 2) {
        throw new Error(t("labs.cmt.needDatasets"));
      }
      setPlan(await planCmt(spec));
    } catch (failure) {
      setError(
        failure instanceof Error ? failure.message : t("labs.cmt.planFailed"),
      );
    } finally {
      setPlanning(false);
    }
  };

  const start = async () => {
    if (!plan?.runnable) return;
    setRunning(true);
    setError(null);
    setResult(null);
    try {
      const spec = buildSpec();
      const job = await runCmt(spec);
      setStatus(t("labs.cmt.starting"));
      const finished = await waitForCmt(job, setStatus);
      setResult(finished);
      toast.success(t("labs.cmt.completed"));
    } catch (failure) {
      const message =
        failure instanceof Error ? failure.message : t("labs.cmt.failed");
      setError(message);
      toast.error(t("labs.cmt.failed"), { description: message });
    } finally {
      setRunning(false);
      setStatus(null);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <section className="elevated-card flex flex-col gap-4 p-5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="text-ui-14 font-semibold text-foreground">
                {t("labs.cmt.title")}
              </h2>
              <Badge variant="outline" className="text-ui-9">
                {t("labs.experimental")}
              </Badge>
            </div>
            <p className="mt-1 max-w-3xl text-ui-11 leading-relaxed text-muted-foreground">
              {t("labs.cmt.description")}
            </p>
          </div>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => {
              setPlan(null);
              setResult(null);
              setError(null);
            }}
          >
            <HugeiconsIcon icon={Refresh01Icon} className="size-3.5" />
            {t("labs.cmt.reset")}
          </Button>
        </div>

        <div className="grid gap-3 @2xl/train-card:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
          <label className="flex min-w-0 flex-col gap-1.5">
            <span className="text-ui-10 uppercase tracking-[0.05em] text-muted-foreground">
              {t("labs.cmt.source")}
            </span>
            <input
              className={INPUT_CLASS}
              value={source}
              onChange={(event) => {
                setSource(event.target.value);
                setPlan(null);
              }}
              placeholder={t("labs.cmt.sourcePlaceholder")}
            />
            <span className="text-ui-10 text-muted-foreground/70">
              {t("labs.cmt.sourceHint")}
            </span>
          </label>
          <div className="flex min-w-0 flex-col gap-1.5">
            <span className="text-ui-10 uppercase tracking-[0.05em] text-muted-foreground">
              {t("labs.cmt.mode")}
            </span>
            <div className="flex gap-1.5">
              {(
                [
                  ["cpt", t("labs.cmt.modeCpt")],
                  ["finetune", t("labs.cmt.modeFinetune")],
                ] as const
              ).map(([value, label]) => (
                <button
                  key={value}
                  type="button"
                  aria-pressed={mode === value}
                  onClick={() => {
                    setMode(value);
                    setPlan(null);
                  }}
                  className={cn(
                    "rounded-full border px-3 py-1 text-ui-11 transition-colors",
                    mode === value
                      ? "border-border bg-muted text-foreground"
                      : "border-border/60 text-muted-foreground hover:text-foreground",
                  )}
                >
                  {label}
                </button>
              ))}
            </div>
            <span className="text-ui-10 text-muted-foreground/70">
              {mode === "cpt"
                ? t("labs.cmt.modeCptHint")
                : t("labs.cmt.modeFinetuneHint")}
            </span>
          </div>
        </div>

        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between gap-2">
            <span className="text-ui-10 uppercase tracking-[0.05em] text-muted-foreground">
              {t("labs.cmt.datasets")}
            </span>
            <Button
              size="sm"
              variant="outline"
              onClick={() => setRows((current) => [...current, newRow()])}
              disabled={rows.length >= 8}
            >
              <HugeiconsIcon icon={PlusSignIcon} className="size-3.5" />
              {t("labs.cmt.addDataset")}
            </Button>
          </div>
          <div className="flex flex-col gap-2">
            {rows.map((row, index) => (
              <div
                key={row.id}
                className="grid gap-2 rounded-lg border border-border/70 bg-background/40 p-2.5 @2xl/train-card:grid-cols-[minmax(0,2fr)_minmax(0,1fr)_5rem_5rem_2rem]"
              >
                <input
                  className={INPUT_CLASS}
                  value={row.path}
                  onChange={(event) =>
                    updateRow(row.id, { path: event.target.value })
                  }
                  placeholder={t("labs.cmt.datasetPath", {
                    index: index + 1,
                  })}
                />
                <input
                  className={INPUT_CLASS}
                  value={row.name}
                  onChange={(event) =>
                    updateRow(row.id, { name: event.target.value })
                  }
                  placeholder={t("labs.cmt.datasetName")}
                />
                <label className="flex items-center gap-1">
                  <input
                    className={cn(INPUT_CLASS, "w-full")}
                    inputMode="decimal"
                    value={row.evalPercent}
                    onChange={(event) =>
                      updateRow(row.id, { evalPercent: event.target.value })
                    }
                    aria-label={t("labs.cmt.evalPercent")}
                  />
                  <span className="text-ui-10 text-muted-foreground">%</span>
                </label>
                <input
                  className={INPUT_CLASS}
                  value={row.textColumn}
                  onChange={(event) =>
                    updateRow(row.id, { textColumn: event.target.value })
                  }
                  placeholder={t("labs.cmt.textColumn")}
                />
                <Button
                  size="icon-sm"
                  variant="ghost"
                  aria-label={t("labs.cmt.removeDataset")}
                  disabled={rows.length <= 2}
                  onClick={() =>
                    setRows((current) =>
                      current.filter((item) => item.id !== row.id),
                    )
                  }
                >
                  <HugeiconsIcon icon={Delete02Icon} className="size-3.5" />
                </Button>
              </div>
            ))}
          </div>
          <p className="text-ui-10 text-muted-foreground/70">
            {t("labs.cmt.datasetHint")}
          </p>
        </div>

        <div className="grid gap-3 sm:grid-cols-3 @3xl/train-card:grid-cols-6">
          {(
            [
              [t("labs.cmt.added"), added, setAdded, "4.6M"],
              [t("labs.cmt.steps"), steps, setSteps, "100"],
              [t("labs.cmt.batch"), batch, setBatch, "1"],
              [t("labs.cmt.accum"), accum, setAccum, "8"],
              [t("labs.cmt.sequence"), sequence, setSequence, "2048"],
              [t("labs.cmt.lr"), learningRate, setLearningRate, "0.0002"],
            ] as const
          ).map(([label, value, setter, placeholder]) => (
            <label key={label} className="flex min-w-0 flex-col gap-1">
              <span className="text-ui-10 text-muted-foreground">{label}</span>
              <input
                className={INPUT_CLASS}
                inputMode="decimal"
                value={value}
                onChange={(event) => {
                  setter(event.target.value);
                  setPlan(null);
                }}
                placeholder={placeholder}
              />
            </label>
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Button
            size="sm"
            variant="outline"
            onClick={() => void makePlan()}
            disabled={planning || running}
          >
            {planning ? (
              <Spinner />
            ) : (
              <HugeiconsIcon icon={ArrowDown01Icon} className="size-3.5" />
            )}
            {planning ? t("labs.cmt.planning") : t("labs.cmt.plan")}
          </Button>
          <Button
            size="sm"
            onClick={() => void start()}
            disabled={!plan?.runnable || running || planning}
          >
            {running ? (
              <Spinner />
            ) : (
              <HugeiconsIcon icon={PlayIcon} className="size-3.5" />
            )}
            {running ? t("labs.cmt.running") : t("labs.cmt.run")}
          </Button>
          <span className="text-ui-10 text-muted-foreground/70">
            {t("labs.cmt.oneJob")}
          </span>
        </div>

        {status ? (
          <div className="flex items-center gap-2 rounded-lg border border-border/70 bg-muted/30 px-3 py-2 text-ui-11 text-muted-foreground">
            <Spinner />
            {status}
          </div>
        ) : null}
        {error ? (
          <p className="whitespace-pre-wrap text-ui-11 text-destructive">
            {error}
          </p>
        ) : null}
      </section>

      {plan ? (
        <section className="elevated-card flex flex-col gap-3 p-5">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="text-ui-13 font-semibold text-foreground">
              {t("labs.cmt.planTitle")}
            </h3>
            <Badge
              variant="outline"
              className={cn(
                plan.runnable
                  ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
                  : "border-destructive/30 bg-destructive/10 text-destructive",
              )}
            >
              {plan.runnable
                ? t("labs.cmt.runnable")
                : t("labs.cmt.blocked")}
            </Badge>
            {plan.expansion.floor_bypassed ? (
              <Badge variant="outline" className="text-ui-9">
                {t("labs.cmt.floorBypassed")}
              </Badge>
            ) : null}
          </div>

          {plan.blocked_reason ? (
            <p className="rounded-lg border border-destructive/30 bg-destructive/5 p-3 text-ui-11 text-destructive">
              {plan.blocked_reason}
            </p>
          ) : null}

          <div className="grid gap-2 sm:grid-cols-2 @3xl/train-card:grid-cols-4">
            {[
              [
                t("labs.cmt.baseParams"),
                params(plan.expansion.base_params),
              ],
              [t("labs.cmt.addedParams"), plan.expansion.added_label],
              [t("labs.cmt.targetParams"), plan.expansion.target_label],
              [
                t("labs.cmt.newLayers"),
                `${plan.expansion.new_layers} · ${params(
                  plan.expansion.params_per_layer,
                )}`,
              ],
            ].map(([label, value]) => (
              <div
                key={label}
                className="rounded-lg border border-border/70 bg-background/40 px-3 py-2"
              >
                <div className="text-ui-10 uppercase tracking-[0.05em] text-muted-foreground">
                  {label}
                </div>
                <div className="mt-0.5 font-mono text-ui-13 font-semibold text-foreground">
                  {value}
                </div>
              </div>
            ))}
          </div>

          <div className="overflow-x-auto rounded-lg border border-border/70">
            <table className="w-full text-ui-11">
              <thead>
                <tr className="text-left text-ui-10 uppercase tracking-[0.04em] text-muted-foreground">
                  <th className="px-3 py-2 font-medium">
                    {t("labs.cmt.dataset")}
                  </th>
                  <th className="px-3 py-2 text-right font-medium">
                    {t("labs.cmt.rows")}
                  </th>
                  <th className="px-3 py-2 text-right font-medium">
                    {t("labs.cmt.trainRows")}
                  </th>
                  <th className="px-3 py-2 text-right font-medium">
                    {t("labs.cmt.evalRows")}
                  </th>
                  <th className="px-3 py-2 text-right font-medium">
                    {t("labs.cmt.tokens")}
                  </th>
                </tr>
              </thead>
              <tbody>
                {plan.datasets.map((dataset) => (
                  <tr
                    key={dataset.path}
                    className="border-t border-border/50"
                  >
                    <td className="max-w-[22rem] truncate px-3 py-1.5">
                      <span className="text-foreground">{dataset.name}</span>
                      <span className="ml-2 font-mono text-ui-9 text-muted-foreground">
                        {dataset.path}
                      </span>
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono">
                      {dataset.rows.toLocaleString()}
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono">
                      {dataset.train_rows.toLocaleString()}
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono">
                      {dataset.eval_rows.toLocaleString()}
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono">
                      {params(dataset.tokens_estimate)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="flex flex-wrap gap-x-4 gap-y-1 text-ui-10 text-muted-foreground">
            <span>
              {t("labs.cmt.tokensTotal")}: {params(plan.totals.tokens_estimate)}
            </span>
            <span>
              {t("labs.cmt.suggestedSteps")}:{" "}
              {plan.training.suggested_steps.toLocaleString()}
            </span>
            <span>{plan.sizing.note}</span>
          </div>

          {plan.warnings.length > 0 ? (
            <ul className="flex flex-col gap-1 rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-ui-10 text-amber-700 dark:text-amber-300">
              {plan.warnings.map((warning) => (
                <li key={warning} className="flex gap-1.5">
                  <HugeiconsIcon
                    icon={Alert02Icon}
                    className="mt-0.5 size-3 shrink-0"
                  />
                  <span>{warning}</span>
                </li>
              ))}
            </ul>
          ) : null}
        </section>
      ) : null}

      {result ? (
        <section className="elevated-card flex flex-col gap-3 p-5">
          <div className="flex flex-wrap items-center gap-2">
            <HugeiconsIcon
              icon={Tick02Icon}
              className="size-4 text-emerald-600 dark:text-emerald-400"
            />
            <h3 className="text-ui-13 font-semibold text-foreground">
              {t("labs.cmt.resultTitle")}
            </h3>
            <Badge variant="outline" className="text-ui-9">
              {result.new_layers ?? 0} {t("labs.cmt.newLayers")}
            </Badge>
            <span className="truncate font-mono text-ui-10 text-muted-foreground">
              {result.output_dir}
            </span>
          </div>

          <div className="grid gap-2 sm:grid-cols-2 @3xl/train-card:grid-cols-4">
            {[
              [t("labs.cmt.baseParams"), params(result.base_params)],
              [t("labs.cmt.addedParams"), params(result.added_params)],
              [t("labs.cmt.finalParams"), params(result.final_params)],
              [
                t("labs.cmt.growth"),
                result.growth_factor
                  ? `${result.growth_factor.toFixed(3)}×`
                  : "—",
              ],
            ].map(([label, value]) => (
              <div
                key={label}
                className="rounded-lg border border-border/70 bg-background/40 px-3 py-2"
              >
                <div className="text-ui-10 uppercase tracking-[0.05em] text-muted-foreground">
                  {label}
                </div>
                <div className="mt-0.5 font-mono text-ui-13 font-semibold text-foreground">
                  {value}
                </div>
              </div>
            ))}
          </div>

          <div className="overflow-x-auto rounded-lg border border-border/70">
            <table className="w-full text-ui-11">
              <thead>
                <tr className="text-left text-ui-10 uppercase tracking-[0.04em] text-muted-foreground">
                  <th className="px-3 py-2 font-medium">
                    {t("labs.cmt.dataset")}
                  </th>
                  <th className="px-3 py-2 text-right font-medium">
                    {t("labs.cmt.before")}
                  </th>
                  <th className="px-3 py-2 text-right font-medium">
                    {t("labs.cmt.after")}
                  </th>
                  <th className="px-3 py-2 text-right font-medium">
                    {t("labs.cmt.delta")}
                  </th>
                </tr>
              </thead>
              <tbody>
                {Object.values(result.per_dataset ?? {}).map((entry) => (
                  <tr key={entry.name} className="border-t border-border/50">
                    <td className="px-3 py-1.5 font-mono text-ui-10">
                      {entry.name}
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono">
                      {entry.base_score?.toFixed(3) ?? "—"}
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono">
                      {entry.best_score?.toFixed(3) ?? "—"}
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono">
                      {entry.delta === null || entry.delta === undefined
                        ? "—"
                        : `${entry.delta > 0 ? "+" : ""}${entry.delta.toFixed(
                            3,
                          )}`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="flex flex-wrap gap-x-4 gap-y-1 text-ui-10 text-muted-foreground">
            <span>
              {t("labs.cmt.stepsRun")}: {result.training?.steps_run ?? "—"}
            </span>
            <span>
              {t("labs.cmt.tokensSeen")}:{" "}
              {params(result.training?.tokens_seen)}
            </span>
            <span>
              {t("labs.cmt.gpuHours")}:{" "}
              {result.training?.gpu_hours !== null &&
              result.training?.gpu_hours !== undefined
                ? result.training.gpu_hours.toFixed(2)
                : "—"}
            </span>
            <span>
              {t("labs.cmt.peakMemory")}:{" "}
              {result.training?.peak_mem_gb
                ? `${result.training.peak_mem_gb.toFixed(1)} GB`
                : "—"}
            </span>
            <span className="truncate font-mono">
              {result.training?.final_checkpoint ??
                result.training?.best_checkpoint ??
                ""}
            </span>
          </div>

          {result.warnings && result.warnings.length > 0 ? (
            <ul className="flex flex-col gap-1 rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-ui-10 text-amber-700 dark:text-amber-300">
              {result.warnings.map((warning) => (
                <li key={warning} className="flex gap-1.5">
                  <HugeiconsIcon
                    icon={Alert02Icon}
                    className="mt-0.5 size-3 shrink-0"
                  />
                  <span>{warning}</span>
                </li>
              ))}
            </ul>
          ) : null}
        </section>
      ) : null}
    </div>
  );
}
