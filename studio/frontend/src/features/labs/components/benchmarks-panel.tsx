// SPDX-License-Identifier: NRC-1
// Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-1.md

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import {
  type AdvancedCapabilities,
  fetchAdvancedCapabilities,
  fetchJob,
} from "@/features/advanced/api/advanced-api";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  Alert02Icon,
  Delete02Icon,
  PlayIcon,
  Refresh01Icon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type ReactElement,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";
import {
  type Artifact,
  type BenchmarkRecord,
  deleteBenchmark,
  loadArtifacts,
  loadBenchmarks,
  runBenchmark,
} from "../api";

const NOISE = 0.015;
const INPUT_CLASS =
  "h-8 min-w-0 rounded-lg border border-border bg-background px-2 text-sm text-foreground outline-none focus-visible:ring-1 focus-visible:ring-ring";

function percent(value: number | undefined): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(1)}`;
}

function deltaLabel(value: number | null | undefined): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "—";
  const points = value * 100;
  if (Math.abs(points) < NOISE * 100) return "within noise";
  return `${points > 0 ? "+" : ""}${points.toFixed(1)} pts`;
}

function verdictClass(verdict: string): string {
  if (verdict === "pareto-positive") {
    return "border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300";
  }
  if (verdict === "regressed") {
    return "border-destructive/30 bg-destructive/10 text-destructive";
  }
  return "border-border bg-muted/40 text-muted-foreground";
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForJob(
  jobId: string,
  onProgress: (message: string) => void,
): Promise<Record<string, unknown>> {
  for (let attempt = 0; attempt < 7200; attempt += 1) {
    const job = await fetchJob(jobId);
    const message = job.progress?.message ?? job.progress?.stage ?? "Running…";
    onProgress(String(message));
    if (job.status === "completed") return job.result ?? {};
    if (job.status === "failed") {
      throw new Error(job.error || "The benchmark job failed.");
    }
    await sleep(2000);
  }
  throw new Error("The benchmark job did not finish in four hours.");
}

export function BenchmarksPanel(): ReactElement {
  const t = useT();
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [capabilities, setCapabilities] =
    useState<AdvancedCapabilities | null>(null);
  const [records, setRecords] = useState<BenchmarkRecord[]>([]);
  const [baseline, setBaseline] = useState("");
  const [customBaseline, setCustomBaseline] = useState("");
  const [selected, setSelected] = useState<string[]>([]);
  const [suites, setSuites] = useState<string[]>(["mmlu", "gsm8k"]);
  const [limit, setLimit] = useState("200");
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refreshRecords = useCallback(async () => {
    try {
      setRecords(await loadBenchmarks(100));
    } catch (failure) {
      setError(
        failure instanceof Error
          ? failure.message
          : "Benchmark records could not be read.",
      );
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    Promise.all([
      loadArtifacts(100, controller.signal),
      fetchAdvancedCapabilities(),
      loadBenchmarks(100, undefined, controller.signal),
    ])
      .then(([artifactsResult, capabilityResult, recordResult]) => {
        setArtifacts(artifactsResult.artifacts);
        setCapabilities(capabilityResult);
        setRecords(recordResult);
        const first = artifactsResult.artifacts[0]?.path;
        if (first) setBaseline((current) => current || first);
      })
      .catch((failure: unknown) => {
        if (controller.signal.aborted) return;
        setError(
          failure instanceof Error
            ? failure.message
            : "The benchmark workspace could not be loaded.",
        );
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, []);

  const suiteOptions = useMemo(
    () =>
      (capabilities?.eval_suites ?? []).filter(
        (suite) => suite.id !== "custom",
      ),
    [capabilities],
  );

  const baselineValue = (customBaseline.trim() || baseline).trim();
  const canRun =
    !running &&
    Boolean(baselineValue) &&
    selected.length > 0 &&
    suites.length > 0 &&
    Boolean(capabilities?.eval_harness.available);

  const toggleSuite = (id: string) => {
    setSuites((current) =>
      current.includes(id)
        ? current.filter((suite) => suite !== id)
        : [...current, id],
    );
  };

  const run = async () => {
    if (!canRun) return;
    setRunning(true);
    setError(null);
    try {
      for (let index = 0; index < selected.length; index += 1) {
        const candidate = selected[index];
        const label = `${index + 1}/${selected.length}`;
        setStatus(`${label} · ${candidate.split(/[\\/]/).pop() ?? candidate}`);
        const job = await runBenchmark({
          artifact: candidate,
          baseline: baselineValue,
          suites,
          limit: limit.trim() === "" ? null : Number(limit),
          batch_size: 4,
          timeout_seconds: 3600,
        });
        await waitForJob(job.job_id, (message) =>
          setStatus(`${label} · ${message}`),
        );
        await refreshRecords();
      }
      toast.success(t("labs.benchmarks.completed"));
    } catch (failure) {
      const message =
        failure instanceof Error
          ? failure.message
          : t("labs.benchmarks.failed");
      setError(message);
      toast.error(t("labs.benchmarks.failed"), { description: message });
    } finally {
      setRunning(false);
      setStatus(null);
    }
  };

  const remove = async (id: string) => {
    try {
      await deleteBenchmark(id);
      setRecords((current) => current.filter((record) => record.id !== id));
    } catch (failure) {
      toast.error(t("labs.benchmarks.deleteFailed"), {
        description: failure instanceof Error ? failure.message : undefined,
      });
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <section className="elevated-card flex flex-col gap-4 p-5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            <h2 className="text-ui-14 font-semibold text-foreground">
              {t("labs.benchmarks.title")}
            </h2>
            <p className="mt-1 max-w-3xl text-ui-11 leading-relaxed text-muted-foreground">
              {t("labs.benchmarks.description")}
            </p>
          </div>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => void refreshRecords()}
          >
            <HugeiconsIcon icon={Refresh01Icon} className="size-3.5" />
            {t("labs.refresh")}
          </Button>
        </div>

        {loading ? (
          <div className="flex items-center gap-2 text-ui-11 text-muted-foreground">
            <Spinner />
            {t("labs.loading")}
          </div>
        ) : null}

        {capabilities && !capabilities.eval_harness.available ? (
          <div className="flex items-start gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-ui-11 text-amber-700 dark:text-amber-300">
            <HugeiconsIcon icon={Alert02Icon} className="mt-0.5 size-3.5" />
            <span>
              {t("labs.benchmarks.harnessMissing")}{" "}
              <code className="font-mono">
                {capabilities.eval_harness.install_hint}
              </code>
            </span>
          </div>
        ) : null}

        <div className="grid gap-4 @2xl/train-card:grid-cols-2">
          <label className="flex min-w-0 flex-col gap-1.5">
            <span className="text-ui-10 uppercase tracking-[0.05em] text-muted-foreground">
              {t("labs.benchmarks.baseline")}
            </span>
            <select
              className={INPUT_CLASS}
              value={baseline}
              onChange={(event) => setBaseline(event.target.value)}
            >
              <option value="">{t("labs.benchmarks.pickBaseline")}</option>
              {artifacts.map((artifact) => (
                <option key={artifact.path} value={artifact.path}>
                  {artifact.name}
                </option>
              ))}
            </select>
            <input
              className={INPUT_CLASS}
              placeholder={t("labs.benchmarks.baselinePlaceholder")}
              value={customBaseline}
              onChange={(event) => setCustomBaseline(event.target.value)}
            />
            <span className="text-ui-10 text-muted-foreground/70">
              {t("labs.benchmarks.baselineHint")}
            </span>
          </label>

          <div className="flex min-w-0 flex-col gap-1.5">
            <span className="text-ui-10 uppercase tracking-[0.05em] text-muted-foreground">
              {t("labs.benchmarks.candidates")}
            </span>
            {artifacts.length === 0 ? (
              <p className="text-ui-11 text-muted-foreground">
                {t("labs.artifacts.empty")}
              </p>
            ) : (
              <div className="max-h-44 overflow-auto rounded-lg border border-border/70 bg-background/40 p-2">
                {artifacts.map((artifact) => {
                  const checked = selected.includes(artifact.path);
                  return (
                    <label
                      key={artifact.path}
                      className="flex cursor-pointer items-center gap-2 rounded-md px-1.5 py-1 text-ui-11 text-foreground hover:bg-muted/40"
                    >
                      <input
                        type="checkbox"
                        className="size-3.5 accent-primary"
                        checked={checked}
                        onChange={() =>
                          setSelected((current) =>
                            checked
                              ? current.filter((path) => path !== artifact.path)
                              : [...current, artifact.path],
                          )
                        }
                      />
                      <span className="truncate">{artifact.name}</span>
                      <span className="ml-auto shrink-0 text-ui-10 text-muted-foreground">
                        {artifact.params
                          ? `${(artifact.params / 1e6).toFixed(0)}M`
                          : ""}
                      </span>
                    </label>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        <div className="flex flex-wrap items-end gap-4">
          <div className="flex min-w-0 flex-col gap-1.5">
            <span className="text-ui-10 uppercase tracking-[0.05em] text-muted-foreground">
              {t("labs.benchmarks.suites")}
            </span>
            <div className="flex flex-wrap gap-1.5">
              {suiteOptions.map((suite) => {
                const on = suites.includes(suite.id);
                return (
                  <button
                    key={suite.id}
                    type="button"
                    aria-pressed={on}
                    title={suite.description}
                    onClick={() => toggleSuite(suite.id)}
                    className={cn(
                      "rounded-full border px-2.5 py-1 text-ui-11 transition-colors",
                      on
                        ? "border-border bg-muted text-foreground"
                        : "border-border/60 text-muted-foreground hover:text-foreground",
                    )}
                  >
                    {suite.label}
                  </button>
                );
              })}
            </div>
          </div>

          <label className="flex w-28 flex-col gap-1.5">
            <span className="text-ui-10 uppercase tracking-[0.05em] text-muted-foreground">
              {t("labs.benchmarks.limit")}
            </span>
            <input
              className={INPUT_CLASS}
              inputMode="numeric"
              value={limit}
              onChange={(event) => setLimit(event.target.value)}
              placeholder={t("labs.benchmarks.limitPlaceholder")}
            />
          </label>

          <Button
            size="sm"
            disabled={!canRun}
            onClick={() => void run()}
            className="mb-0.5"
          >
            {running ? (
              <Spinner />
            ) : (
              <HugeiconsIcon icon={PlayIcon} className="size-3.5" />
            )}
            {running
              ? t("labs.benchmarks.running")
              : t("labs.benchmarks.run")}
          </Button>
        </div>

        {status ? (
          <div className="flex items-center gap-2 rounded-lg border border-border/70 bg-muted/30 px-3 py-2 text-ui-11 text-muted-foreground">
            <Spinner />
            {status}
          </div>
        ) : null}
        {error ? (
          <p className="text-ui-11 text-destructive">{error}</p>
        ) : null}
        <p className="text-ui-10 leading-relaxed text-muted-foreground/70">
          {t("labs.benchmarks.noise")}
        </p>
      </section>

      <section className="elevated-card flex flex-col gap-3 p-5">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-ui-13 font-semibold text-foreground">
            {t("labs.benchmarks.history")}
          </h3>
          <Badge variant="outline">{records.length}</Badge>
        </div>
        {records.length === 0 ? (
          <p className="text-ui-11 text-muted-foreground">
            {t("labs.benchmarks.noHistory")}
          </p>
        ) : (
          <div className="flex flex-col gap-2">
            {records.map((record) => {
              const open = expanded === record.id;
              const entries = record.suites.map((suite) => {
                const base = record.baseline_scores[suite];
                const candidate = record.candidate_scores[suite];
                const delta =
                  typeof base === "number" && typeof candidate === "number"
                    ? candidate - base
                    : null;
                return { suite, base, candidate, delta };
              });
              return (
                <div
                  key={record.id}
                  className="rounded-lg border border-border/70 bg-background/40"
                >
                  <button
                    type="button"
                    className="flex w-full flex-wrap items-center gap-2 px-3 py-2 text-left"
                    onClick={() => setExpanded(open ? null : record.id)}
                  >
                    <span className="min-w-0 truncate text-ui-12 font-medium text-foreground">
                      {record.artifact_name}
                    </span>
                    <span className="text-ui-10 text-muted-foreground">
                      vs {record.baseline_name}
                    </span>
                    <span
                      className={cn(
                        "rounded-full border px-2 py-0.5 text-ui-10",
                        verdictClass(record.verdict),
                      )}
                    >
                      {record.verdict}
                    </span>
                    <span className="text-ui-10 text-muted-foreground">
                      {deltaLabel(record.comparison.aggregate_delta)}
                    </span>
                    <span className="ml-auto shrink-0 text-ui-10 text-muted-foreground">
                      {new Date(record.created_at).toLocaleString()}
                    </span>
                  </button>
                  {open ? (
                    <div className="border-t border-border/70 px-3 py-3">
                      <table className="w-full text-ui-11">
                        <thead>
                          <tr className="text-left text-ui-10 uppercase tracking-[0.04em] text-muted-foreground">
                            <th className="pb-1 font-medium">
                              {t("labs.benchmarks.suite")}
                            </th>
                            <th className="pb-1 text-right font-medium">
                              {t("labs.benchmarks.baselineScore")}
                            </th>
                            <th className="pb-1 text-right font-medium">
                              {t("labs.benchmarks.candidateScore")}
                            </th>
                            <th className="pb-1 text-right font-medium">
                              {t("labs.benchmarks.delta")}
                            </th>
                          </tr>
                        </thead>
                        <tbody>
                          {entries.map((entry) => (
                            <tr
                              key={entry.suite}
                              className="border-t border-border/50"
                            >
                              <td className="py-1 font-mono text-ui-10">
                                {entry.suite}
                              </td>
                              <td className="py-1 text-right">
                                {percent(entry.base)}
                              </td>
                              <td className="py-1 text-right">
                                {percent(entry.candidate)}
                              </td>
                              <td className="py-1 text-right">
                                {deltaLabel(entry.delta)}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                      <div className="mt-2 flex flex-wrap items-center gap-3 text-ui-10 text-muted-foreground">
                        {record.duration_seconds ? (
                          <span>
                            {t("labs.benchmarks.duration")}:{" "}
                            {Math.round(record.duration_seconds)}s
                          </span>
                        ) : null}
                        <span className="truncate font-mono">
                          {record.report_dir}
                        </span>
                        <Button
                          size="sm"
                          variant="ghost"
                          className="ml-auto h-6 px-2 text-ui-10"
                          onClick={() => void remove(record.id)}
                        >
                          <HugeiconsIcon
                            icon={Delete02Icon}
                            className="size-3"
                          />
                          {t("labs.benchmarks.delete")}
                        </Button>
                      </div>
                      {record.errors.length > 0 ? (
                        <p className="mt-2 text-ui-10 text-amber-600 dark:text-amber-400">
                          {record.errors.join(" · ")}
                        </p>
                      ) : null}
                    </div>
                  ) : null}
                </div>
              );
            })}
          </div>
        )}
        {records.length > 0 ? (
          <p className="flex items-center gap-1.5 text-ui-10 text-muted-foreground/70">
            <HugeiconsIcon icon={Tick02Icon} className="size-3" />
            {t("labs.benchmarks.stored")}
          </p>
        ) : null}
      </section>
    </div>
  );
}
