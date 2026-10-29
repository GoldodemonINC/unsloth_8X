import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import { getTrainingRunDisplayTitle } from "@/features/training";
import { getTrainingRun, listTrainingRuns } from "@/features/training/api/history-api";
import type { TrainingRunDetailResponse, TrainingRunSummary } from "@/features/training/types/history";
import { type TranslationKey, useT } from "@/i18n";
import { type ReactElement, useEffect, useMemo, useState } from "react";
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts";
import type { ChartConfig } from "@/components/ui/chart";

const RUN_LIMIT = 100;
const COMPARISON_LIMIT = 4;
const RUN_COLORS = ["var(--chart-1)", "var(--chart-2)", "var(--chart-3)", "var(--chart-4)"];
const CONFIG_FIELDS: ReadonlyArray<{
  key: string;
  fallbackKey?: string;
  labelKey: TranslationKey;
}> = [
  { key: "learning_rate", labelKey: "studio.progress.learningRate" },
  { key: "batch_size", labelKey: "studio.progress.batchSize" },
  { key: "gradient_accumulation_steps", labelKey: "studio.params.gradAccum" },
  { key: "random_seed", fallbackKey: "seed", labelKey: "studio.params.seed" },
  { key: "num_epochs", labelKey: "studio.progress.epochs" },
  { key: "max_steps", labelKey: "studio.progress.maxSteps" },
  { key: "optim", labelKey: "studio.progress.optimizer" },
  { key: "max_seq_length", labelKey: "studio.progress.contextLength" },
  { key: "lora_r", labelKey: "studio.progress.rank" },
];

interface LoadedRun {
  summary: TrainingRunSummary;
  detail: TrainingRunDetailResponse;
}

function configValue(run: LoadedRun, key: string, fallbackKey?: string): unknown {
  const config = run.detail.config;
  return config[key] ?? (fallbackKey ? config[fallbackKey] : undefined);
}

function datasetFingerprintSha(run: LoadedRun): string | null {
  const fingerprint = run.detail.config.dataset_fingerprint;
  if (!fingerprint || typeof fingerprint !== "object") return null;
  const candidate = fingerprint as { status?: unknown; sha256?: unknown };
  return candidate.status === "hashed" && typeof candidate.sha256 === "string"
    ? candidate.sha256
    : null;
}

function displayValue(value: unknown): string {
  if (value === undefined || value === null || value === "") return "—";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function lossCurve(run: LoadedRun): { step: number; loss: number }[] {
  return run.detail.metrics.loss_step_history
    .map((step, index) => ({ step, loss: run.detail.metrics.loss_history[index] }))
    .filter((point): point is { step: number; loss: number } => Number.isFinite(point.step) && Number.isFinite(point.loss));
}

function chartRows(runs: LoadedRun[]): Record<string, number>[] {
  const steps = new Set<number>();
  const values = new Map<number, Record<string, number>>();
  for (const run of runs) {
    for (const point of lossCurve(run)) {
      steps.add(point.step);
      const row = values.get(point.step) ?? { step: point.step };
      row[run.summary.id] = point.loss;
      values.set(point.step, row);
    }
  }
  return [...steps].sort((left, right) => left - right).map((step) => values.get(step) ?? { step });
}

export function RunComparisonBoard(): ReactElement {
  const t = useT();
  const [runs, setRuns] = useState<TrainingRunSummary[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [details, setDetails] = useState<Record<string, TrainingRunDetailResponse>>({});
  const [winnerId, setWinnerId] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [activeTag, setActiveTag] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    listTrainingRuns(RUN_LIMIT, 0, controller.signal)
      .then((response) => setRuns(response.runs))
      .catch((requestError: unknown) => {
        if (!(requestError instanceof DOMException && requestError.name === "AbortError")) setError(true);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const missing = selectedIds.filter((id) => !details[id]);
    if (missing.length === 0) return;
    const controller = new AbortController();
    Promise.all(missing.map(async (id) => [id, await getTrainingRun(id, controller.signal)] as const))
      .then((loaded) => setDetails((current) => ({
        ...current,
        ...Object.fromEntries(loaded.map(([id, detail]) => [id, detail])),
      })))
      .catch((requestError: unknown) => {
        if (!(requestError instanceof DOMException && requestError.name === "AbortError")) setError(true);
      });
    return () => controller.abort();
  }, [details, selectedIds]);

  const knownTags = useMemo(
    () => [...new Set(runs.flatMap((run) => run.tags ?? []))].sort((left, right) => left.localeCompare(right)),
    [runs],
  );
  const visibleRuns = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return runs.filter((run) => {
      if (activeTag !== null && !(run.tags ?? []).includes(activeTag)) return false;
      if (!needle) return true;
      return [
        getTrainingRunDisplayTitle(run),
        run.model_name,
        run.dataset_name,
        run.notes ?? "",
        ...(run.tags ?? []),
      ].some((value) => value.toLowerCase().includes(needle));
    });
  }, [activeTag, query, runs]);
  const selectedRuns = useMemo(
    () => selectedIds.flatMap((id) => {
      const summary = runs.find((run) => run.id === id);
      const detail = details[id];
      return summary && detail ? [{ summary, detail }] : [];
    }),
    [details, runs, selectedIds],
  );
  const plotData = useMemo(() => chartRows(selectedRuns), [selectedRuns]);
  const chartConfig: ChartConfig = useMemo(
    () => Object.fromEntries(selectedRuns.map((run, index) => [
      run.summary.id,
      { label: getTrainingRunDisplayTitle(run.summary), color: RUN_COLORS[index] },
    ])),
    [selectedRuns],
  );
  const differentFields = useMemo(() => {
    if (selectedRuns.length < 2) return [];
    const fields: ReadonlyArray<{
      key: string;
      labelKey: TranslationKey;
      read: (run: LoadedRun) => unknown;
    }> = [
      { key: "dataset_name", labelKey: "studio.wizard.datasetLabel", read: (run: LoadedRun) => run.summary.dataset_name },
      ...CONFIG_FIELDS.map((field) => ({
        key: field.key,
        labelKey: field.labelKey,
        read: (run: LoadedRun) =>
          configValue(run, field.key, field.fallbackKey),
      })),
    ];
    return fields.filter((field) => new Set(selectedRuns.map((run) => displayValue(field.read(run)))).size > 1);
  }, [selectedRuns]);
  const datasetDrift = useMemo(() => {
    if (selectedRuns.length < 2) return false;
    const names = new Set(selectedRuns.map((run) => run.summary.dataset_name));
    if (names.size !== 1) return false;
    const hashes = new Set(
      selectedRuns
        .map(datasetFingerprintSha)
        .filter((value): value is string => value !== null),
    );
    return hashes.size > 1;
  }, [selectedRuns]);
  const isLoadingDetails = selectedIds.some((id) => !details[id]);

  const toggleRun = (runId: string, checked: boolean) => {
    setSelectedIds((current) => checked
      ? current.length < COMPARISON_LIMIT ? [...current, runId] : current
      : current.filter((id) => id !== runId));
    if (!checked && winnerId === runId) setWinnerId(null);
  };

  return (
    <section className="flex flex-col gap-5" aria-labelledby="run-comparison-title">
      <header className="flex flex-col gap-1">
        <h2 id="run-comparison-title" className="text-base font-medium">{t("studio.comparison.title")}</h2>
        <p className="text-sm text-muted-foreground">{t("studio.comparison.description")}</p>
      </header>
      <div className="flex flex-col gap-3 rounded-4xl corner-squircle bg-card p-5 ring-1 ring-[color-mix(in_oklab,var(--foreground)_calc(10%*var(--contrast-edge-gain,1)),transparent)]">
        <p className="text-sm font-medium">{t("studio.comparison.selectionHint", { count: selectedIds.length })}</p>
        <div className="flex flex-col gap-2">
          <Input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder={t("studio.history.searchPlaceholder")}
            aria-label={t("studio.history.searchPlaceholder")}
          />
          {knownTags.length > 0 ? (
            <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label={t("studio.history.filterByTag")}>
              <Button size="xs" variant={activeTag === null ? "default" : "outline"} className="h-7 px-2.5 text-ui-11" aria-pressed={activeTag === null} onClick={() => setActiveTag(null)}>
                {t("studio.history.allRuns")}
              </Button>
              {knownTags.map((tag) => (
                <Button key={tag} size="xs" variant={activeTag === tag ? "default" : "outline"} className="h-7 px-2.5 text-ui-11" aria-pressed={activeTag === tag} onClick={() => setActiveTag(activeTag === tag ? null : tag)}>
                  #{tag}
                </Button>
              ))}
            </div>
          ) : null}
        </div>
        {loading ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground"><Spinner className="size-4" />{t("studio.comparison.loading")}</div>
        ) : error ? (
          <p role="alert" className="text-sm text-destructive">{t("studio.comparison.loadError")}</p>
        ) : visibleRuns.length === 0 ? (
          <p className="text-sm text-muted-foreground">{t("studio.comparison.noRuns")}</p>
        ) : (
          <div className="grid max-h-72 grid-cols-1 gap-2 overflow-y-auto sm:grid-cols-2 xl:grid-cols-3">
            {visibleRuns.map((run) => {
              const checked = selectedIds.includes(run.id);
              const blocked = !checked && selectedIds.length >= COMPARISON_LIMIT;
              return (
                <label key={run.id} className="flex min-w-0 cursor-pointer items-start gap-3 rounded-xl bg-muted/30 p-3">
                  <Checkbox checked={checked} disabled={blocked} onCheckedChange={(value) => toggleRun(run.id, value === true)} aria-label={t("studio.comparison.selectRun", { run: getTrainingRunDisplayTitle(run) })} />
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm font-medium">{getTrainingRunDisplayTitle(run)}</span>
                    <span className="block truncate text-xs text-muted-foreground">{run.model_name} · {run.dataset_name}</span>
                    <span className="block truncate text-xs text-muted-foreground">{t("studio.comparison.finalLoss")}: {run.final_loss == null ? "—" : run.final_loss.toFixed(4)}</span>
                    {(run.tags ?? []).length > 0 ? (
                      <span className="mt-1 block truncate text-ui-10 text-muted-foreground">
                        {(run.tags ?? []).map((tag) => `#${tag}`).join(" ")}
                      </span>
                    ) : null}
                    {run.notes ? (
                      <span className="mt-0.5 block truncate text-ui-10 text-muted-foreground">{run.notes}</span>
                    ) : null}
                  </span>
                </label>
              );
            })}
          </div>
        )}
      </div>
      {selectedIds.length < 2 ? (
        <p className="text-sm text-muted-foreground">{t("studio.comparison.chooseTwo")}</p>
      ) : isLoadingDetails ? (
        <div className="flex items-center gap-2 text-sm text-muted-foreground"><Spinner className="size-4" />{t("studio.comparison.loading")}</div>
      ) : selectedRuns.length >= 2 ? (
        <>
          <div className="rounded-4xl corner-squircle bg-card p-5 ring-1 ring-[color-mix(in_oklab,var(--foreground)_calc(10%*var(--contrast-edge-gain,1)),transparent)]">
            <h3 className="mb-3 text-sm font-medium">{t("studio.comparison.chartTitle")}</h3>
            <ChartContainer config={chartConfig} className="h-[calc(300px*var(--ui-space-scale,1))] w-full">
              <LineChart data={plotData} margin={{ top: 12, right: 12, bottom: 2, left: 4 }}>
                <CartesianGrid vertical={false} strokeDasharray="3 3" />
                <XAxis dataKey="step" type="number" domain={["dataMin", "dataMax"]} allowDecimals={false} tickLine={false} axisLine={false} />
                <YAxis width={56} tickLine={false} axisLine={false} />
                <ChartTooltip content={<ChartTooltipContent labelFormatter={(value) => t("studio.charts.step", { step: value })} />} />
                {selectedRuns.map((run) => (
                  <Line key={run.summary.id} dataKey={run.summary.id} name={run.summary.id} type="linear" stroke={`var(--color-${run.summary.id})`} strokeWidth={2} dot={false} connectNulls={true} isAnimationActive={false} />
                ))}
              </LineChart>
            </ChartContainer>
            <div className="mt-3 flex flex-wrap gap-2">
              {selectedRuns.map((run, index) => (
                <Button key={run.summary.id} size="xs" variant={winnerId === run.summary.id ? "default" : "outline"} aria-pressed={winnerId === run.summary.id} onClick={() => setWinnerId(run.summary.id)}>
                  <span className="size-2 rounded-full" style={{ backgroundColor: RUN_COLORS[index] }} />
                  {winnerId === run.summary.id ? t("studio.comparison.winner") : t("studio.comparison.markWinner")}: {getTrainingRunDisplayTitle(run.summary)}
                </Button>
              ))}
            </div>
          </div>
          {datasetDrift ? (
            <p
              role="status"
              className="rounded-xl border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-ui-11 leading-relaxed text-amber-800 dark:text-amber-200"
            >
              {t("studio.comparison.datasetDrift")}
            </p>
          ) : null}
          <div className="rounded-4xl corner-squircle bg-card p-5 ring-1 ring-[color-mix(in_oklab,var(--foreground)_calc(10%*var(--contrast-edge-gain,1)),transparent)]">
            <h3 className="mb-3 text-sm font-medium">{t("studio.comparison.diffTitle")}</h3>
            {differentFields.length === 0 ? (
              <p className="text-sm text-muted-foreground">{t("studio.comparison.noDifferences")}</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full min-w-[36rem] text-left text-sm">
                  <thead><tr><th className="py-2 pr-4 font-medium text-muted-foreground">{t("studio.comparison.field")}</th>{selectedRuns.map((run) => <th key={run.summary.id} className="max-w-56 truncate px-3 py-2 font-medium">{getTrainingRunDisplayTitle(run.summary)}</th>)}</tr></thead>
                  <tbody>{differentFields.map((field) => <tr key={field.key} className="border-t border-border/50"><th className="py-2 pr-4 font-medium text-muted-foreground">{t(field.labelKey)}</th>{selectedRuns.map((run) => <td key={run.summary.id} className="max-w-56 truncate px-3 py-2 font-mono text-ui-11" title={displayValue(field.read(run))}>{displayValue(field.read(run))}</td>)}</tr>)}</tbody>
                </table>
              </div>
            )}
          </div>
        </>
      ) : null}
    </section>
  );
}
