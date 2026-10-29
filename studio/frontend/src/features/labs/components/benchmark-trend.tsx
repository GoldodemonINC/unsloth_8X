// SPDX-License-Identifier: NRC-2
// Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-2.md

import { Badge } from "@/components/ui/badge";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import type { ChartConfig } from "@/components/ui/chart";
import { useT } from "@/i18n";
import { type ReactElement, useMemo } from "react";
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts";
import type { BenchmarkRecord } from "../api";

const NOISE_POINTS = 1.5;
const SERIES_COLORS = [
  "var(--chart-1)",
  "var(--chart-2)",
  "var(--chart-3)",
  "var(--chart-4)",
] as const;

interface TrendPoint {
  when: number;
  label: string;
  score: number;
}

interface TrendSeries {
  name: string;
  color: string;
  points: TrendPoint[];
}

function meanScore(scores: Record<string, number>): number | null {
  const values = Object.values(scores).filter((value) =>
    Number.isFinite(value),
  );
  if (values.length === 0) return null;
  return (values.reduce((total, value) => total + value, 0) / values.length) * 100;
}

function shortDate(when: number): string {
  return new Date(when).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}

function buildSeries(records: readonly BenchmarkRecord[]): TrendSeries[] {
  const byArtifact = new Map<string, TrendPoint[]>();
  for (const record of records) {
    const score = meanScore(record.candidate_scores);
    const when = Date.parse(record.created_at);
    if (score === null || Number.isNaN(when)) continue;
    const points = byArtifact.get(record.artifact_name) ?? [];
    points.push({ when, label: shortDate(when), score });
    byArtifact.set(record.artifact_name, points);
  }
  const series: TrendSeries[] = [];
  for (const [name, points] of byArtifact) {
    if (points.length < 2) continue;
    series.push({
      name,
      color: SERIES_COLORS[series.length % SERIES_COLORS.length],
      points: [...points].sort((left, right) => left.when - right.when),
    });
  }
  return series
    .sort(
      (left, right) =>
        (right.points.at(-1)?.when ?? 0) - (left.points.at(-1)?.when ?? 0),
    )
    .slice(0, SERIES_COLORS.length);
}

function latestVerdict(series: TrendSeries): {
  kind: "best" | "regressed" | "flat";
  delta: number;
} {
  const latest = series.points.at(-1);
  if (!latest) return { kind: "flat", delta: 0 };
  const previousBest = Math.max(
    ...series.points.slice(0, -1).map((point) => point.score),
  );
  const delta = latest.score - previousBest;
  if (delta >= NOISE_POINTS) return { kind: "best", delta };
  if (delta <= -NOISE_POINTS) return { kind: "regressed", delta };
  return { kind: "flat", delta };
}

export function BenchmarkTrend({
  records,
}: {
  records: readonly BenchmarkRecord[];
}): ReactElement | null {
  const t = useT();
  const series = useMemo(() => buildSeries(records), [records]);

  const rows = useMemo(() => {
    const byDate = new Map<number, Record<string, number | string>>();
    for (const entry of series) {
      for (const point of entry.points) {
        const row = byDate.get(point.when) ?? { when: point.when, label: point.label };
        row[entry.name] = Number(point.score.toFixed(1));
        byDate.set(point.when, row);
      }
    }
    return [...byDate.values()].sort(
      (left, right) => Number(left.when) - Number(right.when),
    );
  }, [series]);

  if (series.length === 0) {
    return null;
  }

  const config = Object.fromEntries(
    series.map((entry) => [
      entry.name,
      { label: entry.name, color: entry.color },
    ]),
  ) satisfies ChartConfig;

  return (
    <div className="flex flex-col gap-3 rounded-lg border border-border/70 bg-background/40 p-3">
      <div className="flex flex-wrap items-baseline gap-2">
        <h4 className="text-ui-12 font-semibold text-foreground">
          {t("labs.benchmarks.trendTitle")}
        </h4>
        <p className="min-w-0 flex-1 text-ui-10 text-muted-foreground">
          {t("labs.benchmarks.trendDescription")}
        </p>
      </div>
      <ChartContainer config={config} className="h-52 w-full">
        <LineChart data={rows} margin={{ left: 4, right: 8, top: 8, bottom: 0 }}>
          <CartesianGrid vertical={false} strokeDasharray="3 3" />
          <XAxis dataKey="label" tickLine={false} axisLine={false} fontSize={10} />
          <YAxis
            tickLine={false}
            axisLine={false}
            fontSize={10}
            width={36}
            domain={["auto", "auto"]}
            tickFormatter={(value: number) => `${value.toFixed(0)}%`}
          />
          <ChartTooltip content={<ChartTooltipContent indicator="dot" />} />
          {series.map((entry) => (
            <Line
              key={entry.name}
              dataKey={entry.name}
              type="monotone"
              stroke={entry.color}
              strokeWidth={2}
              connectNulls
              dot={{ r: 2.5, strokeWidth: 0, fill: entry.color }}
              activeDot={{ r: 4 }}
            />
          ))}
        </LineChart>
      </ChartContainer>
      <div className="flex flex-wrap gap-2">
        {series.map((entry) => {
          const verdict = latestVerdict(entry);
          return (
            <div
              key={entry.name}
              className="flex min-w-0 items-center gap-2 rounded-md border border-border/60 px-2 py-1"
            >
              <span
                className="size-1.5 shrink-0 rounded-full"
                style={{ backgroundColor: entry.color }}
              />
              <span className="max-w-52 truncate text-ui-10 text-foreground">
                {entry.name}
              </span>
              {verdict.kind === "regressed" ? (
                <Badge variant="destructive" className="text-ui-9">
                  {t("labs.benchmarks.trendRegression", {
                    value: Math.abs(verdict.delta).toFixed(1),
                  })}
                </Badge>
              ) : verdict.kind === "best" ? (
                <Badge
                  variant="outline"
                  className="border-emerald-500/30 bg-emerald-500/10 text-ui-9 text-emerald-700 dark:text-emerald-300"
                >
                  {t("labs.benchmarks.trendBest")}
                </Badge>
              ) : (
                <Badge variant="outline" className="text-ui-9">
                  {t("labs.benchmarks.trendFlat")}
                </Badge>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
