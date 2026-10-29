// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import type { ChartConfig } from "@/components/ui/chart";
import { type TranslationKey, useT } from "@/i18n";
import type { ReactElement } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  XAxis,
  YAxis,
} from "recharts";
import type { LossChartEvent, LossChartEventKind } from "./loss-chart-events";
import type { ScaleMode } from "./types";
import {
  CHART_SYNC_ID,
  CHART_CONTAINER_CLASS,
  DEFAULT_CHART_MARGIN,
  DEFAULT_Y_AXIS_WIDTH,
  formatAxisMetric,
  formatMetric,
  formatStepTick,
  fromLog1p,
} from "./utils";

interface LossChartPoint {
  step: number;
  loss: number;
  smoothed: number;
  displayLoss: number;
  displaySmoothed: number;
}

const EVENT_STROKE: Record<LossChartEventKind, string> = {
  checkpoint: "var(--muted-foreground)",
  resume: "var(--chart-3)",
  lrChange: "var(--chart-2)",
  anomaly: "var(--destructive)",
};

const EVENT_DASH: Record<LossChartEventKind, string> = {
  checkpoint: "2 4",
  resume: "5 3",
  lrChange: "4 3",
  anomaly: "1 2",
};

const EVENT_LABEL: Record<LossChartEventKind, TranslationKey> = {
  checkpoint: "studio.chartEvents.checkpoint",
  resume: "studio.chartEvents.resume",
  lrChange: "studio.chartEvents.lrChange",
  anomaly: "studio.chartEvents.anomaly",
};

export function TrainingLossChartCard({
  data,
  domain,
  visibleStepDomain,
  xAxisTicks,
  avgRaw,
  avgDisplay,
  showRaw,
  showSmoothed,
  showAvgLine,
  scale,
  events = [],
}: {
  data: LossChartPoint[];
  domain: [number, number];
  visibleStepDomain: [number, number];
  xAxisTicks: number[];
  avgRaw: number;
  avgDisplay: number;
  showRaw: boolean;
  showSmoothed: boolean;
  showAvgLine: boolean;
  scale: ScaleMode;
  events?: readonly LossChartEvent[];
}): ReactElement {
  const t = useT();
  const lossConfig = {
    displayLoss: { label: t("studio.charts.loss"), color: "#3b82f6" },
    displaySmoothed: { label: t("studio.charts.smoothed"), color: "#f59e0b" },
  } satisfies ChartConfig;
  const showPoint = data.length <= 1 ? { r: 3, strokeWidth: 0 } : false;
  const visibleEvents = events.filter(
    (event) =>
      event.step >= visibleStepDomain[0] && event.step <= visibleStepDomain[1],
  );
  const presentKinds = [...new Set(visibleEvents.map((event) => event.kind))];

  return (
    <Card data-tour="studio-training-loss" size="sm">
      <CardHeader>
        <CardTitle className="text-sm">
          {t("studio.charts.trainingLoss")}
        </CardTitle>
        {presentKinds.length > 0 && (
          <div className="flex flex-wrap items-center gap-2.5">
            {presentKinds.map((kind) => (
              <span
                key={kind}
                className="inline-flex items-center gap-1 text-ui-10 text-muted-foreground"
              >
                <span
                  className="size-1.5 rounded-full"
                  style={{ backgroundColor: EVENT_STROKE[kind] }}
                />
                {t(EVENT_LABEL[kind])}
              </span>
            ))}
          </div>
        )}
      </CardHeader>
      <CardContent>
        <ChartContainer config={lossConfig} className={CHART_CONTAINER_CLASS}>
          <LineChart
            data={data}
            syncId={CHART_SYNC_ID}
            syncMethod="value"
            accessibilityLayer={true}
            margin={DEFAULT_CHART_MARGIN}
          >
            <CartesianGrid vertical={false} strokeDasharray="3 3" />
            <XAxis
              dataKey="step"
              type="number"
              domain={visibleStepDomain}
              ticks={xAxisTicks}
              allowDataOverflow={true}
              allowDecimals={false}
              minTickGap={28}
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              fontSize={10}
              tickFormatter={(value) => formatStepTick(Number(value))}
              interval="preserveStartEnd"
            />
            <YAxis
              domain={domain}
              allowDataOverflow={true}
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              tickCount={5}
              fontSize={10}
              width={DEFAULT_Y_AXIS_WIDTH}
              tickFormatter={(value) => {
                const num = Number(value);
                if (!Number.isFinite(num)) {
                  return "0";
                }
                const shown = scale === "log" ? fromLog1p(num) : num;
                return formatAxisMetric(shown);
              }}
            />
            <ChartTooltip
              content={
                <ChartTooltipContent
                  labelFormatter={(_value, payload) =>
                    t("studio.charts.step", {
                      step: payload?.[0]?.payload?.step ?? "",
                    })
                  }
                  formatter={(_value, name, item) => {
                    if (name === "displaySmoothed") {
                      return [
                        formatMetric(Number(item?.payload?.smoothed)),
                        t("studio.charts.smoothed"),
                      ];
                    }
                    return [
                      formatMetric(Number(item?.payload?.loss)),
                      t("studio.charts.loss"),
                    ];
                  }}
                />
              }
            />
            {showAvgLine && (
              <ReferenceLine
                y={avgDisplay}
                stroke="#3b82f6"
                strokeDasharray="4 4"
                strokeOpacity={0.5}
                label={{
                  value: t("studio.charts.averageValue", {
                    value: formatMetric(avgRaw),
                  }),
                  position: "insideTopRight",
                  fontSize: "calc(0.625rem * var(--ui-font-scale, 1))",
                  fill: "#3b82f6",
                }}
              />
            )}
            {visibleEvents.map((event) => (
              <ReferenceLine
                key={`${event.kind}-${event.step}`}
                x={event.step}
                stroke={EVENT_STROKE[event.kind]}
                strokeDasharray={EVENT_DASH[event.kind]}
                strokeOpacity={event.kind === "anomaly" ? 0.9 : 0.55}
              />
            ))}
            {showRaw && (
              <Line
                type="linear"
                dataKey="displayLoss"
                stroke="var(--color-displayLoss)"
                strokeWidth={1.2}
                strokeOpacity={showSmoothed ? 0.35 : 1}
                dot={showPoint}
                activeDot={{ r: 3, strokeWidth: 0 }}
                connectNulls={true}
                strokeLinecap="round"
                strokeLinejoin="round"
                isAnimationActive={false}
              />
            )}
            {showSmoothed && (
              <Line
                type="linear"
                dataKey="displaySmoothed"
                stroke="var(--color-displaySmoothed)"
                strokeWidth={2.2}
                dot={showPoint}
                activeDot={{ r: 3, strokeWidth: 0 }}
                connectNulls={true}
                strokeLinecap="round"
                strokeLinejoin="round"
                isAnimationActive={false}
              />
            )}
            <ChartLegend content={<ChartLegendContent />} />
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
