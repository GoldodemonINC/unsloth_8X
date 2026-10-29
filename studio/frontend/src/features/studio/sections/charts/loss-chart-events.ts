// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TrainingSeriesPoint } from "@/features/training";

export type LossChartEventKind = "checkpoint" | "resume" | "lrChange" | "anomaly";

export interface LossChartEvent {
  kind: LossChartEventKind;
  step: number;
}

const MAX_CHECKPOINT_MARKERS = 12;
const MAX_LR_MARKERS = 12;
const LR_CHANGE_RATIO = 1.25;

export function isLrChange(previous: number, next: number): boolean {
  if (!Number.isFinite(previous) || !Number.isFinite(next)) {
    return false;
  }
  if (previous <= 0 || next <= 0) {
    return previous !== next;
  }
  const ratio = next > previous ? next / previous : previous / next;
  return ratio >= LR_CHANGE_RATIO;
}

function sampleEvenly<T>(values: readonly T[], limit: number): T[] {
  if (values.length <= limit) {
    return [...values];
  }
  const picked: T[] = [];
  for (let index = 0; index < limit; index += 1) {
    picked.push(values[Math.round((index * (values.length - 1)) / (limit - 1))]);
  }
  return picked;
}

function lrChangeSteps(history: readonly TrainingSeriesPoint[]): number[] {
  const sorted = [...history]
    .filter((point) => Number.isFinite(point.step) && Number.isFinite(point.value))
    .sort((left, right) => left.step - right.step);
  const steps: number[] = [];
  for (let index = 1; index < sorted.length; index += 1) {
    if (isLrChange(sorted[index - 1].value, sorted[index].value)) {
      steps.push(sorted[index].step);
    }
  }
  return sampleEvenly(steps, MAX_LR_MARKERS);
}

export function resumedAtStep(
  config: Record<string, unknown> | null | undefined,
  history: readonly TrainingSeriesPoint[],
): number | null {
  const resume = config?.resume_from_checkpoint;
  if (typeof resume !== "string" || resume.trim().length === 0) {
    return null;
  }
  const first = [...history]
    .filter((point) => Number.isFinite(point.step))
    .sort((left, right) => left.step - right.step)[0];
  return first && first.step > 1 ? first.step : null;
}

export function buildLossChartEvents({
  checkpointSteps = [],
  lrHistory = [],
  anomalyStep = null,
  resumeStep = null,
}: {
  checkpointSteps?: readonly number[];
  lrHistory?: readonly TrainingSeriesPoint[];
  anomalyStep?: number | null;
  resumeStep?: number | null;
}): LossChartEvent[] {
  const checkpoints = sampleEvenly(
    [...new Set(checkpointSteps.filter((step) => Number.isFinite(step) && step > 0))].sort(
      (left, right) => left - right,
    ),
    MAX_CHECKPOINT_MARKERS,
  );
  const events: LossChartEvent[] = [
    ...checkpoints.map((step) => ({ kind: "checkpoint" as const, step })),
    ...lrChangeSteps(lrHistory).map((step) => ({ kind: "lrChange" as const, step })),
  ];
  if (resumeStep !== null && Number.isFinite(resumeStep) && resumeStep > 0) {
    events.push({ kind: "resume", step: resumeStep });
  }
  if (anomalyStep !== null && Number.isFinite(anomalyStep) && anomalyStep > 0) {
    events.push({ kind: "anomaly", step: anomalyStep });
  }
  const seen = new Set<string>();
  return events
    .sort((left, right) => left.step - right.step)
    .filter((event) => {
      const key = `${event.kind}:${event.step}`;
      if (seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    });
}
