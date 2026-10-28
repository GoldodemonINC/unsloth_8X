// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Reads the loss curve the way a person squints at it: a sudden jump, a curve
 * that has stopped improving, or one that has turned upward. It reports one
 * finding, not a score, because a banner with three competing diagnoses is
 * worse than no banner.
 *
 * Thresholds are deliberately conservative. A run early in training moves a
 * lot, and warning on ordinary noise trains people to ignore the warning.
 */

export interface LossPoint {
  step: number;
  value: number;
}

export type LossAnomalyKind = "nan" | "divergence" | "spike" | "plateau";

export interface LossAnomaly {
  kind: LossAnomalyKind;
  severity: "warning" | "critical";
  step: number;
  value: number;
  /** The measured quantity that triggered the rule, for the tooltip. */
  detail: number;
}

function median(values: readonly number[]): number {
  if (values.length === 0) return Number.NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle];
}

function slope(values: readonly number[]): number {
  const n = values.length;
  if (n < 2) return 0;
  const meanX = (n - 1) / 2;
  const meanY = values.reduce((sum, value) => sum + value, 0) / n;
  let numerator = 0;
  let denominator = 0;
  for (let index = 0; index < n; index += 1) {
    numerator += (index - meanX) * (values[index] - meanY);
    denominator += (index - meanX) ** 2;
  }
  return denominator === 0 ? 0 : numerator / denominator;
}

export function detectLossAnomaly(
  points: readonly LossPoint[],
): LossAnomaly | null {
  if (points.length < 8) return null;

  const bad = points.find((point) => !Number.isFinite(point.value));
  if (bad) {
    return {
      kind: "nan",
      severity: "critical",
      step: bad.step,
      value: bad.value,
      detail: 0,
    };
  }

  const series = [...points]
    .filter((point) => Number.isFinite(point.step))
    .sort((a, b) => a.step - b.step);
  if (series.length < 8) return null;

  // Divergence first: an upward trend makes a "spike" just one noisy point on
  // the way up, and lowering the learning rate is the more useful advice.
  if (series.length >= 30) {
    const firstThird = series.slice(0, Math.floor(series.length / 3));
    const lastThird = series.slice(-Math.floor(series.length / 3));
    const startMedian = median(firstThird.map((point) => point.value));
    const endMedian = median(lastThird.map((point) => point.value));
    const trend = slope(lastThird.map((point) => point.value));
    if (
      Number.isFinite(startMedian) &&
      startMedian > 0 &&
      endMedian > startMedian * 1.25 &&
      trend > 0
    ) {
      const last = lastThird[lastThird.length - 1];
      return {
        kind: "divergence",
        severity: "critical",
        step: last.step,
        value: last.value,
        detail: endMedian / startMedian,
      };
    }
  }

  if (series.length >= 12) {
    let worst: LossAnomaly | null = null;
    for (let index = 5; index < series.length; index += 1) {
      const point = series[index];
      const window = series
        .slice(Math.max(0, index - 20), index)
        .map((previous) => previous.value);
      const baseline = median(window);
      if (!Number.isFinite(baseline) || baseline <= 0) continue;
      const ratio = point.value / baseline;
      const jump = point.value - baseline;
      if (ratio < 2.5 || jump < 0.5) continue;
      if (!worst || ratio > worst.detail) {
        worst = {
          kind: "spike",
          severity: "warning",
          step: point.step,
          value: point.value,
          detail: ratio,
        };
      }
    }
    if (worst) return worst;
  }

  // A plateau is only meaningful when the loss is still high enough to matter;
  // a curve that has settled at 0.03 has converged, not stalled.
  if (series.length >= 40) {
    const tailLength = Math.max(10, Math.floor(series.length * 0.25));
    const tail = series.slice(-tailLength);
    const tailValues = tail.map((point) => point.value);
    const tailMedian = median(tailValues);
    const spread = Math.max(...tailValues) - Math.min(...tailValues);
    const firstTail = tailValues[0];
    const improvement =
      firstTail > 0
        ? (firstTail - tailValues[tailValues.length - 1]) / firstTail
        : 1;
    if (
      Number.isFinite(tailMedian) &&
      tailMedian > 0.1 &&
      spread / tailMedian <= 0.03 &&
      improvement <= 0.01
    ) {
      const last = tail[tail.length - 1];
      return {
        kind: "plateau",
        severity: "warning",
        step: last.step,
        value: last.value,
        detail: spread / tailMedian,
      };
    }
  }

  return null;
}
