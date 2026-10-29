// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TrainingStartRequest } from "../types/api";

export type TrainingQueueStatus =
  | "queued"
  | "running"
  | "done"
  | "failed"
  | "cancelled";

export interface TrainingQueueItem {
  id: string;
  label: string;
  groupId: string | null;
  config: TrainingStartRequest;
  notBefore: string | null;
  status: TrainingQueueStatus;
  error: string | null;
  createdAt: string;
  startedAt: string | null;
  finishedAt: string | null;
}

export interface SweepAxes {
  learningRates: number[];
  loraRanks: number[];
  epochs: number[];
  batchSizes: number[];
}

export interface SweepVariant {
  label: string;
  config: TrainingStartRequest;
}

export const SWEEP_MAX_ITEMS = 24;

export function sweepCombinationCount(axes: SweepAxes): number {
  return (
    Math.max(1, axes.learningRates.length) *
    Math.max(1, axes.loraRanks.length) *
    Math.max(1, axes.epochs.length) *
    Math.max(1, axes.batchSizes.length)
  );
}

export function sweepVariantLabel(
  learningRate: number,
  loraRank: number,
  epochs: number,
  batchSize: number,
): string {
  return `lr ${learningRate} · r${loraRank} · ${epochs}ep · bs${batchSize}`;
}

export function parseSweepNumbers(raw: string): number[] {
  const values = raw
    .split(/[\s,]+/)
    .map((part) => Number(part.trim()))
    .filter((value) => Number.isFinite(value) && value > 0);
  return [...new Set(values)];
}

export function nextReadyQueueItem(
  items: readonly TrainingQueueItem[],
  now: number,
): TrainingQueueItem | null {
  for (const item of items) {
    if (item.status !== "queued") {
      continue;
    }
    if (item.notBefore === null) {
      return item;
    }
    const readyAt = Date.parse(item.notBefore);
    if (Number.isNaN(readyAt) || readyAt <= now) {
      return item;
    }
  }
  return null;
}

export function isScheduledQueueItem(
  item: TrainingQueueItem,
  now: number,
): boolean {
  if (item.status !== "queued" || item.notBefore === null) {
    return false;
  }
  const readyAt = Date.parse(item.notBefore);
  return !Number.isNaN(readyAt) && readyAt > now;
}

export function sanitizeQueueConfig(
  config: TrainingStartRequest,
): TrainingStartRequest {
  return {
    ...config,
    hf_token: null,
    wandb_token: null,
    trust_remote_code: false,
    approved_remote_code_fingerprint: null,
  };
}

export function createQueueItemId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `queue-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}
