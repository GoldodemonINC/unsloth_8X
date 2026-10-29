// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ModelCheckpoints } from "@/features/export/api/export-api";

const CHECKPOINT_STEP = /^checkpoint-(\d+)$/;

function normalizePath(value: string): string {
  return value.replace(/\\/g, "/").replace(/\/+$/, "");
}

export function checkpointStepFromPath(path: string): number | null {
  const segments = normalizePath(path).split("/");
  const match = CHECKPOINT_STEP.exec(segments[segments.length - 1] ?? "");
  if (!match) {
    return null;
  }
  const step = Number(match[1]);
  return Number.isFinite(step) ? step : null;
}

export function checkpointStepsForOutputDir(
  models: readonly ModelCheckpoints[],
  outputDir: string | null | undefined,
): number[] {
  if (!outputDir || outputDir.trim().length === 0) {
    return [];
  }
  const root = `${normalizePath(outputDir)}/`;
  const steps = new Set<number>();
  for (const model of models) {
    for (const checkpoint of model.checkpoints) {
      const path = normalizePath(checkpoint.path);
      if (!path.startsWith(root)) {
        continue;
      }
      if (path.slice(root.length).includes("/")) {
        continue;
      }
      const step = checkpointStepFromPath(path);
      if (step !== null && step > 0) {
        steps.add(step);
      }
    }
  }
  return [...steps].sort((left, right) => left - right);
}

export async function fetchRunCheckpointSteps(
  outputDir: string | null | undefined,
): Promise<number[]> {
  if (!outputDir || outputDir.trim().length === 0) {
    return [];
  }
  try {
    const { fetchCheckpoints } = await import(
      "@/features/export/api/export-api"
    );
    const inventory = await fetchCheckpoints();
    return checkpointStepsForOutputDir(inventory.models, outputDir);
  } catch {
    return [];
  }
}
