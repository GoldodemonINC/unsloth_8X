// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TrainingRunDetailResponse } from "../types/history";

export const REPRODUCIBILITY_BUNDLE_SCHEMA = "unsloth.reproducibility-bundle.v1";

export interface ReproducibilityEnvironment {
  platform: string | null;
  python_version: string | null;
  torch: string | null;
  transformers: string | null;
  device_backend: string | null;
  gpu: string | null;
}

export interface ReproducibilityBundle {
  schema: string;
  created_at: string;
  run: {
    id: string;
    status: TrainingRunDetailResponse["run"]["status"];
    display_name: string | null;
    model_name: string;
    dataset_name: string;
    started_at: string;
    ended_at: string | null;
    total_steps: number | null;
    final_step: number | null;
    final_loss: number | null;
    duration_seconds: number | null;
    output_dir: string | null;
    resumed_later: boolean;
    tags: string[];
    notes: string | null;
  };
  config: Record<string, unknown>;
  environment: ReproducibilityEnvironment | null;
  dataset: {
    name: string;
    source: string | null;
    subset: string | null;
    train_split: string | null;
    eval_split: string | null;
    local_path: string | null;
    snapshot_path: string | null;
    hash: string | null;
  };
  seed: number | null;
  resume_from_checkpoint: string | null;
  metrics: {
    logged_steps: number;
    first_logged_step: number | null;
    last_logged_step: number | null;
    final_learning_rate: number | null;
    final_grad_norm: number | null;
    final_epoch: number | null;
    final_num_tokens: number | null;
    eval_points: number;
  };
}

function finiteOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function textOrNull(value: unknown): string | null {
  return typeof value === "string" && value.trim().length > 0 ? value : null;
}

function lastFinite(values: readonly number[]): number | null {
  for (let index = values.length - 1; index >= 0; index -= 1) {
    const value = values[index];
    if (Number.isFinite(value)) {
      return value;
    }
  }
  return null;
}

export function buildReproducibilityBundle(
  detail: TrainingRunDetailResponse,
  options: {
    now?: Date;
    environment?: ReproducibilityEnvironment | null;
  } = {},
): ReproducibilityBundle {
  const { run, config, metrics } = detail;
  const loggedSteps = metrics.step_history.filter((step) => Number.isFinite(step));
  const seedValue = config.random_seed ?? config.seed;

  return {
    schema: REPRODUCIBILITY_BUNDLE_SCHEMA,
    created_at: (options.now ?? new Date()).toISOString(),
    run: {
      id: run.id,
      status: run.status,
      display_name: run.display_name,
      model_name: run.model_name,
      dataset_name: run.dataset_name,
      started_at: run.started_at,
      ended_at: run.ended_at,
      total_steps: run.total_steps,
      final_step: run.final_step,
      final_loss: run.final_loss,
      duration_seconds: run.duration_seconds,
      output_dir: run.output_dir,
      resumed_later: run.resumed_later,
      tags: [...(run.tags ?? [])],
      notes: run.notes,
    },
    config,
    environment: options.environment ?? null,
    dataset: {
      name: run.dataset_name,
      source: textOrNull(config.dataset_source),
      subset: textOrNull(config.subset),
      train_split: textOrNull(config.train_split),
      eval_split: textOrNull(config.eval_split),
      local_path: textOrNull(config.dataset_local_path),
      snapshot_path: textOrNull(config.dataset_snapshot_path),
      hash: textOrNull(config.dataset_fingerprint),
    },
    seed: finiteOrNull(seedValue),
    resume_from_checkpoint: textOrNull(config.resume_from_checkpoint),
    metrics: {
      logged_steps: loggedSteps.length,
      first_logged_step: loggedSteps.length > 0 ? loggedSteps[0] : null,
      last_logged_step:
        loggedSteps.length > 0 ? loggedSteps[loggedSteps.length - 1] : null,
      final_learning_rate: lastFinite(metrics.lr_history),
      final_grad_norm: lastFinite(metrics.grad_norm_history),
      final_epoch: finiteOrNull(metrics.final_epoch),
      final_num_tokens: finiteOrNull(metrics.final_num_tokens),
      eval_points: metrics.eval_loss_history.filter((value) =>
        Number.isFinite(value),
      ).length,
    },
  };
}

export function reproducibilityBundleFilename(runId: string): string {
  return `unsloth-run-${runId}.json`;
}
