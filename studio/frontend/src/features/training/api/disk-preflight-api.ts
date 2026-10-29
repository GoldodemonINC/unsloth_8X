// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import type { TrainingMethod } from "@/types/training";

export interface DiskPreflightReport {
  status: "ok" | "tight" | "over" | "unknown";
  reason: string | null;
  path: string | null;
  free_gb: number | null;
  required_gb: number | null;
  headroom_gb: number | null;
  reserve_gb: number;
  breakdown: {
    base_weights_gb: number;
    checkpoints_gb: number;
    optimizer_gb: number;
    final_save_gb: number;
    reserve_gb: number;
    total_gb: number;
  };
  checkpoints: {
    count: number;
    size_gb: number;
    known: boolean;
    save_steps?: number;
    max_steps?: number;
  };
  params: number;
  batch_size: number;
  max_seq_length: number;
  method: string;
  load_in_4bit: boolean;
  suggestions: string[];
  assumptions: string[];
  blockers: string[];
}

function diskMethod(method: TrainingMethod): "lora" | "qlora" | "full" | "cpt" {
  if (method === "full" || method === "cpt") return method;
  return method === "qlora" ? "qlora" : "lora";
}

export async function checkDiskPreflight(input: {
  model: string;
  method: TrainingMethod;
  loadIn4Bit: boolean;
  maxSeqLength: number;
  batchSize: number;
  gradientAccumulation: number;
  loraRank: number;
  maxSteps: number;
  saveSteps: number;
  modelKnownCached: boolean;
  signal?: AbortSignal;
}): Promise<DiskPreflightReport> {
  const response = await authFetch("/api/train/disk/preflight", {
    method: "POST",
    signal: input.signal,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: input.model,
      method: diskMethod(input.method),
      load_in_4bit: input.loadIn4Bit,
      max_seq_length: input.maxSeqLength,
      batch_size: input.batchSize,
      gradient_accumulation: input.gradientAccumulation,
      lora_rank: input.loraRank,
      max_steps: input.maxSteps,
      save_steps: input.saveSteps,
      model_known_cached: input.modelKnownCached,
    }),
  });
  if (!response.ok) {
    throw new Error(await readFastApiError(response, "Disk check failed"));
  }
  return (await response.json()) as DiskPreflightReport;
}
