// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import type { GradientCheckpointing, TrainingMethod } from "@/types/training";

export interface TrainingFitReport {
  model: string;
  method: string;
  params: number;
  params_label: string;
  trainable_params: number;
  config: {
    hidden_size: number;
    num_layers: number;
    vocab_size: number;
    intermediate_size: number;
    batch_size: number;
    gradient_accumulation: number;
    max_seq_length: number;
    lora_rank: number;
    load_in_4bit: boolean;
    gradient_checkpointing: boolean;
  };
  budget_gb: number | null;
  required_gb: number;
  headroom_gb: number | null;
  verdict: "fits" | "tight" | "over" | "unknown";
  breakdown: {
    weights_gb: number;
    adapter_gb: number;
    optimizer_gb: number;
    activations_gb: number;
    logits_gb: number;
    overhead_gb: number;
    total_gb: number;
  };
  suggestions: string[];
  assumptions: string[];
}

function fitMethod(method: TrainingMethod): string {
  if (method === "full" || method === "cpt") return method;
  return method === "qlora" ? "qlora" : "lora";
}

export async function fetchTrainingFit(input: {
  model: string;
  method: TrainingMethod;
  loadIn4Bit: boolean;
  maxSeqLength: number;
  batchSize: number;
  gradientAccumulation: number;
  loraRank: number;
  gradientCheckpointing: GradientCheckpointing;
  gpuMemoryGb?: number | null;
}): Promise<TrainingFitReport> {
  const response = await authFetch("/api/train/fit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: input.model,
      method: fitMethod(input.method),
      load_in_4bit: input.loadIn4Bit,
      max_seq_length: input.maxSeqLength,
      batch_size: input.batchSize,
      gradient_accumulation: input.gradientAccumulation,
      lora_rank: input.loraRank,
      gradient_checkpointing: input.gradientCheckpointing !== "none",
      gpu_memory_gb: input.gpuMemoryGb ?? null,
    }),
  });
  if (!response.ok) {
    throw new Error(await readFastApiError(response));
  }
  return (await response.json()) as TrainingFitReport;
}
