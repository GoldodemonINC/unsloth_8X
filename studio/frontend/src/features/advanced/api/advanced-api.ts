// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import type { AdvancedTrainingRequest, MergeMethod } from "../types";

export interface AdvancedCapabilities {
  qat_formats: string[];
  merge_methods: string[];
  rl_methods: string[];
  reward_presets: string[];
  layer_zones: string[];
  eval_suites: Array<{
    id: string;
    label: string;
    kind: string;
    shots: number;
    metric: string;
    description: string;
  }>;
  eval_harness: { available: boolean; install_hint: string };
  memory: {
    gpus: Array<{ index: number; total_bytes: number; free_bytes: number; name: string }>;
    system_ram_total_bytes: number;
    system_ram_free_bytes: number;
    disk_free_bytes: number;
  };
  torch_available: boolean;
  safetensors_available: boolean;
  trl_available: boolean;
}

export interface AdvancedJob {
  job_id: string;
  kind: string;
  status: "running" | "completed" | "failed";
  created_at: number;
  finished_at: number | null;
  result: Record<string, unknown> | null;
  error: string | null;
}

export interface MergeModelSpec {
  source: string;
  architecture: string;
  hidden_size: number | null;
  num_layers: number | null;
  num_heads: number | null;
  intermediate_size: number | null;
  vocab_size: number | null;
  parameter_count: number | null;
}

export interface MergePlanResponse {
  models: [MergeModelSpec, MergeModelSpec];
  alignment: {
    mode: "identical" | "cross_scale" | "cross_architecture";
    target_hidden: number | null;
    target_layers: number | null;
    target_intermediate: number | null;
    target_vocab: number | null;
    hidden_strategy: string;
    layer_strategy: string;
    vocab_strategy: string;
    notes: string[];
    warnings: string[];
    mmoe_required: boolean;
  };
  moe: {
    experts: Array<{
      index: number;
      source: string;
      architecture: string;
      hidden_size: number | null;
      intermediate_size: number | null;
      num_layers: number | null;
      layer_range: [number, number] | null;
    }>;
    top_k: number;
    router_hidden: number;
    router_temperature: number;
    shared_vocab: number | null;
    notes: string[];
    warnings: string[];
  };
  recommended_mode: "standard" | "mmoe";
}

export interface SplitPlanResponse {
  source: string;
  memory_profile: AdvancedCapabilities["memory"];
  budget: {
    max_shard_bytes: number;
    usable_ram_bytes: number;
    usable_vram_bytes: number;
    reserve_fraction: number;
    notes: string[];
  };
  plan: {
    shard_count: number;
    total_bytes: number;
    max_shard_bytes: number;
    whole_tensors: number;
    split_tensors: number;
    oversized_tensors: number;
    warnings: string[];
  };
  fits_on_disk: boolean;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await authFetch(`/api/advanced${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!response.ok) {
    throw new Error(await readFastApiError(response));
  }
  return (await response.json()) as T;
}

export function fetchAdvancedCapabilities(): Promise<AdvancedCapabilities> {
  return request("/capabilities");
}

export function validateQat(body: AdvancedTrainingRequest["qat"]) {
  return request<{ config: Record<string, unknown> }>("/qat/validate", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function validateRl(body: {
  method: "dpo" | "grpo";
  config: Record<string, unknown>;
  dataset_columns?: string[];
  num_rows?: number;
}) {
  return request<Record<string, unknown>>("/rl/validate", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function planCurriculum(body: {
  cadence: string;
  total_steps?: number | null;
  datasets?: string[] | null;
}) {
  return request<{
    stages: Array<{ name: string; steps: number; dataset: string | null }>;
    cycle_steps: number;
    notes: string[];
    warnings: string[];
  }>("/curriculum/plan", { method: "POST", body: JSON.stringify(body) });
}

export function planLayerLr(body: {
  base_learning_rate: number;
  num_layers?: number | null;
  zone_multipliers?: Record<string, number>;
  depth_curve?: [number, number] | null;
}) {
  return request<{
    config: Record<string, unknown>;
    zones: Array<{
      zone: string;
      layers: number[];
      start: number | null;
      end: number | null;
    }>;
  }>("/layer-lr/plan", { method: "POST", body: JSON.stringify(body) });
}

export function planBlend(body: {
  slots: Array<Record<string, unknown>>;
  batch_size: number;
}) {
  return request<{
    ratios: Record<string, number>;
    batch_layout: string[];
    warnings: string[];
  }>("/blend/plan", { method: "POST", body: JSON.stringify(body) });
}

export function planMerge(body: {
  model_a: string;
  model_b: string;
  top_k: number;
  temperature: number;
  hf_token?: string | null;
}): Promise<MergePlanResponse> {
  return request("/merge/plan", { method: "POST", body: JSON.stringify(body) });
}

export function runMerge(body: {
  model_a: string;
  model_b: string;
  output_dir: string;
  method: MergeMethod;
  weight: number;
  density: number;
  max_shard_gb: number;
  hf_token?: string | null;
}): Promise<AdvancedJob> {
  return request("/merge/run", { method: "POST", body: JSON.stringify(body) });
}

export function runMoeMerge(body: {
  models: string[];
  output_dir: string;
  top_k: number;
  temperature: number;
  hf_token?: string | null;
}): Promise<AdvancedJob> {
  return request("/merge/moe", { method: "POST", body: JSON.stringify(body) });
}

export function planSplit(body: {
  source: string;
  max_shard_gb?: number | null;
  reserve_fraction?: number;
  prefer?: "ram" | "vram" | "auto";
}): Promise<SplitPlanResponse> {
  return request("/split/plan", { method: "POST", body: JSON.stringify(body) });
}

export function runSplit(body: {
  source: string;
  output_dir: string;
  max_shard_gb?: number | null;
  reserve_fraction?: number;
  prefer?: "ram" | "vram" | "auto";
}): Promise<AdvancedJob> {
  return request("/split/run", { method: "POST", body: JSON.stringify(body) });
}

export function fetchJob(jobId: string): Promise<AdvancedJob> {
  return request(`/jobs/${jobId}`);
}

export async function waitForJob(
  jobId: string,
  onUpdate?: (job: AdvancedJob) => void,
  intervalMs = 1200,
  timeoutMs = 24 * 60 * 60 * 1000,
): Promise<AdvancedJob> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const job = await fetchJob(jobId);
    onUpdate?.(job);
    if (job.status !== "running") {
      return job;
    }
    if (Date.now() > deadline) {
      throw new Error("The job is still running; check back later.");
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}
