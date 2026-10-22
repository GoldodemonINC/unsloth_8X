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
    gpus: Array<{
      index: number;
      total_bytes: number;
      free_bytes: number;
      name: string;
    }>;
    system_ram_total_bytes: number;
    system_ram_free_bytes: number;
    disk_free_bytes: number;
  };
  torch_available: boolean;
  safetensors_available: boolean;
  trl_available: boolean;
  /** Honest CPU/GPU availability per feature, matching the platform scope table. */
  features?: Array<{
    key: string;
    label: string;
    cpu: boolean;
    gpu: boolean;
    note: string;
    available_here: boolean;
  }>;
  platform_line?: string;
  has_gpu?: boolean;
  merge_streams?: boolean;
  /** cpu without a GPU, gpu with one: decides the merge parent cap. */
  merge_tier?: "cpu" | "gpu";
  /** 3 on a CPU host, 6 with a GPU. */
  max_merge_models?: number;
  merge_cpu_note?: string;
}

export interface MergeValidation {
  ok: boolean;
  blocked: boolean;
  blocked_reason?: string;
  policy: { allow_remote_code: boolean; allow_non_safetensors: boolean };
  sources: Array<{
    label: string;
    source: string;
    allowed?: boolean | null;
    blocked?: boolean;
    blocked_reason?: string;
    warnings?: string[];
    notes?: string[];
    pickles?: string[];
    requires_remote_code?: boolean;
  }>;
  tokenizer: {
    status: "ok" | "warning" | "conflict" | "not_checked";
    ok?: boolean | null;
    conflicts?: string[];
    warnings?: string[];
    reason?: string;
  };
}

export interface StructuralEval {
  verdict:
    | "structurally-valid"
    | "structurally-broken"
    | "unloadable"
    | "pareto-positive"
    | "within-noise"
    | "regressed";
  structurally_valid: boolean;
  problems: string[];
  warnings: string[];
  tier: string;
  scope: string;
}

export interface AdvancedJob {
  job_id: string;
  kind: string;
  status: "running" | "completed" | "failed";
  created_at: number;
  finished_at: number | null;
  /** Where the job writes, resolved by the backend (absolute) when it started. */
  output_dir?: string | null;
  /** True when Studio's Library/Export/Chat scans will list this output as a model. */
  library_visible?: boolean;
  /** The lease the job acquired, and what it was sized for. */
  resources?: {
    status: string;
    need: { vram_bytes: number; ram_bytes: number; disk_bytes: number } | null;
    notes: string[];
  };
  /** Path to the run_manifest.json written beside the artifacts, when there is one. */
  run_manifest?: string | null;
  /** Step the job published for the progress bar; null until it resolves a source. */
  progress?: {
    stage:
      | "queued"
      | "resolving"
      | "scanning"
      | "merging"
      | "loading"
      | "expanding"
      | "saving"
      | "verifying"
      | "finalizing"
      | "completed"
      | "failed";
    done?: number;
    total?: number;
    message?: string;
    blended_tensors?: number;
    copied_tensors?: number;
    skipped_tensors?: number;
    bytes_written?: number;
    shards?: number;
    updated_at?: number;
  } | null;
  result: Record<string, unknown> | null;
  error: string | null;
}

export interface AdvancedOutputLocations {
  /** The folder Studio scans for trained and merged models. */
  outputs_root: string;
  /** Where GGUF and other exports land. */
  exports_root: string;
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
  /** One entry per parent, in request order (2 to 6). */
  models: MergeModelSpec[];
  /** Each parent planned against the first, which is the merge's target. */
  alignments?: MergePlanResponse["alignment"][];
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
  tier?: "cpu" | "gpu";
  max_models?: number;
  parent_cap_reason?: "ram" | "gpu";
  /** Absolute folder the backend will use when the output field is left blank. */
  suggested_output_dir?: string | null;
  /** Plan before compute: peak RAM/disk (a CPU merge is streaming, not full-load). */
  estimates?: {
    need: { vram_bytes: number; ram_bytes: number; disk_bytes: number };
    notes: string[];
    streaming: boolean;
    model_bytes: number;
  };
  tokenizer?: MergeValidation["tokenizer"];
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

/** One reason a Scaler run would be refused, reported before anything loads. */
export interface ScalerRefusal {
  code: string;
  message: string;
  /** False only for the "slider is under the 2x floor" note on an otherwise valid plan. */
  blocks: boolean;
}

export type ScalerMethod = "depth" | "mlp_width" | "depth+mlp_width";

export interface ScalerPlan {
  source: string;
  spec: {
    architecture: string;
    hidden_size: number | null;
    num_layers: number | null;
    intermediate_size: number | null;
    vocab_size: number | null;
  };
  /** Estimator count from config.json; the run reports the measured one. */
  base_params: number;
  /** Slider minimum: the checkpoint as it is. */
  min_params: number;
  /** Slider's amber line: 2x the base, below which growth is refused. */
  floor_params: number;
  /** Slider maximum: base + 3B. */
  max_params: number;
  cap_added_params: number;
  growth_floor: number;
  target_params: number;
  default_target_params: number;
  step_params: number;
  method: ScalerMethod;
  method_label: string;
  methods: Array<{ id: ScalerMethod; label: string }>;
  projection: {
    projected_params: number;
    added_params: number;
    growth_factor: number;
    new_num_layers: number;
    new_intermediate_size: number;
    current_num_layers: number;
    current_intermediate_size: number;
    params_per_layer: number;
  } | null;
  target_below_floor: boolean;
  refusals: ScalerRefusal[];
  /** A hard stop: the checkpoint itself cannot be expanded. */
  blocked: boolean;
  blocked_reason: string | null;
  /** Nothing stands between this plan and a run. */
  runnable: boolean;
  engine: { available: boolean; error: string | null };
  notes: string[];
}

export interface ScalerResult {
  output_dir: string;
  source: string;
  method: ScalerMethod;
  base_params: number;
  final_params: number;
  added_params: number;
  growth_factor: number;
  new_num_layers: number | null;
  new_intermediate_size: number | null;
  step0_max_logit_diff: number;
  verified: boolean;
  reload_max_logit_diff: number | null;
  peak_mem_gb: number;
  elapsed_seconds: number;
  notes: string[];
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

/** Where a merge has to write for Studio to list, quantize and serve the result. */
export function fetchOutputLocations(): Promise<AdvancedOutputLocations> {
  return request("/outputs");
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
  kind_multipliers?: Record<string, number>;
  /** Per-layer overrides; without these the plan would not match the run. */
  layer_multipliers?: Record<string, number>;
  depth_curve?: [number, number] | null;
  /** A local checkpoint whose parameter names fill in the per-layer rows. */
  model?: string | null;
}) {
  return request<{
    config: Record<string, unknown>;
    zones: Array<{
      zone: string;
      layers: number[];
      start: number | null;
      end: number | null;
    }>;
    num_layers?: number | null;
    notes?: string[];
    preview?: {
      multipliers: Record<string, number>;
      kind_multipliers: Record<string, number>;
      entries?: Array<{
        name: string;
        zone: string;
        kind: string;
        layer: number | null;
        multiplier: number;
        lr: number;
        numel: number;
        trainable: boolean;
      }>;
    };
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
  models: string[];
  top_k: number;
  temperature: number;
  hf_token?: string | null;
}): Promise<MergePlanResponse> {
  return request("/merge/plan", { method: "POST", body: JSON.stringify(body) });
}

export function runMerge(body: {
  /** 2 to 6 parents. */
  models: string[];
  /** One weight per parent; normalized server-side. */
  weights?: number[] | null;
  /** A bare name lands under the backend's outputs root; absolute is used as-is. */
  output_dir?: string | null;
  method: MergeMethod;
  /** Two-parent shorthand, ignored when ``weights`` is given. */
  weight?: number;
  density: number;
  max_shard_gb: number;
  hf_token?: string | null;
  allow_remote_code?: boolean;
  allow_non_safetensors?: boolean;
  allow_tokenizer_conflict?: boolean;
}): Promise<AdvancedJob> {
  return request("/merge/run", { method: "POST", body: JSON.stringify(body) });
}

/** The pre-merge gates: tokenizer agreement, pickle weights, remote code. */
export function validateMerge(body: {
  models: string[];
  hf_token?: string | null;
  allow_remote_code?: boolean;
  allow_non_safetensors?: boolean;
}): Promise<MergeValidation> {
  return request("/merge/validate", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

/** CPU structural merge-eval: loadable, index/files agree, no NaN, tokenizer present. */
export function evalMergedModel(body: {
  source: string;
  sample_tensors?: number;
}): Promise<StructuralEval> {
  return request("/merge/eval", { method: "POST", body: JSON.stringify(body) });
}

export function runMoeMerge(body: {
  models: string[];
  output_dir?: string | null;
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
  output_dir?: string | null;
  max_shard_gb?: number | null;
  reserve_fraction?: number;
  prefer?: "ram" | "vram" | "auto";
}): Promise<AdvancedJob> {
  return request("/split/run", { method: "POST", body: JSON.stringify(body) });
}

/** The Slider's range, the growth floor, and every refusal reason. Reads config.json only. */
export function planScaler(body: {
  source: string;
  target_params?: number | null;
  method?: ScalerMethod;
  hf_token?: string | null;
}): Promise<ScalerPlan> {
  return request("/scale/plan", { method: "POST", body: JSON.stringify(body) });
}

export function runScaler(body: {
  source: string;
  target_params: number;
  method: ScalerMethod;
  output_dir?: string | null;
  insert_strategy?: "even" | "middle" | "tail";
  hf_token?: string | null;
}): Promise<AdvancedJob> {
  return request("/scale/run", { method: "POST", body: JSON.stringify(body) });
}

/** A past merge or MMoE run, read from the manifest beside its output. */
export interface MergeHistoryEntry {
  dir: string;
  name: string;
  stage: string | null;
  created_at: string | null;
  run_id: string | null;
  models: string[];
  method: string | null;
  weights: number[] | null;
  density: number | null;
  top_k: number | null;
  temperature: number | null;
  total_bytes: number | null;
  tokenizer_override: boolean;
}

export function fetchMergeHistory(
  limit = 20,
): Promise<{ entries: MergeHistoryEntry[] }> {
  return request(`/merge/history?limit=${limit}`);
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
