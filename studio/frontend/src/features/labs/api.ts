// SPDX-License-Identifier: NRC-2
// Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-2.md

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

const BASE = "/api/labs";

export type LabsFeatureStatus = "alpha" | "beta";

export interface LabsFeature {
  id: string;
  label: string;
  description: string;
  status: LabsFeatureStatus;
  studio: boolean;
  enabled: boolean;
}

export interface LabsState {
  enabled: boolean;
  features: LabsFeature[];
  note: string;
}

export interface ArtifactAction {
  id: "continue-training" | "scale" | "merge" | "quantize" | "benchmark";
  enabled: boolean;
  reason: string | null;
}

export interface ArtifactManifest {
  stage?: string | null;
  run_id?: string | null;
  created_at?: string | null;
  studio_version?: string | null;
}

export interface Artifact {
  path: string;
  name: string;
  modified: string;
  model_type: string | null;
  params: number | null;
  weight_bytes: number;
  adapter: boolean;
  quantized: boolean;
  moe: boolean;
  supported: boolean;
  manifest: ArtifactManifest | null;
  manifest_path: string | null;
  actions: ArtifactAction[];
  raw_manifest?: unknown;
  manifest_error?: string;
}

export interface ArtifactList {
  artifacts: Artifact[];
  root: string;
}

export type PipelineKind =
  | "merge"
  | "scale"
  | "train"
  | "quantize"
  | "benchmark";

export interface PipelineStageSpec {
  kind: PipelineKind;
  factor?: number;
  method?: string;
  steps?: number;
  batch_size?: number;
  sequence_length?: number;
  dataset?: string;
  level?: string;
  baseline?: string;
  suites?: string[];
  parents?: string[];
}

export interface PipelineSpec {
  name?: string;
  source?: string;
  stages: PipelineStageSpec[];
}

export interface PipelineStagePlan {
  index: number;
  kind: PipelineKind;
  label: string;
  params_before: number | null;
  params_after: number | null;
  output_bytes: number;
  warnings: string[];
  blocked: boolean;
  reason: string | null;
  details: Record<string, unknown>;
}

export interface PipelinePlan {
  source: string | null;
  stages: PipelineStagePlan[];
  final_params: number | null;
  total_output_bytes: number;
  total_output_label: string;
  peak_stage_bytes: number;
  peak_stage_label: string;
  warnings: string[];
  blocked: boolean;
  blocked_reason: string | null;
  notes: string[];
}

export interface SavedPipeline {
  id: string;
  name: string;
  created_at: string;
  spec: PipelineSpec;
  plan: {
    final_params: number | null;
    total_output_bytes: number;
    blocked: boolean;
    blocked_reason: string | null;
    stage_kinds: PipelineKind[];
  };
}

async function readJson<T>(response: Response, what: string): Promise<T> {
  if (!response.ok) {
    throw new Error(await readFastApiError(response));
  }
  try {
    return (await response.json()) as T;
  } catch {
    throw new Error(`The ${what} response was not JSON.`);
  }
}

function jsonRequest(body: unknown, method: "POST" | "PUT"): RequestInit {
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

export async function loadLabsState(signal?: AbortSignal): Promise<LabsState> {
  const response = await authFetch(BASE, { method: "GET", signal });
  return readJson<LabsState>(response, "Labs");
}

export async function saveLabsState(payload: {
  enabled?: boolean;
  features?: Record<string, boolean>;
}): Promise<LabsState> {
  const response = await authFetch(BASE, jsonRequest(payload, "PUT"));
  return readJson<LabsState>(response, "Labs");
}

export async function loadArtifacts(
  limit = 50,
  signal?: AbortSignal,
): Promise<ArtifactList> {
  const response = await authFetch(`${BASE}/artifacts?limit=${limit}`, {
    method: "GET",
    signal,
  });
  return readJson<ArtifactList>(response, "Artifacts");
}

export async function loadArtifactDetail(
  path: string,
  signal?: AbortSignal,
): Promise<Artifact> {
  const query = encodeURIComponent(path);
  const response = await authFetch(`${BASE}/artifacts/detail?path=${query}`, {
    method: "GET",
    signal,
  });
  return readJson<Artifact>(response, "Artifact");
}

export async function planPipeline(
  spec: PipelineSpec,
  signal?: AbortSignal,
): Promise<PipelinePlan> {
  const response = await authFetch(`${BASE}/pipelines/plan`, {
    ...jsonRequest({ spec }, "POST"),
    signal,
  });
  return readJson<PipelinePlan>(response, "pipeline plan");
}

export async function listPipelines(
  signal?: AbortSignal,
): Promise<SavedPipeline[]> {
  const response = await authFetch(`${BASE}/pipelines`, {
    method: "GET",
    signal,
  });
  const body = await readJson<{ pipelines: SavedPipeline[] }>(
    response,
    "pipelines",
  );
  return body.pipelines;
}

export async function savePipeline(spec: PipelineSpec): Promise<SavedPipeline> {
  const response = await authFetch(
    `${BASE}/pipelines`,
    jsonRequest({ spec }, "POST"),
  );
  const body = await readJson<{ pipeline: SavedPipeline }>(
    response,
    "saved pipeline",
  );
  return body.pipeline;
}

export async function deletePipeline(id: string): Promise<void> {
  const response = await authFetch(
    `${BASE}/pipelines/${encodeURIComponent(id)}`,
    { method: "DELETE" },
  );
  if (!response.ok) {
    throw new Error(await readFastApiError(response));
  }
}

export interface LineageNode {
  id: string;
  label: string;
  path: string;
  stage: string;
  created_at: string;
  studio_version: string | null;
  valid: boolean;
  problems: string[];
}

export interface LineageEdge {
  from: string;
  to: string;
  label: string;
}

export interface LineageReport {
  root: string;
  nodes: LineageNode[];
  edges: LineageEdge[];
  roots: string[];
  truncated: boolean;
}

export async function loadLineage(
  signal?: AbortSignal,
): Promise<LineageReport> {
  const response = await authFetch(`${BASE}/lineage`, {
    method: "GET",
    signal,
  });
  return readJson<LineageReport>(response, "lineage");
}

export interface BenchmarkRecord {
  id: string;
  job_id: string;
  artifact: string;
  artifact_name: string;
  baseline: string;
  baseline_name: string;
  suites: string[];
  limit: number | null;
  batch_size: number;
  created_at: string;
  duration_seconds: number | null;
  candidate_scores: Record<string, number>;
  baseline_scores: Record<string, number>;
  verdict: string;
  comparison: {
    entries: Array<{
      suite?: string;
      baseline?: number;
      candidate?: number;
      delta?: number;
    }>;
    aggregate_delta: number | null;
    verdict: string | null;
    regressions: string[];
    improved: string[];
    flat: string[];
  };
  errors: string[];
  warnings: string[];
  report_dir: string;
}

export interface BenchmarkRunBody {
  artifact: string;
  baseline: string;
  suites: string[];
  limit?: number | null;
  batch_size?: number;
  timeout_seconds?: number;
}

export interface LabsJob {
  job_id: string;
  kind: string;
  status: "running" | "completed" | "failed";
  output_dir?: string | null;
  progress?: {
    stage?: string;
    done?: number;
    total?: number;
    message?: string;
    updated_at?: number;
  } | null;
  result: Record<string, unknown> | null;
  error: string | null;
}

export async function loadBenchmarks(
  limit = 50,
  artifact?: string,
  signal?: AbortSignal,
): Promise<BenchmarkRecord[]> {
  const query = new URLSearchParams({ limit: String(limit) });
  if (artifact) query.set("artifact", artifact);
  const response = await authFetch(`${BASE}/benchmarks?${query}`, {
    method: "GET",
    signal,
  });
  const body = await readJson<{ records: BenchmarkRecord[] }>(
    response,
    "benchmarks",
  );
  return body.records;
}

export async function runBenchmark(body: BenchmarkRunBody): Promise<LabsJob> {
  const response = await authFetch(`${BASE}/benchmarks/run`, {
    ...jsonRequest(body, "POST"),
  });
  return readJson<LabsJob>(response, "benchmark run");
}

export async function deleteBenchmark(id: string): Promise<void> {
  const response = await authFetch(
    `${BASE}/benchmarks/${encodeURIComponent(id)}`,
    { method: "DELETE" },
  );
  if (!response.ok) {
    throw new Error(await readFastApiError(response));
  }
}

export interface HousekeepingManifest {
  stage?: string | null;
  run_id?: string | null;
  created_at?: string | null;
}

export interface HousekeepingEntry {
  path: string;
  name: string;
  root: "outputs" | "exports" | "hf-hub" | "hf-datasets";
  kind: string;
  bytes: number;
  bytes_label: string;
  file_count: number;
  modified: string;
  manifest: HousekeepingManifest | null;
  deletable: boolean;
  protected_reason: string | null;
}

export interface HousekeepingRoot {
  id: string;
  label: string;
  note: string;
  path: string;
  exists: boolean;
  entry_count: number;
  bytes: number;
}

export interface HousekeepingReport {
  roots: HousekeepingRoot[];
  entries: HousekeepingEntry[];
  total_bytes: number;
  total_label: string;
  reclaimable_bytes: number;
  reclaimable_label: string;
  free_bytes: number;
  free_label: string;
  disk_total_bytes: number;
  protected: Array<{ path: string; reason: string }>;
  notes: string[];
}

export interface HousekeepingDeleteResult {
  deleted: Array<{
    path: string;
    bytes: number;
    bytes_label: string;
    file_count: number;
  }>;
  refused: Array<{ path: string; reason: string }>;
  freed_bytes: number;
  freed_label: string;
}

export async function scanHousekeeping(
  signal?: AbortSignal,
): Promise<HousekeepingReport> {
  const response = await authFetch(`${BASE}/housekeeping`, {
    method: "GET",
    signal,
  });
  return readJson<HousekeepingReport>(response, "housekeeping");
}

export async function deleteHousekeeping(
  paths: string[],
): Promise<HousekeepingDeleteResult> {
  const response = await authFetch(`${BASE}/housekeeping/delete`, {
    ...jsonRequest({ paths, confirm: true }, "POST"),
  });
  return readJson<HousekeepingDeleteResult>(response, "delete");
}

export interface CmtDataset {
  path: string;
  repo_id?: string | null;
  /**
   * Whether the plan actually measured this dataset. False for a Hub dataset,
   * whose rows and tokens stay placeholders until the run downloads it. Never
   * infer this from a zero: an empty local file measures as zero rows too.
   */
  measured: boolean;
  name: string;
  text_column: string | null;
  eval_fraction: number;
  bytes: number;
  rows: number;
  eval_rows: number;
  train_rows: number;
  tokens_estimate: number;
  notes: string[];
}

export interface CmtPlan {
  source: string;
  mode: "cpt" | "finetune";
  added_source: "tokens" | "override";
  datasets: CmtDataset[];
  totals: {
    datasets: number;
    /** The figures below are subtotals over the measured datasets only. */
    measured_datasets: number;
    unmeasured_datasets: number;
    rows: number;
    train_rows: number;
    eval_rows: number;
    tokens_estimate: number;
    train_tokens_estimate: number;
    train_tokens_label: string;
  };
  sizing: {
    rule: string;
    added_params: number;
    added_label: string;
    suggested_from_tokens: number;
    suggested_from_rows: number;
    min_added_params: number;
    max_added_params: number;
    note: string;
  };
  expansion: {
    base_params: number;
    base_label: string;
    target_params: number;
    target_label: string;
    added_params: number;
    added_label: string;
    growth_factor: number;
    params_per_layer: number;
    new_layers: number;
    floor_bypassed: boolean;
    growth_floor: number | null;
    supported: boolean;
    notes: string[];
  };
  training: {
    mode: string;
    method: string;
    max_steps: number;
    suggested_steps: number;
    batch_size: number;
    gradient_accumulation: number;
    max_seq_length: number;
    learning_rate: number;
    eval_every: number;
    patience: number;
    seed: number;
    train_tokens_per_step: number;
    note: string;
  };
  dependencies: {
    available: boolean;
    missing: string[];
    install_hint: string;
  };
  warnings: string[];
  blocked: boolean;
  blocked_reason: string | null;
  runnable: boolean;
  notes: string[];
}

export interface CmtDatasetSpec {
  /** A local file. Empty when the entry names a Hub dataset instead. */
  path: string;
  repo_id?: string;
  subset?: string | null;
  split?: string | null;
  name?: string;
  text_column?: string | null;
  eval_fraction?: number;
}

export interface CmtSpec {
  source: string;
  datasets: CmtDatasetSpec[];
  mode: "cpt" | "finetune";
  added_params?: number | null;
  max_steps?: number;
  batch_size?: number;
  gradient_accumulation?: number;
  max_seq_length?: number;
  learning_rate?: number;
  eval_every?: number;
  patience?: number;
  seed?: number;
}

export async function planCmt(
  spec: CmtSpec,
  signal?: AbortSignal,
): Promise<CmtPlan> {
  const response = await authFetch(`${BASE}/cmt/plan`, {
    ...jsonRequest({ spec }, "POST"),
    signal,
  });
  return readJson<CmtPlan>(response, "CMT plan");
}

export async function runCmt(spec: CmtSpec): Promise<LabsJob> {
  const response = await authFetch(`${BASE}/cmt/run`, {
    ...jsonRequest({ spec }, "POST"),
  });
  return readJson<LabsJob>(response, "CMT run");
}

export interface AutoScaleSpec {
  /** The checkpoint to grow: a local folder or a Hub id. */
  model: string;
  /** Local dataset files, or Hub ids the picker committed, the growth is sized from. */
  datasets?: Array<{ path?: string; repo_id?: string }>;
  method?: "depth" | "width";
  insert_strategy?: string;
  /** Blank lets the sizing rule decide; otherwise 1M-700M. */
  added_params?: number | null;
  rows?: number;
  tokens?: number;
}

export interface AutoScalePlan {
  model: string;
  method: string;
  sizing: {
    rule: string;
    rows: number;
    tokens: number;
    added_params: number;
    added_label: string;
    source: string;
    note: string;
  };
  datasets: Array<{
    path?: string;
    repo_id?: string;
    name: string;
    rows: number;
    bytes: number;
    tokens_estimate: number;
  }>;
  expansion: {
    base_params: number;
    base_label: string;
    added_params: number;
    added_label: string;
    target_params: number;
    target_label: string;
    growth_factor: number;
    params_per_layer: number;
    new_layers: number;
    floor_bypassed: boolean;
  };
  warnings: string[];
  blocked: boolean;
  blocked_reason: string | null;
  runnable: boolean;
  notes: string[];
}

export interface AutoScaleResult {
  output_dir: string;
  checkpoint: string;
  base_params: number;
  added_params: number;
  final_params: number;
  growth_factor: number;
  new_layers: number;
  warnings?: string[];
}

export async function planAutoScale(
  spec: AutoScaleSpec,
  signal?: AbortSignal,
): Promise<AutoScalePlan> {
  const response = await authFetch(`${BASE}/auto-scale/plan`, {
    ...jsonRequest({ spec }, "POST"),
    signal,
  });
  return readJson<AutoScalePlan>(response, "scale plan");
}

export async function runAutoScale(spec: AutoScaleSpec): Promise<LabsJob> {
  const response = await authFetch(`${BASE}/auto-scale/run`, {
    ...jsonRequest({ spec }, "POST"),
  });
  return readJson<LabsJob>(response, "scale run");
}
