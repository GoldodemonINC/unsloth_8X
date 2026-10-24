// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
