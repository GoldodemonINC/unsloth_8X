// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import type {
  ResumeSafetyReport,
  TrainingRunDeleteResponse,
  TrainingRunDetailResponse,
  TrainingRunListResponse,
  TrainingRunSummary,
} from "../types/history";

const readError = (r: Response): Promise<string> => readFastApiError(r);

export class HistoryRequestError extends Error {
  readonly status: number | null;
  readonly errorCode: string | null;

  constructor(
    message: string,
    status: number | null,
    errorCode: string | null = null,
  ) {
    super(message);
    this.name = "HistoryRequestError";
    this.status = status;
    this.errorCode = errorCode;
  }
}

async function readHistoryRequestError(
  response: Response,
): Promise<HistoryRequestError> {
  const fallbackResponse = response.clone();
  try {
    const payload = (await response.json()) as { detail?: unknown };
    const detail = payload.detail;
    if (detail && typeof detail === "object" && !Array.isArray(detail)) {
      const structured = detail as { code?: unknown; message?: unknown };
      if (typeof structured.message === "string" && structured.message) {
        return new HistoryRequestError(
          structured.message,
          response.status,
          typeof structured.code === "string" ? structured.code : null,
        );
      }
    }
  } catch {
    return new HistoryRequestError(
      await readError(fallbackResponse),
      response.status,
    );
  }
  return new HistoryRequestError(
    await readError(fallbackResponse),
    response.status,
  );
}

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw await readHistoryRequestError(response);
  }
  return (await response.json()) as T;
}

export interface TrainingRunFilters {
  query?: string;
  tag?: string;
}

export async function listTrainingRuns(
  limit = 50,
  offset = 0,
  signal?: AbortSignal,
  filters: TrainingRunFilters = {},
): Promise<TrainingRunListResponse> {
  const params = new URLSearchParams({
    limit: String(limit),
    offset: String(offset),
  });
  const query = filters.query?.trim();
  const tag = filters.tag?.trim();
  if (query) params.set("q", query);
  if (tag) params.set("tag", tag);
  const response = await authFetch(`/api/train/runs?${params.toString()}`, {
    signal,
  });
  return parseJson<TrainingRunListResponse>(response);
}

export async function getTrainingRun(
  runId: string,
  signal?: AbortSignal,
): Promise<TrainingRunDetailResponse> {
  const response = await authFetch(
    `/api/train/runs/${encodeURIComponent(runId)}`,
    { signal },
  );
  return parseJson<TrainingRunDetailResponse>(response);
}

export async function checkTrainingRunResume(
  runId: string,
  signal?: AbortSignal,
): Promise<ResumeSafetyReport> {
  const response = await authFetch(
    `/api/train/runs/${encodeURIComponent(runId)}/resume-check`,
    { method: "POST", signal },
  );
  return parseJson<ResumeSafetyReport>(response);
}

export async function deleteTrainingRun(
  runId: string,
  options?: { deleteArtifacts?: boolean; signal?: AbortSignal },
): Promise<TrainingRunDeleteResponse> {
  const query = options?.deleteArtifacts ? "?delete_artifacts=true" : "";
  const response = await authFetch(
    `/api/train/runs/${encodeURIComponent(runId)}${query}`,
    { method: "DELETE", signal: options?.signal },
  );
  return parseJson<TrainingRunDeleteResponse>(response);
}

export interface TrainingRunPatch {
  displayName?: string | null;
  tags?: string[];
  notes?: string | null;
}

export async function updateTrainingRun(
  runId: string,
  patch: TrainingRunPatch,
  signal?: AbortSignal,
): Promise<TrainingRunSummary> {
  const body: Record<string, unknown> = {};
  if ("displayName" in patch) body.display_name = patch.displayName;
  if ("tags" in patch) body.tags = patch.tags;
  if ("notes" in patch) body.notes = patch.notes;
  const response = await authFetch(
    `/api/train/runs/${encodeURIComponent(runId)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    },
  );
  return parseJson<TrainingRunSummary>(response);
}

export async function renameTrainingRun(
  runId: string,
  displayName: string | null,
  signal?: AbortSignal,
): Promise<TrainingRunSummary> {
  return updateTrainingRun(runId, { displayName }, signal);
}
