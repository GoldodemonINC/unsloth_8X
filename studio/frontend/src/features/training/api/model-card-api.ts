import { authFetch } from "@/features/auth";

export type ModelCardCheckId =
  | "base_model"
  | "license"
  | "dataset"
  | "evaluation";

export interface ModelCardCheck {
  id: ModelCardCheckId;
  status: "present" | "missing";
  evidence: string | null;
}

export interface ModelCardLintReport {
  card_found: boolean;
  source_path: string | null;
  reason: string | null;
  ok: boolean;
  missing: ModelCardCheckId[];
  checks: ModelCardCheck[];
}

export async function lintCheckpointModelCard(
  path: string,
  signal?: AbortSignal,
): Promise<ModelCardLintReport> {
  const params = new URLSearchParams({ path });
  const response = await authFetch(`/api/train/model-card/lint?${params.toString()}`, {
    signal,
  });
  if (!response.ok) {
    throw new Error(`Model card check failed (${response.status})`);
  }
  return (await response.json()) as ModelCardLintReport;
}

export async function lintTrainingRunModelCard(
  runId: string,
  signal?: AbortSignal,
): Promise<ModelCardLintReport> {
  const response = await authFetch(
    `/api/train/runs/${encodeURIComponent(runId)}/model-card/lint`,
    { signal },
  );
  if (!response.ok) {
    throw new Error(`Model card check failed (${response.status})`);
  }
  return (await response.json()) as ModelCardLintReport;
}
