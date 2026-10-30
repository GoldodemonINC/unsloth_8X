// SPDX-License-Identifier: NRC-3
// Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-3.md

export interface RetentionRun {
  id: string;
  output_dir: string | null;
  project_name: string | null;
  final_loss: number | null;
  final_step: number | null;
  started_at: string;
}

export interface RetentionEntry {
  path: string;
  bytes: number;
  deletable: boolean;
}

export interface RetentionPlan {
  candidates: RetentionEntry[];
  kept: RetentionEntry[];
  groups: number;
  keptRuns: number;
}

export function normalizeRetentionPath(path: string): string {
  return path.replace(/\\/g, "/").replace(/\/+$/, "");
}

function lossRank(loss: number | null): number {
  return loss !== null && Number.isFinite(loss) ? loss : Number.POSITIVE_INFINITY;
}

function compareRuns(left: RetentionRun, right: RetentionRun): number {
  const byLoss = lossRank(left.final_loss) - lossRank(right.final_loss);
  if (byLoss !== 0) return byLoss;
  const byStep = (right.final_step ?? -1) - (left.final_step ?? -1);
  if (byStep !== 0) return byStep;
  return right.started_at.localeCompare(left.started_at);
}

export function planRetention(
  entries: readonly RetentionEntry[],
  runs: readonly RetentionRun[],
  keepPerProject: number,
): RetentionPlan {
  const keep = Math.max(1, Math.floor(keepPerProject));
  const byPath = new Map<string, RetentionEntry>();
  for (const entry of entries) {
    byPath.set(normalizeRetentionPath(entry.path), entry);
  }

  const groups = new Map<string, RetentionRun[]>();
  for (const run of runs) {
    if (!run.output_dir) continue;
    const path = normalizeRetentionPath(run.output_dir);
    if (!byPath.has(path)) continue;
    const project = run.project_name?.trim() || "Unassigned";
    const bucket = groups.get(project) ?? [];
    bucket.push(run);
    groups.set(project, bucket);
  }

  const keptPaths = new Set<string>();
  let keptRuns = 0;
  for (const bucket of groups.values()) {
    bucket.sort(compareRuns);
    for (const [index, run] of bucket.entries()) {
      if (index < keep && run.output_dir) {
        keptPaths.add(normalizeRetentionPath(run.output_dir));
        keptRuns += 1;
      }
    }
  }

  const candidates: RetentionEntry[] = [];
  const kept: RetentionEntry[] = [];
  for (const path of keptPaths) {
    const entry = byPath.get(path);
    if (entry) kept.push(entry);
  }
  for (const bucket of groups.values()) {
    for (const run of bucket) {
      if (!run.output_dir) continue;
      const path = normalizeRetentionPath(run.output_dir);
      if (keptPaths.has(path)) continue;
      const entry = byPath.get(path);
      if (!entry || !entry.deletable) continue;
      if (!candidates.some((candidate) => candidate.path === entry.path)) {
        candidates.push(entry);
      }
    }
  }

  candidates.sort((left, right) => right.bytes - left.bytes);
  return { candidates, kept, groups: groups.size, keptRuns };
}
