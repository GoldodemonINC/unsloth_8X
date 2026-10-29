// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type TrainingFailureKind =
  | "oom"
  | "nan"
  | "disk"
  | "auth"
  | "dataset"
  | "timeout"
  | "unknown";

export interface TrainingFailureDiagnosis {
  kind: TrainingFailureKind;
}

const RULES: ReadonlyArray<{
  kind: TrainingFailureKind;
  patterns: readonly string[];
}> = [
  {
    kind: "oom",
    patterns: [
      "out of memory",
      "outofmemoryerror",
      "out of vram",
      "unable to allocate",
      "cannot allocate memory",
      "memory allocation failed",
      "cublas_status_alloc_failed",
      "cuda error: out of memory",
      "insufficient memory",
      "not enough memory",
    ],
  },
  {
    kind: "nan",
    patterns: ["loss is nan", "loss became nan", "nan or inf", "non-finite loss", "loss: nan"],
  },
  {
    kind: "disk",
    patterns: ["no space left", "disk quota", "errno 28", "os error 112", "not enough space"],
  },
  {
    kind: "auth",
    patterns: [
      "401 client",
      "403 client",
      "gatedrepo",
      "gated repo",
      "access to model",
      "access denied",
      "repository not found",
      "unauthorized",
      "invalid token",
    ],
  },
  {
    kind: "dataset",
    patterns: [
      "failed to check dataset format",
      "could not load dataset",
      "dataset scripts are no longer supported",
      "dataset format",
      "no columns",
      "empty dataset",
    ],
  },
  {
    kind: "timeout",
    patterns: ["timed out", "timeout", "deadline exceeded", "took too long"],
  },
];

export function diagnoseTrainingFailure(
  message: string | null | undefined,
): TrainingFailureDiagnosis | null {
  const text = (message ?? "").trim().toLowerCase();
  if (text.length === 0) {
    return null;
  }
  for (const rule of RULES) {
    if (rule.patterns.some((pattern) => text.includes(pattern))) {
      return { kind: rule.kind };
    }
  }
  return { kind: "unknown" };
}
