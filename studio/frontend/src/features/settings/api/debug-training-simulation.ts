// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

const ENDPOINT = "/api/settings/debug/training-simulation";

export interface DebugTrainingSimulation {
  enabled: boolean;
  /** What the switch falls back to when the server cannot say. */
  defaultEnabled: boolean;
}

interface DebugTrainingSimulationPayload {
  enabled?: unknown;
  default_enabled?: unknown;
}

export function parseDebugTrainingSimulation(
  payload: DebugTrainingSimulationPayload | null | undefined,
): DebugTrainingSimulation {
  return {
    enabled: payload?.enabled === true,
    defaultEnabled: payload?.default_enabled === true,
  };
}

export async function loadDebugTrainingSimulation(
  signal?: AbortSignal,
): Promise<DebugTrainingSimulation> {
  const response = await authFetch(ENDPOINT, { method: "GET", signal });
  if (!response.ok) {
    throw new Error(await readFastApiError(response));
  }
  return parseDebugTrainingSimulation(await response.json());
}

export async function saveDebugTrainingSimulation(
  enabled: boolean,
): Promise<DebugTrainingSimulation> {
  const response = await authFetch(ENDPOINT, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled }),
  });
  if (!response.ok) {
    throw new Error(await readFastApiError(response));
  }
  return parseDebugTrainingSimulation(await response.json());
}
