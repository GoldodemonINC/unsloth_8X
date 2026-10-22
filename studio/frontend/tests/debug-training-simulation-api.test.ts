// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type * as DebugTrainingSimulationApi from "../src/features/settings/api/debug-training-simulation.ts";
import * as formatFastApiError from "../src/lib/format-fastapi-error.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const ENDPOINT = "/api/settings/debug/training-simulation";

function client(response: () => Response) {
  const requests: { path: string; init?: RequestInit }[] = [];
  const api = loadWithStubs<typeof DebugTrainingSimulationApi>(
    new URL(
      "../src/features/settings/api/debug-training-simulation.ts",
      import.meta.url,
    ),
    {
      "@/features/auth": {
        authFetch: async (path: string, init?: RequestInit) => {
          requests.push({ path, init });
          return response();
        },
      },
      "@/lib/format-fastapi-error": formatFastApiError,
    },
  );
  return { api, requests };
}

function json(body: unknown, status = 200) {
  return () =>
    new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    });
}

test("reads the toggle from the debug settings endpoint", async () => {
  const { api, requests } = client(
    json({ enabled: true, default_enabled: false }),
  );

  const state = await api.loadDebugTrainingSimulation();

  assert.deepEqual(state, { enabled: true, defaultEnabled: false });
  assert.equal(requests.length, 1);
  assert.equal(requests[0].path, ENDPOINT);
  assert.equal(requests[0].init?.method, "GET");
});

test("writes the toggle as a JSON boolean", async () => {
  const { api, requests } = client(
    json({ enabled: true, default_enabled: false }),
  );

  const saved = await api.saveDebugTrainingSimulation(true);

  assert.equal(saved.enabled, true);
  assert.equal(requests[0].init?.method, "PUT");
  assert.deepEqual(requests[0].init?.headers, {
    "Content-Type": "application/json",
  });
  assert.equal(requests[0].init?.body, JSON.stringify({ enabled: true }));
});

test("a refusal surfaces the server's message instead of a silent off", async () => {
  const { api } = client(json({ detail: "Could not save the setting." }, 500));

  await assert.rejects(
    () => api.saveDebugTrainingSimulation(true),
    /Could not save the setting\./,
  );
});

test("a payload without booleans reads as off, never as on", () => {
  const { api } = client(json({}));

  assert.deepEqual(api.parseDebugTrainingSimulation(null), {
    enabled: false,
    defaultEnabled: false,
  });
  assert.deepEqual(api.parseDebugTrainingSimulation({ enabled: "yes" }), {
    enabled: false,
    defaultEnabled: false,
  });
  assert.deepEqual(api.parseDebugTrainingSimulation({ enabled: 1 }), {
    enabled: false,
    defaultEnabled: false,
  });
});
