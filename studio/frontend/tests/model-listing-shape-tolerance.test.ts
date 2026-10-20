// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * A model listing must always hand callers an array.
 *
 * The pages read `models.length` / `models.filter(...)` straight off these responses, so a
 * payload that is not the documented envelope (a bare array from a proxy or stub, a null
 * `models`, a payload the backend rejected) used to put `undefined` into state and crash the
 * export page: "Cannot read properties of undefined (reading 'filter')".
 */

import assert from "node:assert/strict";
import test from "node:test";

import { registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();

const { setAuthFetchHandler } = await import("./helpers/store-stubs/auth.ts");

const { listLocalModels: listTrainingLocalModels } = await import(
  "../src/features/training/api/models-api.ts"
);
const { listLocalModels: listHubLocalModels } = await import(
  "../src/features/hub/inventory/api.ts"
);
const { fetchCheckpoints } = await import(
  "../src/features/export/api/export-api.ts"
);

const ROWS = [
  { id: "a", display_name: "A", path: "/models/a", source: "models_dir" },
];

function respondWith(body: unknown): void {
  setAuthFetchHandler(
    () =>
      new Response(JSON.stringify(body), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
  );
}

test("a bare array from /api/models/local is read as rows", async () => {
  respondWith(ROWS);
  assert.deepEqual(await listTrainingLocalModels(), ROWS);
});

test("a listing without usable rows is empty, never undefined", async () => {
  for (const body of [{}, { models_dir: "/models" }, { models: null }]) {
    respondWith(body);
    assert.deepEqual(await listTrainingLocalModels(), []);
  }
});

test("the documented envelope is still read as-is", async () => {
  respondWith({ models_dir: "/models", lmstudio_dirs: [], models: ROWS });
  assert.deepEqual(await listTrainingLocalModels(), ROWS);
});

test("the hub inventory listing normalises the same shapes", async () => {
  respondWith(ROWS);
  const bare = await listHubLocalModels();
  assert.deepEqual(bare.models, ROWS);
  assert.equal(bare.models_dir, "");

  respondWith({ models: null });
  assert.deepEqual((await listHubLocalModels()).models, []);
});

test("the checkpoint listing normalises the same shapes", async () => {
  respondWith(ROWS);
  const bare = await fetchCheckpoints();
  assert.deepEqual(bare.models, ROWS);
  assert.equal(bare.outputs_dir, "");

  respondWith({ outputs_dir: "/out", models: null });
  const envelope = await fetchCheckpoints();
  assert.deepEqual(envelope.models, []);
  assert.equal(envelope.outputs_dir, "/out");
});

test("no fetch handler outlives this file", () => {
  setAuthFetchHandler(null);
});
