// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { importRecipePayload } = await import(
  "../src/features/recipe-studio/utils/import/importer.ts"
);

const RECIPES = [
  "reasoning-distill",
  "code-from-text",
  "tool-call-traces",
  "critique-revise",
] as const;

const SUPPORTED_COLUMN_TYPES = new Set([
  "sampler",
  "expression",
  "llm-text",
  "llm-structured",
  "llm-code",
  "llm-judge",
  "validation",
]);

type Column = Record<string, unknown>;

function load(id: string): Record<string, unknown> {
  const url = new URL(
    `../src/features/data-recipes/learning-recipes/${id}.json`,
    import.meta.url,
  );
  return JSON.parse(readFileSync(url, "utf8")) as Record<string, unknown>;
}

function columnsOf(payload: Record<string, unknown>): Column[] {
  const recipe = payload.recipe as Record<string, unknown>;
  return recipe.columns as Column[];
}

function referencesIn(value: unknown): string[] {
  if (typeof value !== "string") return [];
  return [...value.matchAll(/\{\{\s*([A-Za-z_][\w.]*)/g)].map((hit) => hit[1]);
}

for (const id of RECIPES) {
  test(`${id}: every column is importable and references only earlier columns`, () => {
    const payload = load(id);
    const recipe = payload.recipe as Record<string, unknown>;
    const columns = columnsOf(payload);
    const aliases = new Set(
      (recipe.model_configs as Column[]).map((model) => String(model.alias)),
    );
    const seeded = Boolean(recipe.seed_config);
    const known = new Set<string>(seeded ? ["chunk_text", "source_file"] : []);

    assert.ok(columns.length >= 4, `${id} should have a real flow`);

    for (const column of columns) {
      const type = String(column.column_type);
      assert.ok(
        SUPPORTED_COLUMN_TYPES.has(type),
        `${id}: ${String(column.name)} has unsupported type ${type}`,
      );
      for (const reference of [
        ...referencesIn(column.prompt),
        ...referencesIn(column.expr),
      ]) {
        const root = reference.split(".")[0];
        assert.ok(
          known.has(root),
          `${id}: ${String(column.name)} references {{ ${reference} }} before it exists`,
        );
      }
      if (type.startsWith("llm")) {
        assert.ok(
          aliases.has(String(column.model_alias)),
          `${id}: ${String(column.name)} points at unknown model alias`,
        );
      }
      if (type === "llm-judge") {
        const scores = column.scores as Column[];
        assert.ok(scores.length > 0, `${id}: judge needs at least one score`);
        for (const entry of scores) {
          assert.ok(
            Object.keys(entry.options as Record<string, string>).length > 1,
            `${id}: score ${String(entry.name)} needs graded options`,
          );
        }
      }
      if (type === "validation") {
        for (const target of column.target_columns as string[]) {
          assert.ok(
            known.has(target),
            `${id}: validator targets unknown column ${target}`,
          );
        }
      }
      if (column.sampler_type === "subcategory") {
        const params = column.params as Record<string, unknown>;
        assert.ok(
          known.has(String(params.category)),
          `${id}: subcategory parent ${String(params.category)} does not exist`,
        );
      }
      known.add(String(column.name));
    }
  });

  test(`${id}: the canvas has a node for every block`, () => {
    const payload = load(id);
    const recipe = payload.recipe as Record<string, unknown>;
    const ui = payload.ui as Record<string, unknown>;
    const nodeIds = new Set(
      (ui.nodes as Column[]).map((node) => String(node.id)),
    );
    for (const name of [
      ...columnsOf(payload).map((column) => String(column.name)),
      ...(recipe.model_configs as Column[]).map((model) => String(model.alias)),
      ...(recipe.model_providers as Column[]).map((provider) =>
        String(provider.name),
      ),
      ...(recipe.seed_config ? ["seed"] : []),
    ]) {
      assert.ok(nodeIds.has(name), `${id}: no canvas node for ${name}`);
    }
  });
}

for (const id of RECIPES) {
  test(`${id}: the recipe importer accepts the shipped payload`, () => {
    const url = new URL(
      `../src/features/data-recipes/learning-recipes/${id}.json`,
      import.meta.url,
    );
    const result = importRecipePayload(readFileSync(url, "utf8"));
    assert.deepEqual(result.errors, [], `${id} should import without errors`);
    assert.ok(result.snapshot, `${id} should produce a canvas snapshot`);
    const columns = columnsOf(load(id));
    assert.equal(
      Object.keys(result.snapshot.configs).length >= columns.length,
      true,
      `${id}: every column should become a canvas block`,
    );
    assert.ok(
      result.snapshot.edges.length > 0,
      `${id}: the imported canvas should be wired up`,
    );
  });
}

test("the new recipes are wired into the learning recipe index", () => {
  const index = readFileSync(
    new URL(
      "../src/features/data-recipes/learning-recipes/index.ts",
      import.meta.url,
    ),
    "utf8",
  );
  for (const id of RECIPES) {
    assert.match(index, new RegExp(`id: "${id}"`));
  }
});
