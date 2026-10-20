// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Named merge setups the user saved, kept in this browser.
 *
 * A merge setup is small (five repo ids and a handful of numbers), so localStorage
 * is enough — and it means a saved recipe travels with the browser that made it
 * rather than needing a backend table. The storage object is injected so the
 * round-trip can be tested without a DOM.
 */

import {
  parseRecipe,
  serializeRecipe,
  type MergeSetup,
} from "./merge-tools.ts";

export const MERGE_PRESET_STORAGE_KEY = "unsloth_merge_recipe_presets";
export const MAX_SAVED_PRESETS = 24;

export interface SavedMergePreset {
  id: string;
  name: string;
  /** When it was saved, in ms since the epoch; shown in the list. */
  savedAt: number;
  setup: MergeSetup;
}

export interface PresetStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

function isValidPreset(value: unknown): value is SavedMergePreset {
  if (!value || typeof value !== "object") {
    return false;
  }
  const record = value as Record<string, unknown>;
  if (typeof record.id !== "string" || typeof record.name !== "string") {
    return false;
  }
  // Reuse the recipe parser so a hand-edited or stale entry is measured by the
  // same rules a pasted recipe is.
  const parsed = parseRecipe(JSON.stringify(record.setup));
  return parsed.ok;
}

export function loadSavedPresets(
  storage: PresetStorage | null,
): SavedMergePreset[] {
  if (!storage) {
    return [];
  }
  let raw: string | null = null;
  try {
    raw = storage.getItem(MERGE_PRESET_STORAGE_KEY);
  } catch {
    return [];
  }
  if (!raw) {
    return [];
  }
  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed
      .filter(isValidPreset)
      .sort((a, b) => b.savedAt - a.savedAt)
      .slice(0, MAX_SAVED_PRESETS);
  } catch {
    return [];
  }
}

export function writeSavedPresets(
  storage: PresetStorage | null,
  presets: readonly SavedMergePreset[],
): boolean {
  if (!storage) {
    return false;
  }
  try {
    storage.setItem(
      MERGE_PRESET_STORAGE_KEY,
      JSON.stringify(presets.slice(0, MAX_SAVED_PRESETS)),
    );
    return true;
  } catch {
    return false;
  }
}

export function newPresetId(now = Date.now()): string {
  return `preset-${now.toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

/** Save, replacing a preset of the same name so repeat saves do not pile up. */
export function upsertSavedPreset(
  presets: readonly SavedMergePreset[],
  name: string,
  setup: MergeSetup,
  now = Date.now(),
): SavedMergePreset[] {
  const trimmed = name.trim();
  if (!trimmed) {
    return [...presets];
  }
  const entry: SavedMergePreset = {
    id: newPresetId(now),
    name: trimmed,
    savedAt: now,
    setup,
  };
  const others = presets.filter(
    (preset) => preset.name.trim().toLowerCase() !== trimmed.toLowerCase(),
  );
  return [entry, ...others].slice(0, MAX_SAVED_PRESETS);
}

export function removeSavedPreset(
  presets: readonly SavedMergePreset[],
  id: string,
): SavedMergePreset[] {
  return presets.filter((preset) => preset.id !== id);
}

/** The shareable text for a preset, so it can be pasted into a chat or an issue. */
export function presetRecipeText(preset: SavedMergePreset): string {
  return serializeRecipe(preset.setup, preset.name);
}
