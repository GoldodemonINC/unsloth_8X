// SPDX-License-Identifier: NRC-3
// Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-3.md

import type { GradientCheckpointing, TrainingMethod } from "@/types/training";

export interface TrainingPresetValues {
  trainingMethod: TrainingMethod;
  loraRank: number;
  loraAlpha: number;
  loraDropout: number;
  learningRate: number;
  epochs: number;
  batchSize: number;
  gradientAccumulation: number;
  contextLength: number;
  gradientCheckpointing: GradientCheckpointing;
  packing: boolean;
  trainOnCompletions: boolean;
}

export interface TrainingPreset {
  id: string;
  name: string;
  description: string;
  values: TrainingPresetValues;
}

export interface StoredTrainingPreset extends TrainingPreset {
  createdAt: string;
}

export const BUILTIN_PRESETS: readonly TrainingPreset[] = [
  {
    id: "chat-quick",
    name: "Chat, one pass",
    description: "A first conversational tune that finishes on a small dataset.",
    values: {
      trainingMethod: "qlora",
      loraRank: 16,
      loraAlpha: 32,
      loraDropout: 0,
      learningRate: 2e-4,
      epochs: 1,
      batchSize: 2,
      gradientAccumulation: 4,
      contextLength: 2048,
      gradientCheckpointing: "unsloth",
      packing: false,
      trainOnCompletions: true,
    },
  },
  {
    id: "instruction-code",
    name: "Instruction / code",
    description: "Ranks the adapter up for instruction following and code edits.",
    values: {
      trainingMethod: "qlora",
      loraRank: 32,
      loraAlpha: 64,
      loraDropout: 0,
      learningRate: 1e-4,
      epochs: 2,
      batchSize: 1,
      gradientAccumulation: 8,
      contextLength: 4096,
      gradientCheckpointing: "unsloth",
      packing: true,
      trainOnCompletions: true,
    },
  },
  {
    id: "json-tools",
    name: "JSON / tool calls",
    description: "Tighter learning rate and longer context for structured output.",
    values: {
      trainingMethod: "qlora",
      loraRank: 32,
      loraAlpha: 64,
      loraDropout: 0.05,
      learningRate: 1.5e-4,
      epochs: 3,
      batchSize: 1,
      gradientAccumulation: 8,
      contextLength: 4096,
      gradientCheckpointing: "unsloth",
      packing: false,
      trainOnCompletions: true,
    },
  },
  {
    id: "roleplay",
    name: "Roleplay / voice",
    description: "A larger adapter and a gentler rate, for style rather than facts.",
    values: {
      trainingMethod: "qlora",
      loraRank: 64,
      loraAlpha: 128,
      loraDropout: 0.05,
      learningRate: 1e-4,
      epochs: 2,
      batchSize: 1,
      gradientAccumulation: 8,
      contextLength: 4096,
      gradientCheckpointing: "unsloth",
      packing: false,
      trainOnCompletions: true,
    },
  },
  {
    id: "low-vram",
    name: "Low VRAM",
    description: "Batch 1, short context, more accumulation; the safest first run.",
    values: {
      trainingMethod: "qlora",
      loraRank: 16,
      loraAlpha: 32,
      loraDropout: 0,
      learningRate: 2e-4,
      epochs: 1,
      batchSize: 1,
      gradientAccumulation: 16,
      contextLength: 1024,
      gradientCheckpointing: "unsloth",
      packing: false,
      trainOnCompletions: true,
    },
  },
  {
    id: "full-small",
    name: "Full tune, small model",
    description: "For a model that fits entirely: full weights, a much lower rate.",
    values: {
      trainingMethod: "full",
      loraRank: 16,
      loraAlpha: 32,
      loraDropout: 0,
      learningRate: 2e-5,
      epochs: 1,
      batchSize: 1,
      gradientAccumulation: 16,
      contextLength: 2048,
      gradientCheckpointing: "true",
      packing: false,
      trainOnCompletions: true,
    },
  },
];

export const PRESET_STORAGE_KEY = "unsloth.training.presets.v1";

const NUMERIC_FIELDS: Array<keyof TrainingPresetValues> = [
  "loraRank",
  "loraAlpha",
  "loraDropout",
  "learningRate",
  "epochs",
  "batchSize",
  "gradientAccumulation",
  "contextLength",
];

function isMethod(value: unknown): value is TrainingMethod {
  return (
    value === "cora" ||
    value === "qlora" ||
    value === "lora" ||
    value === "full" ||
    value === "cpt" ||
    value === "rl"
  );
}

function isCheckpointing(value: unknown): value is GradientCheckpointing {
  return value === "none" || value === "true" || value === "unsloth" || value === "mlx";
}

/** Keep only the fields a preset can actually change, with the right shapes. */
export function sanitizePresetValues(
  raw: unknown,
): TrainingPresetValues | null {
  if (typeof raw !== "object" || raw === null) return null;
  const source = raw as Record<string, unknown>;
  if (!isMethod(source.trainingMethod)) return null;
  for (const field of NUMERIC_FIELDS) {
    const value = source[field];
    if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
      return null;
    }
  }
  const checkpointing = source.gradientCheckpointing;
  if (!isCheckpointing(checkpointing)) return null;
  return {
    trainingMethod: source.trainingMethod,
    loraRank: source.loraRank as number,
    loraAlpha: source.loraAlpha as number,
    loraDropout: source.loraDropout as number,
    learningRate: source.learningRate as number,
    epochs: source.epochs as number,
    batchSize: source.batchSize as number,
    gradientAccumulation: source.gradientAccumulation as number,
    contextLength: source.contextLength as number,
    gradientCheckpointing: checkpointing,
    packing: Boolean(source.packing),
    trainOnCompletions: Boolean(source.trainOnCompletions),
  };
}

export function parseStoredPresets(raw: string | null): StoredTrainingPreset[] {
  if (!raw) return [];
  try {
    const payload = JSON.parse(raw) as unknown;
    if (!Array.isArray(payload)) return [];
    const presets: StoredTrainingPreset[] = [];
    for (const entry of payload) {
      if (typeof entry !== "object" || entry === null) continue;
      const record = entry as Record<string, unknown>;
      const values = sanitizePresetValues(record.values);
      if (!values) continue;
      if (typeof record.id !== "string" || typeof record.name !== "string") {
        continue;
      }
      presets.push({
        id: record.id,
        name: record.name,
        description:
          typeof record.description === "string" ? record.description : "",
        values,
        createdAt:
          typeof record.createdAt === "string"
            ? record.createdAt
            : new Date(0).toISOString(),
      });
    }
    return presets.slice(0, 50);
  } catch {
    return [];
  }
}

export function loadStoredPresets(): StoredTrainingPreset[] {
  if (typeof window === "undefined") return [];
  try {
    return parseStoredPresets(window.localStorage.getItem(PRESET_STORAGE_KEY));
  } catch {
    return [];
  }
}

export function writeStoredPresets(presets: StoredTrainingPreset[]): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(PRESET_STORAGE_KEY, JSON.stringify(presets));
  } catch {
    // A full or blocked localStorage must not break the wizard; the preset
    // simply does not survive a reload.
  }
}
