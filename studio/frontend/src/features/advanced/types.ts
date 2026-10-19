// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Advanced Studio: QAT, curriculum, layer-wise LR, dataset blending, auto-eval, RL. */

export const QAT_FORMATS = ["int4", "nf4", "nvfp4"] as const;
export type QatFormat = (typeof QAT_FORMATS)[number];

export const LAYER_ZONES = [
  "embedding",
  "early",
  "middle",
  "late",
  "head",
  "adapter",
] as const;
export type LayerZone = (typeof LAYER_ZONES)[number];

export const RL_METHODS = ["dpo", "grpo"] as const;
export type RlMethod = (typeof RL_METHODS)[number];

export const REWARD_PRESETS = [
  "exact_match",
  "contains",
  "length",
  "format",
  "reasoning_tags",
] as const;
export type RewardPreset = (typeof REWARD_PRESETS)[number];

export const MERGE_METHODS = ["linear", "slerp", "ties", "dare"] as const;
export type MergeMethod = (typeof MERGE_METHODS)[number];

export type BlendSlotSource = "huggingface" | "local";

export interface BlendSlotState {
  id: string;
  name: string;
  dataset: string;
  subset: string;
  split: string;
  source: BlendSlotSource;
  /** Relative weight; normalized to 1.0 on the backend. */
  ratio: number;
  rows: number | null;
}

export interface AdvancedTrainingState {
  // Feature 1: QAT
  qatEnabled: boolean;
  qatFormat: QatFormat;
  qatGroupSize: 16 | 32 | 64 | 128 | 256;

  // Feature 4: alternating curriculum
  curriculumEnabled: boolean;
  curriculumCadence: string;

  // Feature 5: layer-wise LR
  layerLrEnabled: boolean;
  layerLrZones: Record<LayerZone, number>;
  layerLrDepthCurveEnabled: boolean;
  layerLrDepthStart: number;
  layerLrDepthEnd: number;

  // Feature 6: dynamic dataset blending
  blendEnabled: boolean;
  blendSlots: BlendSlotState[];

  // Feature 7: post-run auto eval
  autoEvalEnabled: boolean;
  autoEvalSuites: string[];
  autoEvalLimit: number | null;

  // Feature 3: policy alignment / RL
  rlMethod: RlMethod;
  rlBeta: number;
  rlKlCoefficient: number;
  rlLearningRate: number;
  rlNumGenerations: number;
  rlMaxCompletionLength: number;
  rlTemperature: number;
  rlRewardPreset: RewardPreset;
}

export const DEFAULT_LAYER_ZONE_MULTIPLIERS: Record<LayerZone, number> = {
  embedding: 0.25,
  early: 0.25,
  middle: 0.5,
  late: 0.75,
  head: 1,
  adapter: 1,
};

export function newBlendSlot(index: number): BlendSlotState {
  return {
    id: `slot-${index}-${Math.random().toString(36).slice(2, 8)}`,
    name: `dataset${index}`,
    dataset: "",
    subset: "",
    split: "train",
    source: "huggingface",
    ratio: 1,
    rows: null,
  };
}

export const DEFAULT_ADVANCED_TRAINING: AdvancedTrainingState = {
  qatEnabled: false,
  qatFormat: "int4",
  qatGroupSize: 128,
  curriculumEnabled: false,
  curriculumCadence: "reasoning:2,code:1",
  layerLrEnabled: false,
  layerLrZones: { ...DEFAULT_LAYER_ZONE_MULTIPLIERS },
  layerLrDepthCurveEnabled: false,
  layerLrDepthStart: 0.5,
  layerLrDepthEnd: 1,
  blendEnabled: false,
  blendSlots: [],
  autoEvalEnabled: false,
  autoEvalSuites: ["mmlu", "gsm8k"],
  autoEvalLimit: 200,
  rlMethod: "dpo",
  rlBeta: 0.1,
  rlKlCoefficient: 0,
  rlLearningRate: 5e-6,
  rlNumGenerations: 4,
  rlMaxCompletionLength: 512,
  rlTemperature: 0.9,
  rlRewardPreset: "exact_match",
};

export interface QatRequest {
  enabled: boolean;
  format: QatFormat;
  group_size: number;
  min_numel?: number;
}

export interface CurriculumRequest {
  enabled: boolean;
  cadence: string;
}

export interface LayerLrRequest {
  enabled: boolean;
  zone_multipliers: Partial<Record<LayerZone, number>>;
  depth_curve: [number, number] | null;
}

export interface BlendRequest {
  enabled: boolean;
  slots: Array<{
    name: string;
    dataset: string | null;
    subset: string | null;
    split: string | null;
    ratio: number;
    rows?: number | null;
  }>;
}

export interface AutoEvalRequest {
  enabled: boolean;
  suites: string[];
  eval_dataset: string | null;
  limit: number | null;
  batch_size: number;
}

export interface RlRequest {
  method: RlMethod;
  beta: number;
  learning_rate: number;
  num_generations: number;
  max_completion_length: number;
  temperature: number;
  top_p: number;
  clip_ratio: number;
  kl_coefficient: number;
  reward_preset: RewardPreset;
}

export interface AdvancedTrainingRequest {
  qat?: QatRequest;
  curriculum?: CurriculumRequest;
  layer_lr?: LayerLrRequest;
  blend?: BlendRequest;
  auto_eval?: AutoEvalRequest;
  rl?: RlRequest;
}

/**
 * Build the request block. Returns undefined when nothing is enabled so an
 * ordinary run posts exactly the payload it did before Advanced Studio existed.
 */
export function buildAdvancedRequest(
  state: AdvancedTrainingState,
  options: { batchSize: number; isPolicyAlignment: boolean },
): AdvancedTrainingRequest | undefined {
  const request: AdvancedTrainingRequest = {};

  if (state.qatEnabled) {
    request.qat = {
      enabled: true,
      format: state.qatFormat,
      group_size: state.qatGroupSize,
    };
  }
  if (state.curriculumEnabled) {
    request.curriculum = {
      enabled: true,
      cadence: state.curriculumCadence,
    };
  }
  if (state.layerLrEnabled) {
    request.layer_lr = {
      enabled: true,
      zone_multipliers: { ...state.layerLrZones },
      depth_curve: state.layerLrDepthCurveEnabled
        ? [state.layerLrDepthStart, state.layerLrDepthEnd]
        : null,
    };
  }
  if (state.blendEnabled && state.blendSlots.length > 0) {
    request.blend = {
      enabled: true,
      slots: state.blendSlots.map((slot) => ({
        name: slot.name || slot.id,
        dataset: slot.dataset.trim() || null,
        subset: slot.subset.trim() || null,
        split: slot.split.trim() || null,
        ratio: slot.ratio,
        rows: slot.rows,
      })),
    };
  }
  if (state.autoEvalEnabled && state.autoEvalSuites.length > 0) {
    request.auto_eval = {
      enabled: true,
      suites: state.autoEvalSuites,
      eval_dataset: null,
      limit: state.autoEvalLimit,
      batch_size: Math.max(1, Math.min(64, options.batchSize)),
    };
  }
  if (options.isPolicyAlignment) {
    request.rl = {
      method: state.rlMethod,
      beta: state.rlBeta,
      learning_rate: state.rlLearningRate,
      num_generations: state.rlNumGenerations,
      max_completion_length: state.rlMaxCompletionLength,
      temperature: state.rlTemperature,
      top_p: 0.95,
      clip_ratio: 0.2,
      kl_coefficient: state.rlKlCoefficient,
      reward_preset: state.rlRewardPreset,
    };
  }

  return Object.keys(request).length > 0 ? request : undefined;
}

/** Map the backend's snake_case RL block back into local state (run restore). */
export function advancedStateFromRunConfig(
  raw: unknown,
  current: AdvancedTrainingState,
): AdvancedTrainingState {
  if (!raw || typeof raw !== "object") return current;
  const config = raw as Record<string, unknown>;
  const next: AdvancedTrainingState = { ...current };
  const qat = config.qat as Record<string, unknown> | undefined;
  if (qat) {
    next.qatEnabled = qat.enabled === true;
    if (typeof qat.format === "string" && QAT_FORMATS.includes(qat.format as QatFormat)) {
      next.qatFormat = qat.format as QatFormat;
    }
    if (typeof qat.group_size === "number") {
      next.qatGroupSize = qat.group_size as AdvancedTrainingState["qatGroupSize"];
    }
  }
  const curriculum = config.curriculum as Record<string, unknown> | undefined;
  if (curriculum) {
    next.curriculumEnabled = curriculum.enabled === true;
    if (typeof curriculum.cadence === "string") {
      next.curriculumCadence = curriculum.cadence;
    }
  }
  const layerLr = config.layer_lr as Record<string, unknown> | undefined;
  if (layerLr) {
    next.layerLrEnabled = layerLr.enabled === true;
    const zones = layerLr.zone_multipliers as Record<string, number> | undefined;
    if (zones) {
      next.layerLrZones = { ...next.layerLrZones, ...zones };
    }
    const curve = layerLr.depth_curve as [number, number] | null | undefined;
    if (Array.isArray(curve) && curve.length === 2) {
      next.layerLrDepthCurveEnabled = true;
      next.layerLrDepthStart = curve[0];
      next.layerLrDepthEnd = curve[1];
    }
  }
  const blend = config.blend as Record<string, unknown> | undefined;
  if (blend && Array.isArray(blend.slots)) {
    next.blendEnabled = blend.enabled === true;
    next.blendSlots = (blend.slots as Array<Record<string, unknown>>).map(
      (slot, index) => ({
        id: `run-${index}`,
        name: String(slot.name ?? `dataset${index + 1}`),
        dataset: String(slot.dataset ?? ""),
        subset: String(slot.subset ?? ""),
        split: String(slot.split ?? "train"),
        source: "huggingface",
        ratio: typeof slot.ratio === "number" ? slot.ratio : 1,
        rows: typeof slot.rows === "number" ? slot.rows : null,
      }),
    );
  }
  const autoEval = config.auto_eval as Record<string, unknown> | undefined;
  if (autoEval) {
    next.autoEvalEnabled = autoEval.enabled === true;
    if (Array.isArray(autoEval.suites)) {
      next.autoEvalSuites = autoEval.suites.map(String);
    }
    next.autoEvalLimit =
      typeof autoEval.limit === "number" ? autoEval.limit : null;
  }
  const rl = config.rl as Record<string, unknown> | undefined;
  if (rl) {
    if (rl.method === "dpo" || rl.method === "grpo") next.rlMethod = rl.method;
    if (typeof rl.beta === "number") next.rlBeta = rl.beta;
    if (typeof rl.kl_coefficient === "number") {
      next.rlKlCoefficient = rl.kl_coefficient;
    }
    if (typeof rl.learning_rate === "number") next.rlLearningRate = rl.learning_rate;
    if (typeof rl.num_generations === "number") {
      next.rlNumGenerations = rl.num_generations;
    }
    if (typeof rl.max_completion_length === "number") {
      next.rlMaxCompletionLength = rl.max_completion_length;
    }
    if (typeof rl.temperature === "number") next.rlTemperature = rl.temperature;
    if (
      typeof rl.reward_preset === "string" &&
      REWARD_PRESETS.includes(rl.reward_preset as RewardPreset)
    ) {
      next.rlRewardPreset = rl.reward_preset as RewardPreset;
    }
  }
  return next;
}
