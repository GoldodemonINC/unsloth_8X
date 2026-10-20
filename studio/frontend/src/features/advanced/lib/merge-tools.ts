// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The Merge page's arithmetic, kept out of the component so it can be tested.
 *
 * Everything here is a pure function of the setup, the plan and the job: presets
 * and recipes round-trip through JSON, weight tools keep the list lined up with
 * the slots, and the review/budget/timeline readers turn the backend's plan and
 * progress payloads into the lines the page shows.
 */

export type MergeMethodId = "linear" | "slerp" | "ties" | "dare";
export type MergeMode = "standard" | "mmoe";

export interface MergeSetup {
  models: string[];
  weights: number[];
  method: MergeMethodId;
  density: number;
  mode: MergeMode;
  topK: number;
  temperature: number;
  outputDir: string;
}

export const MIN_MERGE_PARENTS = 2;
export const MAX_MERGE_PARENTS = 6;
export const RECIPE_VERSION = 1;

/** What each method does to the weights, and whether density means anything to it. */
export const MERGE_METHOD_NOTES: Record<
  MergeMethodId,
  { label: string; summary: string; usesDensity: boolean }
> = {
  linear: {
    label: "Linear",
    summary:
      "Weighted average of every tensor. The safe default; weights decide how far the result leans to each parent.",
    usesDensity: false,
  },
  slerp: {
    label: "SLERP",
    summary:
      "Spherical interpolation for two parents: keeps the direction of both vectors instead of averaging them flat.",
    usesDensity: false,
  },
  ties: {
    label: "TIES",
    summary:
      "Keeps the largest-magnitude deltas and drops the rest before averaging, so disagreeing parents stop cancelling out. Density is the share kept.",
    usesDensity: true,
  },
  dare: {
    label: "DARE",
    summary:
      "Randomly drops most deltas and rescales the survivors, then merges. Density is the share kept.",
    usesDensity: true,
  },
};

export function methodUsesDensity(method: MergeMethodId): boolean {
  return MERGE_METHOD_NOTES[method]?.usesDensity ?? false;
}

/**
 * A fusion preset: the engine settings only, never the parents.
 *
 * Choosing a preset while five slots hold models must not empty them, so presets
 * patch the controls and leave the model list alone.
 */
export interface MergePreset {
  id: string;
  label: string;
  description: string;
  method: MergeMethodId;
  density: number;
  mode: MergeMode;
  topK: number;
  temperature: number;
  /** How the preset reshapes the existing weights, if at all. */
  weights: "keep" | "even" | "focusFirst" | "focusLast";
}

export const BUILT_IN_MERGE_PRESETS: readonly MergePreset[] = [
  {
    id: "even-linear",
    label: "Even blend",
    description: "Linear average with every parent weighted the same.",
    method: "linear",
    density: 0.5,
    mode: "standard",
    topK: 2,
    temperature: 1,
    weights: "even",
  },
  {
    id: "slerp-balanced",
    label: "SLERP 50/50",
    description: "Two parents, spherical interpolation at the midpoint.",
    method: "slerp",
    density: 0.5,
    mode: "standard",
    topK: 2,
    temperature: 1,
    weights: "even",
  },
  {
    id: "ties-half",
    label: "TIES 50%",
    description: "Keeps the top half of each parent's deltas before merging.",
    method: "ties",
    density: 0.5,
    mode: "standard",
    topK: 2,
    temperature: 1,
    weights: "even",
  },
  {
    id: "dare-light",
    label: "DARE 30%",
    description:
      "Drops 70% of the deltas and rescales the rest; the gentlest DARE pass.",
    method: "dare",
    density: 0.3,
    mode: "standard",
    topK: 2,
    temperature: 1,
    weights: "even",
  },
  {
    id: "focus-first",
    label: "Lean on first",
    description:
      "Linear, but the first parent carries twice the weight of the rest.",
    method: "linear",
    density: 0.5,
    mode: "standard",
    topK: 2,
    temperature: 1,
    weights: "focusFirst",
  },
  {
    id: "mmoe-top2",
    label: "MMoE top-2",
    description:
      "Route each token to its two best-matching parents instead of blending.",
    method: "linear",
    density: 0.5,
    mode: "mmoe",
    topK: 2,
    temperature: 1,
    weights: "keep",
  },
];

function clamped(
  value: number,
  low: number,
  high: number,
  fallback: number,
): number {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return fallback;
  }
  return Math.min(high, Math.max(low, number));
}

/** Equal weights, in slot order. */
export function evenWeights(count: number): number[] {
  return Array.from({ length: Math.max(0, count) }, () => 1);
}

export type WeightTool = "equal" | "normalize" | "invert";

export const WEIGHT_TOOL_LABELS: Record<
  WeightTool,
  { label: string; about: string }
> = {
  equal: {
    label: "Equal",
    about: "Give every parent the same weight.",
  },
  normalize: {
    label: "Normalize",
    about:
      "Scale the current weights so they sum to 1; the shares stay as they are.",
  },
  invert: {
    label: "Invert",
    about:
      "Flip the weights, so the parent with the smallest share carries the largest.",
  },
};

/**
 * One weight tool applied to the current list.
 *
 * Every branch returns the same length, because the page draws one slider per
 * slot: a short list is what makes weights drift off their parent.
 */
export function applyWeightTool(weights: number[], tool: WeightTool): number[] {
  const count = weights.length;
  if (count === 0) {
    return [];
  }
  const safe = weights.map((value) =>
    Number.isFinite(value) && value > 0 ? value : 0,
  );
  if (tool === "equal") {
    return evenWeights(count);
  }
  if (tool === "normalize") {
    const total = safe.reduce((sum, value) => sum + value, 0);
    if (!(total > 0)) {
      return evenWeights(count);
    }
    return safe.map((value) => value / total);
  }
  const total = safe.reduce((sum, value) => sum + value, 0);
  if (!(total > 0)) {
    return evenWeights(count);
  }
  // Invert around the average, so a 0.2/0.8 pair becomes 0.8/0.2 and an even list
  // stays even instead of every slot collapsing to zero.
  const average = total / count;
  return safe.map((value) => {
    const inverted = 2 * average - value;
    // Six places keeps a 0.2/0.8 pair reading 0.8000000000000002 as 0.2's inverse should be.
    const rounded = Math.round(inverted * 1e6) / 1e6;
    return rounded > 0 ? rounded : 0;
  });
}

/** Presets change the engine settings; the parent list is the user's. */
export function applyPreset(
  setup: MergeSetup,
  preset: MergePreset,
): MergeSetup {
  const weights =
    preset.weights === "even"
      ? evenWeights(setup.models.length)
      : preset.weights === "focusFirst"
        ? setup.weights.map((_, index) => (index === 0 ? 2 : 1))
        : preset.weights === "focusLast"
          ? setup.weights.map((_, index, all) =>
              index === all.length - 1 ? 2 : 1,
            )
          : setup.weights;
  return {
    ...setup,
    method: preset.method,
    density: clamped(preset.density, 0.05, 1, setup.density),
    mode: preset.mode,
    topK: Math.round(clamped(preset.topK, 1, MAX_MERGE_PARENTS, setup.topK)),
    temperature: clamped(preset.temperature, 0.1, 5, setup.temperature),
    weights: fitWeights(weights, setup.models.length),
  };
}

/** The weight list cut or padded to *count*, keeping the values it already had. */
export function fitWeights(weights: number[], count: number): number[] {
  return Array.from({ length: Math.max(0, count) }, (_, index) => {
    const value = weights[index];
    return Number.isFinite(value) && value > 0 ? (value as number) : 1;
  });
}

// ---------------------------------------------------------------------------
// Recipes: a whole merge setup, saved or pasted back
// ---------------------------------------------------------------------------

export interface MergeRecipe {
  version: number;
  name?: string;
  setup: MergeSetup;
}

export function serializeRecipe(setup: MergeSetup, name?: string): string {
  const recipe: MergeRecipe = {
    version: RECIPE_VERSION,
    ...(name ? { name } : {}),
    setup: {
      ...setup,
      models: setup.models.map((model) => model.trim()).filter(Boolean),
      weights: fitWeights(
        setup.weights,
        setup.models.filter((model) => model.trim()).length,
      ),
    },
  };
  return JSON.stringify(recipe, null, 2);
}

export type RecipeParse =
  | { ok: true; recipe: MergeRecipe }
  | { ok: false; error: string };

const METHODS = new Set<MergeMethodId>(["linear", "slerp", "ties", "dare"]);

/**
 * Parse a pasted or restored recipe.
 *
 * Anything it cannot place is an error rather than a silent default: a recipe
 * that quietly drops a parent merges something the user never asked for.
 */
export function parseRecipe(text: string): RecipeParse {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    return { ok: false, error: "That is not valid JSON." };
  }
  if (!raw || typeof raw !== "object") {
    return { ok: false, error: "A recipe is a JSON object." };
  }
  const record = raw as Record<string, unknown>;
  const body = (record.setup ?? record) as Record<string, unknown>;
  if (record.version !== undefined && record.version !== RECIPE_VERSION) {
    return {
      ok: false,
      error: `Recipe version ${String(record.version)} is newer than this build (${RECIPE_VERSION}).`,
    };
  }
  const models = Array.isArray(body.models)
    ? body.models
        .filter((entry): entry is string => typeof entry === "string")
        .map((entry) => entry.trim())
    : [];
  if (models.length < MIN_MERGE_PARENTS) {
    return { ok: false, error: "A recipe needs at least two parents." };
  }
  if (models.length > MAX_MERGE_PARENTS) {
    return {
      ok: false,
      error: `A recipe can hold at most ${MAX_MERGE_PARENTS} parents.`,
    };
  }
  if (models.some((model) => model.length === 0)) {
    return { ok: false, error: "Every parent needs a model." };
  }
  const method = body.method as MergeMethodId;
  if (!METHODS.has(method)) {
    return { ok: false, error: `Unknown merge method ${String(body.method)}.` };
  }
  const mode: MergeMode = body.mode === "mmoe" ? "mmoe" : "standard";
  const rawWeights = Array.isArray(body.weights) ? body.weights : [];
  const weights = models.map((_, index) => {
    const value = Number(rawWeights[index]);
    return Number.isFinite(value) && value > 0 ? value : 1;
  });
  return {
    ok: true,
    recipe: {
      version: RECIPE_VERSION,
      ...(typeof record.name === "string" ? { name: record.name } : {}),
      setup: {
        models,
        weights,
        method,
        density: clamped(Number(body.density), 0.05, 1, 0.5),
        mode,
        topK: Math.round(clamped(Number(body.topK), 1, MAX_MERGE_PARENTS, 2)),
        temperature: clamped(Number(body.temperature), 0.1, 5, 1),
        outputDir: typeof body.outputDir === "string" ? body.outputDir : "",
      },
    },
  };
}

// ---------------------------------------------------------------------------
// Plan readers: comparison, budget, review
// ---------------------------------------------------------------------------

export interface PlanModelLike {
  source: string;
  architecture: string;
  hidden_size: number | null;
  num_layers: number | null;
  num_heads: number | null;
  intermediate_size: number | null;
  vocab_size: number | null;
  parameter_count: number | null;
}

export interface PlanLike {
  models: PlanModelLike[];
  alignment: {
    mode: string;
    hidden_strategy: string;
    layer_strategy: string;
    vocab_strategy: string;
    warnings: string[];
    mmoe_required: boolean;
  };
  moe?: { warnings?: string[]; notes?: string[] };
  estimates?: {
    need: { vram_bytes: number; ram_bytes: number; disk_bytes: number };
    notes: string[];
    streaming: boolean;
    model_bytes: number;
  } | null;
}

export interface ComparisonRow {
  label: string;
  source: string;
  architecture: string;
  parameters: number | null;
  hidden: number | null;
  layers: number | null;
  vocab: number | null;
  sameShape: boolean;
}

export function comparisonRows(plan: PlanLike): ComparisonRow[] {
  const target = plan.models[0];
  return plan.models.map((model, index) => ({
    label:
      plan.models.length === 2
        ? index === 0
          ? "Model A"
          : "Model B"
        : `Model ${index + 1}`,
    source: model.source,
    architecture: model.architecture,
    parameters: model.parameter_count,
    hidden: model.hidden_size,
    layers: model.num_layers,
    vocab: model.vocab_size,
    sameShape:
      index === 0 ||
      (model.hidden_size === target?.hidden_size &&
        model.num_layers === target?.num_layers &&
        model.vocab_size === target?.vocab_size),
  }));
}

export interface MemoryBudget {
  ramBytes: number;
  diskBytes: number;
  vramBytes?: number | null;
}

export interface BudgetVerdict {
  /** null when the plan published no estimate, so the page says "unknown". */
  fits: boolean | null;
  shortfalls: Array<"ram" | "disk" | "vram">;
  notes: string[];
}

export function budgetVerdict(
  plan: PlanLike,
  memory: MemoryBudget,
): BudgetVerdict {
  const need = plan.estimates?.need;
  const notes = [...(plan.estimates?.notes ?? [])];
  if (!need) {
    return { fits: null, shortfalls: [], notes };
  }
  const shortfalls: Array<"ram" | "disk" | "vram"> = [];
  // A zero is "this machine did not report it", not "this machine has none":
  // flagging on it would call every merge impossible on a fresh install.
  if (memory.ramBytes > 0 && need.ram_bytes > memory.ramBytes) {
    shortfalls.push("ram");
  }
  if (memory.diskBytes > 0 && need.disk_bytes > memory.diskBytes) {
    shortfalls.push("disk");
  }
  // A streaming merge holds no two parents at once, so VRAM is advisory only.
  if (
    (memory.vramBytes ?? 0) > 0 &&
    need.vram_bytes > (memory.vramBytes ?? 0)
  ) {
    shortfalls.push("vram");
  }
  if (plan.estimates?.streaming) {
    notes.push(
      "Streaming merge: shards are read and written one at a time, so RAM peaks far below the model size.",
    );
  }
  return { fits: shortfalls.length === 0, shortfalls, notes };
}

export type ReviewLevel = "block" | "warn" | "info";

export interface ReviewItem {
  id: string;
  level: ReviewLevel;
  text: string;
}

export interface ReviewInput {
  setup: Pick<MergeSetup, "models" | "weights" | "mode">;
  plan: PlanLike | null;
  validation: {
    blocked: boolean;
    blocked_reason?: string;
    sources?: Array<{
      label: string;
      blocked_reason?: string;
      warnings?: string[];
      pickles?: string[];
      requires_remote_code?: boolean;
    }>;
    tokenizer: {
      status: string;
      reason?: string;
      conflicts?: string[];
      warnings?: string[];
    };
  } | null;
}

/**
 * Everything that should stop, or should be read, before Run merge.
 *
 * The page already shows these across three cards; this gathers them in the order
 * they matter so nothing in the third card is missed because the first looked fine.
 */
export function reviewItems({
  setup,
  plan,
  validation,
}: ReviewInput): ReviewItem[] {
  const items: ReviewItem[] = [];
  if (setup.models.length < MIN_MERGE_PARENTS) {
    items.push({
      id: "parents",
      level: "block",
      text: "Pick at least two parents.",
    });
  }
  if (setup.mode === "standard" && !setup.weights.some((value) => value > 0)) {
    items.push({
      id: "weights",
      level: "block",
      text: "Every blend weight is zero, so there is nothing to blend.",
    });
  }
  if (validation?.blocked) {
    items.push({
      id: "blocked",
      level: "block",
      text:
        validation.blocked_reason ??
        "The pre-merge policy check blocked this merge.",
    });
  }
  if (validation?.tokenizer.status === "conflict") {
    items.push({
      id: "tokenizer-conflict",
      level: "block",
      text:
        validation.tokenizer.reason ??
        "The parents disagree on special tokens.",
    });
  }
  for (const conflict of validation?.tokenizer.conflicts ?? []) {
    items.push({ id: `tokenizer-${conflict}`, level: "warn", text: conflict });
  }
  for (const warning of validation?.tokenizer.warnings ?? []) {
    items.push({
      id: `tokenizer-warn-${warning}`,
      level: "warn",
      text: warning,
    });
  }
  for (const source of validation?.sources ?? []) {
    if (source.blocked_reason) {
      items.push({
        id: `source-blocked-${source.label}`,
        level: "block",
        text: `${source.label}: ${source.blocked_reason}`,
      });
    }
    for (const warning of source.warnings ?? []) {
      items.push({
        id: `source-warn-${source.label}-${warning}`,
        level: "warn",
        text: `${source.label}: ${warning}`,
      });
    }
    if ((source.pickles ?? []).length > 0) {
      items.push({
        id: `pickles-${source.label}`,
        level: "warn",
        text: `${source.label} ships pickle weights; they can run code while loading.`,
      });
    }
    if (source.requires_remote_code) {
      items.push({
        id: `remote-code-${source.label}`,
        level: "warn",
        text: `${source.label} needs custom code; allow it only if you trust the repo.`,
      });
    }
  }
  for (const warning of plan?.alignment.warnings ?? []) {
    items.push({ id: `align-${warning}`, level: "warn", text: warning });
  }
  if (plan?.alignment.mmoe_required && setup.mode !== "mmoe") {
    items.push({
      id: "mmoe-required",
      level: "warn",
      text: "These architectures disagree too much for weight blending; the MMoE router is the route that keeps both intact.",
    });
  }
  for (const warning of plan?.moe?.warnings ?? []) {
    items.push({ id: `moe-${warning}`, level: "warn", text: warning });
  }
  if (items.length === 0) {
    items.push({
      id: "clear",
      level: "info",
      text: plan
        ? "No blocking checks and no warnings. The parents agree on shapes and tokens."
        : "Inspect the parents to run the shape, tokenizer and policy checks.",
    });
  }
  return items;
}

// ---------------------------------------------------------------------------
// Job readers: timeline, throughput
// ---------------------------------------------------------------------------

export type TimelineState = "done" | "active" | "pending" | "failed";

export interface TimelineStep {
  stage: string;
  label: string;
  state: TimelineState;
  detail: string;
}

const STAGE_ORDER = [
  "queued",
  "resolving",
  "scanning",
  "merging",
  "finalizing",
  "completed",
] as const;

const STAGE_LABELS: Record<string, string> = {
  queued: "Queued",
  resolving: "Resolving sources",
  scanning: "Scanning shard indexes",
  merging: "Blending tensors",
  finalizing: "Writing config and tokenizer",
  completed: "Done",
};

export interface JobLike {
  status: "running" | "completed" | "failed";
  created_at?: number;
  finished_at?: number | null;
  progress?: {
    stage?: string;
    done?: number;
    total?: number;
    message?: string;
    blended_tensors?: number;
    copied_tensors?: number;
    skipped_tensors?: number;
    bytes_written?: number;
    shards?: number;
    updated_at?: number;
  } | null;
  error?: string | null;
}

export function progressTimeline(job: JobLike | null): TimelineStep[] {
  if (!job) {
    return [];
  }
  const current = job.progress?.stage ?? "queued";
  const currentIndex = STAGE_ORDER.indexOf(
    current as (typeof STAGE_ORDER)[number],
  );
  const failedIndex = job.status === "failed" ? Math.max(0, currentIndex) : -1;
  return STAGE_ORDER.map((stage, index) => {
    let state: TimelineState = "pending";
    if (job.status === "completed") {
      state = "done";
    } else if (failedIndex >= 0) {
      // The stage that was running when the job failed is the one that failed;
      // nothing after it ran.
      state =
        index < failedIndex
          ? "done"
          : index === failedIndex
            ? "failed"
            : "pending";
    } else if (index < currentIndex) {
      state = "done";
    } else if (index === currentIndex) {
      state = "active";
    }
    const detail: string[] = [];
    if (stage === "merging" && current === "merging") {
      if (job.progress?.total) {
        detail.push(
          `${job.progress.done ?? 0} of ${job.progress.total} shards`,
        );
      }
      if (job.progress?.blended_tensors != null) {
        detail.push(`${job.progress.blended_tensors} blended`);
      }
      if (job.progress?.skipped_tensors) {
        detail.push(`${job.progress.skipped_tensors} skipped`);
      }
    }
    if (stage === "finalizing" && job.progress?.shards) {
      detail.push(`${job.progress.shards} shards`);
    }
    return {
      stage,
      label: STAGE_LABELS[stage] ?? stage,
      state,
      detail: detail.join(" · "),
    };
  });
}

/** Bytes per second since the job started, or null before it has written anything. */
export function throughputBytesPerSecond(
  job: JobLike,
  now = Date.now(),
): number | null {
  const written = job.progress?.bytes_written;
  if (!written || written <= 0) {
    return null;
  }
  const started = job.finished_at ?? Math.floor(now / 1000);
  const seconds = started - (job.created_at ?? started);
  return seconds > 0 ? written / seconds : null;
}

/** Seconds left at the observed rate, from the shard counts the job published. */
export function etaSeconds(job: JobLike, now = Date.now()): number | null {
  const progress = job.progress;
  if (!progress || job.status !== "running") {
    return null;
  }
  if (!progress.total || progress.done == null || progress.done <= 0) {
    return null;
  }
  const elapsed = Math.max(
    0,
    Math.floor(now / 1000) - (job.created_at ?? Math.floor(now / 1000)),
  );
  const perShard = elapsed / progress.done;
  const left = progress.total - progress.done;
  return left > 0 && perShard > 0 ? Math.round(perShard * left) : null;
}

// ---------------------------------------------------------------------------
// Structural report and output folder
// ---------------------------------------------------------------------------

export interface StructuralLike {
  verdict: string;
  structurally_valid: boolean;
  problems: string[];
  warnings: string[];
  tier: string;
  scope: string;
}

export interface StructuralCheck {
  label: string;
  ok: boolean;
  detail: string;
}

/**
 * The structural tier's verdict as rows.
 *
 * The backend reports problems and warnings, not per-check booleans, so each row
 * names the check and whether the report mentioned it.
 */
const NAN_WEIGHT = /nan|infinit/i;
const TOKENIZER_MISSING = /tokenizer/i;
const SHAPE_MISMATCH = /shape|mismatch/i;

export function structuralChecks(report: StructuralLike): StructuralCheck[] {
  const mentions = (text: string, needle: RegExp) => needle.test(text);
  const problems = report.problems.join(" ");
  const warnings = report.warnings.join(" ");
  return [
    {
      label: "Folder loads as a checkpoint",
      ok: report.structurally_valid,
      detail: report.structurally_valid
        ? "config.json, shard index and weights agree."
        : "The folder does not load.",
    },
    {
      label: "No NaN or infinite weights",
      ok: !mentions(problems, NAN_WEIGHT),
      detail: mentions(problems, NAN_WEIGHT)
        ? "Some tensors are NaN or infinite."
        : "Every sampled tensor is finite.",
    },
    {
      label: "Tokenizer present",
      ok: !mentions(problems, TOKENIZER_MISSING),
      detail: mentions(problems, TOKENIZER_MISSING)
        ? "The merged folder is missing tokenizer files."
        : "Tokenizer files were copied from Model A.",
    },
    {
      label: "Shapes match the plan",
      ok: !mentions(`${problems} ${warnings}`, SHAPE_MISMATCH),
      detail: mentions(`${problems} ${warnings}`, SHAPE_MISMATCH)
        ? "A tensor did not match the planned alignment."
        : "Every sampled tensor matches the planned shape.",
    },
  ];
}

/** The folder a merge would write, and whether that name is already taken. */
export function folderCollisionNotice(
  folderName: string | null,
  existingNames: readonly string[],
): string | null {
  if (!folderName) {
    return null;
  }
  const taken = existingNames.some(
    (name) => name.trim().toLowerCase() === folderName.trim().toLowerCase(),
  );
  return taken
    ? `"${folderName}" already exists in the library. The backend appends a number instead of overwriting it, so the new merge will not replace that model.`
    : null;
}

export function outputFolderName(path: string): string {
  const parts = path.replace(/\\/g, "/").split("/").filter(Boolean);
  return parts[parts.length - 1] ?? path;
}
