// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useAppShellReadySignal } from "@/components/app-readiness";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Spinner } from "@/components/ui/spinner";
import { Textarea } from "@/components/ui/textarea";
import { usePlatformStore } from "@/config/env";
import { hfApiToken, useHfTokenStore } from "@/features/hub";
import { StudioModeNav } from "@/features/studio/studio-mode-nav";
import { cn } from "@/lib/utils";
import {
  Layers01Icon,
  PlayIcon,
  PlusSignIcon,
  Refresh01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import { type ReactElement, useEffect, useState } from "react";
import {
  type AdvancedJob,
  type MergeHistoryEntry,
  type MergePlanResponse,
  type MergeValidation,
  type StructuralEval,
  evalMergedModel,
  fetchAdvancedCapabilities,
  fetchMergeHistory,
  fetchOutputLocations,
  planMerge,
  runMerge,
  runMoeMerge,
  validateMerge,
  waitForJob,
} from "./api/advanced-api";
import { MergeModelSlot } from "./components/merge-model-slot";
import {
  type SavedMergePreset,
  loadSavedPresets,
  removeSavedPreset,
  upsertSavedPreset,
  writeSavedPresets,
} from "./lib/merge-preset-store";
import {
  BUILT_IN_MERGE_PRESETS,
  MERGE_METHOD_NOTES,
  type MergePreset,
  type MergeSetup,
  WEIGHT_TOOL_LABELS,
  applyPreset,
  applyWeightTool,
  budgetVerdict,
  comparisonRows,
  etaSeconds,
  fitWeights,
  folderCollisionNotice,
  methodUsesDensity,
  parseRecipe,
  progressTimeline,
  reviewItems,
  serializeRecipe,
  structuralChecks,
  throughputBytesPerSecond,
} from "./lib/merge-tools";
import {
  MIN_MERGE_MODELS,
  appendWeight,
  blendShares,
  maxMergeModels,
  parentLabel,
  removeWeight,
} from "./merge-slots";
import { MERGE_METHODS, type MergeMethod } from "./types";

const MODE_LABELS: Record<string, string> = {
  identical: "Identical checkpoints",
  cross_scale: "Cross-scale merge",
  cross_architecture: "Cross-architecture merge",
};

const PROGRESS_LABELS: Record<string, string> = {
  queued: "Queued",
  resolving: "Resolving checkpoints",
  scanning: "Reading shard indexes",
  merging: "Blending tensors",
  finalizing: "Writing the model",
  completed: "Complete",
  failed: "Failed",
};

/** A merge reports tensors, not steps: percent is done/total, or null while it resolves. */
function progressPercent(progress: AdvancedJob["progress"]): number | null {
  if (!progress) return null;
  if (progress.stage === "completed") return 100;
  if (progress.stage === "failed") return 100;
  const total = progress.total ?? 0;
  if (!total) return null;
  return Math.max(0, Math.min(100, (100 * (progress.done ?? 0)) / total));
}

function bytes(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let size = value;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(size >= 10 ? 1 : 2)} ${units[unit]}`;
}

// A drive path or a rooted POSIX path; a bare name is resolved by the backend.
function isAbsolutePath(value: string): boolean {
  return /^(?:[A-Za-z]:[\\/]|[\\/])/.test(value.trim());
}

function slashes(value: string): string {
  return value.trim().replace(/\\/g, "/").replace(/\/+$/, "");
}

/** Whether a path sits inside the folder Studio scans for models to list, quantize and run. */
function isInsideOutputsRoot(value: string, root: string | null): boolean {
  if (!root) return false;
  const target = slashes(value);
  const base = slashes(root);
  if (!target || !base) return false;
  return target === base || target.startsWith(`${base}/`);
}

/** The folder name Studio uses as a model name (Export preselects checkpoints by it). */
function outputFolderName(path: string): string {
  const parts = slashes(path).split("/");
  return parts[parts.length - 1] ?? "";
}

function SpecCard({
  label,
  spec,
}: {
  label: string;
  spec: MergePlanResponse["models"][number];
}) {
  return (
    <div className="rounded-2xl border border-border/60 bg-card p-4">
      <div className="flex items-center justify-between gap-2">
        <span className="text-ui-10p5 font-medium uppercase tracking-[0.08em] text-muted-foreground">
          {label}
        </span>
        <Badge variant="secondary">{spec.architecture}</Badge>
      </div>
      <p
        className="mt-1 truncate font-mono text-ui-11 text-foreground"
        title={spec.source}
      >
        {spec.source}
      </p>
      <dl className="mt-3 grid grid-cols-2 gap-x-3 gap-y-1.5 text-ui-11">
        <dt className="text-muted-foreground">Hidden</dt>
        <dd className="text-right font-mono">{spec.hidden_size ?? "—"}</dd>
        <dt className="text-muted-foreground">Layers</dt>
        <dd className="text-right font-mono">{spec.num_layers ?? "—"}</dd>
        <dt className="text-muted-foreground">Heads</dt>
        <dd className="text-right font-mono">{spec.num_heads ?? "—"}</dd>
        <dt className="text-muted-foreground">FFN</dt>
        <dd className="text-right font-mono">
          {spec.intermediate_size ?? "—"}
        </dd>
        <dt className="text-muted-foreground">Vocab</dt>
        <dd className="text-right font-mono">{spec.vocab_size ?? "—"}</dd>
      </dl>
    </div>
  );
}

export function MergePage(): ReactElement {
  const signalReady = useAppShellReadySignal();
  const navigate = useNavigate();
  const hfToken = useHfTokenStore((s) => s.token);
  const accessToken = hfApiToken(hfToken) ?? null;
  // Two slots is the common case; up to six parents on a GPU, three on a CPU.
  const [models, setModels] = useState<string[]>(["", ""]);
  // Capabilities name the GPU; the platform verdict is the fallback, so a failed
  // /capabilities request on a CPU host still lands on the three-parent tier.
  const platformChatOnly = usePlatformStore((s) => s.isChatOnly());
  const platformUnknown = usePlatformStore((s) => s.capabilitiesUnknown());
  const [hasGpu, setHasGpu] = useState<boolean | null>(
    platformChatOnly && !platformUnknown ? false : null,
  );
  // Relative weight per slot, normalized on the backend and in the share label.
  const [weights, setWeights] = useState<number[]>([0.5, 0.5]);
  const typedModels = models.map((entry) => entry.trim());
  const allSlotsFilled = typedModels.length >= 2 && typedModels.every(Boolean);
  const shares = blendShares(weights, models.length);

  const setModelAt = (index: number, value: string) =>
    setModels((current) =>
      current.map((entry, position) => (position === index ? value : entry)),
    );
  // The two updates stay side by side rather than nested inside one updater: a
  // nested setState runs twice under StrictMode and drifts out of step with the
  // slot list.
  const addModel = () => {
    if (models.length >= maxModels) return;
    setModels((current) => [...current, ""]);
    setWeights(appendWeight);
  };
  const removeModel = (index: number) => {
    if (models.length <= MIN_MERGE_MODELS) return;
    setModels((current) => current.filter((_, position) => position !== index));
    setWeights((current) => removeWeight(current, index));
  };
  const [plan, setPlan] = useState<MergePlanResponse | null>(null);
  const maxModels = maxMergeModels(hasGpu, plan?.max_models);
  const [planError, setPlanError] = useState<string | null>(null);
  const [planning, setPlanning] = useState(false);
  const [mode, setMode] = useState<"standard" | "mmoe">("standard");
  const [method, setMethod] = useState<MergeMethod>("linear");
  const [density, setDensity] = useState(0.5);
  const [topK, setTopK] = useState(2);
  const [temperature, setTemperature] = useState(1);
  // Blank means "let the backend pick": it writes a folder named after the parents directly
  // inside the folder Studio scans, so the Library, Export and Chat pick the result up.
  const [outputDir, setOutputDir] = useState("");
  const [outputTouched, setOutputTouched] = useState(false);
  const [libraryRoot, setLibraryRoot] = useState<string | null>(null);
  const [job, setJob] = useState<AdvancedJob | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  // The pre-merge gates, run on Inspect: the tokenizer disagreement that otherwise
  // produces a model which loads fine and answers slightly wrong.
  const [validation, setValidation] = useState<MergeValidation | null>(null);
  const [validating, setValidating] = useState(false);
  const [allowTokenConflict, setAllowTokenConflict] = useState(false);
  const [allowPickle, setAllowPickle] = useState(false);
  const [structural, setStructural] = useState<StructuralEval | null>(null);
  // Recipes and recent runs: one is saved here, the other read from the manifests
  // the finished merges wrote.
  const [savedPresets, setSavedPresets] = useState<SavedMergePreset[]>([]);
  const [presetName, setPresetName] = useState("");
  const [recipeOpen, setRecipeOpen] = useState(false);
  const [recipeText, setRecipeText] = useState("");
  const [recipeNotice, setRecipeNotice] = useState<{
    kind: "ok" | "error";
    text: string;
  } | null>(null);
  const [history, setHistory] = useState<MergeHistoryEntry[]>([]);
  const [historyError, setHistoryError] = useState<string | null>(null);
  // What this machine reports, for the plan's peak-RAM/disk verdict.
  const [memoryBudget, setMemoryBudget] = useState<{
    ramBytes: number;
    diskBytes: number;
    vramBytes: number | null;
  }>({ ramBytes: 0, diskBytes: 0, vramBytes: null });
  // For 3+ parents the server reports each parent against the group target, so
  // Model 1's own entry is dropped. A pair keeps the classic Model A/Model B
  // comparison, whose notes name both sides.
  const alignmentsToShow = plan
    ? plan.models.length > 2
      ? (plan.alignments ?? [plan.alignment]).slice(1)
      : [plan.alignment]
    : [];
  // Every slider at zero leaves nothing to blend; the backend refuses it, so the
  // page says so before the request is sent.
  const weightsValid = mode === "mmoe" || weights.some((value) => value > 0);

  // The setup as the tools see it: one object for presets, recipes and saving.
  const currentSetup: MergeSetup = {
    models: typedModels,
    weights: fitWeights(weights, models.length),
    method,
    density,
    mode,
    topK,
    temperature,
    outputDir,
  };
  const applySetup = (next: MergeSetup) => {
    setModels(next.models);
    setWeights(next.weights);
    setMethod(next.method);
    setDensity(next.density);
    setMode(next.mode);
    setTopK(next.topK);
    setTemperature(next.temperature);
    setOutputDir(next.outputDir);
    setOutputTouched(next.outputDir.trim() !== "");
    // The plan and the tokenizer check describe the old parent list.
    setPlan(null);
    setValidation(null);
    setStructural(null);
  };
  const usePreset = (preset: MergePreset) => {
    const next = applyPreset(currentSetup, preset);
    setMethod(next.method);
    setDensity(next.density);
    setMode(next.mode);
    setTopK(next.topK);
    setTemperature(next.temperature);
    setWeights(next.weights);
  };
  const refreshHistory = () => {
    fetchMergeHistory(20)
      .then((payload) => {
        setHistory(payload.entries);
        setHistoryError(null);
      })
      .catch((error) => {
        setHistory([]);
        setHistoryError(error instanceof Error ? error.message : String(error));
      });
  };
  const savePreset = () => {
    const next = upsertSavedPreset(savedPresets, presetName, currentSetup);
    setSavedPresets(next);
    writeSavedPresets(
      typeof window === "undefined" ? null : window.localStorage,
      next,
    );
    setPresetName("");
  };
  const deletePreset = (id: string) => {
    const next = removeSavedPreset(savedPresets, id);
    setSavedPresets(next);
    writeSavedPresets(
      typeof window === "undefined" ? null : window.localStorage,
      next,
    );
  };
  const copyRecipe = () => {
    const text = serializeRecipe(currentSetup);
    setRecipeText(text);
    setRecipeOpen(true);
    setRecipeNotice({
      kind: "ok",
      text: "Recipe filled in below. It is copied to the clipboard too, so it can be pasted into a chat or an issue.",
    });
    // Clipboard access is refused outside a user gesture and on some browsers; the
    // textarea is the fallback, so a refusal is not an error worth showing.
    if (typeof navigator !== "undefined" && navigator.clipboard) {
      navigator.clipboard.writeText(text).catch(() => undefined);
    }
  };
  const importRecipe = () => {
    const parsed = parseRecipe(recipeText);
    if (!parsed.ok) {
      setRecipeNotice({ kind: "error", text: parsed.error });
      return;
    }
    applySetup(parsed.recipe.setup);
    setRecipeNotice({
      kind: "ok",
      text: parsed.recipe.name
        ? `Loaded \u201c${parsed.recipe.name}\u201d: ${parsed.recipe.setup.models.length} parents. Inspect them before running.`
        : `Loaded ${parsed.recipe.setup.models.length} parents. Inspect them before running.`,
    });
  };

  useEffect(() => {
    signalReady();
  }, [signalReady]);

  // Saved recipes live in this browser, and past merges in their manifests.
  useEffect(() => {
    setSavedPresets(
      loadSavedPresets(
        typeof window === "undefined" ? null : window.localStorage,
      ),
    );
    refreshHistory();
    // A failed probe leaves the zeros, and a zero budget means "unknown" rather
    // than calling every merge too big.
    fetchAdvancedCapabilities()
      .then((capabilities) => {
        const memory = capabilities.memory;
        setHasGpu(capabilities.has_gpu ?? memory.gpus.length > 0);
        setMemoryBudget({
          ramBytes: memory.system_ram_total_bytes,
          diskBytes: memory.disk_free_bytes,
          vramBytes: memory.gpus.length
            ? memory.gpus.reduce((sum, gpu) => sum + gpu.free_bytes, 0)
            : null,
        });
      })
      .catch(() => undefined);
  }, []);

  // A CPU host takes three parents and no MMoE router, so trim a slot list that a
  // recipe or an earlier GPU-sized session left beyond the tier, and fall back to
  // the weight-space blend. The backend enforces the same cap.
  useEffect(() => {
    if (models.length <= maxModels) return;
    setModels((current) => current.slice(0, maxModels));
    setWeights((current) => current.slice(0, maxModels));
  }, [maxModels, models.length]);

  useEffect(() => {
    if (hasGpu === false && mode === "mmoe") setMode("standard");
  }, [hasGpu, mode]);

  // Only used for the hint and the "outside the library" warning; a merge resolves its
  // own default server-side, so a failure here costs nothing.
  useEffect(() => {
    let cancelled = false;
    fetchOutputLocations()
      .then((locations) => {
        if (!cancelled) setLibraryRoot(locations.outputs_root);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, []);

  // The override has to be reachable from the screen that shows the refusal, or the
  // "re-run with allow_tokenizer_conflict" advice is unactionable. It appears when the
  // pre-merge check found the conflict, and also when a run was refused by the gate
  // (e.g. the parents were private and could not be checked up front).
  const tokenizerBlocked =
    Boolean(validation?.blocked) && validation?.tokenizer.status === "conflict";
  const tokenizerRefusedRun =
    Boolean(runError) && /tokenizer|special token/i.test(runError ?? "");
  const showTokenOverride = tokenizerBlocked || tokenizerRefusedRun;

  const typedOutput = outputDir.trim();
  const outsideLibrary =
    typedOutput !== "" &&
    isAbsolutePath(typedOutput) &&
    libraryRoot !== null &&
    !isInsideOutputsRoot(typedOutput, libraryRoot);

  const inspect = async () => {
    setPlanning(true);
    setPlanError(null);
    try {
      const result = await planMerge({
        models: typedModels,
        top_k: topK,
        temperature,
        hf_token: accessToken,
      });
      setPlan(result);
      setMode(result.recommended_mode);
      if (!outputTouched && result.suggested_output_dir) {
        setOutputDir(result.suggested_output_dir);
      }
      if (result.models[1]?.hidden_size) {
        setTopK(Math.min(2, result.moe.experts.length));
      }
    } catch (error) {
      setPlan(null);
      setPlanError(error instanceof Error ? error.message : String(error));
    } finally {
      setPlanning(false);
    }
    // A validation failure must not hide the plan; it just shows nothing extra.
    setValidating(true);
    try {
      setValidation(
        await validateMerge({
          models: typedModels,
          allow_non_safetensors: allowPickle,
          hf_token: accessToken,
        }),
      );
    } catch {
      setValidation(null);
    } finally {
      setValidating(false);
    }
  };

  const start = async () => {
    setRunError(null);
    setJob(null);
    setStructural(null);
    setRunning(true);
    try {
      const created =
        mode === "mmoe"
          ? await runMoeMerge({
              models: typedModels,
              output_dir: typedOutput || null,
              top_k: topK,
              temperature,
              hf_token: accessToken,
            })
          : await runMerge({
              models: typedModels,
              weights,
              output_dir: typedOutput || null,
              method,
              density,
              max_shard_gb: 4,
              hf_token: accessToken,
              allow_non_safetensors: allowPickle,
              allow_tokenizer_conflict: allowTokenConflict,
            });
      setJob(created);
      const finished = await waitForJob(created.job_id, setJob);
      setJob(finished);
      if (!outputTouched && finished.status === "completed") {
        // Hand the field back to the backend so a second run gets its own folder
        // instead of blending into this one; the card below shows where this went.
        setOutputDir("");
      }
      if (finished.status === "completed") {
        // The new run is now the newest manifest on disk.
        refreshHistory();
      }
      const mergedPath =
        finished.output_dir ??
        (finished.result as { output_dir?: string } | null)?.output_dir ??
        null;
      // Structural tier only: it proves the folder loads and is sane, not that the
      // merge is better. Quality needs an eval suite.
      if (
        mode === "standard" &&
        finished.status === "completed" &&
        mergedPath
      ) {
        try {
          setStructural(await evalMergedModel({ source: mergedPath }));
        } catch {
          setStructural(null);
        }
      }
    } catch (error) {
      setRunError(error instanceof Error ? error.message : String(error));
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="flex h-full min-h-0 flex-col bg-background">
      <div className="mx-auto flex w-full max-w-[calc(1180px*var(--ui-space-scale,1))] flex-col gap-6 px-5 pb-20 pt-8 max-sm:px-4 sm:px-9">
        <header className="font-heading flex flex-col gap-4">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <h1 className="page-title-halo text-ui-30 font-semibold leading-[1.04] tracking-[-0.028em] text-foreground sm:text-ui-34">
                Merging Studio
              </h1>
              <p className="page-title-halo text-sm text-muted-foreground">
                Fuse 2 to {maxModels} checkpoints — same family or different
                architectures — by weights or by a trained MMoE router.
              </p>
            </div>
            <StudioModeNav />
          </div>
        </header>

        <section className="elevated-card flex flex-col gap-4 p-5">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h2 className="text-ui-13p5 font-semibold text-foreground">
              Model slots
            </h2>
            <span className="text-ui-11 text-muted-foreground">
              Search the Hugging Face Hub, or point at a local checkpoint
            </span>
          </div>
          <div className="grid gap-3 lg:grid-cols-2">
            {models.map((value, index) => (
              <div key={index} className="flex flex-col gap-1.5">
                <MergeModelSlot
                  label={parentLabel(index, models.length)}
                  value={value}
                  onChange={(next) => setModelAt(index, next)}
                  placeholder={
                    index === 0
                      ? "C:\\models\\llama-3.2-1b"
                      : "C:\\models\\qwen3-2b"
                  }
                  disabled={running}
                />
                <div className="flex items-center justify-between gap-2">
                  <span className="text-ui-10p5 text-muted-foreground/70">
                    {models.length > 2
                      ? `${Math.round((shares[index] ?? 0) * 100)}% of the blend`
                      : ""}
                  </span>
                  {models.length > MIN_MERGE_MODELS ? (
                    <button
                      type="button"
                      onClick={() => removeModel(index)}
                      disabled={running}
                      className="text-ui-10 text-muted-foreground transition-colors hover:text-destructive disabled:opacity-50"
                    >
                      Remove
                    </button>
                  ) : null}
                </div>
              </div>
            ))}
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <Button
              variant="outline"
              size="sm"
              onClick={addModel}
              disabled={running || models.length >= maxModels}
            >
              <HugeiconsIcon icon={PlusSignIcon} className="size-3.5" />
              Add model
            </Button>
            <span className="text-ui-11 text-muted-foreground">
              {models.length} of {maxModels} parents
            </span>
          </div>
          {hasGpu === false ? (
            <p
              data-testid="cpu-merge-note"
              className="text-ui-10p5 leading-ui-14 text-muted-foreground/70"
            >
              CPU merge, {maxModels} parents on this host. The cap follows free
              RAM: every method streams tensors a row chunk at a time, so no
              checkpoint is ever fully resident. The MMoE router needs a GPU.
            </p>
          ) : null}
          <p className="text-ui-10p5 leading-ui-14 text-muted-foreground/70">
            Inspect downloads only the small{" "}
            <span className="font-mono">config.json</span> and tokenizer JSONs
            from the Hub; Run merge fetches the safetensors snapshot first.
            Local checkpoints are used as-is.
          </p>
          <div className="flex flex-wrap items-center gap-3">
            <Button
              onClick={() => void inspect()}
              disabled={planning || !allSlotsFilled}
              size="sm"
            >
              {planning ? (
                <Spinner className="size-3.5" />
              ) : (
                <HugeiconsIcon icon={Layers01Icon} className="size-3.5" />
              )}
              Inspect &amp; align
            </Button>
            {plan ? (
              <Badge
                variant={
                  plan.recommended_mode === "mmoe" ? "destructive" : "secondary"
                }
              >
                {MODE_LABELS[plan.alignment.mode] ?? plan.alignment.mode}
              </Badge>
            ) : null}
          </div>
          {models.length >= MIN_MERGE_MODELS ? (
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-ui-10p5 uppercase tracking-[0.05em] text-muted-foreground/70">
                Weights
              </span>
              {(["equal", "normalize", "invert"] as const).map((tool) => (
                <button
                  key={tool}
                  type="button"
                  title={WEIGHT_TOOL_LABELS[tool].about}
                  onClick={() =>
                    setWeights((current) => applyWeightTool(current, tool))
                  }
                  disabled={running}
                  className="rounded-full border border-border/60 px-2.5 py-1 text-ui-10p5 text-muted-foreground transition-colors hover:border-foreground/40 hover:text-foreground disabled:opacity-50"
                >
                  {WEIGHT_TOOL_LABELS[tool].label}
                </button>
              ))}
              <span className="font-mono text-ui-10p5 text-muted-foreground/70">
                {shares
                  .map((share) => `${Math.round(share * 100)}%`)
                  .join(" / ")}
              </span>
            </div>
          ) : null}
          {planError ? (
            <p className="text-ui-11 text-destructive">{planError}</p>
          ) : null}
          {!plan && !planError ? (
            <p className="text-ui-11 leading-ui-16 text-muted-foreground/80">
              Pick 2 to {maxModels} checkpoints — Hub repos or local paths —
              then inspect. The resolver reports each shape, plans every parent
              against the first, and only then unlocks Run merge.
            </p>
          ) : null}

          {plan ? (
            <div className="flex flex-col gap-4">
              <div className="grid gap-3 lg:grid-cols-2 2xl:grid-cols-3">
                {plan.models.map((spec, index) => (
                  <SpecCard
                    key={spec.source}
                    label={parentLabel(index, plan.models.length)}
                    spec={spec}
                  />
                ))}
              </div>
              <div className="flex flex-col gap-2 rounded-2xl border border-border/60 p-4">
                <span className="text-ui-11p5 font-medium text-foreground">
                  Parents at a glance
                </span>
                <div className="overflow-x-auto">
                  <table className="w-full min-w-[calc(520px*var(--ui-space-scale,1))] text-ui-11">
                    <thead className="text-muted-foreground/70">
                      <tr className="text-left">
                        <th className="py-1 pr-3 font-medium">Slot</th>
                        <th className="py-1 pr-3 font-medium">Params</th>
                        <th className="py-1 pr-3 font-medium">Hidden</th>
                        <th className="py-1 pr-3 font-medium">Layers</th>
                        <th className="py-1 pr-3 font-medium">Vocab</th>
                        <th className="py-1 font-medium">Shape</th>
                      </tr>
                    </thead>
                    <tbody>
                      {comparisonRows(plan).map((row) => (
                        <tr
                          key={row.source}
                          className="border-t border-border/40"
                        >
                          <td className="py-1.5 pr-3 font-medium text-foreground">
                            {row.label}
                          </td>
                          <td className="py-1.5 pr-3 font-mono">
                            {row.parameters != null
                              ? row.parameters >= 1e9
                                ? `${(row.parameters / 1e9).toFixed(2)}B`
                                : `${(row.parameters / 1e6).toFixed(0)}M`
                              : "\u2014"}
                          </td>
                          <td className="py-1.5 pr-3 font-mono">
                            {row.hidden ?? "\u2014"}
                          </td>
                          <td className="py-1.5 pr-3 font-mono">
                            {row.layers ?? "\u2014"}
                          </td>
                          <td className="py-1.5 pr-3 font-mono">
                            {row.vocab?.toLocaleString() ?? "\u2014"}
                          </td>
                          <td
                            className={cn(
                              "py-1.5",
                              row.sameShape
                                ? "text-muted-foreground"
                                : "text-amber-600 dark:text-amber-400",
                            )}
                          >
                            {row.sameShape ? "matches A" : "aligned"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {plan.estimates
                  ? (() => {
                      const budget = budgetVerdict(plan, memoryBudget);
                      return (
                        <p className="text-ui-10p5 leading-ui-14 text-muted-foreground/80">
                          Peak {bytes(plan.estimates.need.ram_bytes)} RAM \u00b7{" "}
                          {bytes(plan.estimates.need.disk_bytes)} disk \u00b7
                          parent weights {bytes(plan.estimates.model_bytes)}.
                          {budget.fits === false
                            ? ` This does not fit the ${budget.shortfalls.join(" and ")} this machine reports.`
                            : plan.estimates.streaming
                              ? " Streamed one shard at a time, so it runs on a CPU tier smaller than the model."
                              : ""}
                        </p>
                      );
                    })()
                  : null}
              </div>
              <div className="flex flex-col gap-2 rounded-2xl border border-border/60 p-4">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="text-ui-11p5 font-medium text-foreground">
                    Alignment resolver
                  </span>
                  <span className="text-ui-10p5 text-muted-foreground/70">
                    {plan.alignment.mode === "identical"
                      ? "Shapes already agree"
                      : "Cropped to the smallest shared shape"}
                  </span>
                </div>
                {alignmentsToShow.map((alignment, position) => (
                  <div
                    key={`alignment-${position + 1}`}
                    className="rounded-xl bg-muted/25 p-3"
                  >
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-ui-11 font-medium text-foreground">
                        {parentLabel(position + 1, plan.models.length)}
                      </span>
                      <Badge variant="outline">{alignment.mode}</Badge>
                      <Badge variant="outline">
                        hidden {alignment.hidden_strategy}
                      </Badge>
                      <Badge variant="outline">
                        layers {alignment.layer_strategy}
                      </Badge>
                      <Badge variant="outline">
                        vocab {alignment.vocab_strategy}
                      </Badge>
                    </div>
                    <p className="mt-2 text-ui-11 leading-ui-16 text-muted-foreground">
                      {alignment.notes.length > 0
                        ? alignment.notes.join(" · ")
                        : "Shares Model 1's shape and can be blended directly."}
                    </p>
                    {alignment.warnings.length > 0 ? (
                      <ul className="mt-2 flex list-disc flex-col gap-1 pl-5 text-ui-11 leading-ui-16 text-amber-600 dark:text-amber-400">
                        {alignment.warnings.map((warning) => (
                          <li key={warning}>{warning}</li>
                        ))}
                      </ul>
                    ) : null}
                  </div>
                ))}
              </div>
            </div>
          ) : null}
        </section>

        <section className="elevated-card flex flex-col gap-4 p-5">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h2 className="text-ui-13p5 font-semibold text-foreground">
              Presets and recipes
            </h2>
            <span className="text-ui-11 text-muted-foreground">
              {savedPresets.length > 0
                ? `${savedPresets.length} saved in this browser`
                : "Presets change the engine, never the parents"}
            </span>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {[
              ...BUILT_IN_MERGE_PRESETS,
              ...savedPresets.map((preset) => ({
                ...preset.setup,
                id: preset.id,
                label: preset.name,
                description: `${preset.setup.mode === "mmoe" ? "MMoE" : MERGE_METHOD_NOTES[preset.setup.method].label} \u00b7 ${preset.setup.models.length} parents`,
                weights: "keep" as const,
              })),
            ].map((preset) => (
              <button
                key={preset.id}
                type="button"
                title={preset.description}
                onClick={() =>
                  preset.id.startsWith("preset-")
                    ? applySetup(
                        savedPresets.find((entry) => entry.id === preset.id)
                          ?.setup ?? currentSetup,
                      )
                    : usePreset(preset)
                }
                disabled={running}
                className="rounded-full border border-border/60 px-2.5 py-1 text-ui-10p5 text-muted-foreground transition-colors hover:border-foreground/40 hover:text-foreground disabled:opacity-50"
              >
                {preset.label}
              </button>
            ))}
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Input
              value={presetName}
              onChange={(event) => setPresetName(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && presetName.trim()) {
                  event.preventDefault();
                  savePreset();
                }
              }}
              placeholder="Name this setup"
              className="h-9 w-56"
              disabled={running}
            />
            <Button
              variant="outline"
              size="sm"
              onClick={savePreset}
              disabled={running || presetName.trim().length === 0}
            >
              Save preset
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={copyRecipe}
              disabled={running}
            >
              Export recipe
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                setRecipeOpen((open) => !open);
                setRecipeNotice(null);
              }}
            >
              {recipeOpen ? "Hide recipe" : "Import / paste recipe"}
            </Button>
          </div>
          {savedPresets.length > 0 ? (
            <ul className="flex flex-wrap gap-2">
              {savedPresets.map((preset) => (
                <li
                  key={preset.id}
                  className="flex items-center gap-2 rounded-full border border-border/50 px-2.5 py-1 text-ui-10p5"
                >
                  <span className="text-foreground">{preset.name}</span>
                  <button
                    type="button"
                    onClick={() => deletePreset(preset.id)}
                    disabled={running}
                    className="text-muted-foreground transition-colors hover:text-destructive disabled:opacity-50"
                    aria-label={`Delete preset ${preset.name}`}
                  >
                    Remove
                  </button>
                </li>
              ))}
            </ul>
          ) : null}
          {recipeNotice ? (
            <p
              className={cn(
                "text-ui-11 leading-ui-15",
                recipeNotice.kind === "error"
                  ? "text-destructive"
                  : "text-muted-foreground",
              )}
            >
              {recipeNotice.text}
            </p>
          ) : null}
          {recipeOpen ? (
            <div className="flex flex-col gap-2">
              <Textarea
                value={recipeText}
                onChange={(event) => setRecipeText(event.target.value)}
                placeholder='{"version": 1, "setup": {"models": ["org/a", "org/b"], ...}}'
                className="min-h-28 font-mono text-ui-11"
                disabled={running}
              />
              <div className="flex flex-wrap items-center gap-2">
                <Button
                  size="sm"
                  onClick={importRecipe}
                  disabled={running || !recipeText.trim()}
                >
                  Apply recipe
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={copyRecipe}
                  disabled={running}
                >
                  Fill from current setup
                </Button>
              </div>
            </div>
          ) : null}
        </section>

        {plan || validation ? (
          <section className="elevated-card flex flex-col gap-3 p-5">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h2 className="text-ui-13p5 font-semibold text-foreground">
                Pre-merge checks
              </h2>
              {validating ? (
                <span className="flex items-center gap-2 text-ui-11 text-muted-foreground">
                  <Spinner className="size-3.5" /> Checking tokenizers
                </span>
              ) : null}
            </div>
            <p className="text-ui-11 leading-ui-16 text-muted-foreground">
              These run before any shard is written. A special-token
              disagreement would produce a model that loads and runs but quietly
              mis-handles prompts, so the merge refuses instead.
            </p>

            <div className="grid gap-3 sm:grid-cols-2">
              <div className="rounded-xl border border-border/60 p-3">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-ui-11 font-medium text-foreground">
                    Tokenizer
                  </span>
                  <Badge
                    variant={
                      validation?.tokenizer.status === "conflict"
                        ? "destructive"
                        : validation?.tokenizer.status === "warning"
                          ? "outline"
                          : "secondary"
                    }
                  >
                    {validation?.tokenizer.status === "not_checked"
                      ? "not checked yet"
                      : (validation?.tokenizer.status ?? "checking")}
                  </Badge>
                </div>
                {validation?.tokenizer.reason ? (
                  <p className="mt-2 text-ui-11 leading-ui-15 text-muted-foreground">
                    {validation.tokenizer.reason}
                  </p>
                ) : null}
                {(validation?.tokenizer.conflicts?.length ?? 0) > 0 ? (
                  <ul className="mt-2 flex list-disc flex-col gap-1 pl-4 text-ui-11 leading-ui-15 text-destructive">
                    {validation?.tokenizer.conflicts?.map((conflict) => (
                      <li key={conflict}>{conflict}</li>
                    ))}
                  </ul>
                ) : null}
                {(validation?.tokenizer.warnings?.length ?? 0) > 0 ? (
                  <ul className="mt-2 flex list-disc flex-col gap-1 pl-4 text-ui-11 leading-ui-15 text-amber-600 dark:text-amber-400">
                    {validation?.tokenizer.warnings?.map((warning) => (
                      <li key={warning}>{warning}</li>
                    ))}
                  </ul>
                ) : null}
                {showTokenOverride ? (
                  <label className="mt-3 flex items-start gap-2 text-ui-11 leading-ui-15 text-muted-foreground">
                    <input
                      type="checkbox"
                      className="mt-0.5"
                      checked={allowTokenConflict}
                      onChange={(event) =>
                        setAllowTokenConflict(event.target.checked)
                      }
                    />
                    Build it anyway. The override is recorded in the run
                    manifest, and the merged folder keeps Model A&apos;s
                    tokenizer files.
                  </label>
                ) : null}
              </div>

              <div className="rounded-xl border border-border/60 p-3">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-ui-11 font-medium text-foreground">
                    Loading policy
                  </span>
                  <Badge
                    variant={validation?.blocked ? "destructive" : "secondary"}
                  >
                    {validation?.blocked
                      ? "blocked"
                      : validation
                        ? "safe"
                        : "checking"}
                  </Badge>
                </div>
                <p className="mt-2 text-ui-11 leading-ui-15 text-muted-foreground">
                  Safetensors only by default (a pickle can run code while
                  loading), and checkpoints that need custom code are refused
                  unless you opt in.
                </p>
                <label className="mt-2 flex items-start gap-2 text-ui-11 leading-ui-15 text-muted-foreground">
                  <input
                    type="checkbox"
                    className="mt-0.5"
                    checked={allowPickle}
                    onChange={(event) => setAllowPickle(event.target.checked)}
                    disabled={running}
                  />
                  Read <span className="font-mono">.bin</span> checkpoints
                  lazily through a memory map. Only enable this for files you
                  trust.
                </label>
                {validation?.sources
                  .filter((entry) => entry.blocked_reason)
                  .map((entry) => (
                    <p
                      key={entry.label}
                      className="mt-2 text-ui-11 leading-ui-15 text-destructive"
                    >
                      {entry.blocked_reason}
                    </p>
                  ))}
                {plan?.estimates ? (
                  <p className="mt-2 text-ui-11 leading-ui-15 text-muted-foreground">
                    Peak {bytes(plan.estimates.need.ram_bytes)} RAM ·{" "}
                    {bytes(plan.estimates.need.disk_bytes)} disk
                    {plan.estimates.streaming
                      ? " · streaming, so no two parents are fully resident at once (an 8 GB CPU tier can merge)"
                      : ""}
                  </p>
                ) : null}
              </div>
            </div>
          </section>
        ) : null}

        <section className="elevated-card flex flex-col gap-3 p-5">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h2 className="text-ui-13p5 font-semibold text-foreground">
              Review before merging
            </h2>
            {(() => {
              const items = reviewItems({
                setup: currentSetup,
                plan,
                validation,
              });
              const blocking = items.filter(
                (item) => item.level === "block",
              ).length;
              return (
                <Badge variant={blocking > 0 ? "destructive" : "secondary"}>
                  {blocking > 0
                    ? `${blocking} blocking`
                    : `${items.filter((item) => item.level === "warn").length} warnings`}
                </Badge>
              );
            })()}
          </div>
          <ul className="flex flex-col gap-1.5">
            {reviewItems({ setup: currentSetup, plan, validation }).map(
              (item) => (
                <li
                  key={item.id}
                  className="flex items-start gap-2 text-ui-11 leading-ui-16"
                >
                  <span
                    aria-hidden={true}
                    className={cn(
                      "mt-1.5 size-1.5 shrink-0 rounded-full",
                      item.level === "block"
                        ? "bg-destructive"
                        : item.level === "warn"
                          ? "bg-amber-500"
                          : "bg-emerald-500",
                    )}
                  />
                  <span
                    className={cn(
                      item.level === "block"
                        ? "text-destructive"
                        : item.level === "warn"
                          ? "text-amber-600 dark:text-amber-400"
                          : "text-muted-foreground",
                    )}
                  >
                    {item.text}
                  </span>
                </li>
              ),
            )}
          </ul>
        </section>

        <section className="elevated-card flex flex-col gap-5 p-5">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 className="text-ui-13p5 font-semibold text-foreground">
                Fusion engine
              </h2>
              <p className="text-ui-11 text-muted-foreground">
                {mode === "mmoe"
                  ? "Each model keeps its own FFN as an expert; a router sends tokens to the best fits."
                  : "Weight-space blending after the alignment resolver matches shapes."}
              </p>
            </div>
            <div className="flex gap-1 rounded-full border border-border/60 p-0.5">
              {(["standard", "mmoe"] as const).map((option) => (
                <button
                  key={option}
                  type="button"
                  onClick={() => setMode(option)}
                  disabled={option === "mmoe" && hasGpu === false}
                  title={
                    option === "mmoe" && hasGpu === false
                      ? "The MMoE router needs a GPU; standard blending runs on a CPU."
                      : undefined
                  }
                  className={cn(
                    "rounded-full px-3 py-1 text-ui-11 transition-colors",
                    mode === option
                      ? "bg-foreground text-background"
                      : "text-muted-foreground hover:text-foreground",
                    "disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:text-muted-foreground",
                  )}
                >
                  {option === "standard" ? "Standard blending" : "MMoE router"}
                </button>
              ))}
            </div>
          </div>

          {mode === "standard" ? (
            <div className="grid gap-4 sm:grid-cols-3">
              <label className="flex flex-col gap-1.5">
                <span className="text-ui-11 font-medium uppercase tracking-[0.05em] text-muted-foreground/70">
                  Method
                </span>
                <Select
                  value={method}
                  onValueChange={(value) => setMethod(value as MergeMethod)}
                >
                  <SelectTrigger className="h-9">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {MERGE_METHODS.map((entry) => (
                      <SelectItem key={entry} value={entry}>
                        {entry.toUpperCase()}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <span className="text-ui-10p5 leading-ui-14 text-muted-foreground/70">
                  {MERGE_METHOD_NOTES[method].summary}
                </span>
              </label>
              {models.length === 2 ? (
                <label className="flex flex-col gap-1.5">
                  <span className="text-ui-11 font-medium uppercase tracking-[0.05em] text-muted-foreground/70">
                    Model B weight
                  </span>
                  <div className="flex h-9 items-center gap-3">
                    <Slider
                      value={[weights[1] ?? 0.5]}
                      min={0}
                      max={1}
                      step={0.05}
                      onValueChange={([value]) =>
                        setWeights([1 - value, value])
                      }
                      aria-label="Model B blend weight"
                    />
                    <span className="w-10 text-right font-mono text-ui-11">
                      {(weights[1] ?? 0.5).toFixed(2)}
                    </span>
                  </div>
                </label>
              ) : null}
              {methodUsesDensity(method) ? (
                <label className="flex flex-col gap-1.5">
                  <span className="text-ui-11 font-medium uppercase tracking-[0.05em] text-muted-foreground/70">
                    {method.toUpperCase()} density
                  </span>
                  <div className="flex h-9 items-center gap-3">
                    <Slider
                      value={[density]}
                      min={0.05}
                      max={1}
                      step={0.05}
                      onValueChange={([value]) => setDensity(value)}
                      aria-label="Merge density"
                    />
                    <span className="w-10 text-right font-mono text-ui-11">
                      {density.toFixed(2)}
                    </span>
                  </div>
                  <span className="text-ui-10p5 leading-ui-14 text-muted-foreground/70">
                    The share of each parent's deltas kept before the average.
                  </span>
                </label>
              ) : null}
              {models.length > 2 ? (
                <div className="flex flex-col gap-2 rounded-xl border border-border/60 p-3 sm:col-span-3">
                  <p className="text-ui-11p5 font-medium text-foreground">
                    Blend weights
                  </p>
                  <p className="text-ui-10p5 leading-ui-14 text-muted-foreground/70">
                    Relative weights, normalized at run time. An even split is
                    the safe default; raise one parent to pull the merge toward
                    it.
                  </p>
                  {models.map((_, index) => (
                    <div
                      key={index}
                      className="grid grid-cols-[minmax(72px,0.6fr)_minmax(0,1.6fr)_52px] items-center gap-3"
                    >
                      <span className="text-ui-11 text-muted-foreground">
                        Model {index + 1}
                      </span>
                      <Slider
                        value={[weights[index] ?? 1]}
                        min={0}
                        max={1}
                        step={0.05}
                        onValueChange={([value]) =>
                          setWeights((current) =>
                            current.map((entry, position) =>
                              position === index ? value : entry,
                            ),
                          )
                        }
                        aria-label={`Model ${index + 1} blend weight`}
                      />
                      <span className="text-right font-mono text-ui-11">
                        {Math.round((shares[index] ?? 0) * 100)}%
                      </span>
                    </div>
                  ))}
                </div>
              ) : null}
            </div>
          ) : (
            <div className="grid gap-4 sm:grid-cols-2">
              <label className="flex flex-col gap-1.5">
                <span className="text-ui-11 font-medium uppercase tracking-[0.05em] text-muted-foreground/70">
                  Experts per token (top-k)
                </span>
                <Input
                  type="number"
                  min={1}
                  max={2}
                  value={topK}
                  onChange={(event) =>
                    setTopK(
                      Math.max(1, Math.min(2, Number(event.target.value) || 1)),
                    )
                  }
                  className="h-9 w-28"
                />
              </label>
              <label className="flex flex-col gap-1.5">
                <span className="text-ui-11 font-medium uppercase tracking-[0.05em] text-muted-foreground/70">
                  Router temperature
                </span>
                <Input
                  type="number"
                  min={0.1}
                  max={5}
                  step={0.1}
                  value={temperature}
                  onChange={(event) =>
                    setTemperature(
                      Math.max(0.1, Number(event.target.value) || 1),
                    )
                  }
                  className="h-9 w-28"
                />
              </label>
              {plan ? (
                <div className="sm:col-span-2 rounded-xl border border-border/60 bg-muted/20 p-3 text-ui-11 leading-ui-16 text-muted-foreground">
                  {plan.moe.experts.length} experts · router width{" "}
                  {plan.moe.router_hidden} · shared vocab{" "}
                  {plan.moe.shared_vocab ?? "—"}. The bundle writes the router
                  and projections; expert weights stay in their source
                  checkpoints.
                </div>
              ) : null}
            </div>
          )}

          <label className="flex flex-col gap-1.5">
            <span className="text-ui-11 font-medium uppercase tracking-[0.05em] text-muted-foreground/70">
              Output folder
            </span>
            <Input
              value={outputDir}
              onChange={(event) => {
                setOutputTouched(true);
                setOutputDir(event.target.value);
              }}
              placeholder="Automatic — named after the parents, inside Unsloth's library"
              className="font-mono text-ui-12"
            />
            {folderCollisionNotice(
              outputFolderName(
                typedOutput || (plan?.suggested_output_dir ?? "") || "",
              ) || null,
              history.map((entry) => entry.name),
            ) ? (
              <span className="text-ui-11 leading-ui-15 text-amber-600 dark:text-amber-400">
                {folderCollisionNotice(
                  outputFolderName(
                    typedOutput || (plan?.suggested_output_dir ?? "") || "",
                  ) || null,
                  history.map((entry) => entry.name),
                )}
              </span>
            ) : null}
            {outsideLibrary ? (
              <span className="text-ui-11 leading-ui-15 text-amber-600 dark:text-amber-400">
                Outside Unsloth&apos;s library ({libraryRoot}). The merge still
                runs, but Library, Export and Chat only list models under that
                folder, so you could not quantize or chat with the result.
              </span>
            ) : (
              <span className="text-ui-11 leading-ui-15 text-muted-foreground/70">
                {libraryRoot
                  ? `Written inside ${libraryRoot}, the folder Studio scans — the merged model shows up in Library, Export (quantize) and Chat as soon as it finishes.`
                  : "Written inside Unsloth's library folder, so the merged model shows up in Library, Export (quantize) and Chat when it finishes."}
              </span>
            )}
          </label>

          <div className="flex flex-wrap items-center gap-3">
            <Button
              onClick={() => void start()}
              disabled={
                running ||
                !plan ||
                !weightsValid ||
                Boolean(validation?.blocked && !allowTokenConflict)
              }
              size="sm"
            >
              {running ? (
                <Spinner className="size-3.5" />
              ) : (
                <HugeiconsIcon icon={PlayIcon} className="size-3.5" />
              )}
              {mode === "mmoe" ? "Build MMoE bundle" : "Run merge"}
            </Button>
            {job ? (
              <span className="flex items-center gap-2 text-ui-11 text-muted-foreground">
                {job.status === "running" ? (
                  <HugeiconsIcon
                    icon={Refresh01Icon}
                    className="size-3.5 animate-spin"
                  />
                ) : null}
                {job.status}
                {job.error ? ` — ${job.error}` : ""}
              </span>
            ) : null}
            {runError && job?.status !== "failed" ? (
              <span className="text-ui-11 text-destructive">{runError}</span>
            ) : null}
            {weightsValid ? null : (
              <span className="text-ui-11 text-destructive">
                Raise at least one blend weight above zero.
              </span>
            )}
          </div>

          {job?.status === "running" ? (
            <div className="flex flex-col gap-2 rounded-2xl border border-border/60 p-4">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="text-ui-11p5 font-medium text-foreground">
                  {PROGRESS_LABELS[job.progress?.stage ?? "queued"] ??
                    "Merging"}
                </span>
                <span className="font-mono text-ui-10p5 text-muted-foreground">
                  {progressPercent(job.progress) === null
                    ? "preparing"
                    : `${Math.round(progressPercent(job.progress) ?? 0)}%`}
                </span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-muted">
                <div
                  className={cn(
                    "h-full rounded-full bg-foreground transition-[width] duration-300",
                    progressPercent(job.progress) === null &&
                      "w-1/3 animate-pulse",
                  )}
                  style={
                    progressPercent(job.progress) === null
                      ? undefined
                      : { width: `${progressPercent(job.progress)}%` }
                  }
                />
              </div>
              <p className="text-ui-10p5 leading-ui-14 text-muted-foreground/80">
                {job.progress?.message ??
                  "Downloading any Hub snapshots and reading the shard indexes"}
                {job.progress?.blended_tensors != null
                  ? ` · ${job.progress.blended_tensors} blended`
                  : ""}
                {job.progress?.copied_tensors
                  ? ` · ${job.progress.copied_tensors} copied`
                  : ""}
                {job.progress?.bytes_written
                  ? ` · ${bytes(job.progress.bytes_written)} written`
                  : ""}
                {(() => {
                  const rate = throughputBytesPerSecond(job);
                  return rate ? ` · ${bytes(rate)}/s` : "";
                })()}
                {(() => {
                  const eta = etaSeconds(job);
                  if (!eta) return "";
                  const minutes = Math.floor(eta / 60);
                  return ` · about ${minutes > 0 ? `${minutes}m ` : ""}${eta % 60}s left`;
                })()}
              </p>
              <ol className="mt-1 flex flex-col gap-1 text-ui-10p5 text-muted-foreground">
                {progressTimeline(job).map((step) => (
                  <li key={step.stage} className="flex items-center gap-2">
                    <span
                      aria-hidden={true}
                      className={cn(
                        "size-1.5 rounded-full",
                        step.state === "done"
                          ? "bg-emerald-500"
                          : step.state === "active"
                            ? "animate-pulse bg-foreground"
                            : step.state === "failed"
                              ? "bg-destructive"
                              : "bg-muted-foreground/30",
                      )}
                    />
                    <span
                      className={step.state === "pending" ? "opacity-50" : ""}
                    >
                      {step.label}
                      {step.detail ? ` — ${step.detail}` : ""}
                    </span>
                  </li>
                ))}
              </ol>
            </div>
          ) : null}

          {job?.status === "completed" && job.result ? (
            <div className="rounded-2xl border border-emerald-500/30 bg-emerald-500/5 p-4 text-ui-11 leading-ui-16">
              {(() => {
                const completedPath =
                  job.output_dir ??
                  (job.result as { output_dir?: string }).output_dir ??
                  outputDir;
                const result = job.result as {
                  metrics?: {
                    blended_tensors?: number;
                    shards?: number;
                    total_bytes?: number;
                  };
                  tokenizer?: { status?: string; override_used?: boolean };
                };
                const metrics = result.metrics;
                const override = result.tokenizer?.override_used === true;
                // Only a weight-space merge is a loadable model; an MMoE bundle is a router
                // Studio trains, so Library and Export have nothing to open for it.
                // From the job, not the mode toggle, which the user can flip afterwards.
                const loadable = job.kind === "merge";
                return (
                  <>
                    <p className="font-medium text-foreground">
                      {loadable ? "Merge complete" : "MMoE bundle written"}
                    </p>
                    <p className="mt-1 break-all font-mono text-muted-foreground">
                      {completedPath}
                    </p>
                    <p className="mt-1 text-muted-foreground">
                      {metrics
                        ? `${metrics.blended_tensors ?? 0} tensors blended in ${metrics.shards ?? 0} shards (${bytes(metrics.total_bytes)})`
                        : "bundle written"}
                    </p>
                    {override ? (
                      <p className="mt-1 text-amber-600 dark:text-amber-400">
                        Built with the tokenizer override: the parents disagreed
                        on a special token and it was accepted. The merged
                        folder keeps Model A&apos;s tokenizer files, so prompt
                        the result the way that parent expects.
                      </p>
                    ) : null}
                    {structural ? (
                      <div className="mt-2 flex flex-col gap-1">
                        <p
                          className={cn(
                            "font-medium",
                            structural.structurally_valid
                              ? "text-emerald-600 dark:text-emerald-400"
                              : "text-destructive",
                          )}
                        >
                          Structural check: {structural.verdict}
                        </p>
                        <ul className="flex flex-col gap-0.5">
                          {structuralChecks(structural).map((check) => (
                            <li
                              key={check.label}
                              className="text-ui-10p5 leading-ui-14"
                            >
                              <span
                                className={
                                  check.ok
                                    ? "text-muted-foreground"
                                    : "text-destructive"
                                }
                              >
                                {check.ok ? "PASS" : "FAIL"} · {check.label}
                              </span>{" "}
                              <span className="text-muted-foreground/70">
                                {check.detail}
                              </span>
                            </li>
                          ))}
                        </ul>
                        {structural.problems.length > 0 ? (
                          <p className="text-ui-10p5 leading-ui-14 text-destructive">
                            {structural.problems.join(" ")}
                          </p>
                        ) : null}
                      </div>
                    ) : null}
                    {job.run_manifest ? (
                      <p className="mt-1 break-all font-mono text-ui-10p5 text-muted-foreground/70">
                        manifest: {job.run_manifest}
                      </p>
                    ) : null}
                    {job.library_visible === false ? (
                      <p className="mt-2 text-amber-600 dark:text-amber-400">
                        Stored outside Unsloth&apos;s library, so Library,
                        Export and Chat will not list it. Merge into{" "}
                        {libraryRoot ? (
                          <span className="font-mono">{libraryRoot}</span>
                        ) : (
                          "Unsloth's outputs folder"
                        )}{" "}
                        to quantize or chat with it.
                      </p>
                    ) : loadable ? (
                      <div className="mt-3 flex flex-wrap items-center gap-2">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() =>
                            void navigate({
                              to: "/library",
                              search: { show: "models" },
                            })
                          }
                        >
                          Open in Library
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() =>
                            void navigate({
                              to: "/export",
                              search: { run: outputFolderName(completedPath) },
                            })
                          }
                        >
                          Quantize in Export
                        </Button>
                        <span className="text-ui-10p5 text-muted-foreground/70">
                          Pick a GGUF level there, then Chat with the file it
                          writes.
                        </span>
                      </div>
                    ) : null}
                  </>
                );
              })()}
            </div>
          ) : null}
          {job?.status === "failed" ? (
            <p className="text-ui-11 text-destructive">{job.error}</p>
          ) : null}
        </section>

        <section className="elevated-card flex flex-col gap-3 p-5">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h2 className="text-ui-13p5 font-semibold text-foreground">
              Recent merges
            </h2>
            <Button
              variant="ghost"
              size="sm"
              onClick={refreshHistory}
              disabled={running}
            >
              <HugeiconsIcon icon={Refresh01Icon} className="size-3.5" />
              Refresh
            </Button>
          </div>
          <p className="text-ui-11 leading-ui-16 text-muted-foreground">
            Read from the run manifests beside each output, so loading one
            restores the parents and settings that produced it.
          </p>
          {historyError ? (
            <p className="text-ui-11 text-destructive">{historyError}</p>
          ) : null}
          {!historyError && history.length === 0 ? (
            <p className="text-ui-11 text-muted-foreground/70">
              No completed merges in Unsloth&apos;s library yet.
            </p>
          ) : null}
          <ul className="flex flex-col divide-y divide-border/50">
            {history.map((entry) => (
              <li
                key={entry.dir}
                className="flex flex-wrap items-center gap-3 py-2 text-ui-11"
              >
                <div className="min-w-0 flex-1">
                  <p className="truncate font-mono text-foreground">
                    {entry.name}
                  </p>
                  <p className="text-ui-10p5 text-muted-foreground/80">
                    {entry.stage === "moe"
                      ? "MMoE bundle"
                      : (entry.method ?? "merge")}
                    {entry.models.length > 0
                      ? ` · ${entry.models.map((model) => model.split("/").pop()).join(" + ")}`
                      : ""}
                    {entry.total_bytes ? ` · ${bytes(entry.total_bytes)}` : ""}
                  </p>
                  <p className="text-ui-10p5 text-muted-foreground/60">
                    {entry.created_at
                      ? new Date(entry.created_at).toLocaleString()
                      : "timestamp not recorded"}
                    {entry.tokenizer_override ? " · tokenizer override" : ""}
                  </p>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    applySetup({
                      models: entry.models,
                      weights: fitWeights(
                        entry.weights ?? [],
                        entry.models.length,
                      ), // An unknown method in an old manifest must not reach the run payload.
                      method:
                        entry.method === "slerp" ||
                        entry.method === "ties" ||
                        entry.method === "dare" ||
                        entry.method === "linear"
                          ? entry.method
                          : "linear",
                      density: entry.density ?? 0.5,
                      mode: entry.stage === "moe" ? "mmoe" : "standard",
                      topK: entry.top_k ?? 2,
                      temperature: entry.temperature ?? 1,
                      outputDir: "",
                    });
                    setRecipeNotice({
                      kind: "ok",
                      text: `Loaded ${entry.name}. Inspect the parents to re-check them before running.`,
                    });
                  }}
                  disabled={running}
                >
                  Load into slots
                </Button>
              </li>
            ))}
          </ul>
        </section>
      </div>
    </div>
  );
}
