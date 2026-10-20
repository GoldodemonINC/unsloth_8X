// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useNavigate } from "@tanstack/react-router";
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
import { hfApiToken, useHfTokenStore } from "@/features/hub";
import { StudioModeNav } from "@/features/studio/studio-mode-nav";
import { useAppShellReadySignal } from "@/components/app-readiness";
import { cn } from "@/lib/utils";
import {
  ArrowRight01Icon,
  Layers01Icon,
  PlayIcon,
  Refresh01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useState, type ReactElement } from "react";
import {
  evalMergedModel,
  fetchOutputLocations,
  planMerge,
  runMerge,
  runMoeMerge,
  validateMerge,
  waitForJob,
  type AdvancedJob,
  type MergePlanResponse,
  type MergeValidation,
  type StructuralEval,
} from "./api/advanced-api";
import { MergeModelSlot } from "./components/merge-model-slot";
import { MERGE_METHODS, type MergeMethod } from "./types";

const MODE_LABELS: Record<string, string> = {
  identical: "Identical checkpoints",
  cross_scale: "Cross-scale merge",
  cross_architecture: "Cross-architecture merge",
};

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
      <p className="mt-1 truncate font-mono text-ui-11 text-foreground" title={spec.source}>
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
        <dd className="text-right font-mono">{spec.intermediate_size ?? "—"}</dd>
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
  const [modelA, setModelA] = useState("");
  const [modelB, setModelB] = useState("");
  const [plan, setPlan] = useState<MergePlanResponse | null>(null);
  const [planError, setPlanError] = useState<string | null>(null);
  const [planning, setPlanning] = useState(false);
  const [mode, setMode] = useState<"standard" | "mmoe">("standard");
  const [method, setMethod] = useState<MergeMethod>("linear");
  const [weight, setWeight] = useState(0.5);
  const [density, setDensity] = useState(0.5);
  const [topK, setTopK] = useState(2);
  const [temperature, setTemperature] = useState(1);
  // Blank means "let the backend pick": it writes a folder named after both models directly
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
  const [structural, setStructural] = useState<StructuralEval | null>(null);

  useEffect(() => {
    signalReady();
  }, [signalReady]);

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
        model_a: modelA.trim(),
        model_b: modelB.trim(),
        top_k: topK,
        temperature,
        hf_token: accessToken,
      });
      setPlan(result);
      setMode(result.recommended_mode);
      if (!outputTouched && result.suggested_output_dir) {
        setOutputDir(result.suggested_output_dir);
      }
      if (result.models[1].hidden_size) {
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
          model_a: modelA.trim(),
          model_b: modelB.trim(),
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
              models: [modelA.trim(), modelB.trim()],
              output_dir: typedOutput || null,
              top_k: topK,
              temperature,
              hf_token: accessToken,
            })
          : await runMerge({
              model_a: modelA.trim(),
              model_b: modelB.trim(),
              output_dir: typedOutput || null,
              method,
              weight,
              density,
              max_shard_gb: 4,
              hf_token: accessToken,
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
      const mergedPath =
        finished.output_dir ??
        (finished.result as { output_dir?: string } | null)?.output_dir ??
        null;
      // Structural tier only: it proves the folder loads and is sane, not that the
      // merge is better. Quality needs an eval suite.
      if (mode === "standard" && finished.status === "completed" && mergedPath) {
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
                Fuse two checkpoints — same family or different architectures — by
                weights or by a trained MMoE router.
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
          <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] lg:items-start">
            <MergeModelSlot
              label="Model A"
              value={modelA}
              onChange={setModelA}
              placeholder="C:\\models\\llama-3.2-1b"
              disabled={running}
            />
            <div className="hidden justify-center pt-10 lg:flex">
              <HugeiconsIcon
                icon={ArrowRight01Icon}
                className="size-4 text-muted-foreground"
              />
            </div>
            <MergeModelSlot
              label="Model B"
              value={modelB}
              onChange={setModelB}
              placeholder="C:\\models\\qwen3-2b"
              disabled={running}
            />
          </div>
          <p className="text-ui-10p5 leading-ui-14 text-muted-foreground/70">
            Inspect downloads only <span className="font-mono">config.json</span>{" "}
            from the Hub; Run merge fetches the safetensors snapshot first. Local
            checkpoints are used as-is.
          </p>
          <div className="flex flex-wrap items-center gap-3">
            <Button
              onClick={() => void inspect()}
              disabled={planning || !modelA.trim() || !modelB.trim()}
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
                variant={plan.recommended_mode === "mmoe" ? "destructive" : "secondary"}
              >
                {MODE_LABELS[plan.alignment.mode] ?? plan.alignment.mode}
              </Badge>
            ) : null}
          </div>
          {planError ? (
            <p className="text-ui-11 text-destructive">{planError}</p>
          ) : null}
          {!plan && !planError ? (
            <p className="text-ui-11 leading-ui-16 text-muted-foreground/80">
              Pick both checkpoints — Hub repos or local paths — then inspect. The
              resolver reports each shape and the strategy it will use to align
              them, and only then unlocks Run merge.
            </p>
          ) : null}

          {plan ? (
            <div className="flex flex-col gap-4">
              <div className="grid gap-3 lg:grid-cols-2">
                <SpecCard label="Model A" spec={plan.models[0]} />
                <SpecCard label="Model B" spec={plan.models[1]} />
              </div>
              <div className="rounded-2xl border border-border/60 p-4">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-ui-11p5 font-medium text-foreground">
                    Alignment resolver
                  </span>
                  <Badge variant="outline">
                    hidden {plan.alignment.hidden_strategy}
                  </Badge>
                  <Badge variant="outline">
                    layers {plan.alignment.layer_strategy}
                  </Badge>
                  <Badge variant="outline">
                    vocab {plan.alignment.vocab_strategy}
                  </Badge>
                </div>
                <p className="mt-2 text-ui-11 leading-ui-16 text-muted-foreground">
                  {plan.alignment.notes.length > 0
                    ? plan.alignment.notes.join(" · ")
                    : "The two checkpoints share a shape and can be blended directly."}
                </p>
                {plan.alignment.warnings.length > 0 ? (
                  <ul className="mt-2 flex list-disc flex-col gap-1 pl-5 text-ui-11 leading-ui-16 text-amber-600 dark:text-amber-400">
                    {plan.alignment.warnings.map((warning) => (
                      <li key={warning}>{warning}</li>
                    ))}
                  </ul>
                ) : null}
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
              These run before any shard is written. A special-token disagreement
              would produce a model that loads and runs but quietly mis-handles
              prompts, so the merge refuses instead.
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
                {validation?.blocked && validation.tokenizer.status === "conflict" ? (
                  <label className="mt-3 flex items-start gap-2 text-ui-11 leading-ui-15 text-muted-foreground">
                    <input
                      type="checkbox"
                      className="mt-0.5"
                      checked={allowTokenConflict}
                      onChange={(event) => setAllowTokenConflict(event.target.checked)}
                    />
                    Build it anyway. The override is recorded in the run manifest.
                  </label>
                ) : null}
              </div>

              <div className="rounded-xl border border-border/60 p-3">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-ui-11 font-medium text-foreground">
                    Loading policy
                  </span>
                  <Badge variant={validation?.blocked ? "destructive" : "secondary"}>
                    {validation?.blocked ? "blocked" : validation ? "safe" : "checking"}
                  </Badge>
                </div>
                <p className="mt-2 text-ui-11 leading-ui-15 text-muted-foreground">
                  Safetensors only by default (a pickle can run code while loading),
                  and checkpoints that need custom code are refused unless you opt in.
                </p>
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
                      ? " · streaming, so the parents are never both resident (an 8 GB CPU tier can merge)"
                      : ""}
                  </p>
                ) : null}
              </div>
            </div>
          </section>
        ) : null}

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
                  className={cn(
                    "rounded-full px-3 py-1 text-ui-11 transition-colors",
                    mode === option
                      ? "bg-foreground text-background"
                      : "text-muted-foreground hover:text-foreground",
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
              </label>
              <label className="flex flex-col gap-1.5">
                <span className="text-ui-11 font-medium uppercase tracking-[0.05em] text-muted-foreground/70">
                  Model B weight
                </span>
                <div className="flex h-9 items-center gap-3">
                  <Slider
                    value={[weight]}
                    min={0}
                    max={1}
                    step={0.05}
                    onValueChange={([value]) => setWeight(value)}
                    aria-label="Model B blend weight"
                  />
                  <span className="w-10 text-right font-mono text-ui-11">
                    {weight.toFixed(2)}
                  </span>
                </div>
              </label>
              <label className="flex flex-col gap-1.5">
                <span className="text-ui-11 font-medium uppercase tracking-[0.05em] text-muted-foreground/70">
                  TIES / DARE density
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
              </label>
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
                  {plan.moe.shared_vocab ?? "—"}. The bundle writes the router and
                  projections; expert weights stay in their source checkpoints.
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
              placeholder="Automatic — named after both models, inside Unsloth's library"
              className="font-mono text-ui-12"
            />
            {outsideLibrary ? (
              <span className="text-ui-11 leading-ui-15 text-amber-600 dark:text-amber-400">
                Outside Unsloth&apos;s library ({libraryRoot}). The merge still runs,
                but Library, Export and Chat only list models under that folder, so
                you could not quantize or chat with the result.
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
            {runError ? (
              <span className="text-ui-11 text-destructive">{runError}</span>
            ) : null}
          </div>

          {job?.status === "completed" && job.result ? (
            <div className="rounded-2xl border border-emerald-500/30 bg-emerald-500/5 p-4 text-ui-11 leading-ui-16">
              {(() => {
                const completedPath =
                  job.output_dir ??
                  (job.result as { output_dir?: string }).output_dir ??
                  outputDir;
                const metrics = (job.result as {
                  metrics?: { blended_tensors?: number; shards?: number; total_bytes?: number };
                }).metrics;
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
                    {structural ? (
                      <p
                        className={cn(
                          "mt-1",
                          structural.structurally_valid
                            ? "text-emerald-600 dark:text-emerald-400"
                            : "text-destructive",
                        )}
                      >
                        Structural check: {structural.verdict}
                        {structural.problems.length > 0
                          ? ` — ${structural.problems.join(" ")}`
                          : " — loadable, index and shards agree, no NaN weights, tokenizer present."}
                      </p>
                    ) : null}
                    {job.run_manifest ? (
                      <p className="mt-1 break-all font-mono text-ui-10p5 text-muted-foreground/70">
                        manifest: {job.run_manifest}
                      </p>
                    ) : null}
                    {job.library_visible === false ? (
                      <p className="mt-2 text-amber-600 dark:text-amber-400">
                        Stored outside Unsloth&apos;s library, so Library, Export and Chat
                        will not list it. Merge into{" "}
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
                            void navigate({ to: "/library", search: { show: "models" } })
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
      </div>
    </div>
  );
}
