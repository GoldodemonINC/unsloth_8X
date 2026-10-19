// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
  planMerge,
  runMerge,
  runMoeMerge,
  waitForJob,
  type AdvancedJob,
  type MergePlanResponse,
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
  const [outputDir, setOutputDir] = useState("");
  const [job, setJob] = useState<AdvancedJob | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    signalReady();
  }, [signalReady]);

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
      if (result.models[1].hidden_size) {
        setTopK(Math.min(2, result.moe.experts.length));
      }
    } catch (error) {
      setPlan(null);
      setPlanError(error instanceof Error ? error.message : String(error));
    } finally {
      setPlanning(false);
    }
  };

  const start = async () => {
    setRunError(null);
    setJob(null);
    if (!outputDir.trim()) {
      setRunError("Choose an output folder for the merged checkpoint.");
      return;
    }
    setRunning(true);
    try {
      const created =
        mode === "mmoe"
          ? await runMoeMerge({
              models: [modelA.trim(), modelB.trim()],
              output_dir: outputDir.trim(),
              top_k: topK,
              temperature,
              hf_token: accessToken,
            })
          : await runMerge({
              model_a: modelA.trim(),
              model_b: modelB.trim(),
              output_dir: outputDir.trim(),
              method,
              weight,
              density,
              max_shard_gb: 4,
              hf_token: accessToken,
            });
      setJob(created);
      const finished = await waitForJob(created.job_id, setJob);
      setJob(finished);
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
              onChange={(event) => setOutputDir(event.target.value)}
              placeholder="C:\\models\\merged-output"
              className="font-mono text-ui-12"
            />
          </label>

          <div className="flex flex-wrap items-center gap-3">
            <Button
              onClick={() => void start()}
              disabled={running || !plan || !outputDir.trim()}
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
              <p className="font-medium text-foreground">Merge complete</p>
              <p className="mt-1 text-muted-foreground">
                {(job.result as { output_dir?: string }).output_dir ??
                  outputDir}{" "}
                ·{" "}
                {(() => {
                  const metrics = (job.result as {
                    metrics?: { blended_tensors?: number; shards?: number; total_bytes?: number };
                  }).metrics;
                  return metrics
                    ? `${metrics.blended_tensors ?? 0} tensors blended in ${metrics.shards ?? 0} shards (${bytes(metrics.total_bytes)})`
                    : "bundle written";
                })()}
              </p>
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
