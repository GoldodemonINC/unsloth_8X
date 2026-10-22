// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";
import {
  Alert02Icon,
  ArrowExpand02Icon,
  Cancel01Icon,
  PlayIcon,
  Refresh01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useRef, useState } from "react";
import { MergeModelSlot } from "./merge-model-slot";
import {
  planScaler,
  runScaler,
  type AdvancedJob,
  type ScalerMethod,
  type ScalerPlan,
  type ScalerResult,
  waitForJob,
} from "../api/advanced-api";

/** 134.50M / 1.02B: the slider reads in billions, the checkpoint not always. */
function params(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return `${Math.round(value)}`;
}

export function ScalerPanel({
  defaultSource = "",
  onUseCheckpoint,
}: {
  /** The model the surrounding page is working on; the panel follows it until edited. */
  defaultSource?: string;
  /** Present when the caller can put the expanded checkpoint somewhere useful. */
  onUseCheckpoint?: (path: string) => void;
}) {
  const [source, setSource] = useState(defaultSource);
  const [method, setMethod] = useState<ScalerMethod>("depth");
  // The slider's quantity: parameters added on top of the checkpoint's own count.
  const [added, setAdded] = useState<number | null>(null);
  const [plan, setPlan] = useState<ScalerPlan | null>(null);
  const [planError, setPlanError] = useState<string | null>(null);
  const [planning, setPlanning] = useState(false);
  const [job, setJob] = useState<AdvancedJob | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [adopted, setAdopted] = useState(false);

  // Follow the page's selection until the user types their own source.
  const previousDefault = useRef(defaultSource);
  useEffect(() => {
    setSource((current) =>
      current === "" || current === previousDefault.current
        ? defaultSource
        : current,
    );
    previousDefault.current = defaultSource;
  }, [defaultSource]);

  // Plan on every change, debounced: the response carries the projection and the
  // refusal list, so nothing about the run is guessed on the client.
  useEffect(() => {
    const trimmed = source.trim();
    if (!trimmed) {
      setPlan(null);
      setPlanError(null);
      setPlanning(false);
      return;
    }
    let cancelled = false;
    setPlanning(true);
    const timer = setTimeout(() => {
      planScaler({ source: trimmed, method, added_params: added })
        .then((result) => {
          if (cancelled) return;
          setPlan(result);
          setPlanError(null);
          setAdded((current) => {
            if (current == null) return result.default_added_params;
            if (current < result.min_added_params) return result.min_added_params;
            if (current > result.max_added_params) return result.max_added_params;
            return current;
          });
        })
        .catch((error: unknown) => {
          if (cancelled) return;
          setPlan(null);
          setPlanError(error instanceof Error ? error.message : String(error));
        })
        .finally(() => {
          if (!cancelled) setPlanning(false);
        });
    }, 450);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [source, method, added]);

  const start = async () => {
    setRunError(null);
    setAdopted(false);
    if (!plan || added == null) {
      setRunError("Plan a checkpoint first.");
      return;
    }
    setRunning(true);
    try {
      const created = await runScaler({
        source: plan.source,
        added_params: added,
        method,
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

  const result =
    job?.status === "completed" && job.result
      ? (job.result as unknown as ScalerResult)
      : null;
  const floorPercent = plan
    ? Math.min(
        100,
        Math.max(
          0,
          (plan.floor_added_params / Math.max(1, plan.max_added_params)) * 100,
        ),
      )
    : 0;
  // While the slider moves, the readout follows the thumb instead of waiting
  // for the debounced plan to come back.
  const addedValue = plan ? (added ?? plan.default_added_params) : 0;
  const targetPreview = plan ? plan.base_params + addedValue : 0;
  const growthPreview = plan
    ? targetPreview / Math.max(1, plan.base_params)
    : 1;
  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-start gap-3">
          <span className="mt-0.5 flex size-9 items-center justify-center rounded-full bg-muted/60">
            <HugeiconsIcon icon={ArrowExpand02Icon} className="size-4" />
          </span>
          <div className="min-w-0">
            <p className="text-ui-13p5 font-semibold text-foreground">
              Scaler
            </p>
            <p className="text-ui-11 leading-ui-15 text-muted-foreground/85">
              Grow the checkpoint before training it. At step 0 the expanded
              model computes exactly what the base does; the new capacity is
              unlocked by the continued pretraining that follows.
            </p>
          </div>
        </div>
        {plan ? (
          <Badge variant="outline" className="font-mono">
            {params(plan.base_params)} base
          </Badge>
        ) : null}
      </div>

      <div className="flex flex-col gap-1.5">
        <MergeModelSlot
          label="Checkpoint to expand"
          value={source}
          onChange={(next) => {
            setSource(next);
            setAdded(null);
            setJob(null);
          }}
          placeholder="C:\\models\\llama-3.2-1b"
          disabled={running}
        />
        <span className="text-ui-11 leading-ui-15 text-muted-foreground/70">
          Search the Hugging Face Hub, or point at a local checkpoint. Only
          config.json is read to plan; the weights load when you run.
        </span>
      </div>

      {planError ? (
        <p className="text-ui-11 text-destructive">{planError}</p>
      ) : null}

      {plan ? (
        <>
          <div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_220px]">
            <div className="flex flex-col gap-2">
              <div className="flex items-baseline justify-between gap-3">
                <span className="text-ui-11 font-medium uppercase tracking-[0.05em] text-muted-foreground/70">
                  Added parameters
                </span>
                <span className="font-mono text-ui-13 text-foreground">
                  +{params(addedValue)}
                  <span className="ml-2 font-normal text-muted-foreground">
                    {params(plan.base_params)} → {params(targetPreview)} (
                    {growthPreview.toFixed(2)}x)
                  </span>
                </span>
              </div>

              {/* The refused zone, drawn under the Radix track: 0 added up to the floor. */}
              <div className="relative py-1">
                <div
                  aria-hidden={true}
                  className="pointer-events-none absolute inset-x-0 top-1/2 h-2 -translate-y-1/2 overflow-hidden rounded-4xl bg-[rgb(0_0_0_/_0.1)] dark:bg-[rgb(0_0_0_/_0.12)]"
                >
                  <div
                    className="h-full bg-amber-500/50"
                    style={{ width: `${floorPercent}%` }}
                  />
                </div>
                <div
                  aria-hidden={true}
                  className="pointer-events-none absolute top-1/2 h-4 w-px -translate-y-1/2 bg-amber-600 dark:bg-amber-400"
                  style={{ left: `${floorPercent}%` }}
                />
                <div className="relative [&_[data-slot=slider-track]]:bg-transparent">
                  <Slider
                    value={[addedValue]}
                    min={plan.min_added_params}
                    max={plan.max_added_params}
                    step={plan.step_params}
                    disabled={plan.blocked || running}
                    onValueChange={([value]) => setAdded(value)}
                    thumbValueText={(value) =>
                      `plus ${params(value)} parameters, ${(
                        (plan.base_params + value) /
                        Math.max(1, plan.base_params)
                      ).toFixed(2)}x the base`
                    }
                    aria-label="Parameters to add"
                  />
                </div>
                <div className="mt-1 flex items-center justify-between text-ui-10p5 text-muted-foreground/70">
                  <span className="font-mono">+0</span>
                  <span
                    className="absolute -translate-x-1/2 font-mono text-amber-700 dark:text-amber-400"
                    style={{ left: `${floorPercent}%` }}
                  >
                    {plan.growth_floor}x floor
                  </span>
                  <span className="font-mono">
                    +{params(plan.max_added_params)}
                  </span>
                </div>
              </div>
              <p className="text-ui-10p5 leading-ui-14 text-muted-foreground/70">
                Slider adds 0 to {params(plan.max_added_params)} parameters on top
                of the checkpoint. Adding less than{" "}
                {params(plan.floor_added_params)} (below {plan.growth_floor}x) is
                refused: no growth study reports a gain there, and the run costs
                as much as a real one.
              </p>
            </div>

            <label className="flex flex-col gap-1.5">
              <span className="text-ui-11 font-medium uppercase tracking-[0.05em] text-muted-foreground/70">
                Growth axis
              </span>
              <Select
                value={method}
                onValueChange={(value) => {
                  setMethod(value as ScalerMethod);
                  setAdded(null);
                }}
              >
                <SelectTrigger className="h-9">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {plan.methods.map((option) => (
                    <SelectItem key={option.id} value={option.id}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <span className="text-ui-10p5 leading-ui-14 text-muted-foreground/70">
                Depth is the best-evidenced axis; width only widens the
                feed-forward, so every other shape stays loadable.
              </span>
            </label>
          </div>

          {plan.projection ? (
            <div className="grid grid-cols-2 gap-3 rounded-2xl border border-border/60 p-4 sm:grid-cols-4">
              <div>
                <p className="text-ui-10p5 uppercase tracking-[0.06em] text-muted-foreground/70">
                  Projected
                </p>
                <p className="font-mono text-ui-14 text-foreground">
                  {params(plan.projection.projected_params)}
                </p>
              </div>
              <div>
                <p className="text-ui-10p5 uppercase tracking-[0.06em] text-muted-foreground/70">
                  Added
                </p>
                <p className="font-mono text-ui-14 text-foreground">
                  +{params(plan.projection.added_params)}
                </p>
              </div>
              <div>
                <p className="text-ui-10p5 uppercase tracking-[0.06em] text-muted-foreground/70">
                  Layers
                </p>
                <p className="font-mono text-ui-14 text-foreground">
                  {plan.projection.current_num_layers} →{" "}
                  {plan.projection.new_num_layers}
                </p>
              </div>
              <div>
                <p className="text-ui-10p5 uppercase tracking-[0.06em] text-muted-foreground/70">
                  MLP width
                </p>
                <p className="font-mono text-ui-14 text-foreground">
                  {plan.projection.current_intermediate_size} →{" "}
                  {plan.projection.new_intermediate_size}
                </p>
              </div>
            </div>
          ) : null}

          {plan.refusals.length > 0 ? (
            <div className="flex flex-col gap-2">
              {plan.refusals.map((refusal) => (
                <div
                  key={refusal.code}
                  className={cn(
                    "flex items-start gap-2 rounded-xl border p-2.5 text-ui-11 leading-ui-16",
                    refusal.blocks
                      ? "border-destructive/40 bg-destructive/5 text-foreground"
                      : "border-amber-500/40 bg-amber-500/5 text-foreground",
                  )}
                >
                  <HugeiconsIcon
                    icon={refusal.blocks ? Cancel01Icon : Alert02Icon}
                    className={cn(
                      "mt-0.5 size-3.5 shrink-0",
                      refusal.blocks
                        ? "text-destructive"
                        : "text-amber-600 dark:text-amber-400",
                    )}
                  />
                  <p>{refusal.message}</p>
                </div>
              ))}
            </div>
          ) : null}

          <div className="flex flex-wrap items-center gap-3">
            <Button
              size="sm"
              onClick={() => void start()}
              disabled={running || !plan.runnable}
            >
              {running ? (
                <Spinner className="size-3.5" />
              ) : (
                <HugeiconsIcon icon={PlayIcon} className="size-3.5" />
              )}
              Expand checkpoint
            </Button>
            {planning ? (
              <span className="flex items-center gap-2 text-ui-11 text-muted-foreground">
                <HugeiconsIcon
                  icon={Refresh01Icon}
                  className="size-3.5 animate-spin"
                />
                planning
              </span>
            ) : null}
            {running && job?.progress?.message ? (
              <span className="text-ui-11 text-muted-foreground">
                {job.progress.message}
              </span>
            ) : null}
            {runError ? (
              <span className="text-ui-11 text-destructive">{runError}</span>
            ) : null}
          </div>

          {result ? (
            <div className="flex flex-col gap-2 rounded-2xl border border-emerald-500/30 bg-emerald-500/5 p-4 text-ui-11 leading-ui-16">
              <p className="font-medium text-foreground">
                Expanded to {params(result.final_params)} (
                {result.growth_factor.toFixed(2)}x)
              </p>
              <p className="font-mono break-all text-muted-foreground">
                {result.output_dir}
              </p>
              <p className="text-muted-foreground">
                Function preserved at step 0: max |delta logit|{" "}
                {result.step0_max_logit_diff.toExponential(1)}
                {result.reload_max_logit_diff != null
                  ? ` (${result.reload_max_logit_diff.toExponential(1)} after reload)`
                  : ""}
                {result.new_num_layers != null
                  ? ` · ${result.new_num_layers} layers`
                  : ""}
                {result.elapsed_seconds
                  ? ` · ${result.elapsed_seconds.toFixed(0)}s`
                  : ""}
              </p>
              {onUseCheckpoint ? (
                <div className="flex items-center gap-2">
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => {
                      onUseCheckpoint(result.output_dir);
                      setAdopted(true);
                    }}
                  >
                    Use in Training Studio
                  </Button>
                  {adopted ? (
                    <span className="text-ui-10p5 text-muted-foreground">
                      Set as the training model.
                    </span>
                  ) : null}
                </div>
              ) : null}
            </div>
          ) : null}

          {job?.status === "failed" ? (
            <p className="text-ui-11 text-destructive">{job.error}</p>
          ) : null}

          {plan.notes.length > 0 ? (
            <ul className="flex list-disc flex-col gap-1 pl-5 text-ui-10p5 leading-ui-14 text-muted-foreground/70">
              {plan.notes.map((note) => (
                <li key={note}>{note}</li>
              ))}
            </ul>
          ) : null}
        </>
      ) : null}
    </div>
  );
}
