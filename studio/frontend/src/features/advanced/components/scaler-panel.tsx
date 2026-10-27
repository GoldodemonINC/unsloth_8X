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
  type ScalerDirection,
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
  const [direction, setDirection] = useState<ScalerDirection>("grow");
  const [method, setMethod] = useState<ScalerMethod>("depth");
  // The growth slider's quantity: parameters added on top of the checkpoint's own count.
  const [added, setAdded] = useState<number | null>(null);
  // Shrinking speaks in absolute size, because the target is below the base.
  const [target, setTarget] = useState<number | null>(null);
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
    // A run error belongs to the inputs that produced it: any change here makes it
    // stale, and leaving it on screen next to a runnable plan reads as a live failure.
    setRunError(null);
    if (!trimmed) {
      setPlan(null);
      setPlanError(null);
      setPlanning(false);
      return;
    }
    let cancelled = false;
    setPlanning(true);
    const timer = setTimeout(() => {
      planScaler(
        direction === "shrink"
          ? {
              source: trimmed,
              method,
              direction,
              target_params: target,
            }
          : { source: trimmed, method, direction, added_params: added },
      )
        .then((result) => {
          if (cancelled) return;
          setPlan(result);
          setPlanError(null);
          if (result.direction === "shrink") {
            setTarget((current) => {
              if (current == null) return result.default_target_params;
              if (current < result.min_params) return result.min_params;
              if (current > result.max_params) return result.max_params;
              return current;
            });
            return;
          }
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
  }, [source, method, added, direction, target]);

  const start = async () => {
    setRunError(null);
    setAdopted(false);
    const shrinkTarget =
      direction === "shrink" ? (target ?? plan?.default_target_params ?? null) : null;
    if (!plan || (direction === "grow" && added == null) || shrinkTarget == null) {
      setRunError("Choose a checkpoint first.");
      return;
    }
    setRunning(true);
    try {
      const created = await runScaler(
        shrinkTarget != null
          ? {
              source: plan.source,
              target_params: shrinkTarget,
              // Run what was planned: after a method change the plan can be one
              // response ahead of the select.
              method: plan.method,
              direction,
            }
          : {
              source: plan.source,
              added_params: added as number,
              method: plan.method,
              direction,
            },
      );
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
  const shrink = plan?.direction === "shrink";
  // Width shrinking removes channels, not layers. Read the axis off the plan, not the
  // select: during the debounce the two can disagree.
  const widthShrink = plan?.method === "mlp_width";
  const removedParams = plan?.projection
    ? (plan.projection.removed_params ??
      Math.max(0, plan.base_params - plan.projection.projected_params))
    : 0;
  const removedChannels = plan?.projection
    ? Math.max(
        0,
        (plan.projection.current_intermediate_size ?? 0) -
          (plan.projection.new_intermediate_size ?? 0),
      )
    : 0;
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
  // Shrink mode's slider quantity is the absolute target, and it steps one block.
  const targetValue = shrink ? (target ?? plan?.default_target_params ?? 0) : 0;
  const paramsPerLayer = plan?.projection?.params_per_layer ?? 0;
  const removedPreview =
    shrink && plan && paramsPerLayer
      ? Math.max(
          0,
          Math.min(
            plan.projection?.max_removable_layers ?? 0,
            Math.round((plan.base_params - targetValue) / paramsPerLayer),
          ),
        )
      : 0;
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
              Change the parameter count before training. Growth starts
              function-preserving and is unlocked by continued pretraining;
              shrinking drops whole blocks and needs that training to recover.
            </p>
          </div>
        </div>
        {plan ? (
          <Badge variant="outline" className="font-mono">
            {params(plan.base_params)} base
          </Badge>
        ) : null}
      </div>

      <div className="flex flex-wrap items-center gap-1.5">
        {(["grow", "shrink"] as const).map((mode) => (
          <Button
            key={mode}
            size="sm"
            variant={direction === mode ? "default" : "outline"}
            disabled={running}
            onClick={() => {
              setDirection(mode);
              // "depth" is the one axis both directions offer, so switching cannot
              // leave the select on a value the new plan does not list.
              setMethod("depth");
              setAdded(null);
              setTarget(null);
              setJob(null);
              setAdopted(false);
            }}
          >
            {mode === "grow" ? "Grow" : "Shrink"}
          </Button>
        ))}
        <span className="text-ui-10p5 leading-ui-14 text-muted-foreground/70">
          {direction === "grow"
            ? "adds capacity; the new blocks start inert"
            : "removes capacity; needs recovery training"}
        </span>
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
                  {shrink ? "Target size" : "Added parameters"}
                </span>
                <span className="font-mono text-ui-13 text-foreground">
                  {shrink ? (
                    <>
                      {params(plan.base_params)} → {params(targetValue)} (
                      {(targetValue / Math.max(1, plan.base_params)).toFixed(2)}x)
                      <span className="ml-2 font-normal text-muted-foreground">
                        {widthShrink
                          ? `−${removedChannels} channels`
                          : `−${removedPreview} blocks`}
                      </span>
                    </>
                  ) : (
                    <>
                      +{params(addedValue)}
                      <span className="ml-2 font-normal text-muted-foreground">
                        {params(plan.base_params)} → {params(targetPreview)} (
                        {growthPreview.toFixed(2)}x)
                      </span>
                    </>
                  )}
                </span>
              </div>

              {/* The refused band only exists when a floor above 1x is configured. */}
              <div className="relative py-1">
                {!shrink && plan.floor_added_params > 0 ? (
                  <>
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
                  </>
                ) : null}
                <div className="relative [&_[data-slot=slider-track]]:bg-transparent">
                  {shrink ? (
                    <Slider
                      value={[targetValue]}
                      min={plan.min_params}
                      max={plan.max_params}
                      step={plan.projection?.params_per_layer ?? plan.step_params}
                      disabled={plan.blocked || running}
                      onValueChange={([value]) => setTarget(value)}
                      thumbValueText={(value) =>
                        `${params(value)}, ${(
                          value / Math.max(1, plan.base_params)
                        ).toFixed(2)}x the base`
                      }
                      aria-label="Target parameters"
                    />
                  ) : (
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
                  )}
                </div>
                <div className="mt-1 flex items-center justify-between text-ui-10p5 text-muted-foreground/70">
                  <span className="font-mono">
                    {shrink ? params(plan.min_params) : "+0"}
                  </span>
                  <span className="font-mono">
                    {shrink
                      ? params(plan.max_params)
                      : `+${params(plan.max_added_params)}`}
                  </span>
                </div>
              </div>
              <p className="text-ui-10p5 leading-ui-14 text-muted-foreground/70">
                {shrink
                  ? `Drops whole transformer blocks. The first and last are protected, and at least ${Math.round(
                      (plan.shrink_floor ?? 0.5) * 100,
                    )}% of the depth survives; the output needs recovery training.`
                  : `Slider adds 0 to ${params(
                      plan.max_added_params,
                    )} parameters. Any growth above the base is allowed; below 2x the literature has no measured gain, so a small step is an experiment.`}
              </p>
            </div>

            <label className="flex flex-col gap-1.5">
              <span className="text-ui-11 font-medium uppercase tracking-[0.05em] text-muted-foreground/70">
                {shrink ? "Axis" : "Growth axis"}
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
                {shrink
                  ? "Blocks or feed-forward width. Neither preserves the function, so both report the drift they measured and hand off to recovery training."
                  : "Depth is the best-evidenced axis; width only widens the feed-forward, so every other shape stays loadable."}
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
                  {shrink ? "Removed" : "Added"}
                </p>
                <p className="font-mono text-ui-14 text-foreground">
                  {shrink
                    ? `−${params(removedParams)}`
                    : `+${params(plan.projection.added_params ?? 0)}`}
                </p>
              </div>
              <div>
                <p className="text-ui-10p5 uppercase tracking-[0.06em] text-muted-foreground/70">
                  {widthShrink ? "Params / channel" : "Layers"}
                </p>
                <p className="font-mono text-ui-14 text-foreground">
                  {widthShrink
                    ? params(plan.projection.params_per_channel)
                    : `${plan.projection.current_num_layers} → ${plan.projection.new_num_layers}`}
                </p>
              </div>
              <div>
                <p className="text-ui-10p5 uppercase tracking-[0.06em] text-muted-foreground/70">
                  {shrink && !widthShrink ? "Params / block" : "MLP width"}
                </p>
                <p className="font-mono text-ui-14 text-foreground">
                  {shrink && !widthShrink
                    ? params(plan.projection.params_per_layer)
                    : `${plan.projection.current_intermediate_size} → ${plan.projection.new_intermediate_size}`}
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
              disabled={running || planning || !plan.runnable}
            >
              {running ? (
                <Spinner className="size-3.5" />
              ) : (
                <HugeiconsIcon icon={PlayIcon} className="size-3.5" />
              )}
              {shrink ? "Shrink checkpoint" : "Expand checkpoint"}
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
                {result.recovery_required
                  ? `Shrunk to ${params(result.final_params)} (${(
                      result.shrink_factor ?? result.growth_factor
                    ).toFixed(2)}x)`
                  : `Expanded to ${params(result.final_params)} (${result.growth_factor.toFixed(2)}x)`}
              </p>
              <p className="font-mono break-all text-muted-foreground">
                {result.output_dir}
              </p>
              <p className="text-muted-foreground">
                {result.recovery_required
                  ? `${result.removed_layers?.length ?? 0} blocks dropped (${
                      result.removed_layers?.join(", ") || "—"
                    }), ranked by ${
                      result.scored_by === "block_influence"
                        ? "measured block influence"
                        : "output-projection weight"
                    }. Drift from the base: max |delta logit| ${result.step0_max_logit_diff.toExponential(
                      1,
                    )}.`
                  : `Function preserved at step 0: max |delta logit| ${result.step0_max_logit_diff.toExponential(
                      1,
                    )}`}
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
              {result.recovery_required ? (
                <p className="text-amber-700 dark:text-amber-400">
                  Not a drop-in replacement: run continued pretraining or a
                  fine-tune on this checkpoint before using it.
                </p>
              ) : null}
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
