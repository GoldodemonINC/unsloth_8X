// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Spinner } from "@/components/ui/spinner";
import { planLayerLr } from "@/features/advanced/api/advanced-api";
import type { ReactElement } from "react";
import { useEffect, useMemo, useState } from "react";

/** The layer-LR block a run was started with (the request shape, not the UI state). */
export type AppliedLayerLr = {
  enabled?: boolean;
  zone_multipliers?: Record<string, number>;
  kind_multipliers?: Record<string, number>;
  layer_multipliers?: Record<string, number>;
  depth_curve?: [number, number] | null;
};

type Plan = Awaited<ReturnType<typeof planLayerLr>>;

type Row = {
  key: string;
  label: string;
  detail: string;
  lr: number;
  params: number | null;
};

type LayerRow = Row & { range?: string };

const ZONE_LABELS: Record<string, string> = {
  embedding: "Embeddings",
  early: "Early layers",
  middle: "Middle layers",
  late: "Late layers",
  head: "Output head",
  adapter: "Adapters",
};

const KIND_LABELS: Record<string, string> = {
  attention: "Attention",
  mlp: "MLP / FFN",
  norm: "Layer norms",
  moe_expert: "MoE experts",
  moe_router: "MoE routers",
  hybrid_block: "Mamba / hybrid",
  other: "Other",
};

function formatLr(value: number): string {
  if (!Number.isFinite(value)) return "—";
  if (value === 0) return "frozen";
  return value.toExponential(2);
}

/**
 * The planned rate for a group, and -- once the run is stepping -- the rate the
 * scheduler is actually at for it. A frozen group never gets a "now": it has no
 * rate to report at any step.
 */
function formatCount(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return "0";
  if (value >= 1e9) return `${(value / 1e9).toFixed(1)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}k`;
  return String(Math.round(value));
}

function LrCell({ planned, live }: { planned: number; live: number | null }) {
  return (
    <span className="text-right font-mono text-ui-10p5 text-foreground">
      {formatLr(planned)}
      {live !== null && planned !== 0 ? (
        <span className="block text-ui-10 text-muted-foreground/70">
          {formatLr(live)} now
        </span>
      ) : null}
    </span>
  );
}

/**
 * The learning-rate plan a Layer-LR run is training with.
 *
 * The rows come from the same planner the trainer uses, so what is shown is the
 * zone / module-group table the optimizer was built from, not a re-derivation in
 * the browser. When the run's checkpoint is local the planner also returns every
 * parameter, which is what makes the per-layer rows possible.
 */
export function LayerLrPanel({
  baseLearningRate,
  layerLr,
  model,
  liveLearningRate,
}: {
  baseLearningRate: number;
  layerLr: AppliedLayerLr | null | undefined;
  model?: string | null;
  liveLearningRate?: number | null;
}): ReactElement | null {
  const enabled = layerLr?.enabled === true;
  const [plan, setPlan] = useState<Plan | null>(null);
  const [failed, setFailed] = useState(false);

  const requestKey = useMemo(
    () =>
      enabled
        ? JSON.stringify({
            base: baseLearningRate,
            zones: layerLr?.zone_multipliers ?? {},
            kinds: layerLr?.kind_multipliers ?? {},
            layers: layerLr?.layer_multipliers ?? {},
            curve: layerLr?.depth_curve ?? null,
            model: model ?? null,
          })
        : null,
    [enabled, baseLearningRate, layerLr, model],
  );

  useEffect(() => {
    if (!requestKey || !(baseLearningRate > 0)) {
      setPlan(null);
      return;
    }
    const controller = new AbortController();
    let cancelled = false;
    setFailed(false);
    planLayerLr({
      base_learning_rate: baseLearningRate,
      zone_multipliers: layerLr?.zone_multipliers,
      kind_multipliers: layerLr?.kind_multipliers,
      layer_multipliers: layerLr?.layer_multipliers,
      depth_curve: layerLr?.depth_curve ?? null,
      model: model ?? null,
    })
      .then((result) => {
        if (!cancelled && !controller.signal.aborted) setPlan(result);
      })
      .catch(() => {
        if (!cancelled && !controller.signal.aborted) {
          setPlan(null);
          setFailed(true);
        }
      });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [requestKey, baseLearningRate, layerLr, model]);

  const rows = useMemo<Row[]>(() => {
    if (!plan) return [];
    const entries = plan.preview?.entries ?? [];
    // One row per optimizer group: same zone, same module kind, same rate. The
    // parameter count is summed so the row says how much of the model it covers.
    const groups = new Map<string, Row>();
    for (const entry of entries) {
      const key = `${entry.zone}|${entry.kind}|${entry.lr.toFixed(12)}`;
      const existing = groups.get(key);
      if (existing) {
        existing.params = (existing.params ?? 0) + entry.numel;
        continue;
      }
      groups.set(key, {
        key,
        label: ZONE_LABELS[entry.zone] ?? entry.zone,
        detail:
          entry.kind === "other"
            ? "all weights"
            : (KIND_LABELS[entry.kind] ?? entry.kind),
        lr: entry.lr,
        params: entry.numel,
      });
    }
    if (groups.size > 0) return [...groups.values()];
    // No parameter names (a Hub id, or the plan is still resolving): report the
    // zone table the multipliers describe.
    const multipliers = plan.preview?.multipliers ?? {};
    return Object.entries(multipliers)
      .filter(([, value]) => value !== 1)
      .map(([zone, value]) => ({
        key: zone,
        label: ZONE_LABELS[zone] ?? zone,
        detail: "zone × base",
        lr: baseLearningRate * value,
        params: null,
      }));
  }, [plan, baseLearningRate]);

  const perLayer = useMemo<LayerRow[]>(() => {
    const entries = plan?.preview?.entries ?? [];
    if (entries.length === 0) return [];
    const hasPerLayer =
      layerLr?.depth_curve != null ||
      Object.keys(layerLr?.layer_multipliers ?? {}).length > 0;
    if (!hasPerLayer) return [];
    const byLayer = new Map<number, { zone: string; lrs: Set<number> }>();
    for (const entry of entries) {
      if (entry.layer == null) continue;
      const bucket = byLayer.get(entry.layer) ?? {
        zone: entry.zone,
        lrs: new Set<number>(),
      };
      bucket.lrs.add(entry.lr);
      byLayer.set(entry.layer, bucket);
    }
    return [...byLayer.entries()]
      .sort(([left], [right]) => left - right)
      .map(([layer, bucket]) => {
        const values = [...bucket.lrs].sort((a, b) => a - b);
        const low = values[0];
        const high = values[values.length - 1];
        return {
          key: `layer-${layer}`,
          label: `Layer ${layer}`,
          detail: ZONE_LABELS[bucket.zone] ?? bucket.zone,
          lr: high,
          params: null,
          range:
            values.length > 1
              ? `${formatLr(low)} – ${formatLr(high)}`
              : undefined,
        };
      });
  }, [plan, layerLr]);

  if (!enabled) return null;

  const kinds = Object.entries(layerLr?.kind_multipliers ?? {}).filter(
    ([, value]) => value !== 1,
  );
  // The scheduler scales every group by the same factor, so the base's ratio to
  // the rate in force turns a planned group rate into the one training with now.
  // A run that has not stepped yet reports 0, which is not "everything is
  // frozen" -- it is "no reading yet", so there is no live column to show.
  const liveRate =
    typeof liveLearningRate === "number" && liveLearningRate > 0
      ? liveLearningRate
      : null;
  const liveFactor =
    liveRate !== null && baseLearningRate > 0
      ? liveRate / baseLearningRate
      : null;
  const liveFor = (planned: number): number | null =>
    liveFactor === null ? null : planned * liveFactor;

  return (
    <section className="elevated-card flex flex-col gap-3 p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <h2 className="text-ui-12p5 font-semibold text-foreground">
            Learning rates in force
          </h2>
          <Badge variant="secondary">Layer LR</Badge>
        </div>
        <div className="flex items-center gap-2 text-ui-10p5 text-muted-foreground">
          {!plan && !failed ? <Spinner className="size-3" /> : null}
          <span>
            base{" "}
            <span className="font-mono text-foreground">
              {formatLr(baseLearningRate)}
            </span>
          </span>
          {liveRate !== null ? (
            <span>
              · scheduler now{" "}
              <span className="font-mono text-foreground">
                {formatLr(liveRate)}
              </span>
            </span>
          ) : null}
        </div>
      </div>

      {rows.length > 0 ? (
        <div className="grid gap-1.5">
          {rows.map((row) => (
            <div
              key={row.key}
              className="grid grid-cols-[minmax(96px,0.9fr)_minmax(0,1.2fr)_auto] items-center gap-3 rounded-lg bg-muted/25 px-2.5 py-1.5"
            >
              <span className="text-ui-11 text-foreground">{row.label}</span>
              <span className="truncate text-ui-10p5 text-muted-foreground">
                {row.detail}
                {row.params != null
                  ? ` · ${formatCount(row.params)} params`
                  : ""}
              </span>
              <LrCell planned={row.lr} live={liveFor(row.lr)} />
            </div>
          ))}
        </div>
      ) : null}

      {perLayer.length > 0 ? (
        <div className="flex max-h-56 flex-col gap-1 overflow-y-auto rounded-lg border border-border/60 p-2">
          <p className="px-1 text-ui-10p5 font-medium uppercase tracking-[0.05em] text-muted-foreground/70">
            Per layer
          </p>
          {perLayer.map((row) => (
            <div
              key={row.key}
              className="grid grid-cols-[minmax(72px,0.6fr)_minmax(0,1fr)_auto] items-center gap-3 px-1 py-0.5"
            >
              <span className="text-ui-10p5 text-foreground">{row.label}</span>
              <span className="truncate text-ui-10p5 text-muted-foreground">
                {row.range ?? row.detail}
              </span>
              <LrCell planned={row.lr} live={liveFor(row.lr)} />
            </div>
          ))}
        </div>
      ) : null}

      <p className="text-ui-10p5 leading-ui-14 text-muted-foreground/70">
        {failed
          ? "The layer-LR plan could not be read for this run; the trainer still applies it."
          : rows.length === 0
            ? "Waiting for the run's parameter names; the zone multipliers are applied regardless."
            : "These are the optimizer groups the trainer built: zone and module kind multiplied together, then decayed by the scheduler. The second line is what that group is training at right now."}
        {kinds.length > 0
          ? ` Module groups: ${kinds
              .map(([kind, value]) => `${KIND_LABELS[kind] ?? kind} ×${value}`)
              .join(", ")}.`
          : ""}
      </p>
    </section>
  );
}
