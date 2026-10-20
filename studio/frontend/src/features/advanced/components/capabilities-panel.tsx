// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";
import { HardDriveIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useState, type ReactElement } from "react";
import { fetchAdvancedCapabilities, type AdvancedCapabilities } from "../api/advanced-api";

/**
 * What this machine can actually do, feature by feature.
 *
 * The honest version of a feature list: merging, sharding and structural checks
 * run on a CPU, and the panel says so; training, RL, QAT, MMoE and benchmarks
 * need a GPU. Saying it once here means no other screen has to hedge.
 */
export function CapabilitiesPanel(): ReactElement | null {
  const [caps, setCaps] = useState<AdvancedCapabilities | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetchAdvancedCapabilities()
      .then((result) => {
        if (!cancelled) setCaps(result);
      })
      .catch(() => {
        if (!cancelled) setFailed(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (failed) return null;
  if (!caps) {
    return (
      <section className="elevated-card mt-6 flex items-center gap-2 p-5 text-ui-11 text-muted-foreground">
        <Spinner className="size-3.5" /> Reading this machine&apos;s capabilities
      </section>
    );
  }

  const features = caps.features ?? [];
  const gpuFree = (caps.memory.gpus[0]?.free_bytes ?? 0) / 1024 ** 3;
  const ramFree = caps.memory.system_ram_free_bytes / 1024 ** 3;

  return (
    <section className="elevated-card mt-6 flex flex-col gap-4 p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex items-start gap-3">
          <span className="mt-0.5 flex size-9 items-center justify-center rounded-full bg-muted/60">
            <HugeiconsIcon icon={HardDriveIcon} className="size-4" />
          </span>
          <div>
            <h2 className="text-ui-13p5 font-semibold text-foreground">
              What this machine can run
            </h2>
            <p className="text-ui-11 leading-ui-15 text-muted-foreground/85">
              {caps.platform_line ??
                "Merging, sharding and structural validation run on a CPU; training, RL, QAT, MMoE and benchmarks need a GPU."}
            </p>
          </div>
        </div>
        <div className="flex flex-wrap gap-1.5">
          <Badge variant={caps.has_gpu ? "secondary" : "outline"}>
            {caps.has_gpu ? `GPU · ${gpuFree.toFixed(1)} GB free` : "CPU only"}
          </Badge>
          <Badge variant="outline">{ramFree.toFixed(1)} GB RAM free</Badge>
        </div>
      </div>

      <ul className="grid gap-1.5 sm:grid-cols-2">
        {features.map((feature) => (
          <li
            key={feature.key}
            className="flex items-start justify-between gap-2 rounded-xl border border-border/50 px-3 py-2"
          >
            <div className="min-w-0">
              <p className="truncate text-ui-11p5 text-foreground">{feature.label}</p>
              <p className="text-ui-10p5 leading-ui-14 text-muted-foreground/80">
                {feature.note}
              </p>
            </div>
            <span
              className={cn(
                "shrink-0 rounded-full px-2 py-0.5 text-ui-10",
                feature.available_here
                  ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
                  : "bg-muted text-muted-foreground",
              )}
            >
              {feature.available_here ? "available" : "needs GPU"}
            </span>
          </li>
        ))}
      </ul>

      {caps.merge_cpu_note ? (
        <p className="text-ui-10p5 leading-ui-14 text-muted-foreground/70">
          {caps.merge_cpu_note}
        </p>
      ) : null}
    </section>
  );
}
