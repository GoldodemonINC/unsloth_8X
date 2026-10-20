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
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";
import { HardDriveIcon, PlayIcon, Refresh01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useState } from "react";
import {
  planSplit,
  runSplit,
  waitForJob,
  type AdvancedJob,
  type SplitPlanResponse,
} from "../api/advanced-api";

function gib(bytes: number | null | undefined): string {
  if (bytes == null || !Number.isFinite(bytes)) return "—";
  return `${(bytes / 1024 ** 3).toFixed(1)} GiB`;
}

export function CheckpointSplitterCard() {
  const [source, setSource] = useState("");
  // A bare name resolves under the backend's own outputs folder, so the default works on
  // Windows and on a Colab runtime; an absolute path is still taken as given.
  const [outputDir, setOutputDir] = useState("sharded");
  const [enabled, setEnabled] = useState(false);
  const [prefer, setPrefer] = useState<"ram" | "vram" | "auto">("auto");
  const [reserve, setReserve] = useState(0.15);
  const [maxShardGb, setMaxShardGb] = useState(4);
  const [plan, setPlan] = useState<SplitPlanResponse | null>(null);
  const [planError, setPlanError] = useState<string | null>(null);
  const [planning, setPlanning] = useState(false);
  const [job, setJob] = useState<AdvancedJob | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  const inspect = async () => {
    setPlanning(true);
    setPlanError(null);
    try {
      const result = await planSplit({
        source: source.trim(),
        max_shard_gb: maxShardGb,
        reserve_fraction: reserve,
        prefer,
      });
      setPlan(result);
    } catch (error) {
      setPlan(null);
      setPlanError(error instanceof Error ? error.message : String(error));
    } finally {
      setPlanning(false);
    }
  };

  const start = async () => {
    setRunError(null);
    if (!outputDir.trim()) {
      setRunError("Choose an output folder.");
      return;
    }
    setRunning(true);
    try {
      const created = await runSplit({
        source: source.trim(),
        output_dir: outputDir.trim(),
        max_shard_gb: maxShardGb,
        reserve_fraction: reserve,
        prefer,
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
    <section className="elevated-card mt-6 flex flex-col gap-4 p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex items-start gap-3">
          <span className="mt-0.5 flex size-9 items-center justify-center rounded-full bg-muted/60">
            <HugeiconsIcon icon={HardDriveIcon} className="size-4" />
          </span>
          <div>
            <h2 className="text-ui-13p5 font-semibold text-foreground">
              VRAM Fit
            </h2>
            <p className="text-ui-11 leading-ui-15 text-muted-foreground/85">
              When a checkpoint does not fit your card, shard it. Streaming splits a
              safetensors model into pieces sized to live free VRAM / RAM, so a 70B
              export or a merged expert stack still loads.
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-ui-11 text-muted-foreground">Smart safeguard</span>
          <Switch
            size="sm"
            checked={enabled}
            onCheckedChange={setEnabled}
            aria-label="Enable VRAM Fit"
          />
        </div>
      </div>

      {enabled ? (
        <>
          <div className="grid gap-3 sm:grid-cols-2">
            <label className="flex flex-col gap-1.5">
              <span className="text-ui-11 font-medium uppercase tracking-[0.05em] text-muted-foreground/70">
                Source checkpoint
              </span>
              <Input
                value={source}
                onChange={(event) => setSource(event.target.value)}
                placeholder="path/to/merged-output"
                className="font-mono text-ui-12"
              />
            </label>
            <label className="flex flex-col gap-1.5">
              <span className="text-ui-11 font-medium uppercase tracking-[0.05em] text-muted-foreground/70">
                Split into
              </span>
              <Input
                value={outputDir}
                onChange={(event) => setOutputDir(event.target.value)}
                placeholder="sharded"
                className="font-mono text-ui-12"
              />
              <span className="text-ui-11 leading-ui-15 text-muted-foreground/70">
                A name lands in Unsloth&apos;s outputs folder; an absolute path is
                used as-is.
              </span>
            </label>
          </div>

          <div className="grid gap-4 sm:grid-cols-3">
            <label className="flex flex-col gap-1.5">
              <span className="text-ui-11 font-medium uppercase tracking-[0.05em] text-muted-foreground/70">
                Memory tier
              </span>
              <Select
                value={prefer}
                onValueChange={(value) =>
                  setPrefer(value as "ram" | "vram" | "auto")
                }
              >
                <SelectTrigger className="h-9">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="auto">Auto (largest free tier)</SelectItem>
                  <SelectItem value="ram">System RAM</SelectItem>
                  <SelectItem value="vram">GPU VRAM</SelectItem>
                </SelectContent>
              </Select>
            </label>
            <label className="flex flex-col gap-1.5">
              <span className="text-ui-11 font-medium uppercase tracking-[0.05em] text-muted-foreground/70">
                Shard cap (GiB)
              </span>
              <Input
                type="number"
                min={0.5}
                max={1024}
                step={0.5}
                value={maxShardGb}
                onChange={(event) =>
                  setMaxShardGb(Math.max(0.5, Number(event.target.value) || 4))
                }
                className="h-9 w-28"
              />
            </label>
            <label className="flex flex-col gap-1.5">
              <span className="text-ui-11 font-medium uppercase tracking-[0.05em] text-muted-foreground/70">
                Headroom reserve
              </span>
              <div className="flex h-9 items-center gap-3">
                <Slider
                  value={[reserve]}
                  min={0}
                  max={0.6}
                  step={0.05}
                  onValueChange={([value]) => setReserve(value)}
                  aria-label="Headroom reserve fraction"
                />
                <span className="w-10 text-right font-mono text-ui-11">
                  {(reserve * 100).toFixed(0)}%
                </span>
              </div>
            </label>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <Button
              variant="outline"
              size="sm"
              onClick={() => void inspect()}
              disabled={planning || !source.trim()}
            >
              {planning ? <Spinner className="size-3.5" /> : null}
              Profile &amp; plan
            </Button>
            <Button
              size="sm"
              onClick={() => void start()}
              disabled={running || !plan || !outputDir.trim()}
            >
              {running ? (
                <Spinner className="size-3.5" />
              ) : (
                <HugeiconsIcon icon={PlayIcon} className="size-3.5" />
              )}
              Split checkpoint
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
              </span>
            ) : null}
            {planError ? (
              <span className="text-ui-11 text-destructive">{planError}</span>
            ) : null}
            {runError ? (
              <span className="text-ui-11 text-destructive">{runError}</span>
            ) : null}
          </div>

          {plan ? (
            <div className="flex flex-col gap-3 rounded-2xl border border-border/60 p-4">
              <div className="grid gap-3 sm:grid-cols-3">
                <div>
                  <p className="text-ui-10p5 uppercase tracking-[0.06em] text-muted-foreground/70">
                    Shards
                  </p>
                  <p className="font-mono text-ui-14 text-foreground">
                    {plan.plan.shard_count}
                  </p>
                </div>
                <div>
                  <p className="text-ui-10p5 uppercase tracking-[0.06em] text-muted-foreground/70">
                    Shard cap
                  </p>
                  <p className="font-mono text-ui-14 text-foreground">
                    {gib(plan.plan.max_shard_bytes)}
                  </p>
                </div>
                <div>
                  <p className="text-ui-10p5 uppercase tracking-[0.06em] text-muted-foreground/70">
                    Checkpoint
                  </p>
                  <p className="font-mono text-ui-14 text-foreground">
                    {gib(plan.plan.total_bytes)}
                  </p>
                </div>
              </div>
              <div className="flex flex-wrap gap-1.5">
                <Badge variant="outline">
                  RAM free {gib(plan.memory_profile.system_ram_free_bytes)}
                </Badge>
                <Badge variant="outline">
                  VRAM free{" "}
                  {gib(
                    plan.memory_profile.gpus.reduce(
                      (sum, gpu) => sum + gpu.free_bytes,
                      0,
                    ),
                  )}
                </Badge>
                <Badge
                  variant={plan.fits_on_disk ? "secondary" : "destructive"}
                >
                  {plan.fits_on_disk ? "Fits on disk" : "Disk too small"}
                </Badge>
              </div>
              {plan.plan.warnings.length > 0 ? (
                <ul
                  className={cn(
                    "flex list-disc flex-col gap-1 pl-5 text-ui-11 leading-ui-16",
                    "text-amber-600 dark:text-amber-400",
                  )}
                >
                  {plan.plan.warnings.map((warning) => (
                    <li key={warning}>{warning}</li>
                  ))}
                </ul>
              ) : null}
            </div>
          ) : null}

          {job?.status === "completed" && job.result ? (
            <div className="rounded-2xl border border-emerald-500/30 bg-emerald-500/5 p-4 text-ui-11 leading-ui-16">
              <p className="font-medium text-foreground">Split complete</p>
              <p className="mt-1 text-muted-foreground">
                {String(
                  job.output_dir ??
                    (job.result as { output_dir?: string }).output_dir ??
                    outputDir,
                )}{" "}
                ·{" "}
                {(() => {
                  const files = (job.result as { files?: string[] }).files;
                  return files ? `${files.length} shards written` : "done";
                })()}
              </p>
            </div>
          ) : null}
          {job?.status === "failed" ? (
            <p className="text-ui-11 text-destructive">{job.error}</p>
          ) : null}
        </>
      ) : null}
    </section>
  );
}
