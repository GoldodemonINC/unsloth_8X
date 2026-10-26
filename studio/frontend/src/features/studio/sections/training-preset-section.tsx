// SPDX-License-Identifier: NRC-2
// Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-2.md

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import {
  type TrainingFitReport,
  fetchTrainingFit,
} from "@/features/training/api/fit-api";
import {
  BUILTIN_PRESETS,
  type StoredTrainingPreset,
  applyPreset,
  capturePresetValues,
  loadStoredPresets,
  writeStoredPresets,
} from "@/features/training/lib/training-presets";
import { useTrainingConfigStore } from "@/features/training";
import { useInferenceGpuInfo } from "@/hooks/use-gpu-info";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  Alert02Icon,
  DashboardSpeed01Icon,
  Delete02Icon,
  FloppyDiskIcon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useEffect, useState } from "react";

function verdictTone(verdict: TrainingFitReport["verdict"]): string {
  if (verdict === "fits") {
    return "border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300";
  }
  if (verdict === "tight") {
    return "border-amber-500/30 bg-amber-500/10 text-amber-700 dark:text-amber-300";
  }
  if (verdict === "over") {
    return "border-destructive/30 bg-destructive/10 text-destructive";
  }
  return "border-border bg-muted/40 text-muted-foreground";
}

export function TrainingPresetSection(): ReactElement {
  const t = useT();
  const gpu = useInferenceGpuInfo();
  const selectedModel = useTrainingConfigStore((state) => state.selectedModel);
  const trainingMethod = useTrainingConfigStore((state) => state.trainingMethod);
  const contextLength = useTrainingConfigStore((state) => state.contextLength);
  const batchSize = useTrainingConfigStore((state) => state.batchSize);
  const gradientAccumulation = useTrainingConfigStore(
    (state) => state.gradientAccumulation,
  );
  const loraRank = useTrainingConfigStore((state) => state.loraRank);
  const gradientCheckpointing = useTrainingConfigStore(
    (state) => state.gradientCheckpointing,
  );

  const [custom, setCustom] = useState<StoredTrainingPreset[]>([]);
  const [saving, setSaving] = useState(false);
  const [name, setName] = useState("");
  const [fit, setFit] = useState<TrainingFitReport | null>(null);
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setCustom(loadStoredPresets());
  }, []);

  const saveCurrent = () => {
    const cleaned = name.trim();
    if (!cleaned) return;
    const preset: StoredTrainingPreset = {
      id: crypto.randomUUID(),
      name: cleaned.slice(0, 60),
      description: "",
      values: capturePresetValues(),
      createdAt: new Date().toISOString(),
    };
    const next = [preset, ...custom].slice(0, 50);
    setCustom(next);
    writeStoredPresets(next);
    setName("");
    setSaving(false);
    toast.success(t("studio.presets.saved"), { description: preset.name });
  };

  const remove = (id: string) => {
    const next = custom.filter((preset) => preset.id !== id);
    setCustom(next);
    writeStoredPresets(next);
  };

  const use = (preset: StoredTrainingPreset | (typeof BUILTIN_PRESETS)[number]) => {
    applyPreset(preset);
    setFit(null);
    toast.success(t("studio.presets.applied"), { description: preset.name });
  };

  const check = async () => {
    if (!selectedModel) {
      setError(t("studio.presets.needModel"));
      return;
    }
    setChecking(true);
    setError(null);
    try {
      const report = await fetchTrainingFit({
        model: selectedModel,
        method: trainingMethod,
        loadIn4Bit: trainingMethod === "qlora",
        maxSeqLength: contextLength,
        batchSize,
        gradientAccumulation,
        loraRank,
        gradientCheckpointing,
        gpuMemoryGb: gpu.available ? gpu.memoryTotalGb : null,
      });
      setFit(report);
    } catch (failure) {
      setError(
        failure instanceof Error
          ? failure.message
          : t("studio.presets.checkFailed"),
      );
    } finally {
      setChecking(false);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-ui-11 font-medium text-foreground">
            {t("studio.presets.builtin")}
          </span>
          <span className="text-ui-10 text-muted-foreground/70">
            {t("studio.presets.builtinHint")}
          </span>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {BUILTIN_PRESETS.map((preset) => (
            <button
              key={preset.id}
              type="button"
              title={preset.description}
              onClick={() => use(preset)}
              className="rounded-full border border-border/70 bg-background px-2.5 py-1 text-ui-11 text-foreground transition-colors hover:bg-muted/60"
            >
              {preset.name}
            </button>
          ))}
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2 border-t border-border/60 pt-3">
        <span className="text-ui-11 font-medium text-foreground">
          {t("studio.presets.mine")}
        </span>
        {custom.length === 0 ? (
          <span className="text-ui-10 text-muted-foreground/70">
            {t("studio.presets.none")}
          </span>
        ) : (
          custom.map((preset) => (
            <span
              key={preset.id}
              className="flex items-center gap-1 rounded-full border border-border/70 bg-background pl-2.5 pr-1"
            >
              <button
                type="button"
                onClick={() => use(preset)}
                className="py-1 text-ui-11 text-foreground"
              >
                {preset.name}
              </button>
              <Button
                size="icon-xs"
                variant="ghost"
                aria-label={t("studio.presets.delete")}
                onClick={() => remove(preset.id)}
              >
                <HugeiconsIcon icon={Delete02Icon} className="size-3" />
              </Button>
            </span>
          ))
        )}
        {saving ? (
          <span className="flex items-center gap-1">
            <input
              autoFocus
              value={name}
              onChange={(event) => setName(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") saveCurrent();
                if (event.key === "Escape") setSaving(false);
              }}
              placeholder={t("studio.presets.namePlaceholder")}
              className="h-7 w-44 rounded-lg border border-border bg-background px-2 text-ui-11 outline-none focus-visible:ring-1 focus-visible:ring-ring"
            />
            <Button
              size="sm"
              variant="outline"
              onClick={saveCurrent}
              disabled={!name.trim()}
            >
              {t("studio.presets.save")}
            </Button>
          </span>
        ) : (
          <Button
            size="sm"
            variant="outline"
            onClick={() => setSaving(true)}
          >
            <HugeiconsIcon icon={FloppyDiskIcon} className="size-3.5" />
            {t("studio.presets.saveCurrent")}
          </Button>
        )}
      </div>

      <div className="rounded-lg border border-border/70 bg-background/40 p-3">
        <div className="flex flex-wrap items-center gap-2">
          <span className="flex items-center gap-1.5 text-ui-11 font-medium text-foreground">
            <HugeiconsIcon icon={DashboardSpeed01Icon} className="size-3.5" />
            {t("studio.presets.fitTitle")}
          </span>
          <Button
            size="sm"
            variant="outline"
            onClick={() => void check()}
            disabled={checking}
          >
            {checking ? (
              <Spinner />
            ) : (
              <HugeiconsIcon icon={DashboardSpeed01Icon} className="size-3.5" />
            )}
            {checking ? t("studio.presets.checking") : t("studio.presets.check")}
          </Button>
          {fit ? (
            <Badge
              variant="outline"
              className={cn("ml-auto capitalize", verdictTone(fit.verdict))}
            >
              {fit.verdict}
            </Badge>
          ) : null}
        </div>
        <p className="mt-1 text-ui-10 leading-relaxed text-muted-foreground/70">
          {t("studio.presets.fitHint")}
        </p>

        {error ? (
          <p className="mt-2 text-ui-11 text-destructive">{error}</p>
        ) : null}

        {fit ? (
          <div className="mt-3 flex flex-col gap-2">
            <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1 text-ui-11">
              <span className="font-mono text-foreground">
                {fit.required_gb.toFixed(1)} GB
              </span>
              <span className="text-ui-10 text-muted-foreground">
                {t("studio.presets.required")}
              </span>
              {fit.budget_gb ? (
                <span className="text-ui-10 text-muted-foreground">
                  {t("studio.presets.budget", {
                    gb: fit.budget_gb.toFixed(1),
                  })}
                </span>
              ) : null}
              {fit.headroom_gb !== null ? (
                <span className="text-ui-10 text-muted-foreground">
                  {t("studio.presets.headroom", {
                    gb: fit.headroom_gb.toFixed(1),
                  })}
                </span>
              ) : null}
              <span className="text-ui-10 text-muted-foreground/70">
                {fit.params_label} · {fit.trainable_params.toLocaleString()}{" "}
                {t("studio.presets.trainable")}
              </span>
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-muted">
              <div
                className={cn(
                  "h-full rounded-full",
                  fit.verdict === "fits"
                    ? "bg-emerald-500"
                    : fit.verdict === "tight"
                      ? "bg-amber-500"
                      : "bg-destructive",
                )}
                style={{
                  width: `${Math.min(
                    100,
                    fit.budget_gb
                      ? (fit.required_gb / fit.budget_gb) * 100
                      : 100,
                  )}%`,
                }}
              />
            </div>
            {fit.suggestions.length > 0 ? (
              <ul className="flex flex-col gap-1 rounded-lg border border-border/60 bg-muted/20 p-2.5 text-ui-10 text-muted-foreground">
                {fit.suggestions.map((suggestion) => (
                  <li key={suggestion} className="flex gap-1.5">
                    {fit.verdict === "fits" ? (
                      <HugeiconsIcon
                        icon={Tick02Icon}
                        className="mt-0.5 size-3 shrink-0"
                      />
                    ) : (
                      <HugeiconsIcon
                        icon={Alert02Icon}
                        className="mt-0.5 size-3 shrink-0"
                      />
                    )}
                    <span>{suggestion}</span>
                  </li>
                ))}
              </ul>
            ) : null}
            <details className="text-ui-10 text-muted-foreground/80">
              <summary className="cursor-pointer">
                {t("studio.presets.assumptions")}
              </summary>
              <ul className="mt-1 flex list-disc flex-col gap-0.5 pl-4">
                {fit.assumptions.map((assumption) => (
                  <li key={assumption}>{assumption}</li>
                ))}
              </ul>
            </details>
          </div>
        ) : null}
      </div>
    </div>
  );
}
