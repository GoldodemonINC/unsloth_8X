// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SectionCard } from "@/components/section-card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { getHfToken } from "@/features/hub";
import { buildTrainingStartPayload } from "@/features/training/api/mappers";
import {
  SWEEP_MAX_ITEMS,
  type SweepAxes,
  type TrainingQueueItem,
  type TrainingQueueStatus,
  createQueueItemId,
  expandSweep,
  isScheduledQueueItem,
  parseSweepNumbers,
  sweepCombinationCount,
} from "@/features/training/lib/training-queue";
import { useTrainingConfigStore } from "@/features/training/stores/training-config-store";
import { useTrainingQueueStore } from "@/features/training/stores/training-queue-store";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  Delete02Icon,
  PlayIcon,
  StopIcon,
  TaskDone01Icon,
  ZapIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useMemo, useState } from "react";
import { useShallow } from "zustand/react/shallow";

const SWEEP_INPUT_CLASS =
  "h-8 w-full min-w-0 rounded-lg border border-border bg-background px-2 text-ui-11 text-foreground outline-none focus-visible:ring-1 focus-visible:ring-ring";

const statusClass: Record<TrainingQueueStatus, string> = {
  queued: "border-border bg-muted/40 text-muted-foreground",
  running:
    "border-control-accent/30 bg-control-accent/10 text-control-accent",
  done: "border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300",
  failed: "border-destructive/30 bg-destructive/10 text-destructive",
  cancelled: "border-border bg-muted/20 text-muted-foreground/70",
};

function statusKey(status: TrainingQueueStatus) {
  return `studio.queue.status.${status}` as const;
}

function formatSchedule(iso: string | null): string | null {
  if (!iso) return null;
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return null;
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function TrainingQueueCard(): ReactElement {
  const t = useT();
  const { items, paused, setPaused, remove, clearFinished } =
    useTrainingQueueStore(
      useShallow((state) => ({
        items: state.items,
        paused: state.paused,
        setPaused: state.setPaused,
        remove: state.remove,
        clearFinished: state.clearFinished,
      })),
    );
  const [schedule, setSchedule] = useState("");
  const [learningRates, setLearningRates] = useState("");
  const [loraRanks, setLoraRanks] = useState("");
  const [epochs, setEpochs] = useState("");
  const [batchSizes, setBatchSizes] = useState("");

  const now = Date.now();
  const pendingCount = useMemo(
    () => items.filter((item) => item.status === "queued").length,
    [items],
  );
  const hasFinished = items.some(
    (item) => item.status === "done" || item.status === "cancelled",
  );

  const axes: SweepAxes = useMemo(
    () => ({
      learningRates: parseSweepNumbers(learningRates),
      loraRanks: parseSweepNumbers(loraRanks),
      epochs: parseSweepNumbers(epochs),
      batchSizes: parseSweepNumbers(batchSizes),
    }),
    [learningRates, loraRanks, epochs, batchSizes],
  );
  const sweepCount = sweepCombinationCount(axes);

  const notBefore = schedule ? new Date(schedule).toISOString() : null;

  const queueCurrent = () => {
    const config = useTrainingConfigStore.getState();
    if (!config.selectedModel) {
      toast.error(t("studio.queue.addFailed"));
      return;
    }
    const payload = buildTrainingStartPayload(config, getHfToken(), null);
    const label = [config.selectedModel, config.dataset || config.uploadedFile]
      .filter(Boolean)
      .join(" · ");
    useTrainingQueueStore.getState().enqueue([{ label, config: payload }], {
      notBefore,
    });
    toast.success(t("studio.queue.added", { count: 1 }));
  };

  const queueSweep = () => {
    const config = useTrainingConfigStore.getState();
    if (!config.selectedModel) {
      toast.error(t("studio.queue.addFailed"));
      return;
    }
    const variants = expandSweep(config, axes, getHfToken());
    if (variants.length === 0) {
      toast.error(t("studio.queue.addFailed"));
      return;
    }
    useTrainingQueueStore.getState().enqueue(variants, {
      groupId: createQueueItemId(),
      notBefore,
    });
    toast.success(t("studio.queue.added", { count: variants.length }));
  };

  const startNow = (item: TrainingQueueItem) => {
    useTrainingQueueStore.setState((state) => ({
      items: [
        { ...item, status: "queued", notBefore: null },
        ...state.items.filter((entry) => entry.id !== item.id),
      ],
    }));
  };

  return (
    <SectionCard
      icon={<HugeiconsIcon icon={TaskDone01Icon} className="size-4" />}
      title={t("studio.queue.title")}
      description={t("studio.queue.description")}
      badge={pendingCount > 0 ? String(pendingCount) : undefined}
      headerAction={
        <Button
          type="button"
          size="sm"
          variant="ghost"
          className="gap-1.5"
          onClick={() => setPaused(!paused)}
        >
          <HugeiconsIcon
            icon={paused ? PlayIcon : StopIcon}
            className="size-3.5"
          />
          {paused ? t("studio.queue.resume") : t("studio.queue.pause")}
        </Button>
      }
    >
      <div className="flex flex-wrap items-end gap-2">
        <label className="flex min-w-0 flex-1 flex-col gap-1 text-ui-10 text-muted-foreground">
          {t("studio.queue.scheduleLabel")}
          <input
            type="datetime-local"
            value={schedule}
            onChange={(event) => setSchedule(event.target.value)}
            className={cn(SWEEP_INPUT_CLASS, "max-w-56")}
          />
        </label>
        <Button
          type="button"
          size="sm"
          variant="outline"
          className="gap-1.5"
          onClick={queueCurrent}
        >
          <HugeiconsIcon icon={ZapIcon} className="size-3.5" />
          {t("studio.queue.addCurrent")}
        </Button>
      </div>

      <details className="group rounded-xl border border-border/70 bg-background/40">
        <summary className="cursor-pointer list-none px-3 py-2 text-ui-11 font-medium text-foreground marker:hidden">
          {t("studio.queue.sweepTitle")}
          <span className="ml-2 text-ui-10 font-normal text-muted-foreground">
            {t("studio.queue.sweepCount", { count: sweepCount })}
          </span>
        </summary>
        <div className="grid gap-2 border-t border-border/60 p-3 sm:grid-cols-2 lg:grid-cols-4">
          {[
            {
              label: t("studio.queue.sweepLearningRates"),
              value: learningRates,
              onChange: setLearningRates,
              placeholder: "2e-4, 1e-4",
            },
            {
              label: t("studio.queue.sweepLoraRanks"),
              value: loraRanks,
              onChange: setLoraRanks,
              placeholder: "16, 32",
            },
            {
              label: t("studio.queue.sweepEpochs"),
              value: epochs,
              onChange: setEpochs,
              placeholder: "3",
            },
            {
              label: t("studio.queue.sweepBatchSizes"),
              value: batchSizes,
              onChange: setBatchSizes,
              placeholder: "2, 4",
            },
          ].map((field) => (
            <label
              key={field.label}
              className="flex min-w-0 flex-col gap-1 text-ui-10 text-muted-foreground"
            >
              {field.label}
              <input
                value={field.value}
                onChange={(event) => field.onChange(event.target.value)}
                placeholder={field.placeholder}
                className={SWEEP_INPUT_CLASS}
              />
            </label>
          ))}
          <div className="flex items-center gap-2 sm:col-span-2 lg:col-span-4">
            <p className="min-w-0 flex-1 text-ui-10 text-muted-foreground">
              {sweepCount > SWEEP_MAX_ITEMS
                ? t("studio.queue.sweepLimit", { count: SWEEP_MAX_ITEMS })
                : t("studio.queue.sweepHint")}
            </p>
            <Button
              type="button"
              size="sm"
              variant="outline"
              className="gap-1.5"
              disabled={sweepCount > SWEEP_MAX_ITEMS}
              onClick={queueSweep}
            >
              <HugeiconsIcon icon={ZapIcon} className="size-3.5" />
              {t("studio.queue.addSweep")}
            </Button>
          </div>
        </div>
      </details>

      {items.length === 0 ? (
        <p className="text-ui-11 text-muted-foreground">
          {t("studio.queue.empty")}
        </p>
      ) : (
        <div className="overflow-hidden rounded-xl border border-border/70">
          {items.map((item) => {
            const when = formatSchedule(item.notBefore);
            const scheduled = isScheduledQueueItem(item, now);
            return (
              <div
                key={item.id}
                className="flex flex-wrap items-start gap-2 border-b border-border/50 bg-background/40 px-3 py-2 last:border-b-0"
              >
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-ui-11 font-medium text-foreground">
                      {item.label}
                    </span>
                    <Badge
                      variant="outline"
                      className={cn("text-ui-9", statusClass[item.status])}
                    >
                      {scheduled
                        ? t("studio.queue.status.scheduled")
                        : t(statusKey(item.status))}
                    </Badge>
                    {when ? (
                      <span className="text-ui-10 text-muted-foreground">
                        {scheduled
                          ? t("studio.queue.scheduledFor", { time: when })
                          : when}
                      </span>
                    ) : null}
                  </div>
                  {item.error ? (
                    <p className="mt-1 break-words text-ui-10 text-destructive">
                      {item.error}
                    </p>
                  ) : null}
                </div>
                <div className="ml-auto flex items-center gap-1">
                  {item.status === "queued" && scheduled ? (
                    <Button
                      type="button"
                      size="sm"
                      variant="ghost"
                      onClick={() => startNow(item)}
                    >
                      {t("studio.queue.startNow")}
                    </Button>
                  ) : null}
                  <Button
                    type="button"
                    size="icon-sm"
                    variant="ghost"
                    aria-label={t("studio.queue.remove", { run: item.label })}
                    onClick={() => remove(item.id)}
                  >
                    <HugeiconsIcon icon={Delete02Icon} className="size-3.5" />
                  </Button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2">
        {paused ? (
          <p className="min-w-0 flex-1 text-ui-10 text-amber-600 dark:text-amber-400">
            {t("studio.queue.pausedHint")}
          </p>
        ) : (
          <p className="min-w-0 flex-1 text-ui-10 text-muted-foreground">
            {t("studio.queue.hint")}
          </p>
        )}
        {hasFinished ? (
          <Button
            type="button"
            size="sm"
            variant="ghost"
            onClick={clearFinished}
          >
            {t("studio.queue.clearFinished")}
          </Button>
        ) : null}
      </div>
    </SectionCard>
  );
}
