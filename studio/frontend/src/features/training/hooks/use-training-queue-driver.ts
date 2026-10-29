// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { usePlatformStore } from "@/config/env";
import { getHfToken } from "@/features/hub";
import { translate } from "@/i18n";
import { useEffect } from "react";
import { startTraining } from "../api/train-api";
import { syncTrainingRuntimeFromBackend } from "../lib/sync-runtime";
import { nextReadyQueueItem } from "../lib/training-queue";
import {
  releaseTrainingStart,
  settleAcceptedTrainingStart,
  tryAcquireTrainingStart,
} from "../lib/training-start-runtime";
import { useTrainingConfigStore } from "../stores/training-config-store";
import { useTrainingQueueStore } from "../stores/training-queue-store";
import {
  isTrainingStartPending,
  useTrainingRuntimeStore,
} from "../stores/training-runtime-store";

const POLL_MS = 15000;
let starting = false;

async function startQueuedItem(itemId: string): Promise<void> {
  const queue = useTrainingQueueStore.getState();
  const item = queue.items.find((entry) => entry.id === itemId);
  if (!item || item.status !== "queued") {
    return;
  }
  const lease = tryAcquireTrainingStart();
  if (!lease) {
    return;
  }
  queue.begin(itemId);
  const runtime = useTrainingRuntimeStore.getState();
  const payload = {
    ...item.config,
    hf_token: getHfToken() || null,
    wandb_token: item.config.enable_wandb
      ? useTrainingConfigStore.getState().wandbToken || null
      : null,
  };
  runtime.setStartResources(
    payload.model_name,
    payload.hf_dataset,
    false,
    payload.project_name ?? "",
    payload.hf_token,
  );
  try {
    const response = await startTraining(payload, lease.startRequestId);
    if (response.status === "error") {
      releaseTrainingStart(lease, response.error || response.message);
      queue.settle(
        itemId,
        "failed",
        response.error || response.message || translate("studio.queue.startFailed"),
      );
      return;
    }
    const adopted = await settleAcceptedTrainingStart(
      lease,
      response.job_id,
      response.message,
    );
    if (!adopted) {
      queue.settle(itemId, "failed", translate("studio.queue.startFailed"));
    }
  } catch (error) {
    const message =
      error instanceof Error
        ? error.message
        : translate("studio.queue.startFailed");
    releaseTrainingStart(lease, message);
    queue.settle(itemId, "failed", message);
  }
}

export function useTrainingQueueDriver(): void {
  const pending = useTrainingRuntimeStore(isTrainingStartPending);
  const revision = useTrainingQueueStore((state) => state.revision);
  const paused = useTrainingQueueStore((state) => state.paused);

  useEffect(() => {
    if (pending || paused) {
      return;
    }
    let cancelled = false;

    const tick = async () => {
      if (cancelled || starting) {
        return;
      }
      if (usePlatformStore.getState().capabilitiesUnknown()) {
        return;
      }
      const queue = useTrainingQueueStore.getState();
      if (queue.paused) {
        return;
      }
      const ready = nextReadyQueueItem(queue.items, Date.now());
      if (!ready) {
        return;
      }
      if (!useTrainingRuntimeStore.getState().hasHydrated) {
        await syncTrainingRuntimeFromBackend().catch(() => null);
        if (cancelled) {
          return;
        }
      }
      if (isTrainingStartPending(useTrainingRuntimeStore.getState())) {
        return;
      }
      const item = nextReadyQueueItem(
        useTrainingQueueStore.getState().items,
        Date.now(),
      );
      if (!item) {
        return;
      }
      starting = true;
      try {
        await startQueuedItem(item.id);
      } finally {
        starting = false;
      }
    };

    void tick();
    const timer = window.setInterval(() => void tick(), POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [pending, paused, revision]);
}
