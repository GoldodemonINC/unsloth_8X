// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";
import {
  type SweepVariant,
  type TrainingQueueItem,
  type TrainingQueueStatus,
  createQueueItemId,
  sanitizeQueueConfig,
} from "../lib/training-queue";

interface EnqueueOptions {
  groupId?: string | null;
  notBefore?: string | null;
}

interface TrainingQueueStore {
  items: TrainingQueueItem[];
  paused: boolean;
  revision: number;
  enqueue: (
    variants: readonly SweepVariant[],
    options?: EnqueueOptions,
  ) => TrainingQueueItem[];
  begin: (id: string) => void;
  settle: (
    id: string,
    status: TrainingQueueStatus,
    error?: string | null,
  ) => void;
  remove: (id: string) => void;
  clearFinished: () => void;
  setPaused: (paused: boolean) => void;
}

export const useTrainingQueueStore = create<TrainingQueueStore>()(
  persist(
    (set) => ({
      items: [],
      paused: false,
      revision: 0,
      enqueue: (variants, options = {}) => {
        const createdAt = new Date().toISOString();
        const added = variants.map((variant) => ({
          id: createQueueItemId(),
          label: variant.label,
          groupId: options.groupId ?? null,
          config: sanitizeQueueConfig(variant.config),
          notBefore: options.notBefore ?? null,
          status: "queued" as const,
          error: null,
          createdAt,
          startedAt: null,
          finishedAt: null,
        }));
        set((state) => ({
          items: [...state.items, ...added],
          revision: state.revision + 1,
        }));
        return added;
      },
      begin: (id) =>
        set((state) => {
          const now = new Date().toISOString();
          return {
            revision: state.revision + 1,
            items: state.items.map((item) => {
              if (item.id === id) {
                return {
                  ...item,
                  status: "running",
                  error: null,
                  startedAt: now,
                };
              }
              if (item.status === "running") {
                return { ...item, status: "done", finishedAt: now };
              }
              return item;
            }),
          };
        }),
      settle: (id, status, error = null) =>
        set((state) => ({
          revision: state.revision + 1,
          items: state.items.map((item) =>
            item.id === id
              ? {
                  ...item,
                  status,
                  error,
                  finishedAt:
                    status === "running" || status === "queued"
                      ? null
                      : new Date().toISOString(),
                }
              : item,
          ),
        })),
      remove: (id) =>
        set((state) => ({
          revision: state.revision + 1,
          items: state.items.filter((item) => item.id !== id),
        })),
      clearFinished: () =>
        set((state) => ({
          revision: state.revision + 1,
          items: state.items.filter(
            (item) =>
              item.status === "queued" ||
              item.status === "running" ||
              item.status === "failed",
          ),
        })),
      setPaused: (paused) =>
        set((state) => ({ paused, revision: state.revision + 1 })),
    }),
    {
      name: "unsloth-training-queue",
      version: 1,
      partialize: (state) => ({ items: state.items, paused: state.paused }),
    },
  ),
);
