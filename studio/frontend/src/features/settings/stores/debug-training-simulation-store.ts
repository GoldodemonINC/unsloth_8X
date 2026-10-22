// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { loadDebugTrainingSimulation } from "../api/debug-training-simulation";

interface DebugTrainingSimulationState {
  /** Whether a run started from Training is a simulated one. */
  enabled: boolean;
  apply: (enabled: boolean) => void;
}

/**
 * One value read here rather than fetched per surface: the Debugging tab owns
 * the switch, and the Start button has to follow it without a reload now that
 * Settings is a dialog over the page it changes.
 */
export const useDebugTrainingSimulationStore =
  create<DebugTrainingSimulationState>((set) => ({
    enabled: false,
    apply: (enabled) => set({ enabled }),
  }));

let loaded = false;

/** Read the flag once per session; a failure leaves it off and retries on next mount. */
export function ensureDebugTrainingSimulationLoaded(): void {
  if (loaded) return;
  loaded = true;
  loadDebugTrainingSimulation()
    .then((state) => {
      useDebugTrainingSimulationStore.getState().apply(state.enabled);
    })
    .catch(() => {
      loaded = false;
    });
}

/** Read it again after a save, so a failed write cannot leave the UI claiming on. */
export function setDebugTrainingSimulationEnabled(enabled: boolean): void {
  loaded = true;
  useDebugTrainingSimulationStore.getState().apply(enabled);
}
