// SPDX-License-Identifier: NRC-2
// Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-2.md

import { create } from "zustand";
import { type LabsFeature, type LabsState, loadLabsState } from "./api";
import { labsFeatureOn } from "./lib/feature-gate";

interface LabsStoreState {
  enabled: boolean;
  features: LabsFeature[];
  note: string;
  loaded: boolean;
  loading: boolean;
  error: string | null;
  apply: (state: LabsState) => void;
  begin: () => void;
  fail: (message: string) => void;
  reset: () => void;
}

const EMPTY = {
  enabled: false,
  features: [] as LabsFeature[],
  note: "",
  loaded: false,
  loading: false,
  error: null as string | null,
};

export const useLabsStore = create<LabsStoreState>((set) => ({
  ...EMPTY,
  apply: (state) =>
    set({
      enabled: state.enabled,
      features: state.features,
      note: state.note,
      loaded: true,
      loading: false,
      error: null,
    }),
  begin: () => set({ loading: true, error: null }),
  fail: (message) => set({ loading: false, error: message }),
  reset: () => set({ ...EMPTY }),
}));

export function useLabsEnabled(): boolean {
  return useLabsStore((state) => state.enabled);
}

export function useLabsFeature(featureId: string): boolean {
  return useLabsStore((state) => labsFeatureOn(state, featureId));
}

export function useLabsFeatureEntry(featureId: string): LabsFeature | null {
  return useLabsStore(
    (state) =>
      state.features.find((feature) => feature.id === featureId) ?? null,
  );
}

let loaded = false;
let inFlight: Promise<void> | null = null;

function readLabs(): Promise<void> {
  if (inFlight) return inFlight;
  const store = useLabsStore.getState();
  store.begin();
  inFlight = loadLabsState()
    .then((state) => {
      loaded = true;
      useLabsStore.getState().apply(state);
    })
    .catch((error: unknown) => {
      // Off is the safe answer when the server cannot say: nothing is gated open.
      loaded = false;
      useLabsStore
        .getState()
        .fail(
          error instanceof Error
            ? error.message
            : "Labs settings could not be read.",
        );
    })
    .finally(() => {
      inFlight = null;
    });
  return inFlight;
}

/** Read the switch once per session. Failure leaves it off and retries on next mount. */
export function ensureLabsLoaded(): void {
  if (loaded) return;
  void readLabs();
}

export function refreshLabs(): Promise<void> {
  loaded = false;
  return readLabs();
}

/** A save already returned the authoritative state; adopt it without a second read. */
export function adoptLabsState(state: LabsState): void {
  loaded = true;
  useLabsStore.getState().apply(state);
}

export function resetLabsForTests(): void {
  loaded = false;
  inFlight = null;
  useLabsStore.getState().reset();
}
