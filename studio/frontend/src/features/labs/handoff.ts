// SPDX-License-Identifier: NRC-2
// Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-2.md

import { create } from "zustand";

interface LabsHandoffState {
  /** A checkpoint the Artifacts studio wants the Scaler to open. Consumed once. */
  scaleSource: string | null;
  setScaleSource: (path: string | null) => void;
  consumeScaleSource: () => string | null;
}

/**
 * One-shot page handoff, not saved state: the row's path is usable on the very
 * next Scaler mount and stale after that. Keeping it out of the URL means a
 * refresh cannot silently re-expand a model the user has moved on from.
 */
export const useLabsHandoffStore = create<LabsHandoffState>((set, get) => ({
  scaleSource: null,
  setScaleSource: (path) => set({ scaleSource: path }),
  consumeScaleSource: () => {
    const value = get().scaleSource;
    if (value !== null) set({ scaleSource: null });
    return value;
  },
}));
