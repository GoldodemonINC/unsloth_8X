// SPDX-License-Identifier: NRC-3
// Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-3.md

export interface LabsFeatureLike {
  id: string;
  enabled: boolean;
}

export interface LabsGateState {
  enabled: boolean;
  features: readonly LabsFeatureLike[];
}

/**
 * Labs on *and* the experiment itself on. A per-feature switch means nothing
 * while the master switch is off, so both have to agree.
 */
export function labsFeatureOn(
  state: LabsGateState,
  featureId: string,
): boolean {
  if (!state.enabled) return false;
  return state.features.some(
    (feature) => feature.id === featureId && feature.enabled,
  );
}
