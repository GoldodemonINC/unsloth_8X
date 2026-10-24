// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
