// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect } from "react";
import {
  ensureDebugTrainingSimulationLoaded,
  useDebugTrainingSimulationStore,
} from "../stores/debug-training-simulation-store";

/** Whether Settings -> Debugging turned simulated training on. */
export function useDebugTrainingSimulation(): boolean {
  const enabled = useDebugTrainingSimulationStore((state) => state.enabled);
  useEffect(() => {
    ensureDebugTrainingSimulationLoaded();
  }, []);
  return enabled;
}
