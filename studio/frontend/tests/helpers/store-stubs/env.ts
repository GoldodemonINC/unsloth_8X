// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Minimal settable platform store. */
let deviceType: string | null = null;
// Off by default: a test host that never says otherwise is treated as GPU-capable, which is
// what the training recipes did before a measured CPU backend could downgrade them.
let cpuTrainingHost = false;

export const usePlatformStore = {
  getState: () => ({
    deviceType,
    isCpuTrainingHost: () => cpuTrainingHost,
  }),
  setState: (next: {
    deviceType?: string | null;
    isCpuTrainingHost?: boolean;
  }) => {
    if ("deviceType" in next) {
      deviceType = next.deviceType ?? null;
    }
    if ("isCpuTrainingHost" in next) {
      cpuTrainingHost = next.isCpuTrainingHost === true;
    }
  },
};

export async function fetchDeviceType(_options?: { force?: boolean }): Promise<void> {}
