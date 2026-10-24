// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useLabsHandoffStore } from "@/features/labs/handoff";
import { useTrainingConfigStore } from "@/features/training";
import { useState, type ReactElement } from "react";
import { ScalerPanel } from "./components/scaler-panel";

export function ScalePage(): ReactElement {
  // A row in the Artifacts studio can hand this page one checkpoint. Read it once,
  // so a later visit does not silently reuse a path the user has moved on from.
  const [prefill] = useState(
    () => useLabsHandoffStore.getState().consumeScaleSource() ?? "",
  );
  return (
    <div className="flex h-full min-h-0 flex-col bg-background">
      <div className="mx-auto flex w-full max-w-[calc(1180px*var(--ui-space-scale,1))] flex-col gap-6 px-5 pb-20 pt-8 max-sm:px-4 sm:px-9">
        <header className="font-heading flex flex-col gap-2">
          <h1 className="page-title-halo text-ui-30 font-semibold leading-[1.04] tracking-[-0.028em] text-foreground sm:text-ui-34">
            Scaler
          </h1>
          <p className="page-title-halo text-sm text-muted-foreground">
            Grow a checkpoint before training it: pick a target between its own
            parameter count and base + 3B. At step 0 the expanded model computes
            exactly what the base does, and the later full-parameter training is
            where the new capacity is used.
          </p>
        </header>

        <section className="elevated-card flex flex-col gap-4 p-5">
          <ScalerPanel
            defaultSource={prefill}
            onUseCheckpoint={(path) =>
              useTrainingConfigStore
                .getState()
                .selectTrainingModel(path, null, { localPath: path })
            }
          />
        </section>
      </div>
    </div>
  );
}
