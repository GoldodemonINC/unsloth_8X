// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { useSettingsDialogStore } from "@/features/settings";
import { useT } from "@/i18n";
import { TestTubeOutlineIcon } from "@/lib/hugeicons-derived";
import { cn } from "@/lib/utils";
import { Tick02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useEffect, useState } from "react";
import { ArtifactsPanel } from "./components/artifacts-panel";
import { LabsFeatureGate } from "./components/labs-off-card";
import { PipelinesPanel } from "./components/pipelines-panel";
import {
  ensureLabsLoaded,
  refreshLabs,
  useLabsEnabled,
  useLabsStore,
} from "./store";

type StudioTab = "artifacts" | "pipelines";

export function LabsPage(): ReactElement {
  const t = useT();
  const enabled = useLabsEnabled();
  const loaded = useLabsStore((state) => state.loaded);
  const loading = useLabsStore((state) => state.loading);
  const error = useLabsStore((state) => state.error);
  const [tab, setTab] = useState<StudioTab>("artifacts");

  useEffect(() => {
    ensureLabsLoaded();
  }, []);

  return (
    <div className="flex h-full min-h-0 flex-col bg-background">
      <div className="mx-auto flex w-full max-w-[calc(1180px*var(--ui-space-scale,1))] flex-col gap-6 px-5 pb-20 pt-8 max-sm:px-4 sm:px-9">
        <header className="font-heading flex flex-col gap-2">
          <div className="flex flex-wrap items-center gap-2">
            <h1 className="page-title-halo text-ui-30 font-semibold leading-[1.04] tracking-[-0.028em] text-foreground sm:text-ui-34">
              {t("labs.title")}
            </h1>
            <Badge variant="outline" className="gap-1">
              <HugeiconsIcon
                icon={TestTubeOutlineIcon}
                strokeWidth={1.75}
                className="size-3"
              />
              {t("labs.experimental")}
            </Badge>
          </div>
          <p className="page-title-halo max-w-3xl text-sm leading-relaxed text-muted-foreground">
            {t("labs.description")}
          </p>
        </header>

        {!loaded && loading ? (
          <div className="flex items-center gap-2 text-ui-11 text-muted-foreground">
            <Spinner />
            {t("labs.loading")}
          </div>
        ) : null}

        {!enabled && (loaded || error) ? (
          <section className="elevated-card flex flex-col gap-3 p-5">
            <h2 className="text-sm font-medium text-foreground">
              {t("labs.offTitle")}
            </h2>
            <p className="max-w-2xl text-ui-11 leading-relaxed text-muted-foreground">
              {t("labs.offDescription")}
            </p>
            <div className="flex flex-wrap items-center gap-2">
              <Button
                size="sm"
                onClick={() =>
                  useSettingsDialogStore.getState().openDialog("labs")
                }
              >
                {t("labs.openSettings")}
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => void refreshLabs()}
              >
                {t("labs.reload")}
              </Button>
            </div>
            {error ? (
              <p className="text-ui-11 text-destructive">{error}</p>
            ) : null}
          </section>
        ) : null}

        {enabled ? (
          <>
            <div className="flex flex-wrap items-center gap-1.5">
              {(
                [
                  ["artifacts", "labs.tab.artifacts"],
                  ["pipelines", "labs.tab.pipelines"],
                ] as const
              ).map(([id, key]) => (
                <button
                  key={id}
                  type="button"
                  aria-pressed={tab === id}
                  onClick={() => setTab(id)}
                  className={cn(
                    "relative inline-flex h-8 items-center gap-2 rounded-full px-3.5 text-ui-12 font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
                    tab === id
                      ? "bg-primary/10 text-foreground"
                      : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  {tab === id ? (
                    <HugeiconsIcon
                      icon={Tick02Icon}
                      strokeWidth={2}
                      className="size-3.5"
                    />
                  ) : null}
                  {t(key)}
                </button>
              ))}
            </div>

            {tab === "artifacts" ? (
              <LabsFeatureGate feature="artifacts">
                <ArtifactsPanel />
              </LabsFeatureGate>
            ) : (
              <LabsFeatureGate feature="pipelines">
                <PipelinesPanel />
              </LabsFeatureGate>
            )}
          </>
        ) : null}
      </div>
    </div>
  );
}
