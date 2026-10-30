// SPDX-License-Identifier: NRC-3
// Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-3.md

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { PillTabs } from "@/features/model-picker";
import { useSettingsDialogStore } from "@/features/settings";
import { type TranslationKey, useT } from "@/i18n";
import { TestTubeOutlineIcon } from "@/lib/hugeicons-derived";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useEffect, useState } from "react";
import { ArtifactsPanel } from "./components/artifacts-panel";
import { BenchmarksPanel } from "./components/benchmarks-panel";
import { CmtPanel } from "./components/cmt-panel";
import { JanitorPanel } from "./components/janitor-panel";
import { LabsFeatureGate } from "./components/labs-off-card";
import { LineagePanel } from "./components/lineage-panel";
import { PipelinesPanel } from "./components/pipelines-panel";
import {
  ensureLabsLoaded,
  refreshLabs,
  useLabsEnabled,
  useLabsStore,
} from "./store";

type StudioTab =
  | "artifacts"
  | "pipelines"
  | "benchmarks"
  | "cmt"
  | "lineage"
  | "cleanup";

const TABS: ReadonlyArray<readonly [StudioTab, TranslationKey]> = [
  ["artifacts", "labs.tab.artifacts"],
  ["pipelines", "labs.tab.pipelines"],
  ["benchmarks", "labs.tab.benchmarks"],
  ["cmt", "labs.tab.cmt"],
  ["lineage", "labs.tab.lineage"],
  ["cleanup", "labs.tab.cleanup"],
];

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
      {/* The panels below size their rows with container queries named
          train-card, so this column is that container. Without it every one of
          those rules is dead and each dataset row stacks into five full-width
          fields. */}
      <div className="@container/train-card mx-auto flex w-full max-w-[calc(1180px*var(--ui-space-scale,1))] flex-col gap-6 px-5 pb-20 pt-8 max-sm:px-4 sm:px-9">
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
          <div className="flex flex-col gap-4" aria-busy="true">
            <div className="flex gap-2">
              {["one", "two", "three", "four"].map((key) => (
                <Skeleton key={key} className="h-7 w-24 rounded-full" />
              ))}
            </div>
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              <Skeleton className="h-48 rounded-3xl" />
              <Skeleton className="h-48 rounded-3xl" />
            </div>
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
            {/* Narrow columns must not squeeze the tab labels into
                "Benchm…": a tab keeps its natural width and the strip scrolls
                when five of them cannot fit side by side. */}
            <PillTabs
              ariaLabel={t("labs.title")}
              value={tab}
              onValueChange={(next) => setTab(next as StudioTab)}
              compact={true}
              fit={true}
              className="max-w-full overflow-x-auto [&>button[role=tab]]:shrink-0"
              tabs={TABS.map(([id, key]) => ({ value: id, label: t(key) }))}
            />

            {tab === "artifacts" ? (
              <LabsFeatureGate feature="artifacts">
                <ArtifactsPanel />
              </LabsFeatureGate>
            ) : null}
            {tab === "pipelines" ? (
              <LabsFeatureGate feature="pipelines">
                <PipelinesPanel />
              </LabsFeatureGate>
            ) : null}
            {tab === "benchmarks" ? (
              <LabsFeatureGate feature="benchmarks">
                <BenchmarksPanel />
              </LabsFeatureGate>
            ) : null}
            {tab === "cmt" ? (
              <LabsFeatureGate feature="cmt">
                <CmtPanel />
              </LabsFeatureGate>
            ) : null}
            {tab === "lineage" ? (
              <LabsFeatureGate feature="lineage">
                <LineagePanel />
              </LabsFeatureGate>
            ) : null}
            {tab === "cleanup" ? (
              <LabsFeatureGate feature="housekeeping">
                <JanitorPanel />
              </LabsFeatureGate>
            ) : null}
          </>
        ) : null}
      </div>
    </div>
  );
}
