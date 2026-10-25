// SPDX-License-Identifier: NRC-1
// Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-1.md

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { useSettingsDialogStore } from "@/features/settings";
import { useT } from "@/i18n";
import { TestTubeOutlineIcon } from "@/lib/hugeicons-derived";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, type ReactNode, useState } from "react";
import {
  refreshLabs,
  useLabsEnabled,
  useLabsFeature,
  useLabsFeatureEntry,
} from "../store";

/**
 * What a Labs surface shows while its switch is off. A 404 or a redirect would
 * hide the feature; this says which switch to flip and offers the door to it.
 */
export function LabsOffCard({
  featureId,
}: { featureId: string }): ReactElement {
  const t = useT();
  const labsOn = useLabsEnabled();
  const entry = useLabsFeatureEntry(featureId);
  const [reloading, setReloading] = useState(false);
  const label = entry?.label ?? t("labs.unknownFeature");

  return (
    <section className="elevated-card flex flex-col gap-3 p-5">
      <div className="flex items-center gap-2">
        <span className="flex size-8 items-center justify-center rounded-full bg-primary/10 text-primary">
          <HugeiconsIcon
            icon={TestTubeOutlineIcon}
            strokeWidth={1.75}
            className="size-4"
          />
        </span>
        <div className="flex min-w-0 flex-col">
          <h2 className="truncate text-sm font-medium text-foreground">
            {label}
          </h2>
          <p className="text-ui-11 text-muted-foreground">
            {labsOn ? t("labs.featureOff") : t("labs.masterOff")}
          </p>
        </div>
        <Badge variant="outline" className="ml-auto shrink-0">
          Labs
        </Badge>
      </div>
      <p className="text-ui-11 leading-relaxed text-muted-foreground">
        {entry?.description ?? t("labs.unknownFeatureDescription")}
      </p>
      <div className="flex flex-wrap items-center gap-2">
        <Button
          size="sm"
          onClick={() => useSettingsDialogStore.getState().openDialog("labs")}
        >
          {t("labs.openSettings")}
        </Button>
        <Button
          size="sm"
          variant="ghost"
          disabled={reloading}
          onClick={() => {
            setReloading(true);
            void refreshLabs().finally(() => setReloading(false));
          }}
        >
          {reloading ? <Spinner /> : null}
          {t("labs.reload")}
        </Button>
      </div>
    </section>
  );
}

/** Renders children only while one experiment is on. */
export function LabsFeatureGate({
  feature,
  children,
}: {
  feature: string;
  children: ReactNode;
}): ReactElement {
  const enabled = useLabsFeature(feature);
  if (!enabled) return <LabsOffCard featureId={feature} />;
  return <>{children}</>;
}
