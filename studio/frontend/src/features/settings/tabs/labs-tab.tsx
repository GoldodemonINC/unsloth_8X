// SPDX-License-Identifier: NRC-3
// Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-3.md

import { Badge } from "@/components/ui/badge";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { type LabsFeature, saveLabsState } from "@/features/labs/api";
import {
  adoptLabsState,
  ensureLabsLoaded,
  useLabsStore,
} from "@/features/labs/store";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { useCallback, useEffect, useState } from "react";
import { SettingsRow } from "../components/settings-row";
import { SettingsSection } from "../components/settings-section";

function statusLabel(feature: LabsFeature): string {
  return feature.status === "beta" ? "Beta" : "Alpha";
}

export function LabsTab() {
  const t = useT();
  const enabled = useLabsStore((state) => state.enabled);
  const features = useLabsStore((state) => state.features);
  const loaded = useLabsStore((state) => state.loaded);
  const loading = useLabsStore((state) => state.loading);
  const error = useLabsStore((state) => state.error);
  const note = useLabsStore((state) => state.note);
  const [saving, setSaving] = useState<string | null>(null);

  useEffect(() => {
    ensureLabsLoaded();
  }, []);

  const save = useCallback(
    async (key: string, payload: Parameters<typeof saveLabsState>[0]) => {
      setSaving(key);
      try {
        adoptLabsState(await saveLabsState(payload));
      } catch (failure) {
        toast.error(t("settings.labs.saveFailed"), {
          description: failure instanceof Error ? failure.message : undefined,
        });
      } finally {
        setSaving(null);
      }
    },
    [t],
  );

  return (
    <div className="settings-page">
      <header className="flex min-w-0 flex-col gap-1">
        <h1
          data-settings-label={t("settings.labs.title")}
          className="text-xl font-semibold font-heading"
        >
          {t("settings.labs.title")}
        </h1>
        <p
          data-settings-label={t("settings.labs.description")}
          className="text-xs leading-relaxed text-muted-foreground"
        >
          {t("settings.labs.description")}
        </p>
      </header>

      <SettingsSection
        title={t("settings.labs.section")}
        description={t("settings.labs.sectionHint")}
      >
        <SettingsRow
          label={t("settings.labs.enableLabel")}
          description={
            enabled ? t("settings.labs.enableOn") : t("settings.labs.enableOff")
          }
        >
          <Switch
            checked={enabled}
            disabled={!loaded || saving !== null}
            onCheckedChange={(next) => void save("master", { enabled: next })}
            data-testid="labs-master-switch"
            aria-label={t("settings.labs.enableLabel")}
          />
        </SettingsRow>
        {!loaded && (loading || !error) ? (
          <SettingsRow label={t("settings.labs.loading")}>
            <Spinner />
          </SettingsRow>
        ) : null}
        {error ? (
          <SettingsRow
            label={t("settings.labs.unavailable")}
            description={error}
          >
            <Badge variant="outline">{t("settings.labs.off")}</Badge>
          </SettingsRow>
        ) : null}
      </SettingsSection>

      {enabled ? (
        <SettingsSection
          title={t("settings.labs.features")}
          description={t("settings.labs.featuresHint")}
        >
          {features.map((feature) => (
            <SettingsRow
              key={feature.id}
              label={feature.label}
              description={
                <span className="flex flex-col gap-1.5">
                  <span className="flex flex-wrap items-center gap-2">
                    <Badge variant="outline" className="h-5 px-1.5 text-ui-10">
                      {statusLabel(feature)}
                    </Badge>
                    {feature.studio ? (
                      <Badge
                        variant="outline"
                        className="h-5 border-primary/30 px-1.5 text-ui-10 text-primary"
                      >
                        {t("settings.labs.studio")}
                      </Badge>
                    ) : null}
                  </span>
                  <span>{feature.description}</span>
                </span>
              }
            >
              <Switch
                checked={feature.enabled}
                disabled={saving !== null}
                onCheckedChange={(next) =>
                  void save(feature.id, {
                    features: { [feature.id]: next },
                  })
                }
                data-testid={`labs-feature-${feature.id}`}
                aria-label={feature.label}
              />
            </SettingsRow>
          ))}
        </SettingsSection>
      ) : null}

      <p className="text-ui-10 leading-relaxed text-muted-foreground/80">
        {note || t("settings.labs.note")}
      </p>
    </div>
  );
}
