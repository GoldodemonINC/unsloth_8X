// SPDX-License-Identifier: NRC-1
// Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-1.md

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { formatBytes } from "@/features/hub";
import { useTrainingConfigStore } from "@/features/training";
import { type TranslationKey, useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  Alert02Icon,
  ArrowRight01Icon,
  ArrowUpRight01Icon,
  Refresh01Icon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import { type ReactElement, useCallback, useEffect, useState } from "react";
import {
  type Artifact,
  type ArtifactAction,
  loadArtifactDetail,
  loadArtifacts,
} from "../api";
import { useLabsHandoffStore } from "../handoff";
import { useLabsFeature } from "../store";

const ACTION_LABEL: Record<ArtifactAction["id"], TranslationKey> = {
  "continue-training": "labs.action.continue",
  scale: "labs.action.scale",
  merge: "labs.action.merge",
  quantize: "labs.action.quantize",
  benchmark: "labs.action.benchmark",
};

function params(value: number | null): string {
  if (!value) return "—";
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return `${Math.round(value)}`;
}

function action(record: Artifact, id: ArtifactAction["id"]): ArtifactAction {
  return (
    record.actions.find((item) => item.id === id) ?? {
      id,
      enabled: false,
      reason: null,
    }
  );
}

export function ArtifactsPanel(): ReactElement {
  const t = useT();
  const navigate = useNavigate();
  const [records, setRecords] = useState<Artifact[]>([]);
  const [root, setRoot] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Artifact | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [showManifest, setShowManifest] = useState(false);
  const canContinue = useLabsFeature("training-continuation");
  const canScale = useLabsFeature("scale-from-artifact");
  const canViewManifest = useLabsFeature("manifest-viewer");

  const load = useCallback(
    (signal?: AbortSignal) => {
      setLoading(true);
      setError(null);
      return loadArtifacts(50, signal)
        .then((result) => {
          setRecords(result.artifacts);
          setRoot(result.root);
        })
        .catch((failure: unknown) => {
          if (signal?.aborted) return;
          setError(
            failure instanceof Error ? failure.message : t("labs.loadFailed"),
          );
        })
        .finally(() => {
          if (!signal?.aborted) setLoading(false);
        });
    },
    [t],
  );

  useEffect(() => {
    const controller = new AbortController();
    void load(controller.signal);
    return () => controller.abort();
  }, [load]);

  const select = useCallback(
    (record: Artifact) => {
      setSelected(record);
      setShowManifest(false);
      setDetailLoading(true);
      loadArtifactDetail(record.path)
        .then((detail) => setSelected(detail))
        .catch((failure: unknown) => {
          toast.error(t("labs.loadFailed"), {
            description: failure instanceof Error ? failure.message : undefined,
          });
        })
        .finally(() => setDetailLoading(false));
    },
    [t],
  );

  const continueTraining = (record: Artifact) => {
    useTrainingConfigStore
      .getState()
      .selectTrainingModel(record.path, null, { localPath: record.path });
    toast.success(t("labs.handoff.continue"));
    void navigate({ to: "/studio" });
  };

  const scaleRecord = (record: Artifact) => {
    useLabsHandoffStore.getState().setScaleSource(record.path);
    toast.success(t("labs.handoff.scale"));
    void navigate({ to: "/scale" });
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-2">
        <p className="text-ui-11 text-muted-foreground">
          {t("labs.artifacts.count", { count: String(records.length) })}
        </p>
        <Button
          size="sm"
          variant="ghost"
          className="ml-auto"
          disabled={loading}
          onClick={() => void load()}
        >
          {loading ? (
            <Spinner />
          ) : (
            <HugeiconsIcon icon={Refresh01Icon} strokeWidth={1.75} />
          )}
          {t("labs.refresh")}
        </Button>
      </div>

      {error ? (
        <p className="flex items-start gap-2 text-ui-11 text-destructive">
          <HugeiconsIcon
            icon={Alert02Icon}
            strokeWidth={1.75}
            className="mt-px size-3.5 shrink-0"
          />
          {error}
        </p>
      ) : null}

      {loading && records.length === 0 ? (
        <div className="flex items-center gap-2 py-10 text-ui-11 text-muted-foreground">
          <Spinner />
          {t("labs.loading")}
        </div>
      ) : records.length === 0 && !error ? (
        <div className="rounded-2xl border border-border/60 p-6">
          <p className="text-sm font-medium text-foreground">
            {t("labs.artifacts.empty")}
          </p>
          <p className="mt-1 text-ui-11 leading-relaxed text-muted-foreground">
            {t("labs.artifacts.emptyHint")}
          </p>
        </div>
      ) : (
        <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
          <ul className="flex flex-col gap-2">
            {records.map((record) => {
              const active = selected?.path === record.path;
              return (
                <li key={record.path}>
                  <button
                    type="button"
                    onClick={() => select(record)}
                    aria-pressed={active}
                    className={cn(
                      "flex w-full flex-col gap-1 rounded-2xl border p-3 text-left transition-colors",
                      active
                        ? "border-primary/50 bg-primary/5"
                        : "border-border/60 hover:bg-muted/40",
                    )}
                  >
                    <span className="flex min-w-0 items-center gap-2">
                      <span className="min-w-0 truncate text-sm font-medium text-foreground">
                        {record.name}
                      </span>
                      <span className="ml-auto shrink-0 font-mono text-ui-10 text-muted-foreground">
                        {params(record.params)}
                      </span>
                    </span>
                    <span className="flex flex-wrap items-center gap-1.5">
                      {record.manifest?.stage ? (
                        <Badge
                          variant="outline"
                          className="h-5 px-1.5 text-ui-10"
                        >
                          {record.manifest.stage}
                        </Badge>
                      ) : null}
                      {record.adapter ? (
                        <Badge
                          variant="outline"
                          className="h-5 px-1.5 text-ui-10"
                        >
                          LoRA
                        </Badge>
                      ) : null}
                      {record.quantized ? (
                        <Badge
                          variant="outline"
                          className="h-5 px-1.5 text-ui-10"
                        >
                          quantized
                        </Badge>
                      ) : null}
                      {record.moe ? (
                        <Badge
                          variant="outline"
                          className="h-5 px-1.5 text-ui-10"
                        >
                          MoE
                        </Badge>
                      ) : null}
                      <span className="text-ui-10 text-muted-foreground">
                        {record.model_type ?? "unknown"} ·{" "}
                        {formatBytes(record.weight_bytes)}
                      </span>
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>

          <div className="flex min-w-0 flex-col gap-3">
            {selected ? (
              <>
                <div className="flex min-w-0 flex-col gap-1 rounded-2xl border border-border/60 p-4">
                  <div className="flex min-w-0 items-center gap-2">
                    <h3 className="min-w-0 truncate text-sm font-medium text-foreground">
                      {selected.name}
                    </h3>
                    {detailLoading ? <Spinner /> : null}
                  </div>
                  <p className="font-mono text-ui-10 break-all text-muted-foreground">
                    {selected.path}
                  </p>
                  <div className="mt-2 grid grid-cols-2 gap-2 text-ui-11 sm:grid-cols-4">
                    <div>
                      <p className="text-ui-10 text-muted-foreground">
                        {t("labs.artifacts.params")}
                      </p>
                      <p className="font-mono text-foreground">
                        {params(selected.params)}
                      </p>
                    </div>
                    <div>
                      <p className="text-ui-10 text-muted-foreground">
                        {t("labs.artifacts.size")}
                      </p>
                      <p className="font-mono text-foreground">
                        {formatBytes(selected.weight_bytes)}
                      </p>
                    </div>
                    <div>
                      <p className="text-ui-10 text-muted-foreground">
                        {t("labs.artifacts.family")}
                      </p>
                      <p className="font-mono text-foreground">
                        {selected.model_type ?? "—"}
                      </p>
                    </div>
                    <div>
                      <p className="text-ui-10 text-muted-foreground">
                        {t("labs.artifacts.written")}
                      </p>
                      <p className="font-mono text-foreground">
                        {selected.modified.replace("T", " ").replace("Z", "")}
                      </p>
                    </div>
                  </div>
                  {selected.manifest ? (
                    <p className="mt-2 text-ui-10 text-muted-foreground">
                      {t("labs.artifacts.provenance")}:{" "}
                      <span className="font-mono text-foreground/80">
                        {selected.manifest.stage ?? "unknown stage"} ·{" "}
                        {selected.manifest.run_id ?? "no run id"}
                      </span>
                    </p>
                  ) : (
                    <p className="mt-2 text-ui-10 text-muted-foreground">
                      {t("labs.artifacts.noManifest")}
                    </p>
                  )}

                  <div className="mt-3 flex flex-wrap items-center gap-2">
                    {canContinue &&
                    action(selected, "continue-training").enabled ? (
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => continueTraining(selected)}
                      >
                        {t("labs.action.continue")}
                        <HugeiconsIcon
                          icon={ArrowRight01Icon}
                          strokeWidth={1.75}
                        />
                      </Button>
                    ) : null}
                    {canScale && action(selected, "scale").enabled ? (
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => scaleRecord(selected)}
                      >
                        {t("labs.action.scale")}
                        <HugeiconsIcon
                          icon={ArrowUpRight01Icon}
                          strokeWidth={1.75}
                        />
                      </Button>
                    ) : null}
                    {canViewManifest && selected.manifest_path ? (
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => setShowManifest((value) => !value)}
                      >
                        {showManifest
                          ? t("labs.manifest.hide")
                          : t("labs.manifest.show")}
                      </Button>
                    ) : null}
                  </div>
                </div>

                <div className="rounded-2xl border border-border/60 p-4">
                  <p className="text-ui-11 font-medium text-foreground">
                    {t("labs.actions.title")}
                  </p>
                  <ul className="mt-2 flex flex-col gap-1.5">
                    {selected.actions.map((item) => (
                      <li
                        key={item.id}
                        className="flex items-start gap-2 text-ui-11 leading-snug"
                      >
                        {item.enabled ? (
                          <HugeiconsIcon
                            icon={Tick02Icon}
                            strokeWidth={2}
                            className="mt-0.5 size-3.5 shrink-0 text-emerald-600 dark:text-emerald-400"
                          />
                        ) : (
                          <HugeiconsIcon
                            icon={Alert02Icon}
                            strokeWidth={1.75}
                            className="mt-0.5 size-3.5 shrink-0 text-muted-foreground"
                          />
                        )}
                        <span className="min-w-0">
                          <span
                            className={cn(
                              "font-medium",
                              item.enabled
                                ? "text-foreground"
                                : "text-muted-foreground",
                            )}
                          >
                            {t(ACTION_LABEL[item.id])}
                          </span>
                          {item.reason ? (
                            <span className="block text-ui-10 text-muted-foreground">
                              {item.reason}
                            </span>
                          ) : (
                            <span className="block text-ui-10 text-muted-foreground">
                              {item.id === "quantize" || item.id === "merge"
                                ? t("labs.actions.openStudio")
                                : ""}
                            </span>
                          )}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>

                {showManifest && selected.raw_manifest ? (
                  <pre className="max-h-72 overflow-auto rounded-2xl border border-border/60 bg-muted/20 p-3 font-mono text-ui-10 leading-relaxed text-foreground/90">
                    {JSON.stringify(selected.raw_manifest, null, 2)}
                  </pre>
                ) : null}
                {selected.manifest_error ? (
                  <p className="text-ui-10 text-destructive">
                    {selected.manifest_error}
                  </p>
                ) : null}
              </>
            ) : (
              <p className="rounded-2xl border border-dashed border-border/60 p-6 text-ui-11 text-muted-foreground">
                {t("labs.artifacts.selectHint")}
              </p>
            )}
          </div>
        </div>
      )}

      {root ? (
        <p className="font-mono text-ui-10 break-all text-muted-foreground/70">
          {t("labs.artifacts.root")}: {root}
        </p>
      ) : null}
    </div>
  );
}
