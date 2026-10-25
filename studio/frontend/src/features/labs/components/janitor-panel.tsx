// SPDX-License-Identifier: NRC-1
// Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-1.md

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  Alert02Icon,
  Delete02Icon,
  HardDriveIcon,
  Refresh01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useCallback, useEffect, useState } from "react";
import {
  type HousekeepingEntry,
  type HousekeepingReport,
  deleteHousekeeping,
  scanHousekeeping,
} from "../api";

const KIND_LABEL: Record<string, string> = {
  checkpoint: "Checkpoint",
  run: "Run output",
  export: "Export",
  "hf-model": "HF model",
  "hf-dataset": "HF dataset",
  "hf-cache": "HF cache",
  folder: "Folder",
};

function isSelected(
  selected: Record<string, boolean>,
  entry: HousekeepingEntry,
): boolean {
  return Boolean(selected[entry.path]);
}

export function JanitorPanel(): ReactElement {
  const t = useT();
  const [report, setReport] = useState<HousekeepingReport | null>(null);
  const [selected, setSelected] = useState<Record<string, boolean>>({});
  const [loading, setLoading] = useState(true);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const next = await scanHousekeeping();
      setReport(next);
      setSelected({});
    } catch (failure) {
      setError(
        failure instanceof Error
          ? failure.message
          : "The storage scan could not finish.",
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const chosen = (report?.entries ?? []).filter((entry) =>
    isSelected(selected, entry),
  );
  const chosenBytes = chosen.reduce((total, entry) => total + entry.bytes, 0);

  const toggle = (entry: HousekeepingEntry) => {
    if (!entry.deletable) return;
    setSelected((current) => ({
      ...current,
      [entry.path]: !current[entry.path],
    }));
  };

  const remove = async () => {
    if (chosen.length === 0) return;
    const ok = window.confirm(
      `${t("labs.cleanup.confirm")}\n\n${chosen
        .map((entry) => entry.path)
        .join("\n")}`,
    );
    if (!ok) return;
    setDeleting(true);
    setError(null);
    try {
      const result = await deleteHousekeeping(chosen.map((entry) => entry.path));
      toast.success(t("labs.cleanup.deleted"), {
        description: result.freed_label,
      });
      if (result.refused.length > 0) {
        setError(
          result.refused
            .map((item) => `${item.path}: ${item.reason}`)
            .join("\n"),
        );
      }
      await refresh();
    } catch (failure) {
      const message =
        failure instanceof Error ? failure.message : t("labs.cleanup.failed");
      setError(message);
      toast.error(t("labs.cleanup.failed"), { description: message });
    } finally {
      setDeleting(false);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <section className="elevated-card flex flex-col gap-4 p-5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            <h2 className="text-ui-14 font-semibold text-foreground">
              {t("labs.cleanup.title")}
            </h2>
            <p className="mt-1 max-w-3xl text-ui-11 leading-relaxed text-muted-foreground">
              {t("labs.cleanup.description")}
            </p>
          </div>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => void refresh()}
            disabled={loading || deleting}
          >
            {loading ? (
              <Spinner />
            ) : (
              <HugeiconsIcon icon={Refresh01Icon} className="size-3.5" />
            )}
            {t("labs.refresh")}
          </Button>
        </div>

        {report ? (
          <div className="grid gap-3 sm:grid-cols-3">
            {[
              [t("labs.cleanup.used"), report.total_label],
              [t("labs.cleanup.reclaimable"), report.reclaimable_label],
              [t("labs.cleanup.free"), report.free_label],
            ].map(([label, value]) => (
              <div
                key={label}
                className="rounded-lg border border-border/70 bg-background/40 px-3 py-2"
              >
                <div className="text-ui-10 uppercase tracking-[0.05em] text-muted-foreground">
                  {label}
                </div>
                <div className="mt-0.5 font-mono text-ui-16 font-semibold text-foreground">
                  {value}
                </div>
              </div>
            ))}
          </div>
        ) : null}

        {error ? (
          <pre className="whitespace-pre-wrap rounded-lg border border-destructive/30 bg-destructive/5 p-3 text-ui-10 text-destructive">
            {error}
          </pre>
        ) : null}

        {report && report.entries.length === 0 ? (
          <p className="text-ui-11 text-muted-foreground">
            {t("labs.cleanup.empty")}
          </p>
        ) : null}

        {report?.roots.map((root) => {
          const entries = report.entries.filter(
            (entry) => entry.root === root.id,
          );
          if (entries.length === 0) return null;
          return (
            <div key={root.id} className="flex flex-col gap-2">
              <div className="flex flex-wrap items-baseline gap-2">
                <span className="flex items-center gap-1.5 text-ui-12 font-semibold text-foreground">
                  <HugeiconsIcon
                    icon={HardDriveIcon}
                    className="size-3.5 text-muted-foreground"
                  />
                  {root.label}
                </span>
                <Badge variant="outline">{root.entry_count}</Badge>
                <span className="truncate font-mono text-ui-10 text-muted-foreground/70">
                  {root.path}
                </span>
              </div>
              <p className="text-ui-10 text-muted-foreground/80">{root.note}</p>
              <div className="overflow-hidden rounded-lg border border-border/70">
                {entries.map((entry) => {
                  const on = isSelected(selected, entry);
                  return (
                    <label
                      key={entry.path}
                      className={cn(
                        "flex cursor-pointer flex-wrap items-center gap-2 border-b border-border/50 px-3 py-2 last:border-b-0",
                        on ? "bg-primary/5" : "bg-background/40",
                        !entry.deletable && "cursor-not-allowed opacity-70",
                      )}
                      title={entry.protected_reason ?? entry.path}
                    >
                      <input
                        type="checkbox"
                        className="size-3.5 accent-primary"
                        checked={on}
                        disabled={!entry.deletable}
                        onChange={() => toggle(entry)}
                      />
                      <span className="min-w-0 flex-1 truncate text-ui-11 text-foreground">
                        {entry.name}
                      </span>
                      <Badge variant="outline" className="text-ui-9">
                        {KIND_LABEL[entry.kind] ?? entry.kind}
                      </Badge>
                      <span className="font-mono text-ui-10 text-muted-foreground">
                        {entry.bytes_label}
                      </span>
                      <span className="hidden text-ui-10 text-muted-foreground/70 sm:inline">
                        {new Date(entry.modified).toLocaleDateString()}
                      </span>
                      {!entry.deletable ? (
                        <span className="flex items-center gap-1 text-ui-9 text-muted-foreground">
                          <HugeiconsIcon
                            icon={Alert02Icon}
                            className="size-3"
                          />
                          {entry.protected_reason}
                        </span>
                      ) : null}
                    </label>
                  );
                })}
              </div>
            </div>
          );
        })}

        <div className="flex flex-wrap items-center gap-3 border-t border-border/60 pt-3">
          <span className="text-ui-11 text-muted-foreground">
            {t("labs.cleanup.selected", { count: chosen.length })} ·{" "}
            <span className="font-mono text-foreground">
              {(chosenBytes / 1e9).toFixed(2)} GB
            </span>
          </span>
          <Button
            size="sm"
            variant="destructive"
            disabled={chosen.length === 0 || deleting}
            onClick={() => void remove()}
            className="ml-auto"
          >
            {deleting ? (
              <Spinner />
            ) : (
              <HugeiconsIcon icon={Delete02Icon} className="size-3.5" />
            )}
            {t("labs.cleanup.delete")}
          </Button>
        </div>
        <p className="text-ui-10 leading-relaxed text-muted-foreground/70">
          {t("labs.cleanup.note")}
        </p>
      </section>
    </div>
  );
}
