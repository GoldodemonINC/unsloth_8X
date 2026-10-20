// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Spinner } from "@/components/ui/spinner";
import {
  hfApiToken,
  useHfTokenStore,
  useHubDatasetSearch,
} from "@/features/hub";
import { useTrainingConfigStore } from "@/features/training";
import { useDebouncedValue, useHfDatasetSplits } from "@/hooks";
import { cn } from "@/lib/utils";
import { Search01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useMemo, useState } from "react";
import { useShallow } from "zustand/react/shallow";

const REPO_ID_RE = /^[A-Za-z0-9][A-Za-z0-9_.-]*\/[A-Za-z0-9][A-Za-z0-9_.-]*$/;

function formatCount(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return "0";
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
  return String(value);
}

/**
 * The second Dataset selector: a separate Hugging Face dataset for evaluation.
 *
 * Evaluation used to be a split of the training dataset only, so a run could not
 * hold out a clean set from a different repo. This picks that repo and its
 * subset/split; the backend reads it beside the training data instead of carving
 * a split out of it. Upload and S3 training keep their own eval-file slot, so the
 * selector only appears for a Browse dataset.
 */
export function EvalDatasetSection({
  disabled = false,
}: { disabled?: boolean }) {
  const {
    evalDataset,
    evalDatasetSubset,
    evalDatasetSplit,
    setEvalDatasetSubset,
    setEvalDatasetSplit,
  } = useTrainingConfigStore(
    useShallow((state) => ({
      evalDataset: state.evalDataset,
      evalDatasetSubset: state.evalDatasetSubset,
      evalDatasetSplit: state.evalDatasetSplit,
      setEvalDatasetSubset: state.setEvalDatasetSubset,
      setEvalDatasetSplit: state.setEvalDatasetSplit,
    })),
  );
  const selectEvalHfDataset = useTrainingConfigStore(
    (state) => state.selectEvalHfDataset,
  );
  const hfToken = useHfTokenStore((state) => state.token);
  const accessToken = hfApiToken(hfToken);
  const [query, setQuery] = useState("");
  const debouncedQuery = useDebouncedValue(query, 350);

  const { results, isLoading, error } = useHubDatasetSearch(debouncedQuery, {
    accessToken,
    enabled: debouncedQuery.trim().length > 0 && !disabled,
  });
  const typedRepoId = useMemo(() => {
    const text = debouncedQuery.trim();
    return REPO_ID_RE.test(text) ? text : null;
  }, [debouncedQuery]);
  const visible = useMemo(() => results.slice(0, 6), [results]);

  const {
    subsets,
    splits,
    isLoading: optionsLoading,
  } = useHfDatasetSplits(evalDataset, evalDatasetSubset, { accessToken });

  const commit = (id: string) => {
    const trimmed = id.trim();
    if (!trimmed) return;
    selectEvalHfDataset(trimmed);
    setQuery("");
  };

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="flex items-center gap-1.5 text-ui-11 font-medium uppercase tracking-[0.05em] text-muted-foreground/70">
          Evaluation dataset
          <span className="text-ui-10 normal-case tracking-normal text-muted-foreground/60">
            optional
          </span>
        </span>
        {evalDataset ? (
          <button
            type="button"
            onClick={() => selectEvalHfDataset(null)}
            disabled={disabled}
            className="text-ui-10 text-muted-foreground transition-colors hover:text-foreground disabled:opacity-50"
          >
            Clear
          </button>
        ) : null}
      </div>

      {evalDataset ? (
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-2 rounded-lg border border-border/60 bg-muted/20 px-2.5 py-1.5">
            <span className="truncate font-mono text-ui-11 text-foreground">
              {evalDataset}
            </span>
          </div>
          <div className="grid gap-2 sm:grid-cols-2">
            <div className="flex flex-col gap-1">
              <span className="text-ui-10 text-muted-foreground">Subset</span>
              <Select
                value={evalDatasetSubset ?? ""}
                onValueChange={(value) =>
                  setEvalDatasetSubset(value === "" ? null : value)
                }
                disabled={disabled || subsets.length === 0}
              >
                <SelectTrigger aria-label="Subset" className="h-9 text-ui-11">
                  <SelectValue
                    placeholder={optionsLoading ? "Loading…" : "Default subset"}
                  />
                </SelectTrigger>
                <SelectContent>
                  {subsets.map((subset) => (
                    <SelectItem key={subset} value={subset}>
                      {subset}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="flex flex-col gap-1">
              <span className="text-ui-10 text-muted-foreground">Split</span>
              <Select
                value={evalDatasetSplit ?? ""}
                onValueChange={(value) =>
                  setEvalDatasetSplit(value === "" ? null : value)
                }
                disabled={disabled || splits.length === 0}
              >
                <SelectTrigger aria-label="Split" className="h-9 text-ui-11">
                  <SelectValue
                    placeholder={optionsLoading ? "Loading…" : "train"}
                  />
                </SelectTrigger>
                <SelectContent>
                  {splits.map((split) => (
                    <SelectItem key={split} value={split}>
                      {split}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>
      ) : (
        <div className="flex flex-col gap-2">
          <div className="relative">
            <HugeiconsIcon
              icon={Search01Icon}
              className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted-foreground"
            />
            <Input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && typedRepoId) {
                  event.preventDefault();
                  commit(typedRepoId);
                }
              }}
              placeholder="Search the Hub, or paste org/dataset"
              disabled={disabled}
              className="h-9 pl-8 font-mono text-ui-12"
            />
            {isLoading ? (
              <Spinner className="absolute top-1/2 right-2.5 size-3.5 -translate-y-1/2" />
            ) : null}
          </div>

          {typedRepoId && !visible.some((row) => row.id === typedRepoId) ? (
            <button
              type="button"
              onClick={() => commit(typedRepoId)}
              className="flex items-center justify-between rounded-lg border border-dashed border-border/70 px-2.5 py-1.5 text-left text-ui-11 hover:border-foreground/40"
            >
              <span className="truncate font-mono">{typedRepoId}</span>
              <span className="text-ui-10 text-muted-foreground">use this</span>
            </button>
          ) : null}

          {visible.length > 0 ? (
            <ul className="flex max-h-52 flex-col overflow-y-auto rounded-lg border border-border/60">
              {visible.map((row) => (
                <li key={row.id}>
                  <button
                    type="button"
                    onClick={() => commit(row.id)}
                    className={cn(
                      "flex w-full items-center justify-between gap-3 px-2.5 py-1.5 text-left hover:bg-muted/40",
                      row.id === evalDataset && "bg-muted/50",
                    )}
                  >
                    <span className="truncate font-mono text-ui-11">
                      {row.id}
                    </span>
                    <span className="shrink-0 text-ui-10 text-muted-foreground">
                      {formatCount(row.downloads)} downloads
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          ) : null}

          {error && !isLoading ? (
            <p className="text-ui-10 text-destructive">
              Hub search failed: {error}
            </p>
          ) : null}
        </div>
      )}

      <p className="text-ui-10 leading-ui-14 text-muted-foreground/70">
        {evalDataset
          ? "Evaluated on this dataset instead of a split of the training data; it must use the same columns as the training data and is never trained on."
          : "Leave empty to evaluate on a split of the training dataset, or pick a second dataset to hold out a clean one from another repo."}
      </p>
    </div>
  );
}
