// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import { hfApiToken, useHfTokenStore, useHubModelSearch } from "@/features/hub";
import { filterMergeSearchResults } from "@/features/advanced/merge-search-filter";
import { useDebouncedValue } from "@/hooks";
import { cn } from "@/lib/utils";
import { Search01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useMemo, useState } from "react";

const REPO_ID_RE = /^[A-Za-z0-9][A-Za-z0-9_.-]*\/[A-Za-z0-9][A-Za-z0-9_.-]*$/;

function formatCount(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return "0";
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
  return String(value);
}

/**
 * One merge input: a Hugging Face repo picked from the same Hub search the
 * trainer uses, or a local checkpoint path. The Hub is the default because it
 * is what people actually have; local stays one click away.
 */
export function MergeModelSlot({
  label,
  value,
  onChange,
  placeholder,
  disabled = false,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder: string;
  disabled?: boolean;
}) {
  const looksLikePath = /[\\/]/.test(value) && !REPO_ID_RE.test(value.trim());
  const [mode, setMode] = useState<"hub" | "local">(
    looksLikePath ? "local" : "hub",
  );
  const [query, setQuery] = useState("");
  const debouncedQuery = useDebouncedValue(query, 350);
  const hfToken = useHfTokenStore((s) => s.token);
  const accessToken = hfApiToken(hfToken);
  const searching = mode === "hub" && debouncedQuery.trim().length > 0;

  const { results, isLoading, error } = useHubModelSearch(debouncedQuery, {
    accessToken,
    excludeGguf: true,
    ownerScope: "all",
    enabled: searching && !disabled,
  });

  // A typed repo id is always usable, even when the listing has not caught up.
  const typedRepoId = useMemo(() => {
    const text = debouncedQuery.trim();
    return REPO_ID_RE.test(text) ? text : null;
  }, [debouncedQuery]);

  const visible = useMemo(
    () => filterMergeSearchResults(results).slice(0, 6),
    [results],
  );
  const exactAlreadyListed =
    typedRepoId !== null && visible.some((row) => row.id === typedRepoId);

  useEffect(() => {
    if (value) setQuery("");
  }, [value]);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between gap-2">
        <span className="text-ui-11 font-medium uppercase tracking-[0.05em] text-muted-foreground/70">
          {label}
        </span>
        <div className="flex gap-1 rounded-full border border-border/60 p-0.5">
          {(["hub", "local"] as const).map((option) => (
            <button
              key={option}
              type="button"
              disabled={disabled}
              onClick={() => setMode(option)}
              className={cn(
                "rounded-full px-2 py-0.5 text-ui-10 transition-colors disabled:opacity-50",
                mode === option
                  ? "bg-foreground text-background"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              {option === "hub" ? "Hugging Face" : "Local path"}
            </button>
          ))}
        </div>
      </div>

      {mode === "hub" ? (
        <div className="flex flex-col gap-2">
          <div className="relative">
            <HugeiconsIcon
              icon={Search01Icon}
              className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted-foreground"
            />
            <Input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search the Hub, or paste org/model"
              disabled={disabled}
              className="h-9 pl-8 font-mono text-ui-12"
            />
            {isLoading ? (
              <Spinner className="absolute top-1/2 right-2.5 size-3.5 -translate-y-1/2" />
            ) : null}
          </div>

          {value ? (
            <div className="flex items-center gap-2 rounded-lg border border-border/60 bg-muted/20 px-2.5 py-1.5">
              <span className="truncate font-mono text-ui-11 text-foreground">
                {value}
              </span>
              <button
                type="button"
                onClick={() => onChange("")}
                className="ml-auto text-ui-10 text-muted-foreground hover:text-foreground"
              >
                Clear
              </button>
            </div>
          ) : null}

          {typedRepoId && !exactAlreadyListed ? (
            <button
              type="button"
              onClick={() => onChange(typedRepoId)}
              className="flex items-center justify-between rounded-lg border border-dashed border-border/70 px-2.5 py-1.5 text-left text-ui-11 hover:border-border"
            >
              <span className="truncate font-mono">{typedRepoId}</span>
              <span className="text-ui-10 text-muted-foreground">
                use this repo
              </span>
            </button>
          ) : null}

          {visible.length > 0 ? (
            <ul className="flex max-h-52 flex-col overflow-y-auto rounded-lg border border-border/60">
              {visible.map((row) => (
                <li key={row.id}>
                  <button
                    type="button"
                    onClick={() => onChange(row.id)}
                    className={cn(
                      "flex w-full items-center justify-between gap-3 px-2.5 py-1.5 text-left hover:bg-muted/40",
                      row.id === value && "bg-muted/50",
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
          {!accessToken && searching ? (
            <p className="text-ui-10 text-muted-foreground/70">
              Public repos only. Add a Hugging Face token in Settings for gated
              or private ones.
            </p>
          ) : null}
        </div>
      ) : (
        <Input
          value={value}
          onChange={(event) => onChange(event.target.value)}
          placeholder={placeholder}
          disabled={disabled}
          className="h-9 font-mono text-ui-12"
        />
      )}
    </div>
  );
}
