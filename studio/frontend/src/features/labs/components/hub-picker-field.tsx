// SPDX-License-Identifier: NRC-3
// Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-3.md

import { PathInput } from "@/components/path-input";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
import { Spinner } from "@/components/ui/spinner";
import {
  hfApiToken,
  useHfTokenStore,
  useHubDatasetSearch,
  useHubModelSearch,
} from "@/features/hub";
import { useDebouncedValue } from "@/hooks";
import { useT } from "@/i18n";
import { canPickLocalPath } from "@/lib/native-files";
import { cn } from "@/lib/utils";
import { Search01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useEffect, useMemo, useRef, useState } from "react";
import { parseRepoId } from "./hub-repo-id";

export type PickerMode = "hub" | "local";

export function HubPickerField({
  kind,
  value,
  onChange,
  mode,
  onModeChange,
  placeholder,
  disabled = false,
  extensions,
  commitLocalOnBlur = false,
}: {
  kind: "model" | "dataset";
  value: string;
  onChange: (value: string) => void;
  mode: PickerMode;
  onModeChange: (mode: PickerMode) => void;
  placeholder: string;
  disabled?: boolean;
  extensions?: string[];
  /** Holds a typed path until Enter or blur, for a control that collects one value. */
  commitLocalOnBlur?: boolean;
}): ReactElement {
  const t = useT();
  const anchorRef = useRef<HTMLDivElement>(null);
  const selectingRef = useRef(false);
  const [inputValue, setInputValue] = useState(value);
  const [rejected, setRejected] = useState(false);
  const debouncedQuery = useDebouncedValue(inputValue, 350);
  const hfToken = useHfTokenStore((state) => state.token);
  const accessToken = hfApiToken(hfToken);

  useEffect(() => {
    setInputValue(value);
  }, [value]);

  const commitInput = () => {
    const next = inputValue.trim();
    if (next !== value) onChange(next);
  };

  const commitRepo = () => {
    const text = inputValue.trim();
    if (!text) {
      setRejected(false);
      return;
    }
    const repoId = parseRepoId(text);
    if (!repoId) {
      setRejected(true);
      return;
    }
    setRejected(false);
    setInputValue(repoId);
    if (repoId !== value) onChange(repoId);
  };

  const query = debouncedQuery.trim();
  const searching = mode === "hub" && query.length > 0 && !disabled;

  const modelSearch = useHubModelSearch(query, {
    accessToken,
    excludeGguf: true,
    ownerScope: "all",
    enabled: searching && kind === "model",
  });
  const datasetSearch = useHubDatasetSearch(query, {
    accessToken,
    enabled: searching && kind === "dataset",
  });

  const active = kind === "model" ? modelSearch : datasetSearch;
  const results = active.results;
  const isLoading = active.isLoading;
  const error = active.error;

  const items = useMemo(() => {
    const ids = results.map((row) => row.id).filter((id) => id !== value);
    const typed = parseRepoId(query);
    const merged = typed ? [typed, ...ids] : ids;
    return merged.filter((id, index) => merged.indexOf(id) === index);
  }, [results, query, value]);

  return (
    <div className="flex min-w-0 flex-col gap-1.5">
      {mode === "hub" ? (
        <div
          ref={anchorRef}
          className="min-w-0"
          onBlur={(event) => {
            if (event.currentTarget.contains(event.relatedTarget as Node)) {
              return;
            }
            commitRepo();
          }}
        >
          <Combobox
            items={items}
            filteredItems={items}
            filter={null}
            value={value.trim() ? value : null}
            onValueChange={(next) => {
              const picked = (next ?? "").trim();
              if (!picked) {
                setRejected(false);
                onChange("");
                setInputValue("");
                return;
              }
              const repoId = parseRepoId(picked);
              if (!repoId) {
                setRejected(true);
                return;
              }
              setRejected(false);
              onChange(repoId);
              setInputValue(repoId);
            }}
            onInputValueChange={(next) => {
              if (selectingRef.current) {
                selectingRef.current = false;
                return;
              }
              setRejected(false);
              setInputValue(next);
            }}
            itemToStringValue={(item) => item}
            autoHighlight={true}
          >
            <ComboboxInput
              placeholder={placeholder}
              disabled={disabled}
              showClear={Boolean(value)}
              startAddon={
                <HugeiconsIcon
                  icon={Search01Icon}
                  className="size-3.5 text-muted-foreground"
                />
              }
              className="font-mono text-ui-12"
            />
            <ComboboxContent anchor={anchorRef}>
              {isLoading ? (
                <div className="flex items-center gap-2 px-2 py-3 text-ui-11 text-muted-foreground">
                  <Spinner className="size-3.5" />
                  {t("labs.picker.searching")}
                </div>
              ) : null}
              <ComboboxEmpty>
                {searching ? t("labs.picker.empty") : t("labs.picker.idle")}
              </ComboboxEmpty>
              <ComboboxList>
                {(id: string) => (
                  <ComboboxItem
                    key={id}
                    value={id}
                    onPointerDown={() => {
                      selectingRef.current = true;
                    }}
                  >
                    <span className="truncate font-mono text-ui-11">{id}</span>
                  </ComboboxItem>
                )}
              </ComboboxList>
            </ComboboxContent>
          </Combobox>
        </div>
      ) : (
        <div
          className="min-w-0"
          onKeyDown={(event) => {
            if (event.key !== "Enter" || !commitLocalOnBlur) return;
            event.preventDefault();
            commitInput();
          }}
          onBlur={(event) => {
            if (!commitLocalOnBlur) return;
            if (event.currentTarget.contains(event.relatedTarget as Node)) {
              return;
            }
            commitInput();
          }}
        >
          <PathInput
            value={commitLocalOnBlur ? inputValue : value}
            onChange={commitLocalOnBlur ? setInputValue : onChange}
            placeholder={
              kind === "model"
                ? t("labs.picker.modelPlaceholder")
                : t("labs.picker.datasetPlaceholder")
            }
            kind={kind === "model" ? "folder" : "file"}
            extensions={extensions}
            title={
              kind === "model"
                ? t("labs.picker.checkpointTitle")
                : t("labs.picker.datasetTitle")
            }
            disabled={disabled}
            inputClassName="h-9 text-ui-12"
          />
        </div>
      )}

      <div className="flex items-center gap-1.5">
        {(["hub", "local"] as const).map((option) => (
          <button
            key={option}
            type="button"
            disabled={disabled}
            aria-pressed={mode === option}
            onClick={() => onModeChange(option)}
            className={cn(
              "rounded-full px-2 py-0.5 text-ui-10 transition-colors disabled:opacity-50",
              mode === option
                ? "bg-foreground text-background"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            {option === "hub"
              ? t("labs.picker.hubMode")
              : t("labs.picker.localMode")}
          </button>
        ))}
        {mode === "hub" && rejected ? (
          <span
            className="min-w-0 text-ui-10 leading-snug text-destructive"
            title={t("labs.picker.badRepoId")}
          >
            {t("labs.picker.badRepoId")}
          </span>
        ) : mode === "hub" && error && !isLoading ? (
          <span className="truncate text-ui-10 text-destructive">
            {t("labs.picker.searchFailed")}
          </span>
        ) : null}
        {mode === "local" && !canPickLocalPath() ? (
          <span
            className="min-w-0 text-ui-10 leading-snug text-muted-foreground/70"
            title={t("labs.picker.localHint")}
          >
            {t("labs.picker.localHint")}
          </span>
        ) : null}
      </div>
    </div>
  );
}

export function HubPickerAdder({
  kind,
  placeholder,
  onAdd,
  extensions,
}: {
  kind: "model" | "dataset";
  placeholder: string;
  onAdd: (value: string, mode: PickerMode) => void;
  extensions?: string[];
}): ReactElement {
  const [generation, setGeneration] = useState(0);
  const [mode, setMode] = useState<PickerMode>("hub");

  return (
    <HubPickerField
      key={generation}
      kind={kind}
      value=""
      mode={mode}
      onModeChange={setMode}
      placeholder={placeholder}
      extensions={extensions}
      commitLocalOnBlur={true}
      onChange={(picked) => {
        if (!picked.trim()) return;
        onAdd(picked.trim(), mode);
        setGeneration((current) => current + 1);
      }}
    />
  );
}
