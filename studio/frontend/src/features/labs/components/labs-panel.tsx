// SPDX-License-Identifier: NRC-3
// Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-3.md

import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { useT } from "@/i18n";
import { Refresh01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ReactElement, ReactNode } from "react";

/**
 * How a Labs path field degrades once the panel is narrower than its
 * two-column layout: the Browse button stacks under the input instead of
 * squeezing it. The button is a ~60px sibling, so sharing the row leaves the
 * field ~110px of typing room at the narrowest window. Owned here so every
 * Labs tab degrades the same way.
 */
export const LABS_PATH_INPUT_CLASS = "@max-2xl/train-card:flex-wrap";

/**
 * The one shell every Labs tab renders into: a card, a title and description,
 * and the same refresh control in the same corner. Owns the refresh button so a
 * tab cannot quietly place it somewhere else or leave it icon-only; a tab with
 * nothing to re-read (CMT writes a plan, it does not list one) passes no
 * `onRefresh` and gets no button at all.
 */
export function LabsPanel({
  title,
  description,
  onRefresh,
  refreshing = false,
  refreshDisabled = false,
  children,
}: {
  title: string;
  description?: string;
  onRefresh?: () => void;
  refreshing?: boolean;
  refreshDisabled?: boolean;
  children: ReactNode;
}): ReactElement {
  const t = useT();
  return (
    <section className="elevated-card flex flex-col gap-4 p-5">
      {/* The action shares the title's line at every width and the description
          flows under both, so a wrapped header cannot push Refresh down into
          the copy and leave one tab reading differently from the rest. The
          title wraps instead of truncating: at the narrowest window the card
          is ~190px wide and "Benchmark studio" would otherwise clip mid-word. */}
      <div className="flex flex-col gap-1">
        <div className="flex items-center justify-between gap-3">
          <h2 className="min-w-0 text-ui-14 font-semibold text-foreground">
            {title}
          </h2>
          {onRefresh ? (
            <Button
              size="sm"
              variant="ghost"
              className="shrink-0"
              onClick={onRefresh}
              disabled={refreshing || refreshDisabled}
            >
              {refreshing ? (
                <Spinner />
              ) : (
                <HugeiconsIcon icon={Refresh01Icon} className="size-3.5" />
              )}
              {t("labs.refresh")}
            </Button>
          ) : null}
        </div>
        {description ? (
          <p className="max-w-3xl text-ui-11 leading-relaxed text-muted-foreground">
            {description}
          </p>
        ) : null}
      </div>
      {children}
    </section>
  );
}
