// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Link, useRouterState } from "@tanstack/react-router";
import { cn } from "@/lib/utils";
import type { ReactElement } from "react";

const MODES = [
  { to: "/studio", label: "Training Studio", match: "/studio" },
  { to: "/merge", label: "Merging Studio", match: "/merge" },
  { to: "/export", label: "Export / Splitter", match: "/export" },
] as const;

export function StudioModeNav(): ReactElement {
  const pathname = useRouterState({ select: (state) => state.location.pathname });
  return (
    <nav
      aria-label="Studio mode"
      className="flex flex-wrap items-center gap-1 rounded-full border border-border/60 bg-card/60 p-1"
    >
      {MODES.map((mode) => {
        const active = pathname === mode.match || pathname.startsWith(`${mode.match}/`);
        return (
          <Link
            key={mode.to}
            to={mode.to}
            className={cn(
              "rounded-full px-3.5 py-1.5 text-ui-12 transition-colors",
              active
                ? "bg-foreground text-background"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            {mode.label}
          </Link>
        );
      })}
    </nav>
  );
}
