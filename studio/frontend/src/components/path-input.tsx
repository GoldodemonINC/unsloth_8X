// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Input } from "@/components/ui/input";
import { canPickLocalPath, pickLocalPath } from "@/lib/native-files";
import { cn } from "@/lib/utils";
import { useState } from "react";

/**
 * A path field with a native Browse button. Typing stays the default; the
 * button opens the OS dialog so nobody has to recall a directory string.
 *
 * Browsing is a desktop affordance: the web build has no filesystem to walk, so
 * the button is absent there and this stays a plain text field.
 */
export function PathInput({
  value,
  onChange,
  placeholder,
  kind = "folder",
  extensions,
  title,
  disabled = false,
  className,
  inputClassName,
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  kind?: "file" | "folder";
  /** File filters, lower case and without dots. Ignored for folders. */
  extensions?: string[];
  title?: string;
  disabled?: boolean;
  className?: string;
  inputClassName?: string;
}) {
  const [browsing, setBrowsing] = useState(false);

  const browse = async () => {
    setBrowsing(true);
    try {
      const picked = await pickLocalPath({ kind, extensions, title });
      if (picked) {
        onChange(picked);
      }
    } catch {
      // A dismissed or failed dialog leaves the field exactly as it was.
    } finally {
      setBrowsing(false);
    }
  };

  return (
    <div className={cn("flex min-w-0 items-center gap-1.5", className)}>
      <Input
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder={placeholder}
        disabled={disabled}
        className={cn("h-9 min-w-0 font-mono text-ui-12", inputClassName)}
      />
      {canPickLocalPath() ? (
        <button
          type="button"
          onClick={browse}
          disabled={disabled || browsing}
          className="h-8 shrink-0 rounded-lg border border-border/60 px-2.5 text-ui-11 text-muted-foreground transition-colors hover:border-border hover:text-foreground disabled:opacity-50"
        >
          {browsing ? "…" : "Browse"}
        </button>
      ) : null}
    </div>
  );
}
