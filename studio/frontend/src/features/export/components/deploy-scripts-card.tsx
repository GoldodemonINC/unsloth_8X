// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { Copy01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useMemo, useState } from "react";
import { buildOllamaModelfile, buildVllmCommand } from "../lib/deploy-scripts";

function baseName(path: string): string {
  const trimmed = path.replace(/[/\\]+$/, "");
  const parts = trimmed.split(/[/\\]/);
  return parts[parts.length - 1] || "model";
}

/**
 * Nothing here runs; both scripts are text the user copies into their own
 * shell. That is the point: the desktop app is not a server, and a deploy
 * command the user can read beats a hidden background process.
 */
export function DeployScriptsCard({
  model,
  modelName,
}: {
  model: string | null;
  modelName: string;
}): ReactElement | null {
  const t = useT();
  const [target, setTarget] = useState<"ollama" | "vllm">("ollama");
  const [contextLength, setContextLength] = useState(4096);
  const [port, setPort] = useState(8000);
  const [tensorParallel, setTensorParallel] = useState(1);
  const [copied, setCopied] = useState(false);

  const script = useMemo(() => {
    if (!model) return "";
    const options = {
      model,
      name: modelName.trim() || baseName(model),
      contextLength,
      port,
      tensorParallel,
    };
    return target === "ollama"
      ? buildOllamaModelfile(options)
      : buildVllmCommand(options);
  }, [contextLength, model, modelName, port, target, tensorParallel]);

  if (!model) return null;

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(script);
      setCopied(true);
      toast.success(t("deployScripts.copied"));
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      toast.error(t("deployScripts.copyFailed"));
    }
  };

  return (
    <section className="elevated-card flex flex-col gap-4 p-5">
      <div className="space-y-1">
        <h2 className="text-ui-14 font-semibold text-foreground">
          {t("deployScripts.title")}
        </h2>
        <p className="max-w-3xl text-ui-11 leading-relaxed text-muted-foreground">
          {t("deployScripts.description")}
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-1.5">
        {(["ollama", "vllm"] as const).map((option) => (
          <button
            key={option}
            type="button"
            aria-pressed={target === option}
            onClick={() => setTarget(option)}
            className={cn(
              "rounded-full px-3 py-1 text-ui-11 transition-colors",
              target === option
                ? "bg-foreground text-background"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            {option === "ollama"
              ? t("deployScripts.ollama")
              : t("deployScripts.vllm")}
          </button>
        ))}
      </div>

      <div className="flex flex-wrap items-end gap-3">
        <label
          htmlFor="deploy-context-length"
          className="flex w-32 flex-col gap-1 text-ui-10 text-muted-foreground"
        >
          {t("deployScripts.contextLength")}
          <Input
            id="deploy-context-length"
            type="number"
            min={1}
            value={contextLength}
            onChange={(event) =>
              setContextLength(Number(event.target.value) || 1)
            }
            className="h-8 text-ui-11"
          />
        </label>
        {target === "vllm" ? (
          <>
            <label
              htmlFor="deploy-port"
              className="flex w-24 flex-col gap-1 text-ui-10 text-muted-foreground"
            >
              {t("deployScripts.port")}
              <Input
                id="deploy-port"
                type="number"
                min={1}
                max={65535}
                value={port}
                onChange={(event) => setPort(Number(event.target.value) || 1)}
                className="h-8 text-ui-11"
              />
            </label>
            <label
              htmlFor="deploy-tensor-parallel"
              className="flex w-28 flex-col gap-1 text-ui-10 text-muted-foreground"
            >
              {t("deployScripts.tensorParallel")}
              <Input
                id="deploy-tensor-parallel"
                type="number"
                min={1}
                value={tensorParallel}
                onChange={(event) =>
                  setTensorParallel(Number(event.target.value) || 1)
                }
                className="h-8 text-ui-11"
              />
            </label>
          </>
        ) : null}
      </div>

      <div className="relative">
        <pre className="overflow-x-auto rounded-xl border border-border/60 bg-muted/30 p-3 pr-12 font-mono text-ui-11 leading-relaxed text-foreground">
          <code>{script}</code>
        </pre>
        <Button
          size="icon"
          variant="ghost"
          className="absolute right-2 top-2 size-7"
          aria-label={t("deployScripts.copy")}
          onClick={() => void copy()}
        >
          <HugeiconsIcon icon={Copy01Icon} className="size-3.5" />
        </Button>
      </div>
      <p className="text-ui-10 text-muted-foreground">
        {target === "ollama"
          ? t("deployScripts.ollamaHint")
          : t("deployScripts.vllmHint")}
      </p>
      <span aria-live="polite" className="sr-only">
        {copied ? t("deployScripts.copied") : ""}
      </span>
    </section>
  );
}
