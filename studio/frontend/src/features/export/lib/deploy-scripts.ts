// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The two files people hand-write every time they finish an export: an Ollama
 * Modelfile and a vLLM launch command. Both are plain text, so they are built
 * here rather than by a backend route: no host is touched and the preview is
 * exactly what gets copied.
 */

export interface DeployOptions {
  /** A local checkpoint / GGUF path, or a Hub repo id. */
  model: string;
  /** What the server should call the model. */
  name: string;
  contextLength: number;
  port: number;
  tensorParallel: number;
}

const REPO_ID = /^[A-Za-z0-9][A-Za-z0-9_.-]*\/[A-Za-z0-9][A-Za-z0-9_.-]*$/;

/** A local path stays a path; a bare `owner/name` becomes an Ollama Hub ref. */
export function deployModelReference(model: string): string {
  const value = model.trim();
  if (!value) return "";
  if (REPO_ID.test(value)) return `hf.co/${value}`;
  return value;
}

function quoted(value: string): string {
  return /^[A-Za-z0-9_./:@%+=,-]+$/.test(value)
    ? value
    : `'${value.replace(/'/g, "'\\''")}'`;
}

/** Ollama reads FROM, then PARAMETER lines. A Hub id needs the hf.co prefix. */
export function buildOllamaModelfile(options: DeployOptions): string {
  const reference = deployModelReference(options.model);
  const context = Math.max(1, Math.round(options.contextLength));
  return [`FROM ${quoted(reference)}`, `PARAMETER num_ctx ${context}`, ""].join(
    "\n",
  );
}

/** The current `vllm serve` form; the OpenAI-compatible server is the default. */
export function buildVllmCommand(options: DeployOptions): string {
  const reference = quoted(options.model.trim());
  const context = Math.max(1, Math.round(options.contextLength));
  const port = Math.min(65535, Math.max(1, Math.round(options.port)));
  const parallel = Math.max(1, Math.round(options.tensorParallel));
  const name = options.name.trim() || "unsloth-studio";
  const lines = [
    `vllm serve ${reference} \\`,
    `  --served-model-name ${quoted(name)} \\`,
    `  --max-model-len ${context} \\`,
    `  --port ${port}${parallel > 1 ? " \\" : ""}`,
  ];
  if (parallel > 1) {
    lines.push(`  --tensor-parallel-size ${parallel}`);
  }
  return `${lines.join("\n")}\n`;
}
