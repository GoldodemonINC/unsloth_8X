// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { RecipePayload } from "@/features/recipe-studio";

const structuredOutputsJinjaUrl = new URL(
  "./structured-outputs-jinja.json",
  import.meta.url,
).href;
const pdfGroundedQaUrl = new URL("./pdf-grounded-qa.json", import.meta.url)
  .href;
const instructionFromAnswerUrl = new URL(
  "./instruction-from-answer.json",
  import.meta.url,
).href;
const textToPythonUrl = new URL("./text-to-python.json", import.meta.url).href;
const textToSqlUrl = new URL("./text-to-sql.json", import.meta.url).href;
const ocrDocumentExtractionUrl = new URL(
  "./ocr-document-extraction.json",
  import.meta.url,
).href;
const githubSupportBotUrl = new URL(
  "./github-support-bot.json",
  import.meta.url,
).href;
const reasoningDistillUrl = new URL("./reasoning-distill.json", import.meta.url)
  .href;
const codeFromTextUrl = new URL("./code-from-text.json", import.meta.url).href;
const toolCallTracesUrl = new URL("./tool-call-traces.json", import.meta.url)
  .href;
const critiqueReviseUrl = new URL("./critique-revise.json", import.meta.url)
  .href;

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function toRecordArray(value: unknown): Record<string, unknown>[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((item): item is Record<string, unknown> =>
    isRecord(item),
  );
}

function coerceRecipePayload(value: unknown): RecipePayload {
  if (!isRecord(value)) {
    throw new Error("Template payload is invalid JSON object.");
  }

  const recipeSource = isRecord(value.recipe) ? value.recipe : value;
  if (!Array.isArray(recipeSource.columns)) {
    throw new Error("Template payload must include recipe.columns.");
  }

  if (isRecord(value.recipe) && isRecord(value.run) && isRecord(value.ui)) {
    return value as unknown as RecipePayload;
  }

  const recipe: RecipePayload["recipe"] = {
    // biome-ignore lint/style/useNamingConvention: api schema
    model_providers: toRecordArray(recipeSource.model_providers),
    // biome-ignore lint/style/useNamingConvention: api schema
    mcp_providers: toRecordArray(recipeSource.mcp_providers),
    // biome-ignore lint/style/useNamingConvention: api schema
    model_configs: toRecordArray(recipeSource.model_configs),
    // biome-ignore lint/style/useNamingConvention: api schema
    seed_config: isRecord(recipeSource.seed_config)
      ? recipeSource.seed_config
      : undefined,
    // biome-ignore lint/style/useNamingConvention: api schema
    tool_configs: toRecordArray(recipeSource.tool_configs),
    columns: toRecordArray(recipeSource.columns),
    processors: toRecordArray(recipeSource.processors),
  };

  return {
    recipe,
    run: {
      rows: 5,
      preview: true,
      // biome-ignore lint/style/useNamingConvention: api schema
      output_formats: ["jsonl"],
    },
    ui: {
      nodes: [],
      edges: [],
    },
  };
}

async function loadPayloadFromUrl(url: string): Promise<RecipePayload> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch template payload (${response.status})`);
  }
  const json = (await response.json()) as unknown;
  return coerceRecipePayload(json);
}

export type LearningRecipeDef = {
  id: string;
  title: string;
  description: string;
  loadPayload: () => Promise<RecipePayload>;
};

export const LEARNING_RECIPES: LearningRecipeDef[] = [
  {
    id: "structured-outputs-jinja",
    title: "Structured Outputs + Jinja Expressions",
    description:
      "Support ticket triage with structured JSON outputs and Jinja conditionals.",
    loadPayload: () => loadPayloadFromUrl(structuredOutputsJinjaUrl),
  },
  {
    id: "pdf-grounded-qa",
    title: "PDF Document QA",
    description: "Build grounded question-answer examples from PDF chunks.",
    loadPayload: () => loadPayloadFromUrl(pdfGroundedQaUrl),
  },
  {
    id: "instruction-from-answer",
    title: "Instruction from Answer",
    description:
      "Use seed answer columns to generate high-quality instruction targets.",
    loadPayload: () => loadPayloadFromUrl(instructionFromAnswerUrl),
  },
  {
    id: "text-to-python",
    title: "Text to Python",
    description:
      "Generate instruction-to-code data with category sampling and LLM judging.",
    loadPayload: () => loadPayloadFromUrl(textToPythonUrl),
  },
  {
    id: "text-to-sql",
    title: "Text to SQL",
    description:
      "Generate SQL tasks and runnable SQL outputs with prompt-driven generation.",
    loadPayload: () => loadPayloadFromUrl(textToSqlUrl),
  },
  {
    id: "ocr-document-extraction",
    title: "OCR Document Extraction",
    description:
      "Use image context to generate OCR-style document extraction output.",
    loadPayload: () => loadPayloadFromUrl(ocrDocumentExtractionUrl),
  },
  {
    id: "github-support-bot",
    title: "GitHub Crawler",
    description:
      "Crawl real GitHub issues and PRs and turn each thread into a {User, Assistant} training pair.",
    loadPayload: () => loadPayloadFromUrl(githubSupportBotUrl),
  },
  {
    id: "reasoning-distill",
    title: "Reasoning Distill",
    description:
      "Distil a strong model's step-by-step traces into reasoning + answer rows, kept only where a judge passes them.",
    loadPayload: () => loadPayloadFromUrl(reasoningDistillUrl),
  },
  {
    id: "code-from-text",
    title: "Code Dataset from Text",
    description:
      "Upload documents and turn each chunk into a coding task, a checked Python answer, and a faithfulness score.",
    loadPayload: () => loadPayloadFromUrl(codeFromTextUrl),
  },
  {
    id: "tool-call-traces",
    title: "Tool-Call Traces",
    description:
      "Invent a typed tool schema per domain, then generate the request and the single valid call it needs.",
    loadPayload: () => loadPayloadFromUrl(toolCallTracesUrl),
  },
  {
    id: "critique-revise",
    title: "Critique and Revise",
    description:
      "Generate {instruction, draft, chosen} triples for DPO or GRPO, plus the critique that explains the fix.",
    loadPayload: () => loadPayloadFromUrl(critiqueReviseUrl),
  },
];
