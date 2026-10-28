// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  buildOllamaModelfile,
  buildVllmCommand,
  deployModelReference,
} from "../src/features/export/lib/deploy-scripts.ts";

const base = {
  model: "unsloth/Llama-3.2-1B",
  name: "my-model",
  contextLength: 4096,
  port: 8000,
  tensorParallel: 1,
};

test("a Hub id becomes an Ollama hub reference, a path stays a path", () => {
  assert.equal(
    deployModelReference("unsloth/Llama-3.2-1B"),
    "hf.co/unsloth/Llama-3.2-1B",
  );
  assert.equal(
    deployModelReference("C:\\models\\Llama-3.2-1B-GGUF"),
    "C:\\models\\Llama-3.2-1B-GGUF",
  );
  assert.equal(deployModelReference("  "), "");
});

test("the Modelfile names the model and the context", () => {
  assert.equal(
    buildOllamaModelfile(base),
    "FROM hf.co/unsloth/Llama-3.2-1B\nPARAMETER num_ctx 4096\n",
  );
});

test("a path with spaces is quoted once, not double-quoted", () => {
  assert.equal(
    buildOllamaModelfile({ ...base, model: "/models/my model" }),
    "FROM '/models/my model'\nPARAMETER num_ctx 4096\n",
  );
});

test("vLLM uses the current serve form and omits single-GPU parallelism", () => {
  assert.equal(
    buildVllmCommand(base),
    [
      "vllm serve unsloth/Llama-3.2-1B \\",
      "  --served-model-name my-model \\",
      "  --max-model-len 4096 \\",
      "  --port 8000",
      "",
    ].join("\n"),
  );
});

test("tensor parallel adds the flag only when it can split the model", () => {
  const command = buildVllmCommand({ ...base, tensorParallel: 2 });
  assert.match(command, /--port 8000 \\\n {2}--tensor-parallel-size 2\n$/);
});

test("numbers are clamped so a typed zero cannot produce a broken command", () => {
  const command = buildVllmCommand({
    ...base,
    contextLength: 0,
    port: 99999,
    tensorParallel: 0,
    name: " ",
  });
  assert.match(command, /--max-model-len 1 /);
  assert.match(command, /--port 65535/);
  assert.match(command, /--served-model-name unsloth-studio/);
});
