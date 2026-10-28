// SPDX-License-Identifier: NRC-2
// Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-2.md

import assert from "node:assert/strict";
import test from "node:test";
import type * as Field from "../src/features/labs/components/hub-picker-field.tsx";
import { parseRepoId } from "../src/features/labs/components/hub-repo-id.ts";
import { loadWithStubs, stubJsxRuntime } from "./helpers/module-stubs.ts";

const FIELD_URL = new URL(
  "../src/features/labs/components/hub-picker-field.tsx",
  import.meta.url,
);

type Element = { type: unknown; props?: Record<string, unknown> };

function* walk(node: unknown): Generator<Element> {
  if (Array.isArray(node)) {
    for (const child of node) yield* walk(child);
    return;
  }
  if (!node || typeof node !== "object") return;
  const element = node as Element;
  yield element;
  yield* walk(element.props?.children);
}

/**
 * Drives the shipped component with a React this test steps by hand, so the
 * blur path runs its real handler rather than a copy of it.
 */
function driveField(value: string) {
  const changes: string[] = [];
  const states: unknown[] = [];
  let cursor = 0;

  const react = {
    useState: (initial: unknown) => {
      const index = cursor++;
      if (states.length <= index) states[index] = initial;
      return [
        states[index],
        (next: unknown) => {
          states[index] =
            typeof next === "function"
              ? (next as (current: unknown) => unknown)(states[index])
              : next;
        },
      ];
    },
    useRef: (initial: unknown) => ({ current: initial }),
    useMemo: (factory: () => unknown) => factory(),
    useCallback: (fn: unknown) => fn,
    useEffect: () => undefined,
  };

  const field = loadWithStubs<typeof Field>(
    FIELD_URL,
    {
      react,
      "react/jsx-runtime": stubJsxRuntime(),
      "@/components/path-input": { PathInput: "PathInput" },
      "@/components/ui/combobox": {
        Combobox: "Combobox",
        ComboboxContent: "ComboboxContent",
        ComboboxEmpty: "ComboboxEmpty",
        ComboboxInput: "ComboboxInput",
        ComboboxItem: "ComboboxItem",
        ComboboxList: "ComboboxList",
      },
      "@/components/ui/spinner": { Spinner: "Spinner" },
      "@/features/hub": {
        hfApiToken: () => null,
        useHfTokenStore: () => null,
        useHubDatasetSearch: () => ({
          results: [],
          isLoading: false,
          error: null,
        }),
        useHubModelSearch: () => ({
          results: [],
          isLoading: false,
          error: null,
        }),
      },
      "@/hooks": { useDebouncedValue: (next: string) => next },
      "@/i18n": { useT: () => (key: string) => key },
      "@/lib/native-files": { canPickLocalPath: () => false },
      "@/lib/utils": {
        cn: (...parts: unknown[]) => parts.filter(Boolean).join(" "),
      },
      "@hugeicons/core-free-icons": { Search01Icon: "SearchIcon" },
      "@hugeicons/react": { HugeiconsIcon: "HugeiconsIcon" },
    },
    { relativePassthrough: true },
  );

  const render = () => {
    cursor = 0;
    return field.HubPickerField({
      kind: "dataset",
      value,
      onChange: (next: string) => changes.push(next),
      mode: "hub",
      onModeChange: () => undefined,
      placeholder: "Search the Hub, or paste org/dataset",
    });
  };

  const type = (text: string) => {
    const tree = render();
    const box = [...walk(tree)].find(
      (element) =>
        element.type === "Combobox" &&
        typeof element.props?.onInputValueChange === "function",
    );
    assert.ok(box, "the Hub mode renders a combobox");
    (box.props?.onInputValueChange as (next: string) => void)(text);
  };

  const blur = () => {
    const tree = render();
    const wrapper = [...walk(tree)].find(
      (element) =>
        element.props?.className === "min-w-0" &&
        typeof element.props?.onBlur === "function",
    );
    assert.ok(wrapper, "the Hub mode renders a blur wrapper");
    (wrapper.props?.onBlur as (event: unknown) => void)({
      currentTarget: { contains: () => false },
      relatedTarget: null,
    });
  };

  return { changes, type, blur };
}

for (const [input, expected] of [
  ["owner/name", "owner/name"],
  ["Unsloth/Qwen2.5-7B-Instruct", "Unsloth/Qwen2.5-7B-Instruct"],
  [" owner/name ", "owner/name"],
  ["https://huggingface.co/owner/name", "owner/name"],
  ["http://huggingface.co/owner/name", "owner/name"],
  ["https://www.huggingface.co/owner/name", "owner/name"],
  ["https://hf.co/owner/name", "owner/name"],
  ["https://huggingface.co/datasets/owner/name", "owner/name"],
  ["https://huggingface.co/models/owner/name", "owner/name"],
  ["https://huggingface.co/datasets/owner/name/tree/main", "owner/name"],
  ["https://huggingface.co/owner/name/", "owner/name"],
  ["https://huggingface.co/owner/name?tab=files", "owner/name"],
  ["https://huggingface.co/owner/name#readme", "owner/name"],
  ["https://huggingface.co/owner/name.git", "owner/name"],
  ["owner/name.git", "owner/name"],
] as const) {
  test(`normalizes ${JSON.stringify(input)} to ${expected}`, () => {
    assert.equal(parseRepoId(input), expected);
  });
}

for (const input of [
  "",
  "   ",
  "owner",
  "owner/",
  "/name",
  "not a repo id",
  "owner/name/extra",
  "owner/na me",
  "https://huggingface.co/owner",
  "https://huggingface.co/owner/name/blob/main/train.jsonl",
  "https://huggingface.co/datasets/owner/name/resolve/main/train.parquet",
  "https://example.com/owner/name",
] as const) {
  test(`refuses ${JSON.stringify(input)}`, () => {
    assert.equal(parseRepoId(input), null);
  });
}

test("a pasted page URL lands as its repo id", () => {
  const view = driveField("");
  view.type("https://huggingface.co/datasets/owner/name/tree/main");
  view.blur();
  assert.deepEqual(view.changes, ["owner/name"]);
});

test("a refused value never replaces the last valid one", () => {
  const view = driveField("");
  view.type("owner/name");
  view.blur();
  assert.deepEqual(view.changes, ["owner/name"]);

  view.type("not a repo id");
  view.blur();
  view.type("https://huggingface.co/owner/name/blob/main/train.jsonl");
  view.blur();
  assert.deepEqual(view.changes, ["owner/name"]);
});

test("an empty field commits nothing", () => {
  const view = driveField("");
  view.type("");
  view.blur();
  assert.deepEqual(view.changes, []);
});

test("the selected value is not re-committed on blur", () => {
  const view = driveField("owner/name");
  view.type("owner/name");
  view.blur();
  assert.deepEqual(view.changes, []);
});
