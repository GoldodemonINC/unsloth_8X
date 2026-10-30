// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { checkWebhookUrl } from "../src/features/settings/lib/webhook-url.ts";
import { en } from "../src/i18n/locales/en.ts";
import {
  type StubElement,
  loadWithStubs,
  stubJsxRuntime,
} from "./helpers/module-stubs.ts";

const cardUrl = new URL(
  "../src/features/settings/components/webhook-settings-section.tsx",
  import.meta.url,
);
const tick = () => new Promise<void>((resolve) => setImmediate(resolve));

type Delivery = {
  deliveryId: string;
  event: string;
  runId: string | null;
  status: string | null;
  createdAt: string | null;
  attempts: number;
  ok: boolean;
  statusCode: number | null;
  error: string | null;
  failureKind: string | null;
  durationMs: number;
  signed: boolean;
  anomalyKind: string | null;
  test: boolean;
};

type Settings = {
  enabled: boolean;
  url: string;
  hasSecret: boolean;
  secretUnavailable: boolean;
  signature: {
    algorithm: string;
    header: string;
    signedContent: string;
  };
  deliveries: Delivery[];
};

type TestResult = {
  ok: boolean;
  url: string;
  statusCode: number | null;
  attempts: number;
  error: string | null;
  durationMs: number;
  signed: boolean;
  signatureHeader: string | null;
  signatureVerified: boolean;
  deliveryId: string;
};

const componentStub = new Proxy(
  {},
  { get: (_target, property) => String(property) },
);

function translate(key: string, values: Record<string, string | number> = {}) {
  const message = key
    .split(".")
    .reduce<unknown>(
      (value, part) => (value as Record<string, unknown>)[part],
      en,
    ) as string;
  return message.replace(/\{([^}]+)\}/g, (_match, name: string) =>
    String(values[name] ?? ""),
  );
}

function expand(node: unknown): unknown {
  if (Array.isArray(node)) {
    return node.map(expand);
  }
  if (!node || typeof node !== "object" || !("props" in node)) {
    return node;
  }
  const element = node as StubElement;
  if (typeof element.type === "function") {
    return expand(
      (element.type as (props: Record<string, unknown>) => unknown)(
        element.props,
      ),
    );
  }
  return {
    ...element,
    props: {
      ...element.props,
      children: expand(element.props.children),
      below: expand(element.props.below),
    },
  };
}

function nodes(node: unknown): StubElement[] {
  const expanded = expand(node);
  if (Array.isArray(expanded)) {
    return expanded.flatMap(nodes);
  }
  if (!expanded || typeof expanded !== "object" || !("props" in expanded)) {
    return [];
  }
  const element = expanded as StubElement;
  return [element, ...nodes(element.props.children)];
}

function content(node: unknown): string {
  const expanded = expand(node);
  if (typeof expanded === "string" || typeof expanded === "number") {
    return String(expanded);
  }
  if (Array.isArray(expanded)) {
    return expanded.map(content).join(" ");
  }
  if (expanded && typeof expanded === "object" && "props" in expanded) {
    const props = (expanded as StubElement).props;
    return [
      content(props.label),
      content(props.title),
      content(props.description),
      content(props.children),
      content(props.below),
    ]
      .filter(Boolean)
      .join(" ");
  }
  return "";
}

function sameDeps(left: unknown[] | undefined, right: unknown[] | undefined) {
  return (
    left !== undefined &&
    right !== undefined &&
    left.length === right.length &&
    left.every((value, index) => Object.is(value, right[index]))
  );
}

type EffectSlot = {
  cleanup?: () => void;
  deps?: unknown[];
  next?: () => undefined | (() => void);
};

function start(initial: Settings, result: TestResult) {
  const states: unknown[] = [];
  const callbacks: { deps?: unknown[]; fn: (...args: never[]) => unknown }[] =
    [];
  const effects: EffectSlot[] = [];
  let stateCursor = 0;
  let callbackCursor = 0;
  let effectCursor = 0;
  let remote = initial;
  let nextResult = result;
  const saves: { enabled: boolean; url: string; secret?: string }[] = [];
  const tests: { url?: string; secret?: string }[] = [];

  const setState = (index: number, value: unknown) => {
    states[index] =
      typeof value === "function"
        ? (value as (current: unknown) => unknown)(states[index])
        : value;
  };

  const api = loadWithStubs<{ WebhookSettingsSection: () => StubElement }>(
    cardUrl,
    {
      "react/jsx-runtime": stubJsxRuntime(),
      react: {
        useState(initialValue: unknown) {
          const index = stateCursor++;
          if (!(index in states)) {
            states[index] = initialValue;
          }
          return [states[index], (value: unknown) => setState(index, value)];
        },
        useCallback(fn: (...args: never[]) => unknown, deps?: unknown[]) {
          const index = callbackCursor++;
          const slot = callbacks[index];
          if (!slot || !sameDeps(slot.deps, deps)) {
            callbacks[index] = { deps, fn };
          }
          return callbacks[index].fn;
        },
        useEffect(effect: () => undefined | (() => void), deps?: unknown[]) {
          const index = effectCursor++;
          const slot = effects[index] ?? {};
          if (!sameDeps(slot.deps, deps)) {
            slot.next = effect;
            slot.deps = deps;
          }
          effects[index] = slot;
        },
      },
      "@/components/ui/badge": componentStub,
      "@/components/ui/button": componentStub,
      "@/components/ui/checkbox": componentStub,
      "@/components/ui/input": componentStub,
      "@/components/ui/switch": componentStub,
      "@/i18n": { useT: () => translate },
      "@/lib/toast": {
        toast: { success: () => undefined, error: () => undefined },
      },
      "../api/webhooks": {
        loadWebhookSettings: () => Promise.resolve(remote),
        saveWebhookSettings: (input: {
          enabled: boolean;
          url: string;
          secret?: string;
        }) => {
          saves.push(input);
          remote = {
            ...remote,
            enabled: input.enabled,
            url: input.url,
            ...(input.secret === undefined
              ? {}
              : {
                  hasSecret: input.secret.length > 0,
                  secretUnavailable: false,
                }),
          };
          return remote;
        },
        sendWebhookTest: (input: { url?: string; secret?: string }) => {
          tests.push(input);
          return nextResult;
        },
      },
      "../lib/webhook-url": { checkWebhookUrl },
      "./settings-row": componentStub,
      "./settings-section": componentStub,
    },
  );

  const flushEffects = () => {
    for (const slot of effects) {
      if (!slot.next) {
        continue;
      }
      slot.cleanup?.();
      const effect = slot.next;
      slot.next = undefined;
      slot.cleanup = effect() ?? undefined;
    }
  };

  const render = () => {
    stateCursor = 0;
    callbackCursor = 0;
    effectCursor = 0;
    return api.WebhookSettingsSection();
  };

  const settle = async () => {
    render();
    flushEffects();
    await tick();
    await tick();
    return render();
  };

  return {
    render,
    settle,
    saves,
    setResult(next: TestResult) {
      nextResult = next;
    },
    tests,
  };
}

function button(tree: StubElement, label: string) {
  const found = nodes(tree).find(
    (node) => node.type === "Button" && content(node).includes(label),
  );
  if (!found) {
    throw new Error(`missing ${label} button`);
  }
  return found;
}

function field(tree: StubElement, predicate: (node: StubElement) => boolean) {
  const found = nodes(tree).find(
    (node) => node.type === "Input" && predicate(node),
  );
  if (!found) {
    throw new Error("missing input");
  }
  return found;
}

function change(node: StubElement, value: string) {
  const handler = node.props.onChange as (event: {
    target: { value: string };
  }) => void;
  handler({ target: { value } });
}

function click(node: StubElement) {
  const handler = node.props.onClick as () => void;
  handler();
}

function settings(overrides: Partial<Settings> = {}): Settings {
  return {
    enabled: true,
    url: "http://127.0.0.1:9100/training-hook",
    hasSecret: false,
    secretUnavailable: false,
    signature: {
      algorithm: "hmac-sha256",
      header: "X-Unsloth-Signature-256",
      signedContent: "{timestamp}.{raw request body}",
    },
    deliveries: [],
    ...overrides,
  };
}

function delivery(overrides: Partial<Delivery>): Delivery {
  return {
    deliveryId: "delivery-1",
    event: "training.anomaly",
    runId: "run-1",
    status: null,
    createdAt: null,
    attempts: 1,
    ok: false,
    statusCode: null,
    error: null,
    failureKind: null,
    durationMs: 12,
    signed: false,
    anomalyKind: null,
    test: false,
    ...overrides,
  };
}

function testResult(overrides: Partial<TestResult> = {}): TestResult {
  return {
    ok: true,
    url: "http://127.0.0.1:9100/training-hook",
    statusCode: 200,
    attempts: 1,
    error: null,
    durationMs: 12,
    signed: false,
    signatureHeader: null,
    signatureVerified: false,
    deliveryId: "delivery-1",
    ...overrides,
  };
}

test("the card never calls an unreadable secret absent, and clears its result when config changes", async () => {
  const card = start(
    settings({ secretUnavailable: true }),
    testResult({ signed: false }),
  );
  let tree = await card.settle();

  assert.match(content(tree), /Saved secret can't be read/);
  assert.doesNotMatch(content(tree), /No shared secret is set/);
  click(button(tree, "Send test"));
  tree = await card.settle();
  assert.equal(card.tests.length, 1, "test request was not sent");

  assert.match(
    content(tree),
    /The saved secret can't be read, so this request is unsigned\./,
  );
  assert.doesNotMatch(
    content(tree),
    /No shared secret is set, so this request is unsigned\./,
  );

  change(
    field(tree, (node) => node.props.type === "password"),
    "fresh-secret",
  );
  tree = card.render();
  assert.doesNotMatch(content(tree), /request is unsigned/);

  click(button(tree, "Save"));
  tree = await card.settle();
  assert.match(content(tree), /Stored secret/);
  assert.doesNotMatch(content(tree), /request is unsigned/);
  assert.equal(card.saves.at(-1)?.secret, "fresh-secret");

  card.setResult(
    testResult({
      signed: true,
      signatureHeader: "sha256=abc",
      signatureVerified: true,
    }),
  );
  click(button(tree, "Send test"));
  tree = await card.settle();
  assert.match(content(tree), /Signed request/);
  assert.match(content(tree), /HTTP 200 in 12 ms · attempts: 1/);
  assert.doesNotMatch(content(tree), /Signature verified|receiver accepted/);

  click(button(tree, "Save"));
  tree = await card.settle();
  assert.doesNotMatch(content(tree), /Signed request/);
});

test("no secret and an unreadable secret get different unsigned explanations", async () => {
  const card = start(settings(), testResult({ signed: false }));
  let tree = await card.settle();

  assert.doesNotMatch(content(tree), /Saved secret can't be read/);
  assert.doesNotMatch(content(tree), /Stored secret/);

  click(button(tree, "Send test"));
  tree = await card.settle();

  assert.match(
    content(tree),
    /No shared secret is set, so this request is unsigned\./,
  );
  assert.doesNotMatch(content(tree), /The saved secret can't be read/);
});

test("changing the endpoint or secret clears a settled test result", async () => {
  const card = start(
    settings({ hasSecret: true }),
    testResult({
      signed: true,
      signatureHeader: "sha256=abc",
      signatureVerified: true,
    }),
  );
  let tree = await card.settle();

  click(button(tree, "Send test"));
  tree = await card.settle();
  assert.match(content(tree), /Signed request/);

  change(
    field(tree, (node) => node.props["aria-label"] === "Endpoint URL"),
    "http://127.0.0.1:9200/other",
  );
  tree = card.render();
  assert.doesNotMatch(content(tree), /Signed request/);

  click(button(tree, "Send test"));
  tree = await card.settle();
  assert.match(content(tree), /Signed request/);

  change(
    field(tree, (node) => node.props.type === "password"),
    "replacement",
  );
  tree = card.render();
  assert.doesNotMatch(content(tree), /Signed request/);
});

test("every failure kind renders its own wording instead of the raw error", async () => {
  const rows: [kind: string, wording: string][] = [
    ["invalid", "Invalid endpoint URL"],
    ["unresolvable", "Host name could not be resolved"],
    ["blocked", "Host resolves to a blocked address"],
    ["proxy", "Proxy intercepted the request"],
    ["certificate", "TLS certificate rejected"],
    ["http", "Server returned an error status"],
  ];
  const card = start(
    settings({
      deliveries: rows.map(([kind]) =>
        delivery({
          deliveryId: `delivery-${kind}`,
          failureKind: kind,
          error: `raw ${kind} message`,
        }),
      ),
    }),
    testResult(),
  );
  const tree = await card.settle();

  for (const [kind, wording] of rows) {
    assert.match(content(tree), new RegExp(wording), kind);
    assert.doesNotMatch(content(tree), new RegExp(`raw ${kind} message`), kind);
  }
});

test("transport gets its wording, and an unknown or absent kind falls back to the raw error", async () => {
  const card = start(
    settings({
      deliveries: [
        delivery({
          deliveryId: "delivery-transport",
          failureKind: "transport",
          error: "raw transport message",
        }),
        delivery({
          deliveryId: "delivery-unknown",
          failureKind: "future-kind",
          error: "raw future message",
        }),
        delivery({
          deliveryId: "delivery-absent",
          failureKind: null,
          error: "raw absent message",
        }),
        delivery({
          deliveryId: "delivery-silent",
          failureKind: null,
          error: null,
        }),
      ],
    }),
    testResult(),
  );
  const tree = await card.settle();

  assert.match(content(tree), /Connection failed or timed out/);
  assert.doesNotMatch(content(tree), /raw transport message/);
  assert.match(content(tree), /raw future message/);
  assert.match(content(tree), /raw absent message/);
  assert.match(content(tree), /Delivery failed/);
});
