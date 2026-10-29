// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { checkWebhookUrl } from "../src/features/settings/lib/webhook-url.ts";

test("accepts the endpoints the backend accepts", () => {
  for (const url of [
    "http://example.com/hook",
    "https://example.com/hooks/unsloth?run=1",
    "http://127.0.0.1:8080/webhook",
    "https://[::1]:9443/hook",
  ]) {
    assert.equal(checkWebhookUrl(url), null, url);
  }
});

test("pins every reason a webhook endpoint is refused", () => {
  const cases: [string, string][] = [
    ["", "empty"],
    ["   ", "empty"],
    ["ftp://example.com/hook", "scheme"],
    ["file:///etc/passwd", "scheme"],
    ["javascript:alert(1)", "scheme"],
    ["example.com/hook", "scheme"],
    ["https://", "malformed"],
    ["https://user:secret@example.com/hook", "credentials"],
    ["http://example.com:notaport/hook", "malformed"],
    ["http://example.com/hook with space", "spaces"],
  ];
  for (const [url, problem] of cases) {
    assert.equal(checkWebhookUrl(url), problem, url);
  }
});
