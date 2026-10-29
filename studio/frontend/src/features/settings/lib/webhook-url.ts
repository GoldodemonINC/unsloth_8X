// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type WebhookUrlProblem =
  | "empty"
  | "scheme"
  | "host"
  | "credentials"
  | "spaces"
  | "malformed";

const WHITESPACE = /\s/u;

function hasSpaceOrControlCharacter(value: string): boolean {
  for (const character of value) {
    const point = character.codePointAt(0) ?? 0;
    if (WHITESPACE.test(character) || point < 0x20 || point === 0x7f) {
      return true;
    }
  }
  return false;
}

export function checkWebhookUrl(value: string): WebhookUrlProblem | null {
  const candidate = value.trim();
  if (!candidate) {
    return "empty";
  }
  if (hasSpaceOrControlCharacter(candidate)) {
    return "spaces";
  }

  let parsed: URL;
  try {
    parsed = new URL(candidate);
  } catch {
    return candidate.includes("://") ? "malformed" : "scheme";
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    return "scheme";
  }
  if (!parsed.hostname) {
    return "host";
  }
  if (parsed.username || parsed.password) {
    return "credentials";
  }
  return null;
}
