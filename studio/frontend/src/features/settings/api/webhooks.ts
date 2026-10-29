// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

type ApiSignature = {
  algorithm: string;
  header: string;
  // biome-ignore lint/style/useNamingConvention: API schema
  signed_content: string;
};

type ApiDelivery = {
  // biome-ignore lint/style/useNamingConvention: API schema
  delivery_id: string;
  event: string;
  // biome-ignore lint/style/useNamingConvention: API schema
  run_id: string | null;
  status: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  created_at: string | null;
  attempts: number;
  ok: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  status_code: number | null;
  error: string | null;
  response: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  duration_ms: number;
  signed: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  anomaly_kind: string | null;
  test: boolean;
};

type ApiSettings = {
  enabled: boolean;
  url: string;
  // biome-ignore lint/style/useNamingConvention: API schema
  has_secret: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  secret_unavailable: boolean;
  signature: ApiSignature;
  deliveries: ApiDelivery[];
};

type ApiTestResult = {
  ok: boolean;
  url: string;
  // biome-ignore lint/style/useNamingConvention: API schema
  status_code: number | null;
  attempts: number;
  error: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  duration_ms: number;
  signed: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  signature_header: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  signature_verified: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  delivery_id: string;
};

export type WebhookDelivery = {
  deliveryId: string;
  event: string;
  runId: string | null;
  status: string | null;
  createdAt: string | null;
  attempts: number;
  ok: boolean;
  statusCode: number | null;
  error: string | null;
  durationMs: number;
  signed: boolean;
  anomalyKind: string | null;
  test: boolean;
};

export type WebhookSettings = {
  enabled: boolean;
  url: string;
  hasSecret: boolean;
  secretUnavailable: boolean;
  signature: {
    algorithm: string;
    header: string;
    signedContent: string;
  };
  deliveries: WebhookDelivery[];
};

export type WebhookTestResult = {
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

function fromDelivery(delivery: ApiDelivery): WebhookDelivery {
  return {
    deliveryId: delivery.delivery_id,
    event: delivery.event,
    runId: delivery.run_id,
    status: delivery.status,
    createdAt: delivery.created_at,
    attempts: delivery.attempts,
    ok: delivery.ok,
    statusCode: delivery.status_code,
    error: delivery.error,
    durationMs: delivery.duration_ms,
    signed: delivery.signed,
    anomalyKind: delivery.anomaly_kind,
    test: delivery.test,
  };
}

function fromSettings(settings: ApiSettings): WebhookSettings {
  return {
    enabled: settings.enabled,
    url: settings.url,
    hasSecret: settings.has_secret,
    secretUnavailable: settings.secret_unavailable,
    signature: {
      algorithm: settings.signature.algorithm,
      header: settings.signature.header,
      signedContent: settings.signature.signed_content,
    },
    deliveries: settings.deliveries.map(fromDelivery),
  };
}

function fromTestResult(result: ApiTestResult): WebhookTestResult {
  return {
    ok: result.ok,
    url: result.url,
    statusCode: result.status_code,
    attempts: result.attempts,
    error: result.error,
    durationMs: result.duration_ms,
    signed: result.signed,
    signatureHeader: result.signature_header,
    signatureVerified: result.signature_verified,
    deliveryId: result.delivery_id,
  };
}

export async function loadWebhookSettings(): Promise<WebhookSettings> {
  const response = await authFetch("/api/settings/webhooks");
  if (!response.ok) {
    throw new Error(
      await readFastApiError(response, "Failed to load webhook settings"),
    );
  }
  return fromSettings((await response.json()) as ApiSettings);
}

export async function saveWebhookSettings(input: {
  enabled: boolean;
  url: string;
  secret?: string;
}): Promise<WebhookSettings> {
  const response = await authFetch("/api/settings/webhooks", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!response.ok) {
    throw new Error(
      await readFastApiError(response, "Failed to save webhook settings"),
    );
  }
  return fromSettings((await response.json()) as ApiSettings);
}

export async function sendWebhookTest(input: {
  url?: string;
  secret?: string;
}): Promise<WebhookTestResult> {
  const response = await authFetch("/api/settings/webhooks/test", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!response.ok) {
    throw new Error(
      await readFastApiError(response, "Failed to send the test event"),
    );
  }
  return fromTestResult((await response.json()) as ApiTestResult);
}
