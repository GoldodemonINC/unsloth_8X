// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { type TranslationKey, useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { useCallback, useEffect, useState } from "react";
import {
  type WebhookSettings,
  type WebhookTestResult,
  loadWebhookSettings,
  saveWebhookSettings,
  sendWebhookTest,
} from "../api/webhooks";
import { type WebhookUrlProblem, checkWebhookUrl } from "../lib/webhook-url";
import { SettingsRow } from "./settings-row";
import { SettingsSection } from "./settings-section";

const PROBLEM_KEYS: Record<WebhookUrlProblem, TranslationKey> = {
  empty: "settings.general.webhooks.invalidEmpty",
  scheme: "settings.general.webhooks.invalidScheme",
  host: "settings.general.webhooks.invalidHost",
  credentials: "settings.general.webhooks.invalidCredentials",
  spaces: "settings.general.webhooks.invalidSpaces",
  malformed: "settings.general.webhooks.invalidMalformed",
};

const FAILURE_KIND_KEYS: Partial<Record<string, TranslationKey>> = {
  invalid: "settings.general.webhooks.failureInvalid",
  unresolvable: "settings.general.webhooks.failureUnresolvable",
  blocked: "settings.general.webhooks.failureBlocked",
  proxy: "settings.general.webhooks.failureProxy",
  certificate: "settings.general.webhooks.failureCertificate",
  http: "settings.general.webhooks.failureHttp",
  transport: "settings.general.webhooks.failureTransport",
};

function deliveryTime(value: string | null): string {
  if (!value) {
    return "";
  }
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function failureText(
  t: ReturnType<typeof useT>,
  delivery: WebhookSettings["deliveries"][number],
): string {
  const key = delivery.failureKind
    ? FAILURE_KIND_KEYS[delivery.failureKind]
    : undefined;
  if (key) {
    return t(key);
  }
  return delivery.error ?? t("settings.general.webhooks.failureOther");
}

function TestResultBlock({
  result,
  secretUnavailable,
}: {
  result: WebhookTestResult;
  secretUnavailable: boolean;
}) {
  const t = useT();
  const summary = result.ok
    ? t("settings.general.webhooks.testHttpResult", {
        status: result.statusCode ?? "—",
        ms: result.durationMs,
        attempts: result.attempts,
      })
    : t("settings.general.webhooks.testFailedResult", {
        attempts: result.attempts,
        error: result.error ?? "—",
      });
  let signature = secretUnavailable
    ? t("settings.general.webhooks.testUnsignedUnavailable")
    : t("settings.general.webhooks.testUnsigned");
  if (result.signed) {
    signature = result.signatureVerified
      ? t("settings.general.webhooks.testSignatureChecked")
      : t("settings.general.webhooks.testSignatureFailed");
  }
  return (
    <div className="flex flex-col gap-1 py-2 pl-[calc(2.75rem*var(--ui-space-scale,1))] text-xs">
      <span
        className={
          result.ok
            ? "text-emerald-600 dark:text-emerald-400"
            : "text-destructive"
        }
      >
        {summary}
      </span>
      <span className="text-muted-foreground">{signature}</span>
      {result.signatureHeader ? (
        <code className="break-all text-[11px] text-muted-foreground">
          {result.signatureHeader}
        </code>
      ) : null}
    </div>
  );
}

function DeliveryRow({
  delivery,
}: { delivery: WebhookSettings["deliveries"][number] }) {
  const t = useT();
  return (
    <div className="flex flex-wrap items-center gap-x-2 gap-y-1 border-t border-border/40 py-2 text-xs first:border-t-0">
      <Badge
        variant={delivery.ok ? "secondary" : "destructive"}
        className="h-4 px-1.5 text-[10px]"
      >
        {delivery.ok
          ? t("settings.general.webhooks.delivered")
          : t("settings.general.webhooks.failed")}
      </Badge>
      <span className="font-medium text-foreground">{delivery.event}</span>
      <span className="text-muted-foreground">
        {delivery.statusCode === null ? "—" : `HTTP ${delivery.statusCode}`}
      </span>
      <span className="text-muted-foreground">
        {t("settings.general.webhooks.attempts", { count: delivery.attempts })}
      </span>
      {delivery.anomalyKind ? (
        <span className="text-amber-600 dark:text-amber-400">
          {delivery.anomalyKind}
        </span>
      ) : null}
      {delivery.runId ? (
        <code className="max-w-full truncate text-[10px] text-muted-foreground">
          {delivery.runId}
        </code>
      ) : null}
      {delivery.ok ? null : (
        <span className="max-w-full truncate text-destructive">
          {failureText(t, delivery)}
        </span>
      )}
      <span className="ml-auto text-muted-foreground">
        {deliveryTime(delivery.createdAt)}
      </span>
    </div>
  );
}

function DeliveryHistory({
  deliveries,
  loading,
  onRefresh,
}: {
  deliveries: WebhookSettings["deliveries"];
  loading: boolean;
  onRefresh: () => void;
}) {
  const t = useT();
  return (
    <div className="mt-2 border-t border-border/60 pt-3">
      <div className="flex items-center justify-between gap-3">
        <h3 className="text-sm font-medium text-foreground">
          {t("settings.general.webhooks.historyTitle")}
        </h3>
        <Button
          size="xs"
          variant="ghost"
          disabled={loading}
          onClick={onRefresh}
          type="button"
        >
          {t("settings.general.webhooks.refresh")}
        </Button>
      </div>
      {deliveries.length > 0 ? (
        <div className="mt-1 flex flex-col">
          {deliveries.slice(0, 6).map((delivery) => (
            <DeliveryRow key={delivery.deliveryId} delivery={delivery} />
          ))}
        </div>
      ) : (
        <p className="py-2 text-xs text-muted-foreground">
          {t("settings.general.webhooks.historyEmpty")}
        </p>
      )}
    </div>
  );
}

export function WebhookSettingsSection() {
  const t = useT();
  const [settings, setSettings] = useState<WebhookSettings | null>(null);
  const [enabled, setEnabled] = useState(false);
  const [url, setUrl] = useState("");
  const [secret, setSecret] = useState("");
  const [clearSecret, setClearSecret] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [testResult, setTestResult] = useState<WebhookTestResult | null>(null);

  const hydrate = useCallback((next: WebhookSettings) => {
    setSettings(next);
    setEnabled(next.enabled);
    setUrl(next.url);
    setSecret("");
    setClearSecret(false);
    setTestResult(null);
  }, []);

  const refresh = useCallback(async () => {
    try {
      setSettings(await loadWebhookSettings());
      setTestResult(null);
      setError(null);
    } catch (loadError) {
      setError(
        loadError instanceof Error
          ? loadError.message
          : t("settings.general.webhooks.loadError"),
      );
    } finally {
      setLoading(false);
    }
  }, [t]);

  useEffect(() => {
    let cancelled = false;
    loadWebhookSettings()
      .then((next) => {
        if (!cancelled) {
          hydrate(next);
        }
      })
      .catch((loadError) => {
        if (!cancelled) {
          setError(
            loadError instanceof Error
              ? loadError.message
              : t("settings.general.webhooks.loadError"),
          );
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [hydrate, t]);

  const validate = (): string | null => {
    const problem = checkWebhookUrl(url);
    if (problem === "empty" && !enabled) {
      return null;
    }
    return problem ? t(PROBLEM_KEYS[problem]) : null;
  };

  const save = async () => {
    const problem = validate();
    if (problem) {
      setError(problem);
      return;
    }
    setSaving(true);
    setError(null);
    try {
      const input: { enabled: boolean; url: string; secret?: string } = {
        enabled,
        url: url.trim(),
      };
      if (clearSecret) {
        input.secret = "";
      } else if (secret.trim()) {
        input.secret = secret.trim();
      }
      hydrate(await saveWebhookSettings(input));
      toast.success(t("settings.general.webhooks.saved"));
    } catch (saveError) {
      setError(
        saveError instanceof Error
          ? saveError.message
          : t("settings.general.webhooks.saveError"),
      );
    } finally {
      setSaving(false);
    }
  };

  const test = async () => {
    const problem = checkWebhookUrl(url);
    if (problem) {
      setError(t(PROBLEM_KEYS[problem]));
      return;
    }
    setTesting(true);
    setError(null);
    setTestResult(null);
    try {
      const result = await sendWebhookTest({
        url: url.trim() || undefined,
        secret: clearSecret ? "" : secret.trim() || undefined,
      });
      setTestResult(result);
      setSettings(await loadWebhookSettings());
    } catch (testError) {
      setError(
        testError instanceof Error
          ? testError.message
          : t("settings.general.webhooks.testFailed"),
      );
    } finally {
      setTesting(false);
    }
  };

  const disabled = loading || saving || testing;

  return (
    <SettingsSection
      title={t("settings.general.webhooks.sectionTitle")}
      description={t("settings.general.webhooks.description")}
    >
      <SettingsRow
        label={t("settings.general.webhooks.enabled")}
        description={t("settings.general.webhooks.enabledDescription")}
      >
        <Switch
          checked={enabled}
          disabled={disabled}
          onCheckedChange={(checked) => {
            setEnabled(checked);
            setTestResult(null);
          }}
        />
      </SettingsRow>

      <SettingsRow
        label={t("settings.general.webhooks.url")}
        description={t("settings.general.webhooks.urlDescription")}
        below={
          error ? (
            <span className="max-w-[calc(320px*var(--ui-space-scale,1))] text-right text-xs text-destructive">
              {error}
            </span>
          ) : null
        }
      >
        <div className="flex items-center gap-2">
          <Input
            type="url"
            value={url}
            placeholder={t("settings.general.webhooks.urlPlaceholder")}
            aria-label={t("settings.general.webhooks.url")}
            disabled={disabled}
            onChange={(event) => {
              setUrl(event.target.value);
              setTestResult(null);
            }}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                save().catch(() => undefined);
              }
            }}
            className="h-8 w-60"
          />
          <Button
            size="sm"
            disabled={disabled}
            onClick={() => {
              save().catch(() => undefined);
            }}
            type="button"
          >
            {saving
              ? t("settings.general.webhooks.saving")
              : t("settings.general.webhooks.save")}
          </Button>
        </div>
      </SettingsRow>

      <SettingsRow
        label={t("settings.general.webhooks.secret")}
        description={t("settings.general.webhooks.secretDescription")}
      >
        <Input
          type="password"
          value={secret}
          placeholder={
            settings?.hasSecret
              ? t("settings.general.webhooks.secretPlaceholderStored")
              : t("settings.general.webhooks.secretPlaceholder")
          }
          aria-label={t("settings.general.webhooks.secret")}
          disabled={disabled}
          onChange={(event) => {
            setSecret(event.target.value);
            setTestResult(null);
          }}
          className="h-8 w-60"
        />
      </SettingsRow>

      {settings?.secretUnavailable && !settings.hasSecret ? (
        <SettingsRow
          label={t("settings.general.webhooks.secretUnavailable")}
          description={t(
            "settings.general.webhooks.secretUnavailableDescription",
          )}
        />
      ) : null}

      {settings?.hasSecret ? (
        <SettingsRow
          label={t("settings.general.webhooks.secretStored")}
          description={t("settings.general.webhooks.secretStoredDescription")}
        >
          <div className="flex items-center gap-2">
            <Checkbox
              id="webhook-clear-secret"
              checked={clearSecret}
              disabled={disabled}
              onCheckedChange={(checked) => {
                setClearSecret(checked === true);
                setTestResult(null);
              }}
            />
            <label
              htmlFor="webhook-clear-secret"
              className="text-sm text-muted-foreground"
            >
              {t("settings.general.webhooks.clearSecret")}
            </label>
          </div>
        </SettingsRow>
      ) : null}

      <SettingsRow
        label={t("settings.general.webhooks.test")}
        description={t("settings.general.webhooks.testDescription")}
      >
        <Button
          size="sm"
          variant="outline"
          disabled={disabled}
          onClick={() => {
            test().catch(() => undefined);
          }}
          type="button"
        >
          {testing
            ? t("settings.general.webhooks.testing")
            : t("settings.general.webhooks.test")}
        </Button>
      </SettingsRow>

      {testResult ? (
        <TestResultBlock
          result={testResult}
          secretUnavailable={settings?.secretUnavailable === true}
        />
      ) : null}

      <div className="flex flex-col gap-1 py-2 pl-[calc(2.75rem*var(--ui-space-scale,1))] text-xs text-muted-foreground">
        <p>
          {t("settings.general.webhooks.signatureDescription", {
            algorithm: settings?.signature.algorithm ?? "hmac-sha256",
            signedContent:
              settings?.signature.signedContent ?? "{timestamp}.{raw body}",
            header: settings?.signature.header ?? "X-Unsloth-Signature-256",
          })}
        </p>
      </div>

      <DeliveryHistory
        deliveries={settings?.deliveries ?? []}
        loading={loading}
        onRefresh={() => {
          refresh().catch(() => undefined);
        }}
      />
    </SettingsSection>
  );
}
