# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import ipaddress
from typing import Any
from urllib.parse import urlsplit

SETTINGS_KEY = "training_webhooks"
SECRET_KIND = "training_webhook_secret"
SECRET_SCOPE = "default"
MAX_URL_LENGTH = 2048
MAX_SECRET_LENGTH = 512
_UNSET = object()


LOOPBACK = "loopback"
ALLOWED = "allowed"

_BLOCKED_CATEGORIES = (
    ("is_link_local", "link-local"),
    ("is_multicast", "multicast"),
    ("is_unspecified", "unspecified"),
    ("is_reserved", "reserved"),
)


def classify_webhook_host(value: Any) -> str:
    candidate = str(value).strip().strip("[]").split("%", 1)[0]
    if candidate.lower() in {"localhost", "localhost.localdomain"}:
        return LOOPBACK
    try:
        address = ipaddress.ip_address(candidate)
    except ValueError:
        return ALLOWED
    if address.version == 6 and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    if address.is_loopback:
        return LOOPBACK
    for attribute, category in _BLOCKED_CATEGORIES:
        if getattr(address, attribute):
            return category
    return ALLOWED


def validate_webhook_url(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("Webhook URL must be text.")
    candidate = value.strip()
    if not candidate:
        raise ValueError("Webhook URL is required.")
    if len(candidate) > MAX_URL_LENGTH:
        raise ValueError(f"Webhook URL must be at most {MAX_URL_LENGTH} characters.")
    if any(char.isspace() or ord(char) < 0x20 or ord(char) == 0x7F for char in candidate):
        raise ValueError("Webhook URL must not contain spaces or control characters.")
    try:
        parts = urlsplit(candidate)
        port = parts.port
    except ValueError as exc:
        raise ValueError("Webhook URL has an invalid port or address.") from exc
    if parts.scheme.lower() not in {"http", "https"}:
        raise ValueError("Webhook URL must start with http:// or https://.")
    if not parts.hostname:
        raise ValueError("Webhook URL must include a host.")
    category = classify_webhook_host(parts.hostname)
    if category not in {LOOPBACK, ALLOWED}:
        raise ValueError(f"Webhook URL must not point at a {category} address.")
    if parts.username is not None or parts.password is not None:
        raise ValueError("Webhook URL must not include credentials.")
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("Webhook URL has an invalid port.")
    return candidate


def _normalize_secret(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("Webhook secret must be text.")
    secret = value.strip()
    if len(secret) > MAX_SECRET_LENGTH:
        raise ValueError(f"Webhook secret must be at most {MAX_SECRET_LENGTH} characters.")
    return secret


def _read_raw() -> dict:
    try:
        from storage.studio_db import get_app_setting

        stored = get_app_setting(SETTINGS_KEY, None)
    except Exception:
        stored = None
    return stored if isinstance(stored, dict) else {}


def _legacy_secret(stored: dict) -> str:
    value = stored.get("secret")
    if not isinstance(value, str):
        return ""
    return value.strip()


def _migrate_legacy_secret(stored: dict) -> dict:
    legacy = _legacy_secret(stored)
    if not legacy:
        return stored
    try:
        from storage import credential_secrets

        credential_secrets.insert_secret_if_absent(SECRET_KIND, SECRET_SCOPE, legacy)
        cleaned = {key: value for key, value in stored.items() if key != "secret"}
        from storage.studio_db import upsert_app_settings

        upsert_app_settings({SETTINGS_KEY: cleaned}, reclaim_free_pages = True)
        return cleaned
    except Exception:
        return stored


def _stored_secret() -> "tuple[str, bool]":
    try:
        from storage import credential_secrets

        secret, stored = credential_secrets.get_secret_with_presence(
            SECRET_KIND, SECRET_SCOPE
        )
        return (secret or "", bool(stored and secret is None))
    except Exception:
        return ("", False)


def get_webhook_config() -> dict:
    stored = _migrate_legacy_secret(_read_raw())
    secret, unavailable = _stored_secret()
    if not secret:
        legacy = _legacy_secret(stored)
        if legacy:
            secret = legacy
            unavailable = False
    return {
        "enabled": stored.get("enabled") is True,
        "url": stored.get("url") if isinstance(stored.get("url"), str) else "",
        "secret": secret,
        "secret_unavailable": bool(unavailable and not secret),
    }


def get_webhook_settings() -> dict:
    config = get_webhook_config()
    return {
        "enabled": config["enabled"],
        "url": config["url"],
        "has_secret": bool(config["secret"]),
        "secret_unavailable": config["secret_unavailable"],
        "signature": {
            "algorithm": "hmac-sha256",
            "header": "X-Unsloth-Signature-256",
            "signed_content": "{timestamp}.{raw request body}",
        },
    }


def set_webhook_settings(
    *,
    enabled: bool,
    url: Any,
    secret: Any = _UNSET,
) -> dict:
    if not isinstance(enabled, bool):
        raise ValueError("Webhook enabled must be true or false.")
    if not enabled and (not isinstance(url, str) or not url.strip()):
        normalized_url = ""
    else:
        normalized_url = validate_webhook_url(url)

    stored = _migrate_legacy_secret(_read_raw())
    legacy = _legacy_secret(stored)

    from storage import credential_secrets

    if secret is _UNSET:
        if legacy:
            credential_secrets.upsert_secret(SECRET_KIND, SECRET_SCOPE, legacy)
            legacy = ""
    else:
        normalized_secret = _normalize_secret(secret)
        if normalized_secret:
            credential_secrets.upsert_secret(SECRET_KIND, SECRET_SCOPE, normalized_secret)
        else:
            credential_secrets.delete_secret(SECRET_KIND, SECRET_SCOPE)
        legacy = ""

    payload = {"enabled": enabled, "url": normalized_url}
    if legacy:
        payload["secret"] = legacy
    from storage.studio_db import upsert_app_settings

    upsert_app_settings({SETTINGS_KEY: payload})
    return get_webhook_settings()
