# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import hashlib
import hmac
import http.client
import json
import socket
import ssl
import threading
import time
from collections import deque
from typing import NamedTuple, Optional
from urllib import request as urlrequest
from urllib.parse import urlsplit

from core.training.webhook_payload import (
    DELIVERY_HEADER,
    EVENT_HEADER,
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
)
from utils.webhook_settings import (
    ALLOWED,
    LOOPBACK,
    classify_webhook_host,
)

MAX_ATTEMPTS = 3
REQUEST_TIMEOUT_S = 5.0
RETRY_DELAYS = (0.5, 1.5)
MAX_DELIVERIES = 25

FAILURE_INVALID = "invalid"
FAILURE_UNRESOLVABLE = "unresolvable"
FAILURE_BLOCKED = "blocked"
FAILURE_PROXY = "proxy"
FAILURE_CERTIFICATE = "certificate"
FAILURE_HTTP = "http"
FAILURE_TRANSPORT = "transport"


class _Refusal(NamedTuple):
    kind: str
    message: str


class _Attempt(NamedTuple):
    status_code: Optional[int]
    body: Optional[str]
    error: Optional[str]
    kind: Optional[str]


_deliveries: deque[dict] = deque(maxlen = MAX_DELIVERIES)
_delivery_lock = threading.Lock()
_threads: set[threading.Thread] = set()


def _canonical_body(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys = True, separators = (",", ":"), ensure_ascii = False).encode(
        "utf-8"
    )


def sign_body(secret: str, timestamp: int, body: bytes) -> str:
    message = f"{timestamp}.".encode("utf-8") + body
    digest = hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def verify_signature(secret: str, timestamp: int, body: bytes, signature: str) -> bool:
    if not secret or not signature:
        return False
    return hmac.compare_digest(sign_body(secret, timestamp, body), signature)


class _PinnedConnection(http.client.HTTPConnection):
    def __init__(self, host, port, *, sockaddr, timeout, context = None):
        super().__init__(host, port, timeout = timeout)
        self._pinned = sockaddr
        self._context = context

    def connect(self):
        family, address = self._pinned
        sock = socket.socket(family, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        try:
            sock.connect(address)
            self.sock = (
                self._context.wrap_socket(sock, server_hostname = self.host)
                if self._context is not None
                else sock
            )
        except BaseException:
            sock.close()
            raise


def _resolve_endpoint(url: str) -> tuple[list[tuple[int, tuple]], Optional[_Refusal]]:
    parts = urlsplit(url)
    host = parts.hostname
    if not host:
        return [], _Refusal(FAILURE_INVALID, "Webhook URL has no host")
    try:
        port = parts.port
    except ValueError:
        return [], _Refusal(FAILURE_INVALID, "Webhook URL has an invalid port")
    if port is None:
        port = 443 if parts.scheme.lower() == "https" else 80
    try:
        infos = socket.getaddrinfo(host, port, type = socket.SOCK_STREAM)
    except OSError as exc:
        return [], _Refusal(
            FAILURE_UNRESOLVABLE, str(exc) or "Webhook host could not be resolved"
        )
    endpoints: list[tuple[int, tuple]] = []
    for family, _kind, _protocol, _canonical, sockaddr in infos:
        category = classify_webhook_host(sockaddr[0])
        if category not in {LOOPBACK, ALLOWED}:
            return [], _Refusal(
                FAILURE_BLOCKED, f"Webhook host resolves to a {category} address"
            )
        endpoints.append((family, sockaddr))
    if not endpoints:
        return [], _Refusal(
            FAILURE_UNRESOLVABLE, "Webhook host resolved to no address"
        )
    return endpoints, None


def _proxy_blocked_reason(url: str) -> Optional[_Refusal]:
    try:
        parts = urlsplit(url)
        host = parts.hostname
        if (
            not host
            or classify_webhook_host(host) == LOOPBACK
            or urlrequest.proxy_bypass(host)
        ):
            return None
        scheme = parts.scheme.lower()
        proxies = urlrequest.getproxies()
        proxy = proxies.get(scheme) or proxies.get("all")
        if not proxy:
            return None
        parsed = urlsplit(proxy if "://" in proxy else f"http://{proxy}")
        label = parsed.hostname or proxy
        if parsed.port:
            label = f"{label}:{parsed.port}"
        environment = urlrequest.getproxies_environment()
        if environment.get(scheme) or environment.get("all"):
            variable = f"{scheme.upper()}_PROXY" if environment.get(scheme) else "ALL_PROXY"
            return _Refusal(
                FAILURE_PROXY,
                "Webhook delivery is direct-only so the verified address is always used; "
                f"{variable} {label} is configured for this machine and was not used. "
                f"Add {host} to NO_PROXY to deliver directly.",
            )
        return _Refusal(
            FAILURE_PROXY,
            "Webhook delivery is direct-only so the verified address is always used; "
            f"this machine's system proxy {label} is configured and was not used. "
            "NO_PROXY is not read for the system proxy on this platform; "
            f"add {host} to the operating system's proxy bypass list to deliver directly.",
        )
    except Exception:
        return None


def _post(
    url: str,
    endpoints: list[tuple[int, tuple]],
    body: bytes,
    headers: dict[str, str],
    timeout: float,
) -> _Attempt:
    parts = urlsplit(url)
    host = parts.hostname or ""
    secure = parts.scheme.lower() == "https"
    default_port = 443 if secure else 80
    port = parts.port or default_port
    target = parts.path or "/"
    if parts.query:
        target = f"{target}?{parts.query}"
    authority = f"[{host}]" if ":" in host else host
    if port != default_port:
        authority = f"{authority}:{port}"
    request_headers = dict(headers)
    request_headers["Host"] = authority
    context = ssl.create_default_context() if secure else None
    last_error: Optional[str] = None
    for family, sockaddr in endpoints:
        connection = _PinnedConnection(
            host,
            port,
            sockaddr = (family, sockaddr),
            timeout = timeout,
            context = context,
        )
        try:
            connection.request("POST", target, body = body, headers = request_headers)
            response = connection.getresponse()
            text = response.read(512).decode("utf-8", errors = "replace")
            return _Attempt(int(response.status), text, None, FAILURE_HTTP)
        except ssl.SSLCertVerificationError as exc:
            return _Attempt(
                None,
                None,
                str(exc) or "certificate verification failed",
                FAILURE_CERTIFICATE,
            )
        except Exception as exc:
            last_error = str(exc) or exc.__class__.__name__
        finally:
            connection.close()
    return _Attempt(None, None, last_error, FAILURE_TRANSPORT)


def get_deliveries() -> list[dict]:
    with _delivery_lock:
        return list(reversed(_deliveries))


def deliver(
    *,
    event: str,
    payload: dict,
    url: str,
    secret: str,
    max_attempts: Optional[int] = None,
) -> dict:
    timeout = REQUEST_TIMEOUT_S
    max_attempts = MAX_ATTEMPTS if max_attempts is None else max_attempts
    delivery_id = payload["delivery_id"]
    timestamp = payload["signature"]["timestamp"]
    body = _canonical_body(payload)
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Unsloth-Studio-Webhooks/1.0",
        EVENT_HEADER: event,
        DELIVERY_HEADER: delivery_id,
        TIMESTAMP_HEADER: str(timestamp),
    }
    signature = sign_body(secret, timestamp, body) if secret else ""
    if signature:
        headers[SIGNATURE_HEADER] = signature

    started = time.monotonic()
    attempts = 0
    status_code: Optional[int] = None
    response_text: Optional[str] = None
    last_error: Optional[str] = None
    failure_kind: Optional[str] = None
    ok = False
    endpoints, resolution_refusal = _resolve_endpoint(url)
    refusal = resolution_refusal or _proxy_blocked_reason(url)
    if refusal is not None:
        attempts = 1
        last_error = refusal.message
        failure_kind = refusal.kind
    else:
        for attempt in range(1, max_attempts + 1):
            attempts = attempt
            outcome = _post(url, endpoints, body, headers, timeout)
            status_code = outcome.status_code
            response_text = outcome.body
            last_error = outcome.error
            failure_kind = outcome.kind
            if status_code is not None and 200 <= status_code < 300:
                ok = True
                break
            if outcome.kind == FAILURE_CERTIFICATE:
                break
            if status_code is not None and 300 <= status_code < 400:
                break
            if status_code is not None and 400 <= status_code < 500 and status_code not in {408, 425, 429}:
                break
            if attempt < max_attempts:
                time.sleep(RETRY_DELAYS[min(attempt - 1, len(RETRY_DELAYS) - 1)])

    record_data = payload.get("run") or {}
    entry = {
        "delivery_id": delivery_id,
        "event": event,
        "run_id": (record_data.get("id") if isinstance(record_data, dict) else None),
        "status": (record_data.get("status") if isinstance(record_data, dict) else None),
        "created_at": payload.get("sent_at"),
        "attempts": attempts,
        "ok": ok,
        "status_code": status_code,
        "error": None if ok else (last_error or (f"HTTP {status_code}" if status_code else "Request failed")),
        "failure_kind": None if ok else failure_kind,
        "response": (response_text or "")[:200] or None,
        "duration_ms": int((time.monotonic() - started) * 1000),
        "signed": bool(signature),
        "anomaly_kind": (payload.get("anomaly") or {}).get("kind"),
        "test": bool(payload.get("test")),
    }
    with _delivery_lock:
        _deliveries.append(entry)
    entry["signature_header"] = signature or None
    entry["signature_verified"] = (
        verify_signature(secret, timestamp, body, signature) if signature else False
    )
    return entry


def spawn_delivery(event: str, payload: dict, url: str, secret: str) -> None:
    def target() -> None:
        try:
            deliver(event = event, payload = payload, url = url, secret = secret)
        except Exception:
            pass
        finally:
            with _delivery_lock:
                _threads.discard(threading.current_thread())

    thread = threading.Thread(target = target, daemon = True, name = f"webhook-{event}")
    with _delivery_lock:
        _threads.add(thread)
    thread.start()


def wait_for_deliveries(timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        with _delivery_lock:
            pending = [thread for thread in _threads if thread.is_alive()]
        if not pending:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        pending[0].join(timeout = min(remaining, 0.25))
