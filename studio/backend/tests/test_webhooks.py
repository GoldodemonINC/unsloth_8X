# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import datetime
import ipaddress
import json
import math
import socket
import sqlite3
import ssl
import threading
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.training import webhooks
from core.training import loss_anomaly
from core.training import webhook_delivery
from routes import settings as settings_routes
from utils import webhook_settings


class _Sink:

    def __init__(
        self,
        statuses: tuple[int, ...] = (200,),
        delay: float = 0.0,
        certfile: Path | None = None,
        keyfile: Path | None = None,
    ):
        self.statuses = list(statuses)
        self.delay = delay
        self.requests: list[dict] = []
        sink = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                sink.requests.append(
                    {
                        "body": body,
                        "headers": {key.lower(): value for key, value in self.headers.items()},
                        "path": self.path,
                    }
                )
                if sink.delay:
                    time.sleep(sink.delay)
                status = sink.statuses[min(len(sink.requests) - 1, len(sink.statuses) - 1)]
                self.send_response(status)
                self.end_headers()
                self.wfile.write(b"ok")

            def log_message(self, *_args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        if certfile is not None and keyfile is not None:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(certfile, keyfile)
            self.server.socket = context.wrap_socket(self.server.socket, server_side = True)
        self.thread = threading.Thread(target = self.server.serve_forever, daemon = True)
        self.thread.start()

    @property
    def port(self) -> int:
        return int(self.server.server_address[1])

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/hook"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def _restore_stdlib_ssl(monkeypatch) -> None:
    stdlib = next(c for c in ssl._SSLContext.__subclasses__() if c.__module__ == "ssl")
    monkeypatch.setattr(ssl, "SSLContext", stdlib)


def _mint_certificate(
    directory: Path, sans: list, common_name: str
) -> tuple[Path, Path, Path]:
    now = datetime.datetime.now(datetime.timezone.utc)
    ca_key = rsa.generate_private_key(public_exponent = 65537, key_size = 2048)
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Webhook Test CA")])
    ca = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes = 5))
        .not_valid_after(now + datetime.timedelta(days = 2))
        .add_extension(x509.BasicConstraints(ca = True, path_length = None), critical = True)
        .sign(ca_key, hashes.SHA256())
    )
    leaf_key = rsa.generate_private_key(public_exponent = 65537, key_size = 2048)
    leaf = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)]))
        .issuer_name(ca_name)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes = 5))
        .not_valid_after(now + datetime.timedelta(days = 2))
        .add_extension(x509.SubjectAlternativeName(sans), critical = False)
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical = False
        )
        .sign(ca_key, hashes.SHA256())
    )
    ca_path = directory / "ca.pem"
    cert_path = directory / "leaf.pem"
    key_path = directory / "leaf.key"
    ca_path.write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    cert_path.write_bytes(leaf.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        leaf_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    return ca_path, cert_path, key_path


@pytest.fixture
def sink():
    receiver = _Sink()
    yield receiver
    receiver.close()


@pytest.fixture
def settings_store(monkeypatch):
    store: dict = {}
    monkeypatch.setattr(
        "storage.studio_db.get_app_setting",
        lambda key, default = None: store.get(key, default),
    )
    monkeypatch.setattr(
        "storage.studio_db.upsert_app_settings", lambda values, **_: store.update(values)
    )
    return store


@pytest.fixture(autouse = True)
def encrypted_store(tmp_path, monkeypatch):
    from auth import storage as auth_storage
    from storage import credential_secrets

    studio_db = tmp_path / "studio.db"
    monkeypatch.setattr(auth_storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(auth_storage, "_credential_encryption_key_cache", None)
    monkeypatch.setattr(credential_secrets, "studio_db_path", lambda: studio_db)
    monkeypatch.setattr(
        credential_secrets,
        "ensure_dir",
        lambda path: path.mkdir(parents = True, exist_ok = True),
    )
    monkeypatch.setattr(
        credential_secrets,
        "get_or_create_credential_encryption_key",
        auth_storage.get_or_create_credential_encryption_key,
    )
    credential_secrets._schema_ready = set()
    yield studio_db
    credential_secrets._schema_ready = set()
    auth_storage._credential_encryption_key_cache = None


@pytest.fixture(autouse = True)
def fast_delivery(monkeypatch):
    monkeypatch.setattr(webhook_delivery, "RETRY_DELAYS", (0.0, 0.0))
    monkeypatch.setattr(webhook_delivery, "REQUEST_TIMEOUT_S", 0.5)


@pytest.fixture(autouse = True)
def isolated_proxy_environment(monkeypatch):
    monkeypatch.setattr(webhook_delivery.urlrequest, "getproxies", lambda: {})
    monkeypatch.setattr(webhook_delivery.urlrequest, "getproxies_environment", lambda: {})
    monkeypatch.setattr(webhook_delivery.urlrequest, "proxy_bypass", lambda _host: False)


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/hook",
        "https://example.com/hook?run=1",
        "http://127.0.0.1:8080/webhook",
        "https://[::1]:9443/hook",
    ],
)
def test_accepts_http_and_https_endpoints(url):
    assert webhook_settings.validate_webhook_url(url) == url


@pytest.mark.parametrize(
    "url, reason",
    [
        ("", "required"),
        ("   ", "required"),
        ("ftp://example.com/hook", "http"),
        ("file:///etc/passwd", "http"),
        ("javascript:alert(1)", "http"),
        ("example.com/hook", "http"),
        ("https://", "host"),
        ("https://user:secret@example.com/hook", "credentials"),
        ("http://example.com:notaport/hook", "port"),
        ("http://example.com/hook with space", "spaces"),
    ],
)
def test_refuses_non_http_or_malformed_endpoints(url, reason):
    with pytest.raises(ValueError) as error:
        webhook_settings.validate_webhook_url(url)
    assert reason in str(error.value).lower()


@pytest.mark.parametrize(
    "host, address, category",
    [
        ("127.0.0.1", "127.0.0.1", webhook_settings.LOOPBACK),
        ("[::1]", "::1", webhook_settings.LOOPBACK),
        ("localhost", "127.0.0.1", webhook_settings.LOOPBACK),
        ("169.254.169.254", "169.254.169.254", "link-local"),
        ("224.0.0.1", "224.0.0.1", "multicast"),
        ("0.0.0.0", "0.0.0.0", "unspecified"),
        ("240.0.0.1", "240.0.0.1", "reserved"),
        ("[::ffff:169.254.169.254]", "::ffff:169.254.169.254", "link-local"),
        ("[::ffff:127.0.0.1]", "::ffff:127.0.0.1", webhook_settings.LOOPBACK),
        ("8.8.8.8", "8.8.8.8", webhook_settings.ALLOWED),
    ],
)
def test_literal_and_resolved_paths_share_one_address_policy(
    host, address, category, monkeypatch
):
    assert webhook_settings.classify_webhook_host(host) == category

    url = f"http://{host}/hook"
    permitted = category in {webhook_settings.LOOPBACK, webhook_settings.ALLOWED}
    if permitted:
        assert webhook_settings.validate_webhook_url(url) == url
    else:
        with pytest.raises(ValueError, match = category):
            webhook_settings.validate_webhook_url(url)

    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    monkeypatch.setattr(
        webhook_delivery.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (family, socket.SOCK_STREAM, 6, "", (address, 80))
        ],
    )

    endpoints, refusal = webhook_delivery._resolve_endpoint(url)

    assert (refusal is None) is permitted
    if permitted:
        assert endpoints
    else:
        assert endpoints == []
        assert refusal.kind == webhook_delivery.FAILURE_BLOCKED
        assert category in refusal.message


def test_webhooks_start_disabled_with_no_endpoint(settings_store):
    stored = webhook_settings.get_webhook_settings()
    assert stored["enabled"] is False
    assert stored["url"] == ""
    assert stored["has_secret"] is False
    assert stored["secret_unavailable"] is False


def test_disabled_alerts_can_be_saved_without_an_endpoint(settings_store):
    saved = webhook_settings.set_webhook_settings(enabled = False, url = "")
    assert saved["enabled"] is False
    assert saved["url"] == ""
    with pytest.raises(ValueError):
        webhook_settings.set_webhook_settings(enabled = True, url = "")


def test_a_redirect_is_not_followed_to_its_target(sink):

    class Redirector(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.send_response(302)
            self.send_header("Location", sink.url)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Redirector)
    thread = threading.Thread(target = server.serve_forever, daemon = True)
    thread.start()
    try:
        result = webhooks.send_test_event(
            url = f"http://127.0.0.1:{server.server_address[1]}/hook",
            secret = "topsecret",
            max_attempts = 1,
        )
    finally:
        server.shutdown()
        server.server_close()

    assert result["ok"] is False
    assert result["status_code"] == 302
    assert sink.requests == []


def test_a_second_resolution_cannot_redirect_delivery_to_an_internal_address(
    sink, monkeypatch
):
    safe = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", sink.port))]
    unsafe = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 80))]
    calls: list[tuple] = []

    def resolver(*args, **_kwargs):
        calls.append(args)
        return safe if len(calls) == 1 else unsafe

    monkeypatch.setattr(webhook_delivery.socket, "getaddrinfo", resolver)

    result = webhooks.send_test_event(
        url = "http://rebind.example/hook", secret = "topsecret", max_attempts = 1
    )

    assert result["ok"] is True
    assert len(calls) == 1
    assert len(sink.requests) == 1
    assert sink.requests[0]["headers"]["host"] == "rebind.example"


@pytest.mark.parametrize(
    "url, address",
    [
        ("http://metadata.example/hook", 80),
        ("https://metadata.example/hook", 443),
    ],
    ids = ["http", "https"],
)
def test_a_blocked_resolution_fails_once_without_backoff(url, address, sink, monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(webhook_delivery.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(
        webhook_delivery.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", address))
        ],
    )

    result = webhooks.send_test_event(url = url, secret = "topsecret")

    assert result["ok"] is False
    assert result["attempts"] == 1
    assert result["status_code"] is None
    assert "link-local" in result["error"]
    assert sleeps == []
    assert sink.requests == []


def test_https_delivery_verifies_the_host_name_while_connecting_to_the_pinned_address(
    tmp_path, monkeypatch
):
    _restore_stdlib_ssl(monkeypatch)
    ca, cert, key = _mint_certificate(tmp_path, [x509.DNSName("localhost")], "localhost")
    monkeypatch.setenv("SSL_CERT_FILE", str(ca))
    receiver = _Sink(certfile = cert, keyfile = key)
    try:
        monkeypatch.setattr(
            webhook_delivery.socket,
            "getaddrinfo",
            lambda *_args, **_kwargs: [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", receiver.port))
            ],
        )
        result = webhooks.send_test_event(
            url = f"https://localhost:{receiver.port}/hook",
            secret = "topsecret",
            max_attempts = 1,
        )
    finally:
        receiver.close()

    assert result["ok"] is True
    assert result["signature_verified"] is True
    assert len(receiver.requests) == 1
    sent = receiver.requests[0]
    assert sent["headers"]["host"] == f"localhost:{receiver.port}"
    assert webhook_delivery.verify_signature(
        "topsecret",
        int(sent["headers"]["x-unsloth-timestamp"]),
        sent["body"],
        sent["headers"]["x-unsloth-signature-256"],
    )

def test_a_blocked_resolution_on_https_fails_once_without_retrying(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(webhook_delivery.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(
        webhook_delivery.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 443))
        ],
    )

    result = webhooks.send_test_event(url = "https://metadata.example/hook", secret = "topsecret")

    assert result["ok"] is False
    assert result["attempts"] == 1
    assert result["status_code"] is None
    assert "link-local" in result["error"]
    assert sleeps == []


def test_a_configured_proxy_fails_delivery_with_an_attributable_error(sink, monkeypatch):
    monkeypatch.setattr(
        webhook_delivery.urlrequest, "getproxies", lambda: {"http": "http://proxy.internal:3128"}
    )
    monkeypatch.setattr(
        webhook_delivery.urlrequest,
        "getproxies_environment",
        lambda: {"http": "http://proxy.internal:3128"},
    )
    sleeps: list[float] = []
    monkeypatch.setattr(webhook_delivery.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(
        webhook_delivery.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", sink.port))
        ],
    )

    result = webhooks.send_test_event(url = "http://webhook.example/hook", secret = "topsecret")

    assert result["ok"] is False
    assert result["attempts"] == 1
    assert "direct-only" in result["error"]
    assert "HTTP_PROXY" in result["error"]
    assert "proxy.internal:3128" in result["error"]
    assert "NO_PROXY" in result["error"]
    assert sleeps == []
    assert sink.requests == []


def test_a_loopback_receiver_is_delivered_while_a_proxy_is_configured(sink, monkeypatch):
    monkeypatch.setattr(
        webhook_delivery.urlrequest, "getproxies", lambda: {"http": "http://proxy.internal:3128"}
    )

    result = webhooks.send_test_event(url = sink.url, secret = "topsecret", max_attempts = 1)

    assert result["ok"] is True
    assert len(sink.requests) == 1


def test_a_registry_proxy_reports_the_system_source_and_no_no_proxy_advice(
    sink, settings_store, monkeypatch
):
    monkeypatch.setattr(
        webhook_delivery.urlrequest,
        "getproxies",
        lambda: {"http": "http://system-proxy.internal:3128"},
    )
    sleeps: list[float] = []
    monkeypatch.setattr(webhook_delivery.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(
        webhook_delivery.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", sink.port))
        ],
    )
    webhook_settings.set_webhook_settings(
        enabled = True, url = "http://webhook.example/hook"
    )
    client = _client(monkeypatch)

    response = client.post("/api/settings/webhooks/test", json = {})

    body = response.json()
    assert response.status_code == 200
    assert body["ok"] is False
    assert body["attempts"] == 1
    assert "system proxy" in body["error"]
    assert "system-proxy.internal:3128" in body["error"]
    assert "HTTP_PROXY" not in body["error"]
    assert "NO_PROXY is not read" in body["error"]
    assert sleeps == []
    assert sink.requests == []
    assert webhooks.get_deliveries()[0]["failure_kind"] == webhook_delivery.FAILURE_PROXY

    history = client.get("/api/settings/webhooks").json()["deliveries"]
    recorded = next(
        item for item in history if item["delivery_id"] == body["delivery_id"]
    )
    assert recorded["failure_kind"] == webhook_delivery.FAILURE_PROXY


def test_a_blocked_target_is_refused_before_the_proxy_path(monkeypatch):
    monkeypatch.setattr(
        webhook_delivery.urlrequest, "getproxies", lambda: {"http": "http://proxy.internal:3128"}
    )

    with pytest.raises(ValueError, match = "link-local"):
        webhooks.send_test_event(url = "http://169.254.169.254/hook")


def test_a_certificate_failure_is_reported_once_without_backoff_through_the_route(
    tmp_path, sink, settings_store, monkeypatch
):
    _restore_stdlib_ssl(monkeypatch)
    ca, cert, key = _mint_certificate(
        tmp_path, [x509.IPAddress(ipaddress.IPv4Address("127.0.0.1"))], "127.0.0.1"
    )
    monkeypatch.setenv("SSL_CERT_FILE", str(ca))
    receiver = _Sink(certfile = cert, keyfile = key)
    sleeps: list[float] = []
    monkeypatch.setattr(webhook_delivery.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(
        webhook_delivery.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", receiver.port))
        ],
    )
    webhook_settings.set_webhook_settings(
        enabled = True,
        url = f"https://webhook.example:{receiver.port}/hook",
        secret = "topsecret",
    )
    client = _client(monkeypatch)
    try:
        response = client.post("/api/settings/webhooks/test", json = {})
    finally:
        receiver.close()

    body = response.json()
    assert response.status_code == 200
    assert body["ok"] is False
    assert body["attempts"] == 1
    assert "certificate verify failed" in body["error"].lower()
    assert sleeps == []
    assert receiver.requests == []
    assert webhooks.get_deliveries()[0]["failure_kind"] == webhook_delivery.FAILURE_CERTIFICATE

    history = client.get("/api/settings/webhooks").json()["deliveries"]
    recorded = next(
        item for item in history if item["delivery_id"] == body["delivery_id"]
    )
    assert recorded["failure_kind"] == webhook_delivery.FAILURE_CERTIFICATE

def test_divergence_is_detected_deep_into_a_stream_after_old_points_are_dropped():
    run_id = "run-deep-divergence"
    values = [2.0] * 24000 + [2.0 + 2.0 * offset / 6000 for offset in range(6000)]
    divergence = None
    for step, loss in enumerate(values):
        found = webhooks.observe_progress(run_id = run_id, step = step, loss = loss)
        if found is not None and found["kind"] == "divergence":
            divergence = found

    assert divergence is not None
    assert divergence["step"] > loss_anomaly.MAX_RETAINED_POINTS + loss_anomaly.BASELINE_WINDOW


def test_a_long_run_retains_a_flat_amount_of_state():
    run_id = "run-flat-retention"
    for step in range(30000):
        webhooks.observe_progress(run_id = run_id, step = step, loss = 2.0)

    state = webhooks._progress[run_id]
    tracker = state["tracker"]
    assert state["total"] == 30000
    assert len(state["points"]) == loss_anomaly.MAX_RETAINED_POINTS
    assert tracker.prefix_count <= loss_anomaly.BASELINE_WINDOW
    assert tracker.last.entries() <= loss_anomaly.RECENT_WINDOW
    assert tracker.plateau.entries() <= loss_anomaly.PLATEAU_WINDOW


def test_the_retention_bound_stays_above_every_detector_window():
    assert loss_anomaly.MAX_RETAINED_POINTS >= loss_anomaly.retained_points_floor()
    for window in (
        "BASELINE_WINDOW",
        "RECENT_WINDOW",
        "PLATEAU_WINDOW",
        "SPIKE_BASELINE",
    ):
        original = getattr(loss_anomaly, window)
        setattr(loss_anomaly, window, loss_anomaly.MAX_RETAINED_POINTS)
        try:
            assert loss_anomaly.retained_points_floor() > loss_anomaly.MAX_RETAINED_POINTS, window
        finally:
            setattr(loss_anomaly, window, original)


def test_the_import_refusal_names_the_offending_window_and_its_value():
    assert loss_anomaly._retention_refusal() is None
    for window in (
        "BASELINE_WINDOW",
        "RECENT_WINDOW",
        "PLATEAU_WINDOW",
        "SPIKE_BASELINE",
    ):
        moved = loss_anomaly.MAX_RETAINED_POINTS
        original = getattr(loss_anomaly, window)
        setattr(loss_anomaly, window, moved)
        try:
            refusal = loss_anomaly._retention_refusal()
            required = loss_anomaly.retained_points_floor()
        finally:
            setattr(loss_anomaly, window, original)
        assert refusal is not None, window
        assert f"floor {required} required by {window} = {moved}" in refusal, refusal
        assert f"MAX_RETAINED_POINTS = {loss_anomaly.MAX_RETAINED_POINTS}" in refusal, refusal

    floor = loss_anomaly.retained_points_floor()
    original = loss_anomaly.MAX_RETAINED_POINTS
    loss_anomaly.MAX_RETAINED_POINTS = floor - 1
    try:
        refusal = loss_anomaly._retention_refusal()
    finally:
        loss_anomaly.MAX_RETAINED_POINTS = original
    assert refusal is not None
    assert refusal.startswith(f"MAX_RETAINED_POINTS = {floor - 1}"), refusal
    assert f"required by BASELINE_WINDOW = {loss_anomaly.BASELINE_WINDOW}" in refusal, refusal

def test_a_late_anomaly_in_a_very_long_run_is_attributed_without_indexing_below_the_window(
    monkeypatch,
):
    errors = []
    advance = loss_anomaly.LossAnomalyTracker.advance
    detect = loss_anomaly.LossAnomalyTracker.detect

    def recording_advance(self, points):
        try:
            return advance(self, points)
        except Exception as exc:
            errors.append(exc)
            raise

    def recording_detect(self, points):
        try:
            return detect(self, points)
        except Exception as exc:
            errors.append(exc)
            raise

    monkeypatch.setattr(loss_anomaly.LossAnomalyTracker, "advance", recording_advance)
    monkeypatch.setattr(loss_anomaly.LossAnomalyTracker, "detect", recording_detect)

    run_id = "run-very-long"
    total = 200000
    spike_step = 199500
    spikes = []
    for step in range(total):
        found = webhooks.observe_progress(
            run_id = run_id,
            step = step,
            loss = 9.0 if step == spike_step else 2.0,
        )
        if found is not None and found["kind"] == "spike":
            spikes.append(found)

    state = webhooks._progress[run_id]
    assert errors == []
    assert len(spikes) == 1
    assert spikes[0]["step"] == spike_step
    assert spikes[0]["value"] == 9.0
    assert state["total"] == total
    assert state["base"] == total - loss_anomaly.MAX_RETAINED_POINTS
    assert len(state["points"]) == loss_anomaly.MAX_RETAINED_POINTS


def test_abandoned_runs_do_not_retain_unbounded_progress_history():
    for index in range(webhooks.MAX_TRACKED_RUNS * 3):
        webhooks.observe_progress(run_id = f"run-abandoned-{index}", step = 1, loss = 2.0)

    assert len(webhooks._progress) <= webhooks.MAX_TRACKED_RUNS


def test_signature_is_hmac_over_timestamp_and_raw_body():
    body = b'{"event":"training.completed"}'
    signature = webhook_delivery.sign_body("topsecret", 1_700_000_000, body)
    assert signature.startswith("sha256=")
    assert webhook_delivery.verify_signature("topsecret", 1_700_000_000, body, signature)
    assert not webhook_delivery.verify_signature("wrong", 1_700_000_000, body, signature)
    assert not webhook_delivery.verify_signature(
        "topsecret", 1_700_000_001, body, signature
    )
    assert not webhook_delivery.verify_signature(
        "topsecret", 1_700_000_000, body + b" ", signature
    )
    assert webhook_delivery.verify_signature("", 1, body, "") is False


def _configure(settings_store, sink: _Sink, *, enabled: bool = True, secret: str = "topsecret"):
    webhook_settings.set_webhook_settings(enabled = enabled, url = sink.url, secret = secret)


def test_terminal_alert_has_run_metrics_and_a_verifiable_signature(settings_store, sink):
    _configure(settings_store, sink)
    metrics = {
        "step": 120,
        "loss": 0.42,
        "eval_loss": 0.5,
        "duration_seconds": 90.0,
        "peak_memory_gb": 7.5,
    }
    assert webhooks.notify_terminal(
        run_id = "run-complete-1",
        status = "completed",
        metrics = metrics,
        output_dir = "/models/run-complete-1",
    )
    webhooks.wait_for_deliveries(5)

    assert len(sink.requests) == 1
    request = sink.requests[0]
    payload = json.loads(request["body"])
    assert payload["event"] == "training.completed"
    assert payload["run"] == {
        "id": "run-complete-1",
        "status": "completed",
        "error": None,
        "output_dir": "/models/run-complete-1",
    }
    assert payload["metrics"] == metrics
    assert payload["locator"] == {
        "kind": "studio_run",
        "run_id": "run-complete-1",
        "path": "/studio?run=run-complete-1",
    }
    assert payload["signature"]["algorithm"] == "hmac-sha256"
    assert payload["signature"]["header"] == "X-Unsloth-Signature-256"

    timestamp = int(request["headers"]["x-unsloth-timestamp"])
    signature = request["headers"]["x-unsloth-signature-256"]
    assert webhook_delivery.verify_signature("topsecret", timestamp, request["body"], signature)
    assert request["headers"]["x-unsloth-event"] == "training.completed"
    assert request["headers"]["x-unsloth-delivery"] == payload["delivery_id"]

    delivery = next(
        item for item in webhooks.get_deliveries() if item["delivery_id"] == payload["delivery_id"]
    )
    assert delivery["ok"] is True
    assert delivery["attempts"] == 1
    assert delivery["status_code"] == 200


def test_failure_and_stop_use_their_own_events(settings_store, sink):
    _configure(settings_store, sink)
    webhooks.notify_terminal(
        run_id = "run-failed-1",
        status = "failed",
        metrics = {"step": 4},
        error_message = "CUDA out of memory",
    )
    webhooks.notify_terminal(run_id = "run-stopped-1", status = "stopped", metrics = {"step": 8})
    webhooks.wait_for_deliveries(5)

    events = [json.loads(request["body"]) for request in sink.requests]
    assert [payload["event"] for payload in events] == [
        "training.failed",
        "training.stopped",
    ]
    assert events[0]["run"]["error"] == "CUDA out of memory"


def test_disabled_endpoint_sends_nothing(settings_store, sink):
    _configure(settings_store, sink, enabled = False)
    assert (
        webhooks.notify_terminal(
            run_id = "run-disabled-1", status = "completed", metrics = {"step": 1}
        )
        is False
    )
    webhooks.wait_for_deliveries(1)
    assert sink.requests == []
    assert not any(
        item["run_id"] == "run-disabled-1" for item in webhooks.get_deliveries()
    )


def test_anomaly_fires_once_for_the_run(settings_store, sink):
    _configure(settings_store, sink)
    values = [1.0 + (index % 2) * 0.01 for index in range(11)] + [20.0] + [1.0] * 8
    for step, loss in enumerate(values, start = 1):
        webhooks.observe_progress(
            run_id = "run-spike-1",
            step = step,
            loss = loss,
            metrics = {"step": step, "loss": loss},
        )
    webhooks.wait_for_deliveries(5)

    anomalies = [
        json.loads(request["body"]) for request in sink.requests
    ]
    assert len(anomalies) == 1
    payload = anomalies[0]
    assert payload["event"] == "training.anomaly"
    assert payload["run"]["status"] == "running"
    assert payload["anomaly"]["kind"] == "spike"
    assert payload["anomaly"]["severity"] == "warning"

    for step, loss in enumerate(values, start = len(values) + 1):
        webhooks.observe_progress(run_id = "run-spike-1", step = step, loss = loss)
    webhooks.wait_for_deliveries(2)
    assert len(sink.requests) == 1


_WINDOW_CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "loss_anomaly_window_cases.json").read_text(
        encoding = "utf-8"
    )
)["cases"]


def _window_values(case: dict) -> list[float]:
    values: list[float] = []
    for segment in case["segments"]:
        if "values" in segment:
            values.extend(map(float, segment["values"]))
        elif "value" in segment:
            values.extend([float(segment["value"])] * int(segment["count"]))
        else:
            count = int(segment["count"])
            start, end = float(segment["from"]), float(segment["to"])
            values.extend(start + (end - start) * (i / (count - 1)) for i in range(count))
    return values


@pytest.mark.parametrize("case", _WINDOW_CASES, ids = lambda case: case["name"])
def test_progress_entry_point_matches_the_chart_window_table(case, settings_store, sink):
    _configure(settings_store, sink)
    values = _window_values(case)
    run_id = "window-" + case["name"].replace(" ", "-")
    fired: tuple[int, dict] | None = None
    for step, loss in enumerate(values, start = 1):
        anomaly = webhooks.observe_progress(
            run_id = run_id,
            step = step,
            loss = loss,
            metrics = {"step": step, "loss": loss},
        )
        if anomaly is not None:
            assert fired is None, f"{case['name']}: expected one anomaly alert"
            fired = step, anomaly
    webhooks.wait_for_deliveries(30)

    alerts: list[dict] = []
    for request in sink.requests:
        payload = json.loads(request["body"])
        if payload["event"] == "training.anomaly" and payload["run"]["id"] == run_id:
            alerts.append(payload)
    expected = case["alert"]
    assert len(alerts) == expected["count"], (
        f"{case['name']}: {case['why']} expected {expected['count']} alert(s), got {len(alerts)}"
    )
    assert (fired is not None) is (expected["count"] == 1)
    if fired is None:
        assert expected["count"] == 0
        return
    fired_at, anomaly = fired
    assert fired_at == expected["fired_at"]
    for field in ("kind", "severity", "step"):
        assert anomaly[field] == expected[field]
        assert alerts[0]["anomaly"][field] == expected[field]

    chart = loss_anomaly.detect_loss_anomaly(list(enumerate(values, start = 1)))
    if case["chart"] is None:
        assert chart is None
    else:
        for field in ("kind", "severity", "step"):
            assert chart[field] == case["chart"][field]


_ORACLE_PARITY_CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "loss_anomaly_cases.json").read_text(
        encoding = "utf-8"
    )
)["cases"]


def _oracle_streams() -> list[tuple[str, list[tuple[float, float]]]]:
    streams = []
    for case in _ORACLE_PARITY_CASES:
        raw = case.get("points") or [
            (index + 1, value) for index, value in enumerate(case["values"])
        ]
        points = [
            (float(step), math.nan if value is None else float(value))
            for step, value in raw
        ]
        streams.append((case["name"], sorted(points, key = lambda point: point[0])))

    for name in ("spike after the old window boundary", "divergence cut by the old window"):
        case = next(item for item in _WINDOW_CASES if item["name"] == name)
        points = [(index + 1, value) for index, value in enumerate(_window_values(case))]
        streams.append((f"long run: {name}", points))
    return streams


def _reference_median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _reference_slope(values: list[float]) -> float:
    count = len(values)
    if count < 2:
        return 0.0
    mean_x = (count - 1) / 2
    mean_y = sum(values) / count
    numerator = sum((index - mean_x) * (value - mean_y) for index, value in enumerate(values))
    denominator = sum((index - mean_x) ** 2 for index in range(count))
    return numerator / denominator if denominator else 0.0


def _full_history_reference(points: list[tuple[float, float]]) -> dict | None:
    if len(points) < 8:
        return None

    for step, value in points:
        if not math.isfinite(value):
            return {
                "kind": "nan",
                "severity": "critical",
                "step": step,
                "value": None,
                "detail": 0.0,
            }

    series = sorted(
        ((step, value) for step, value in points if math.isfinite(step)),
        key = lambda point: point[0],
    )
    if len(series) < 8:
        return None

    if len(series) >= 30:
        third = len(series) // 3
        first = [value for _, value in series[:third]]
        last = [value for _, value in series[-third:]]
        first_median = _reference_median(first)
        last_median = _reference_median(last)
        trend = _reference_slope(last)
        if first_median > 0 and last_median > first_median * 1.25 and trend > 0:
            step, value = series[-1]
            return {
                "kind": "divergence",
                "severity": "critical",
                "step": step,
                "value": value,
                "detail": last_median / first_median,
            }

    if len(series) >= 12:
        worst = None
        for index in range(5, len(series)):
            step, value = series[index]
            baseline = _reference_median(
                [previous for _, previous in series[max(0, index - 20) : index]]
            )
            if baseline <= 0:
                continue
            ratio = value / baseline
            if ratio < 2.5 or value - baseline < 0.5:
                continue
            if worst is None or ratio > worst["detail"]:
                worst = {
                    "kind": "spike",
                    "severity": "warning",
                    "step": step,
                    "value": value,
                    "detail": ratio,
                }
        if worst is not None:
            return worst

    if len(series) >= 40:
        tail_length = max(10, len(series) // 4)
        tail = series[-tail_length:]
        tail_values = [value for _, value in tail]
        tail_median = _reference_median(tail_values)
        spread = max(tail_values) - min(tail_values)
        first_value = tail_values[0]
        improvement = (
            (first_value - tail_values[-1]) / first_value if first_value > 0 else 1.0
        )
        if (
            tail_median > 0.1
            and spread / tail_median <= 0.03
            and improvement <= 0.01
        ):
            step, value = tail[-1]
            return {
                "kind": "plateau",
                "severity": "warning",
                "step": step,
                "value": value,
                "detail": spread / tail_median,
            }

    return None


def _full_verdict(anomaly: dict | None) -> tuple | None:
    if anomaly is None:
        return None
    return (
        anomaly["kind"],
        anomaly["severity"],
        anomaly["step"],
        anomaly["value"],
    )


@pytest.mark.parametrize("name,stream", _oracle_streams(), ids = lambda case: case[0])
def test_incremental_tracker_matches_independent_full_history_oracle(name, stream):
    tracker = loss_anomaly.LossAnomalyTracker()
    points = []
    for step, value in stream:
        points.append((step, value))
        count = len(points)
        if count < 8 or (count - 8) % 10:
            continue

        tracker.advance(points)
        actual = tracker.detect(points)
        expected = _full_history_reference(points)
        assert _full_verdict(actual) == _full_verdict(expected), (
            f"{name}: independent full-history oracle and incremental tracker disagree "
            f"at check {count}: expected {_full_verdict(expected)!r}, "
            f"got {_full_verdict(actual)!r}"
        )
        if actual is not None:
            assert actual["detail"] == pytest.approx(expected["detail"], rel = 1e-12, abs = 1e-12), (
                f"{name}: anomaly detail differs at check {count}"
            )


def test_incremental_tracker_mismatches_full_history_on_out_of_order_input():
    case = next(case for case in _ORACLE_PARITY_CASES if case["name"] == "out-of-order spike")
    points = [(float(step), float(value)) for step, value in case["points"]]
    tracker = loss_anomaly.LossAnomalyTracker()

    for index, point in enumerate(points):
        prefix = points[: index + 1]
        count = len(prefix)
        if count < 8 or (count - 8) % 10:
            continue

        tracker.advance(prefix)
        actual = tracker.detect(prefix)
        expected = _full_history_reference(prefix)
        if count == 18:
            assert _full_verdict(actual) is None
            assert _full_verdict(expected) == ("spike", "warning", 21.0, 6.0)
            assert expected["detail"] == 3.0
        else:
            assert _full_verdict(actual) == _full_verdict(expected)
            if actual is not None:
                assert actual["detail"] == pytest.approx(
                    expected["detail"], rel = 1e-12, abs = 1e-12
                )


def test_progress_entry_point_ignores_out_of_order_loss_and_continues_chronologically(
    settings_store, sink
):
    _configure(settings_store, sink)
    run_id = "run-rejects-out-of-order-loss"
    for step in range(1, 21):
        assert webhooks.observe_progress(run_id = run_id, step = step, loss = 2.0) is None

    assert webhooks.observe_progress(run_id = run_id, step = 19, loss = 6.0) is None
    for step in range(21, 29):
        assert webhooks.observe_progress(run_id = run_id, step = step, loss = 2.0) is None

    webhooks.wait_for_deliveries(5)
    alerts = [json.loads(request["body"]) for request in sink.requests]
    assert not any(
        alert["run"]["id"] == run_id and alert["event"] == "training.anomaly"
        for alert in alerts
    )


@pytest.mark.parametrize(
    "values,expected_chart",
    [([1.0] * 17 + [float("nan")], None), ([1.0] * 7, None)],
    ids = ["non-finite loss alerts while chart omits it", "short run is not judged"],
)
def test_progress_entry_point_handles_nonfinite_and_short_runs(
    values, expected_chart, settings_store, sink
):
    _configure(settings_store, sink)
    for step, loss in enumerate(values, start = 1):
        webhooks.observe_progress(run_id = "run-boundary", step = step, loss = loss)
    webhooks.wait_for_deliveries(5)

    expected_alerts = 1 if len(values) >= 8 and not math.isfinite(values[-1]) else 0
    alerts = [request for request in sink.requests if b'"event":"training.anomaly"' in request["body"]]
    assert len(alerts) == expected_alerts
    chart_series = [(step, loss) for step, loss in enumerate(values, 1) if math.isfinite(loss)]
    assert loss_anomaly.detect_loss_anomaly(chart_series) == expected_chart
    if alerts:
        anomaly = json.loads(alerts[0]["body"])["anomaly"]
        assert (anomaly["kind"], anomaly["severity"], anomaly["step"], anomaly["value"]) == (
            "nan", "critical", len(values), None
        )


@pytest.mark.parametrize(
    "endpoint, run_id, expected",
    [
        (
            "unreachable",
            "run-unreachable",
            {"ok": False, "attempts": 3, "status": None, "requests": 0},
        ),
        (
            (404,),
            "run-refused",
            {"ok": False, "attempts": 1, "status": 404, "requests": 1},
        ),
        (
            (500, 200),
            "run-retry-success",
            {"ok": True, "attempts": 2, "status": 200, "requests": 2},
        ),
        (
            (500,),
            "run-retry-fail",
            {"ok": False, "attempts": 3, "status": 500, "requests": 3},
        ),
    ],
    ids = ["unreachable", "refused", "retry-then-success", "retry-exhausted"],
)
def test_the_delivery_outcome_is_recorded_at_the_final_attempt(
    endpoint, run_id, expected, settings_store
):
    if endpoint == "unreachable":
        sink = None
        webhook_settings.set_webhook_settings(
            enabled = True, url = "http://127.0.0.1:1/hook", secret = ""
        )
    else:
        sink = _Sink(statuses = endpoint)
        _configure(settings_store, sink)
    try:
        assert webhooks.notify_terminal(
            run_id = run_id, status = "completed", metrics = {"step": 1}
        )
        webhooks.wait_for_deliveries(5)
        if sink is not None:
            assert len(sink.requests) == expected["requests"]
        history = [
            item for item in webhooks.get_deliveries() if item["run_id"] == run_id
        ]
        assert len(history) == 1
        delivery = history[0]
        assert delivery["ok"] is expected["ok"]
        assert delivery["attempts"] == expected["attempts"]
        assert delivery["status_code"] == expected["status"]
        assert bool(delivery["error"]) is (not expected["ok"])
        if sink is not None and len(sink.requests) > 1:
            delivery_ids = {
                json.loads(request["body"])["delivery_id"] for request in sink.requests
            }
            assert len(delivery_ids) == 1
    finally:
        if sink is not None:
            sink.close()


def test_a_hanging_endpoint_cannot_block_the_caller(settings_store):
    hanging = _Sink(delay = 2.0)
    try:
        _configure(settings_store, hanging)
        started = time.monotonic()
        assert webhooks.notify_terminal(
            run_id = "run-hanging", status = "completed", metrics = {"step": 1}
        )
        elapsed = time.monotonic() - started
        assert elapsed < 0.5, elapsed
        webhooks.wait_for_deliveries(8)
        delivery = next(
            item for item in webhooks.get_deliveries() if item["run_id"] == "run-hanging"
        )
        assert delivery["ok"] is False
        assert delivery["attempts"] == 3
    finally:
        hanging.close()

def _client(monkeypatch) -> TestClient:
    app = FastAPI()
    app.dependency_overrides[settings_routes._require_installation_owner] = lambda: None
    app.dependency_overrides[settings_routes.get_current_subject] = lambda: "test-owner"
    app.include_router(settings_routes.router, prefix = "/api/settings")
    return TestClient(app)


def test_training_backend_dispatch_carries_its_final_metrics(settings_store, sink):
    _configure(settings_store, sink)
    from core.training.training import TrainingBackend, TrainingProgress

    backend = TrainingBackend.__new__(TrainingBackend)
    backend.loss_history = [1.2, 0.8, 0.4]
    backend.eval_loss_history = [0.5]
    backend._progress = TrainingProgress(
        step = 30,
        total_steps = 30,
        epoch = 1.0,
        learning_rate = 2e-4,
        peak_memory_gb = 6.5,
        elapsed_seconds = 75.0,
    )
    backend._dispatch_terminal_webhook(
        run_id = "run-from-training",
        status = "completed",
        output_dir = "/models/run-from-training",
        final_step = 30,
        final_loss = 0.4,
        duration = 75.0,
        loss_history = [1.2, 0.8, 0.4],
    )
    webhooks.wait_for_deliveries(5)

    payload = json.loads(sink.requests[0]["body"])
    assert payload["event"] == "training.completed"
    assert payload["run"]["id"] == "run-from-training"
    assert payload["run"]["output_dir"] == "/models/run-from-training"
    assert payload["metrics"] == {
        "step": 30,
        "loss": 0.4,
        "eval_loss": 0.5,
        "best_loss": 0.4,
        "total_steps": 30,
        "epoch": 1.0,
        "learning_rate": 2e-4,
        "duration_seconds": 75.0,
        "peak_memory_gb": 6.5,
    }


def test_terminal_alert_survives_a_failed_database_finalize(monkeypatch):
    from core.training import training as training_module
    from core.training.training import TrainingBackend, TrainingProgress

    backend = TrainingBackend.__new__(TrainingBackend)
    backend._provenance_lock = threading.Lock()
    backend._lock = threading.Lock()
    backend.current_job_id = "run-db-down"
    backend._db_run_created = True
    backend._run_finalized = False
    backend._db_config = None
    backend._metric_buffer = []
    backend._progress = TrainingProgress(
        step = 5, loss = 0.5, total_steps = 5, elapsed_seconds = 3.0
    )
    backend.loss_history = [0.5]
    backend.eval_loss_history = []
    captured: list[dict] = []
    monkeypatch.setattr(
        training_module, "_DB_FINALIZE_RETRY_S", 0
    )
    monkeypatch.setattr(
        "storage.studio_db.finish_run",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("database unavailable")),
    )
    monkeypatch.setattr(
        webhooks, "notify_terminal", lambda **kwargs: captured.append(kwargs) or True
    )

    backend._finalize_run_in_db(status = "error", error_message = "boom")

    assert len(captured) == 1
    assert captured[0]["run_id"] == "run-db-down"
    assert captured[0]["status"] == "failed"
    assert captured[0]["metrics"]["step"] == 5
    assert captured[0]["error_message"] == "boom"

def test_settings_route_refuses_a_non_http_endpoint(settings_store, monkeypatch):
    webhook_settings.set_webhook_settings(enabled = False, url = "http://127.0.0.1:1/hook")
    client = _client(monkeypatch)
    response = client.put(
        "/api/settings/webhooks", json = {"enabled": True, "url": "ftp://example.com/hook"}
    )
    assert response.status_code == 400
    assert "http" in response.json()["detail"]


def test_test_route_reports_the_real_http_result(settings_store, sink, monkeypatch):
    _configure(settings_store, sink, enabled = False)
    client = _client(monkeypatch)
    response = client.post("/api/settings/webhooks/test", json = {})
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["status_code"] == 200
    assert body["attempts"] == 1
    assert body["signed"] is True
    assert body["signature_verified"] is True
    assert body["signature_header"].startswith("sha256=")
    assert len(sink.requests) == 1

    history = client.get("/api/settings/webhooks").json()["deliveries"]
    assert any(item["delivery_id"] == body["delivery_id"] for item in history)
    assert any(item["test"] for item in history)


@pytest.mark.parametrize(
    "run_id, first_status, expected_event",
    [
        ("run-double-finalize", "completed", "training.completed"),
        ("run-status-retry", "failed", "training.failed"),
    ],
    ids = ["same status twice", "different status second"],
)
def test_only_the_first_terminal_status_for_a_run_is_sent(
    run_id, first_status, expected_event, settings_store, sink
):
    _configure(settings_store, sink)
    assert webhooks.notify_terminal(
        run_id = run_id, status = first_status, metrics = {"step": 2}
    )
    assert (
        webhooks.notify_terminal(
            run_id = run_id, status = "completed", metrics = {"step": 4}
        )
        is False
    )
    webhooks.wait_for_deliveries(5)
    assert len(sink.requests) == 1
    assert json.loads(sink.requests[0]["body"])["event"] == expected_event


def test_concurrent_runs_are_not_cross_wired_or_left_running(settings_store, sink):
    _configure(settings_store, sink)
    with ThreadPoolExecutor(max_workers = 8) as pool:
        accepted = list(
            pool.map(
                lambda index: webhooks.notify_terminal(
                    run_id = f"run-concurrent-{index}",
                    status = "completed",
                    metrics = {"step": index},
                ),
                range(8),
            )
        )
    assert accepted == [True] * 8
    webhooks.wait_for_deliveries(10)

    payloads = [
        json.loads(request["body"])
        for request in sink.requests
        if json.loads(request["body"]).get("run", {}).get("id", "").startswith("run-concurrent-")
    ]
    assert len(payloads) == 8
    assert len({payload["run"]["id"] for payload in payloads}) == 8
    for payload in payloads:
        index = int(payload["run"]["id"].rsplit("-", 1)[1])
        assert payload["metrics"]["step"] == index
    assert webhook_delivery._threads == set()


def test_an_in_flight_delivery_is_drained_within_the_shutdown_budget(settings_store):
    slow = _Sink(delay = 0.25)
    try:
        _configure(settings_store, slow)
        assert webhooks.notify_terminal(
            run_id = "run-drain", status = "completed", metrics = {"step": 1}
        )
        started = time.monotonic()
        webhooks.wait_for_deliveries(2.0)
        assert time.monotonic() - started < 2.0
        assert len(slow.requests) == 1
        delivery = next(
            item for item in webhooks.get_deliveries() if item["run_id"] == "run-drain"
        )
        assert delivery["ok"] is True
    finally:
        slow.close()

def test_an_anomaly_that_stays_true_alerts_once(settings_store, sink):
    _configure(settings_store, sink)
    values = [1.0] * 11 + [
        20.0 if step % 10 == 2 else 1.0 for step in range(12, 61)
    ]
    for step, loss in enumerate(values, start = 1):
        webhooks.observe_progress(run_id = "run-persistent-anomaly", step = step, loss = loss)
    webhooks.wait_for_deliveries(5)

    anomalies = [
        json.loads(request["body"])
        for request in sink.requests
        if json.loads(request["body"])["event"] == "training.anomaly"
    ]
    assert len(anomalies) == 1
    assert anomalies[0]["run"]["id"] == "run-persistent-anomaly"
    assert anomalies[0]["anomaly"]["kind"] == "spike"


class _LogRecorder:
    def __init__(self):
        self.events: list[tuple[str, tuple]] = []

    def info(self, event, *args, **_kwargs):
        self.events.append((event, args))

    def warning(self, event, *args, **_kwargs):
        self.events.append((event, args))


def test_the_secret_is_stored_once_and_masked_everywhere_else(
    settings_store, sink, monkeypatch
):
    recorder = _LogRecorder()
    monkeypatch.setattr(settings_routes, "logger", recorder)
    client = _client(monkeypatch)
    secret = "super-secret-value"

    saved = client.put(
        "/api/settings/webhooks",
        json = {"enabled": True, "url": sink.url, "secret": secret},
    )
    loaded = client.get("/api/settings/webhooks")
    tested = client.post("/api/settings/webhooks/test", json = {})

    assert saved.status_code == 200 and loaded.status_code == 200 and tested.status_code == 200
    for response in (saved, loaded, tested):
        assert secret not in response.text
    assert secret not in json.dumps(webhooks.get_deliveries())
    assert secret not in json.dumps(recorder.events)
    saved_body = saved.json()
    assert saved_body["enabled"] is True
    assert saved_body["url"] == sink.url
    assert saved_body["has_secret"] is True
    assert "secret" not in saved_body
    assert saved_body["signature"]["algorithm"] == "hmac-sha256"
    assert webhook_settings.get_webhook_config()["secret"] == secret
    assert "secret" not in settings_store[webhook_settings.SETTINGS_KEY]
    from storage import credential_secrets

    assert (
        credential_secrets.get_secret(
            webhook_settings.SECRET_KIND, webhook_settings.SECRET_SCOPE
        )
        == secret
    )


def test_the_secret_is_encrypted_in_the_settings_database(settings_store, encrypted_store):
    secret = "only-in-ciphertext"
    webhook_settings.set_webhook_settings(
        enabled = True, url = "http://127.0.0.1:9/hook", secret = secret
    )
    assert "secret" not in settings_store[webhook_settings.SETTINGS_KEY]
    assert secret.encode("utf-8") not in encrypted_store.read_bytes()
    conn = sqlite3.connect(encrypted_store)
    try:
        ciphertext = conn.execute("SELECT ciphertext FROM credential_secrets").fetchone()[0]
    finally:
        conn.close()
    assert secret.encode("utf-8") not in ciphertext
    from storage import credential_secrets

    assert (
        credential_secrets.get_secret(
            webhook_settings.SECRET_KIND, webhook_settings.SECRET_SCOPE
        )
        == secret
    )


def test_a_legacy_secret_migrates_on_either_read_path(
    settings_store, encrypted_store, monkeypatch
):
    secret = "legacy-plaintext-secret"
    legacy = {
        "enabled": True,
        "url": "http://127.0.0.1:9/hook",
        "secret": secret,
    }
    client = _client(monkeypatch)

    settings_store[webhook_settings.SETTINGS_KEY] = dict(legacy)
    response = client.get("/api/settings/webhooks")
    assert response.status_code == 200
    assert response.json()["has_secret"] is True
    assert secret not in response.text
    assert "secret" not in settings_store[webhook_settings.SETTINGS_KEY]
    assert secret.encode("utf-8") not in encrypted_store.read_bytes()
    from storage import credential_secrets

    assert (
        credential_secrets.get_secret(
            webhook_settings.SECRET_KIND, webhook_settings.SECRET_SCOPE
        )
        == secret
    )

    settings_store[webhook_settings.SETTINGS_KEY] = dict(legacy)
    assert webhook_settings.get_webhook_config()["secret"] == secret
    assert "secret" not in settings_store[webhook_settings.SETTINGS_KEY]
    assert secret.encode("utf-8") not in encrypted_store.read_bytes()


def test_saving_without_a_secret_keeps_the_encrypted_one(settings_store, encrypted_store):
    webhook_settings.set_webhook_settings(
        enabled = True, url = "http://127.0.0.1:9/hook", secret = "keep-me"
    )
    saved = webhook_settings.set_webhook_settings(
        enabled = True, url = "http://127.0.0.1:9/other"
    )
    assert saved["has_secret"] is True
    assert webhook_settings.get_webhook_config()["secret"] == "keep-me"

def test_an_unreadable_secret_is_reported_sends_unsigned_and_recovers(
    settings_store, encrypted_store, sink, monkeypatch
):
    broken = "damaged-secret"
    recovered = "recovered-route-secret"
    _configure(settings_store, sink, secret = broken)
    conn = sqlite3.connect(encrypted_store)
    try:
        row = conn.execute("SELECT ciphertext FROM credential_secrets").fetchone()
        damaged = bytearray(row[0])
        damaged[-1] ^= 1
        conn.execute("UPDATE credential_secrets SET ciphertext = ?", (bytes(damaged),))
        conn.commit()
    finally:
        conn.close()

    assert webhook_settings.get_webhook_config()["secret"] == ""
    assert webhook_settings.get_webhook_settings()["has_secret"] is False
    assert webhook_settings.get_webhook_settings()["secret_unavailable"] is True
    assert webhooks.notify_terminal(
        run_id = "run-unreadable-secret", status = "completed", metrics = {"step": 1}
    )
    webhooks.wait_for_deliveries(5)

    request = next(
        item
        for item in sink.requests
        if json.loads(item["body"])["run"]["id"] == "run-unreadable-secret"
    )
    assert "x-unsloth-signature-256" not in request["headers"]
    delivery = next(
        item
        for item in webhooks.get_deliveries()
        if item["run_id"] == "run-unreadable-secret"
    )
    assert delivery["signed"] is False
    assert delivery["ok"] is True

    client = _client(monkeypatch)
    loaded = client.get("/api/settings/webhooks")
    loaded_body = loaded.json()
    assert loaded.status_code == 200
    assert loaded_body["has_secret"] is False
    assert loaded_body["secret_unavailable"] is True
    assert "secret" not in loaded_body
    assert broken not in loaded.text

    unsigned = client.post("/api/settings/webhooks/test", json = {})
    assert unsigned.status_code == 200
    assert unsigned.json()["signed"] is False
    assert unsigned.json()["signature_header"] is None
    assert "x-unsloth-signature-256" not in sink.requests[-1]["headers"]

    saved = client.put(
        "/api/settings/webhooks",
        json = {"enabled": True, "url": sink.url, "secret": recovered},
    )
    saved_body = saved.json()
    assert saved.status_code == 200
    assert saved_body["has_secret"] is True
    assert saved_body["secret_unavailable"] is False
    assert "secret" not in saved_body
    assert recovered not in saved.text

    signed = client.post("/api/settings/webhooks/test", json = {})
    assert signed.status_code == 200
    assert signed.json()["signed"] is True
    assert signed.json()["signature_verified"] is True
    request = sink.requests[-1]
    assert webhook_delivery.verify_signature(
        recovered,
        int(request["headers"]["x-unsloth-timestamp"]),
        request["body"],
        request["headers"]["x-unsloth-signature-256"],
    )


def test_legacy_migration_scrubs_plaintext_from_the_database_files(
    encrypted_store, monkeypatch
):
    import storage.studio_db as studio_db

    db_path = encrypted_store
    monkeypatch.setattr(studio_db, "studio_db_path", lambda: db_path)
    studio_db.reset_schema_state_for_tests()
    studio_db.close_wal_keeper_for(db_path)
    studio_db.open_wal_keeper()
    secret = "legacy-disk-secret"
    wal_path = db_path.with_name(db_path.name + "-wal")
    ledger_path = db_path.with_name(db_path.name + "-shm")
    try:
        studio_db.upsert_app_settings(
            {
                webhook_settings.SETTINGS_KEY: {
                    "enabled": True,
                    "url": "http://127.0.0.1:9/hook",
                    "secret": secret,
                }
            }
        )
        assert secret.encode("utf-8") in wal_path.read_bytes()

        assert webhook_settings.get_webhook_config()["secret"] == secret

        for path in (db_path, wal_path, ledger_path):
            assert secret.encode("utf-8") not in path.read_bytes(), path.name
    finally:
        studio_db.close_wal_keeper_for(db_path)
        studio_db.reset_schema_state_for_tests()


def test_a_read_without_a_legacy_secret_does_not_checkpoint(
    settings_store, encrypted_store, monkeypatch
):
    import storage.studio_db as studio_db

    reclaimed: list[bool] = []
    monkeypatch.setattr(
        studio_db, "_reclaim_freed_pages", lambda _conn: reclaimed.append(True)
    )
    webhook_settings.set_webhook_settings(
        enabled = True,
        url = "http://127.0.0.1:9/hook",
        secret = "already-stored",
    )
    reclaimed.clear()

    assert webhook_settings.get_webhook_config()["secret"] == "already-stored"
    assert reclaimed == []


def test_a_missing_secret_sends_unsigned(settings_store, sink):
    webhook_settings.set_webhook_settings(enabled = True, url = sink.url)
    assert webhooks.notify_terminal(
        run_id = "run-no-secret", status = "completed", metrics = {"step": 1}
    )
    webhooks.wait_for_deliveries(5)

    request = sink.requests[0]
    assert "x-unsloth-signature-256" not in request["headers"]
    delivery = next(
        item for item in webhooks.get_deliveries() if item["run_id"] == "run-no-secret"
    )
    assert delivery["signed"] is False
