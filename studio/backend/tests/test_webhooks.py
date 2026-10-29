# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import math
import sqlite3
import threading
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.training import webhooks
from routes import settings as settings_routes
from utils import webhook_settings


class _Sink:

    def __init__(self, statuses: tuple[int, ...] = (200,), delay: float = 0.0):
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
        self.thread = threading.Thread(target = self.server.serve_forever, daemon = True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/hook"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


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
    monkeypatch.setattr(webhooks, "RETRY_DELAYS", (0.0, 0.0))
    monkeypatch.setattr(webhooks, "REQUEST_TIMEOUT_S", 0.5)


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


def test_signature_is_hmac_over_timestamp_and_raw_body():
    body = b'{"event":"training.completed"}'
    signature = webhooks.sign_body("topsecret", 1_700_000_000, body)
    assert signature.startswith("sha256=")
    assert webhooks.verify_signature("topsecret", 1_700_000_000, body, signature)
    assert not webhooks.verify_signature("wrong", 1_700_000_000, body, signature)
    assert not webhooks.verify_signature(
        "topsecret", 1_700_000_001, body, signature
    )
    assert not webhooks.verify_signature(
        "topsecret", 1_700_000_000, body + b" ", signature
    )
    assert webhooks.verify_signature("", 1, body, "") is False


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
    assert webhooks.verify_signature("topsecret", timestamp, request["body"], signature)
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

    chart = webhooks.detect_loss_anomaly(list(enumerate(values, start = 1)))
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
    tracker = webhooks._LossAnomalyTracker()
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
    tracker = webhooks._LossAnomalyTracker()

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
    assert webhooks.detect_loss_anomaly(chart_series) == expected_chart
    if alerts:
        anomaly = json.loads(alerts[0]["body"])["anomaly"]
        assert (anomaly["kind"], anomaly["severity"], anomaly["step"], anomaly["value"]) == (
            "nan", "critical", len(values), None
        )


def test_an_unreachable_endpoint_is_recorded_after_bounded_retries(settings_store):
    webhook_settings.set_webhook_settings(
        enabled = True, url = "http://127.0.0.1:1/hook", secret = ""
    )
    assert webhooks.notify_terminal(
        run_id = "run-bad-endpoint", status = "failed", metrics = {"step": 0}
    )
    webhooks.wait_for_deliveries(5)
    delivery = next(
        item for item in webhooks.get_deliveries() if item["run_id"] == "run-bad-endpoint"
    )
    assert delivery["ok"] is False
    assert delivery["attempts"] == 3
    assert delivery["status_code"] is None
    assert delivery["error"]


def test_a_refused_request_is_not_retried(settings_store):
    refused = _Sink(statuses = (404,))
    try:
        _configure(settings_store, refused)
        assert webhooks.notify_terminal(
            run_id = "run-refused", status = "completed", metrics = {"step": 1}
        )
        webhooks.wait_for_deliveries(5)
        assert len(refused.requests) == 1
        delivery = next(
            item for item in webhooks.get_deliveries() if item["run_id"] == "run-refused"
        )
        assert delivery["ok"] is False
        assert delivery["attempts"] == 1
        assert delivery["status_code"] == 404
    finally:
        refused.close()


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


def test_retry_until_success_and_retry_until_give_up(settings_store):
    success = _Sink(statuses = (500, 200))
    try:
        _configure(settings_store, success)
        assert webhooks.notify_terminal(
            run_id = "run-retry-success", status = "completed", metrics = {"step": 1}
        )
        webhooks.wait_for_deliveries(5)
        assert len(success.requests) == 2
        delivery = next(
            item
            for item in webhooks.get_deliveries()
            if item["run_id"] == "run-retry-success"
        )
        assert delivery["ok"] is True
        assert delivery["attempts"] == 2
        assert delivery["status_code"] == 200
    finally:
        success.close()

    failure = _Sink(statuses = (500,))
    try:
        _configure(settings_store, failure)
        assert webhooks.notify_terminal(
            run_id = "run-retry-fail", status = "completed", metrics = {"step": 1}
        )
        webhooks.wait_for_deliveries(5)
        assert len(failure.requests) == 3
        delivery = next(
            item for item in webhooks.get_deliveries() if item["run_id"] == "run-retry-fail"
        )
        assert delivery["ok"] is False
        assert delivery["attempts"] == 3
        assert delivery["status_code"] == 500
    finally:
        failure.close()


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


def test_settings_route_saves_without_leaking_the_secret(settings_store, sink, monkeypatch):
    _configure(settings_store, sink, enabled = False)
    client = _client(monkeypatch)

    response = client.put(
        "/api/settings/webhooks",
        json = {"enabled": True, "url": sink.url, "secret": "new-secret"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is True
    assert body["url"] == sink.url
    assert body["has_secret"] is True
    assert "secret" not in body
    assert body["signature"]["algorithm"] == "hmac-sha256"

    assert webhook_settings.get_webhook_config()["secret"] == "new-secret"


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


def test_a_run_finalized_twice_alerts_once(settings_store, sink):
    _configure(settings_store, sink)
    assert webhooks.notify_terminal(
        run_id = "run-double-finalize", status = "completed", metrics = {"step": 3}
    )
    assert (
        webhooks.notify_terminal(
            run_id = "run-double-finalize", status = "completed", metrics = {"step": 3}
        )
        is False
    )
    webhooks.wait_for_deliveries(5)
    assert len(sink.requests) == 1
    assert json.loads(sink.requests[0]["body"])["event"] == "training.completed"


def test_a_second_terminal_status_for_the_same_run_is_not_sent(settings_store, sink):
    _configure(settings_store, sink)
    assert webhooks.notify_terminal(
        run_id = "run-status-retry", status = "failed", metrics = {"step": 2}
    )
    assert (
        webhooks.notify_terminal(
            run_id = "run-status-retry", status = "completed", metrics = {"step": 4}
        )
        is False
    )
    webhooks.wait_for_deliveries(5)
    assert len(sink.requests) == 1
    assert json.loads(sink.requests[0]["body"])["event"] == "training.failed"


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
    assert webhooks._threads == set()


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


def test_retry_success_records_one_outcome_at_the_final_status(settings_store):
    success = _Sink(statuses = (500, 200))
    try:
        _configure(settings_store, success)
        assert webhooks.notify_terminal(
            run_id = "run-retry-record", status = "completed", metrics = {"step": 1}
        )
        webhooks.wait_for_deliveries(5)
        assert len(success.requests) == 2
        delivery_ids = {
            json.loads(request["body"])["delivery_id"] for request in success.requests
        }
        assert len(delivery_ids) == 1
        history = [
            item
            for item in webhooks.get_deliveries()
            if item["run_id"] == "run-retry-record"
        ]
        assert len(history) == 1
        assert history[0]["ok"] is True
        assert history[0]["attempts"] == 2
        assert history[0]["status_code"] == 200
    finally:
        success.close()


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


def test_the_settings_route_migrates_a_legacy_secret_on_read(
    settings_store, encrypted_store, monkeypatch
):
    secret = "legacy-through-get"
    settings_store[webhook_settings.SETTINGS_KEY] = {
        "enabled": True,
        "url": "http://127.0.0.1:9/hook",
        "secret": secret,
    }
    client = _client(monkeypatch)
    response = client.get("/api/settings/webhooks")
    assert response.status_code == 200
    assert response.json()["has_secret"] is True
    assert secret not in response.text
    assert "secret" not in settings_store[webhook_settings.SETTINGS_KEY]
    from storage import credential_secrets

    assert (
        credential_secrets.get_secret(
            webhook_settings.SECRET_KIND, webhook_settings.SECRET_SCOPE
        )
        == secret
    )


def test_saving_without_a_secret_keeps_the_encrypted_one(settings_store, encrypted_store):
    webhook_settings.set_webhook_settings(
        enabled = True, url = "http://127.0.0.1:9/hook", secret = "keep-me"
    )
    saved = webhook_settings.set_webhook_settings(
        enabled = True, url = "http://127.0.0.1:9/other"
    )
    assert saved["has_secret"] is True
    assert webhook_settings.get_webhook_config()["secret"] == "keep-me"


def test_a_legacy_plaintext_secret_migrates_and_is_deleted(settings_store, encrypted_store):
    secret = "legacy-plaintext-secret"
    settings_store[webhook_settings.SETTINGS_KEY] = {
        "enabled": True,
        "url": "http://127.0.0.1:9/hook",
        "secret": secret,
    }
    assert webhook_settings.get_webhook_config()["secret"] == secret
    assert "secret" not in settings_store[webhook_settings.SETTINGS_KEY]
    assert secret.encode("utf-8") not in encrypted_store.read_bytes()
    from storage import credential_secrets

    assert (
        credential_secrets.get_secret(
            webhook_settings.SECRET_KIND, webhook_settings.SECRET_SCOPE
        )
        == secret
    )


def test_an_undecryptable_secret_sends_unsigned_without_crashing(
    settings_store, sink, encrypted_store
):
    webhook_settings.set_webhook_settings(
        enabled = True, url = sink.url, secret = "damaged-secret"
    )
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


def test_the_settings_route_reports_an_unreadable_secret_and_recovers(
    settings_store, encrypted_store, sink, monkeypatch
):
    broken = "broken-route-secret"
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
    assert webhooks.verify_signature(
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
