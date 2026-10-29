# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import hashlib
import heapq
import hmac
import json
import math
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any, Iterable, Optional
from urllib import error as urlerror
from urllib import request as urlrequest

from utils.webhook_settings import get_webhook_config, validate_webhook_url

MAX_ATTEMPTS = 3
REQUEST_TIMEOUT_S = 5.0
RETRY_DELAYS = (0.5, 1.5)
MAX_DELIVERIES = 25
MAX_TERMINAL_KEYS = 100
CHECK_EVERY = 10

SIGNATURE_HEADER = "X-Unsloth-Signature-256"
EVENT_HEADER = "X-Unsloth-Event"
DELIVERY_HEADER = "X-Unsloth-Delivery"
TIMESTAMP_HEADER = "X-Unsloth-Timestamp"

_deliveries: deque[dict] = deque(maxlen = MAX_DELIVERIES)
_delivery_lock = threading.Lock()
_threads: set[threading.Thread] = set()
_terminal_keys: set[str] = set()
_terminal_order: deque[str] = deque(maxlen = MAX_TERMINAL_KEYS)
_progress: dict[str, dict] = {}
_progress_lock = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _median(values: list[float]) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 0:
        return (ordered[middle - 1] + ordered[middle]) / 2
    return ordered[middle]


class _PrefixMedian:
    def __init__(self):
        self.low: list[float] = []
        self.high: list[float] = []

    def add(self, value: float) -> None:
        if self.low and value <= -self.low[0]:
            heapq.heappush(self.low, -value)
        else:
            heapq.heappush(self.high, value)
        while len(self.low) > len(self.high) + 1:
            heapq.heappush(self.high, -heapq.heappop(self.low))
        while len(self.low) < len(self.high):
            heapq.heappush(self.low, -heapq.heappop(self.high))

    def median(self) -> float:
        if not self.low:
            return self.high[0] if self.high else math.nan
        if len(self.low) > len(self.high):
            return -self.low[0]
        if len(self.low) < len(self.high):
            return self.high[0]
        return (-self.low[0] + self.high[0]) / 2


class _WindowMedian:
    def __init__(self):
        self.low: list[tuple[float, int]] = []
        self.high: list[tuple[float, int]] = []
        self.side: dict[int, int] = {}
        self.low_count = 0
        self.high_count = 0

    def add(self, index: int, value: float) -> None:
        if self.low and value <= -self.low[0][0]:
            heapq.heappush(self.low, (-value, index))
            self.side[index] = 0
            self.low_count += 1
        else:
            heapq.heappush(self.high, (value, index))
            self.side[index] = 1
            self.high_count += 1
        self._rebalance()

    def remove(self, index: int) -> None:
        side = self.side.pop(index, None)
        if side is None:
            return
        if side == 0:
            self.low_count -= 1
        else:
            self.high_count -= 1
        self._rebalance()

    def _prune(self) -> None:
        while self.low and self.low[0][1] not in self.side:
            heapq.heappop(self.low)
        while self.high and self.high[0][1] not in self.side:
            heapq.heappop(self.high)

    def _rebalance(self) -> None:
        self._prune()
        while self.low_count > self.high_count + 1:
            value, index = heapq.heappop(self.low)
            self.low_count -= 1
            heapq.heappush(self.high, (-value, index))
            self.side[index] = 1
            self.high_count += 1
            self._prune()
        while self.low_count < self.high_count:
            value, index = heapq.heappop(self.high)
            self.high_count -= 1
            heapq.heappush(self.low, (-value, index))
            self.side[index] = 0
            self.low_count += 1
            self._prune()

    def median(self) -> float:
        self._prune()
        if not self.low:
            return self.high[0][0] if self.high else math.nan
        if self.low_count > self.high_count:
            return -self.low[0][0]
        if self.low_count < self.high_count:
            return self.high[0][0]
        return (-self.low[0][0] + self.high[0][0]) / 2

    def entries(self) -> int:
        return len(self.low) + len(self.high)

    def rebuild(self, points: list[tuple[float, float]], start: int, end: int) -> None:
        self.low.clear()
        self.high.clear()
        self.side.clear()
        self.low_count = 0
        self.high_count = 0
        for index in range(start, end):
            self.add(index, points[index][1])


class _LossAnomalyTracker:
    def __init__(self):
        self.nan_step: Optional[float] = None
        self.spike_scanned = 0
        self.spike_worst: Optional[dict] = None
        self.prefix = _PrefixMedian()
        self.prefix_count = 0
        self.last = _WindowMedian()
        self.last_start = 0
        self.last_sum = 0.0
        self.last_sum_iy = 0.0
        self.last_sums_at = 0
        self.plateau = _WindowMedian()
        self.plateau_start = 0
        self.plateau_min: deque[tuple[int, float]] = deque()
        self.plateau_max: deque[tuple[int, float]] = deque()

    def advance(self, points: list[tuple[float, float]]) -> None:
        total = len(points)
        if self.nan_step is not None:
            return

        third = total // 3
        while self.prefix_count < third:
            self.prefix.add(points[self.prefix_count][1])
            self.prefix_count += 1

        last_start = total - third
        plateau_length = min(total, max(10, total // 4))
        plateau_start = total - plateau_length

        for index in range(self.spike_scanned, total):
            value = points[index][1]
            if not math.isfinite(value):
                self.nan_step = points[index][0]
                return
            self.last.add(index, value)
            self.last_sum_iy += (index - self.last_start) * value
            self.last_sum += value
            self.plateau.add(index, value)
            while self.plateau_min and self.plateau_min[-1][1] >= value:
                self.plateau_min.pop()
            self.plateau_min.append((index, value))
            while self.plateau_max and self.plateau_max[-1][1] <= value:
                self.plateau_max.pop()
            self.plateau_max.append((index, value))

            if index >= 5:
                baseline = _median(
                    [previous for _step, previous in points[max(0, index - 20) : index]]
                )
                if math.isfinite(baseline) and baseline > 0:
                    ratio = value / baseline
                    jump = value - baseline
                    if ratio >= 2.5 and jump >= 0.5:
                        if self.spike_worst is None or ratio > self.spike_worst["detail"]:
                            self.spike_worst = {
                                "kind": "spike",
                                "severity": "warning",
                                "step": points[index][0],
                                "value": value,
                                "detail": ratio,
                            }

        while self.last_start < last_start:
            value = points[self.last_start][1]
            self.last_sum_iy = self.last_sum_iy - self.last_sum + value
            self.last_sum -= value
            self.last.remove(self.last_start)
            self.last_start += 1
        while self.plateau_start < plateau_start:
            self.plateau.remove(self.plateau_start)
            self.plateau_start += 1
        while self.plateau_min and self.plateau_min[0][0] < self.plateau_start:
            self.plateau_min.popleft()
        while self.plateau_max and self.plateau_max[0][0] < self.plateau_start:
            self.plateau_max.popleft()

        if self.last.entries() > int(1.5 * (total - self.last_start)) + 512:
            self.last.rebuild(points, self.last_start, total)
            self._refresh_last_sums(points, total)
        elif total - self.last_sums_at >= 10000:
            self._refresh_last_sums(points, total)
        if self.plateau.entries() > int(1.25 * (total - self.plateau_start)) + 512:
            self.plateau.rebuild(points, self.plateau_start, total)
        self.spike_scanned = total

    def _refresh_last_sums(self, points: list[tuple[float, float]], total: int) -> None:
        self.last_sum = 0.0
        self.last_sum_iy = 0.0
        for offset, index in enumerate(range(self.last_start, total)):
            value = points[index][1]
            self.last_sum += value
            self.last_sum_iy += offset * value
        self.last_sums_at = total

    def detect(self, points: list[tuple[float, float]]) -> Optional[dict]:
        total = len(points)
        if total < 8:
            return None
        if self.nan_step is not None:
            return {
                "kind": "nan",
                "severity": "critical",
                "step": self.nan_step,
                "value": None,
                "detail": 0.0,
            }

        if total >= 30:
            start_median = self.prefix.median()
            end_median = self.last.median()
            if (
                math.isfinite(start_median)
                and start_median > 0
                and end_median > start_median * 1.25
            ):
                count = total - self.last_start
                trend = self.last_sum_iy - (count - 1) / 2 * self.last_sum
                if trend > 0:
                    step, value = points[-1]
                    return {
                        "kind": "divergence",
                        "severity": "critical",
                        "step": step,
                        "value": value,
                        "detail": end_median / start_median,
                    }

        if total >= 12 and self.spike_worst is not None:
            return dict(self.spike_worst)

        if total >= 40:
            tail_median = self.plateau.median()
            tail_values_start = points[self.plateau_start][1]
            step, value = points[-1]
            spread = self.plateau_max[0][1] - self.plateau_min[0][1]
            improvement = (
                (tail_values_start - value) / tail_values_start if tail_values_start > 0 else 1.0
            )
            if (
                math.isfinite(tail_median)
                and tail_median > 0.1
                and spread / tail_median <= 0.03
                and improvement <= 0.01
            ):
                return {
                    "kind": "plateau",
                    "severity": "warning",
                    "step": step,
                    "value": value,
                    "detail": spread / tail_median,
                }

        return None


def detect_loss_anomaly(points: Iterable[tuple[float, float]]) -> Optional[dict]:
    series = list(points)
    if len(series) < 8:
        return None

    for step, value in series:
        if not math.isfinite(value):
            return {
                "kind": "nan",
                "severity": "critical",
                "step": step,
                "value": None,
                "detail": 0.0,
            }

    ordered = sorted(
        ((step, value) for step, value in series if math.isfinite(step)),
        key = lambda point: point[0],
    )
    if len(ordered) < 8:
        return None

    tracker = _LossAnomalyTracker()
    tracker.advance(ordered)
    return tracker.detect(ordered)


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


def _locator(run_id: str) -> dict:
    return {
        "kind": "studio_run",
        "run_id": run_id,
        "path": f"/studio?run={run_id}",
    }


def build_payload(
    *,
    event: str,
    delivery_id: str,
    timestamp: int,
    run_id: Optional[str],
    status: Optional[str],
    metrics: Optional[dict],
    output_dir: Optional[str] = None,
    error_message: Optional[str] = None,
    anomaly: Optional[dict] = None,
    test: bool = False,
) -> dict:
    run_id = (run_id or "").strip() or None
    return {
        "event": event,
        "delivery_id": delivery_id,
        "sent_at": _now_iso(),
        "test": test,
        "run": {
            "id": run_id,
            "status": status,
            "error": error_message,
            "output_dir": output_dir,
        },
        "metrics": metrics or {},
        "anomaly": anomaly,
        "locator": _locator(run_id) if run_id else None,
        "signature": {
            "algorithm": "hmac-sha256",
            "header": SIGNATURE_HEADER,
            "timestamp": timestamp,
            "signed_content": "{timestamp}.{raw request body}",
        },
    }


def _post(
    url: str, body: bytes, headers: dict[str, str], timeout: float
) -> tuple[Optional[int], Optional[str], Optional[str]]:
    request = urlrequest.Request(url, data = body, headers = headers, method = "POST")
    try:
        with urlrequest.urlopen(request, timeout = timeout) as response:
            status = int(response.status)
            text = response.read(512).decode("utf-8", errors = "replace")
            return status, text, None
    except urlerror.HTTPError as exc:
        try:
            text = exc.read(512).decode("utf-8", errors = "replace")
        except Exception:
            text = None
        return int(exc.code), text, f"HTTP {exc.code}"
    except Exception as exc:
        return None, None, str(exc) or exc.__class__.__name__


def _record(record: dict) -> None:
    with _delivery_lock:
        _deliveries.append(record)


def get_deliveries() -> list[dict]:
    with _delivery_lock:
        return list(reversed(_deliveries))


def _deliver(
    *,
    event: str,
    payload: dict,
    url: str,
    secret: str,
    timeout: Optional[float] = None,
    max_attempts: Optional[int] = None,
    record: bool = True,
) -> dict:
    timeout = REQUEST_TIMEOUT_S if timeout is None else timeout
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
    ok = False
    for attempt in range(1, max_attempts + 1):
        attempts = attempt
        status_code, response_text, last_error = _post(url, body, headers, timeout)
        if status_code is not None and 200 <= status_code < 300:
            ok = True
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
        "response": (response_text or "")[:200] or None,
        "duration_ms": int((time.monotonic() - started) * 1000),
        "signed": bool(signature),
        "anomaly_kind": (payload.get("anomaly") or {}).get("kind"),
        "test": bool(payload.get("test")),
    }
    if record:
        _record(entry)
    entry["signature_header"] = signature or None
    entry["signature_verified"] = (
        verify_signature(secret, timestamp, body, signature) if signature else False
    )
    return entry


def _spawn(event: str, payload: dict, url: str, secret: str) -> None:
    def target() -> None:
        try:
            _deliver(event = event, payload = payload, url = url, secret = secret)
        except Exception:
            pass
        finally:
            with _delivery_lock:
                _threads.discard(threading.current_thread())

    thread = threading.Thread(target = target, daemon = True, name = f"webhook-{event}")
    with _delivery_lock:
        _threads.add(thread)
    thread.start()


def _claim_terminal(run_id: str) -> bool:
    with _delivery_lock:
        if run_id in _terminal_keys:
            return False
        if len(_terminal_order) == _terminal_order.maxlen:
            oldest = _terminal_order[0]
            _terminal_keys.discard(oldest)
        _terminal_keys.add(run_id)
        _terminal_order.append(run_id)
    return True


def notify_terminal(
    *,
    run_id: Optional[str],
    status: str,
    metrics: Optional[dict] = None,
    output_dir: Optional[str] = None,
    error_message: Optional[str] = None,
    config: Optional[dict] = None,
) -> bool:
    try:
        run_id = (run_id or "").strip()
        if not run_id:
            return False
        with _progress_lock:
            _progress.pop(run_id, None)
        config = config or get_webhook_config()
        if not config.get("enabled") or not config.get("url"):
            return False
        if not _claim_terminal(run_id):
            return False
        event = {
            "completed": "training.completed",
            "failed": "training.failed",
            "stopped": "training.stopped",
        }.get(status, "training.finished")
        timestamp = int(time.time())
        delivery_id = uuid.uuid4().hex
        payload = build_payload(
            event = event,
            delivery_id = delivery_id,
            timestamp = timestamp,
            run_id = run_id,
            status = status,
            metrics = metrics,
            output_dir = output_dir,
            error_message = error_message,
        )
        _spawn(event, payload, config["url"], config.get("secret", ""))
        return True
    except Exception:
        return False


def observe_progress(
    *,
    run_id: Optional[str],
    step: Any,
    loss: Any,
    metrics: Optional[dict] = None,
    config: Optional[dict] = None,
    parent_guard: Any = None,
) -> Optional[dict]:
    try:
        run_id = (run_id or "").strip()
        if not run_id:
            return None
        try:
            step_number = float(step)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(step_number):
            return None
        try:
            value = float(loss)
        except (TypeError, ValueError):
            return None

        with _progress_lock:
            state = _progress.get(run_id)
            if state is None:
                state = {
                    "points": [],
                    "checked": 0,
                    "seen": set(),
                    "metrics": {},
                    "tracker": _LossAnomalyTracker(),
                }
                _progress[run_id] = state
            points = state["points"]
            if points and step_number <= points[-1][0]:
                return None
            points.append((step_number, value))
            if parent_guard is not None:
                parent_guard.observe(step_number, value)
            if metrics:
                state["metrics"] = metrics
            should_check = len(points) >= 8 and (
                len(points) >= state["checked"] + CHECK_EVERY or state["checked"] < 8
            )
            anomaly = None
            if should_check:
                state["checked"] = len(points)
                tracker = state["tracker"]
                tracker.advance(points)
                anomaly = tracker.detect(points)

        if anomaly is None:
            return None

        with _progress_lock:
            current = _progress.get(run_id)
            if current is None or anomaly["kind"] in current["seen"]:
                return None
            current["seen"].add(anomaly["kind"])
            latest_metrics = dict(current["metrics"])

        config = config or get_webhook_config()
        if not config.get("enabled") or not config.get("url"):
            return anomaly

        timestamp = int(time.time())
        payload = build_payload(
            event = "training.anomaly",
            delivery_id = uuid.uuid4().hex,
            timestamp = timestamp,
            run_id = run_id,
            status = "running",
            metrics = latest_metrics,
            anomaly = anomaly,
        )
        _spawn("training.anomaly", payload, config["url"], config.get("secret", ""))
        return anomaly
    except Exception:
        return None


def send_test_event(
    *,
    url: str,
    secret: str = "",
    timeout: Optional[float] = None,
    max_attempts: Optional[int] = None,
) -> dict:
    normalized_url = validate_webhook_url(url)
    timestamp = int(time.time())
    delivery_id = uuid.uuid4().hex
    payload = build_payload(
        event = "webhook.test",
        delivery_id = delivery_id,
        timestamp = timestamp,
        run_id = None,
        status = "test",
        metrics = {"step": 0},
        test = True,
    )
    result = _deliver(
        event = "webhook.test",
        payload = payload,
        url = normalized_url,
        secret = secret,
        timeout = timeout,
        max_attempts = max_attempts,
    )
    result["url"] = normalized_url
    return result


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
