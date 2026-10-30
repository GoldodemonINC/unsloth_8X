# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import math
import threading
import time
import uuid
from collections import deque
from typing import Any, Optional

from core.training.loss_anomaly import (
    CHECK_EVERY,
    MAX_RETAINED_POINTS,
    LossAnomalyTracker,
    RetainedSeries,
)
from core.training.webhook_delivery import (
    deliver,
    get_deliveries,
    spawn_delivery,
    wait_for_deliveries,
)
from core.training.webhook_payload import build_payload
from utils.webhook_settings import get_webhook_config, validate_webhook_url

MAX_TRACKED_RUNS = 8
MAX_TERMINAL_KEYS = 100

_progress: dict[str, dict] = {}
_progress_lock = threading.Lock()
_terminal_keys: set[str] = set()
_terminal_order: deque[str] = deque(maxlen = MAX_TERMINAL_KEYS)
_terminal_lock = threading.Lock()


def _enabled_config(config: Optional[dict]) -> Optional[dict]:
    config = config or get_webhook_config()
    if config.get("enabled") and config.get("url"):
        return config
    return None


def claim_terminal(run_id: str) -> bool:
    with _terminal_lock:
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
        config = _enabled_config(config)
        if config is None:
            return False
        if not claim_terminal(run_id):
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
        spawn_delivery(event, payload, config["url"], config.get("secret", ""))
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
                if len(_progress) >= MAX_TRACKED_RUNS:
                    stale = min(_progress, key = lambda key: _progress[key]["touched"])
                    _progress.pop(stale, None)
                state = {
                    "points": [],
                    "base": 0,
                    "total": 0,
                    "checked": 0,
                    "seen": set(),
                    "metrics": {},
                    "tracker": LossAnomalyTracker(),
                    "touched": 0.0,
                }
                _progress[run_id] = state
            state["touched"] = time.monotonic()
            points = state["points"]
            if points and step_number <= points[-1][0]:
                return None
            points.append((step_number, value))
            state["total"] += 1
            if len(points) > MAX_RETAINED_POINTS:
                dropped = len(points) - MAX_RETAINED_POINTS
                del points[:dropped]
                state["base"] += dropped
            if parent_guard is not None:
                parent_guard.observe(step_number, value)
            if metrics:
                state["metrics"] = metrics
            should_check = state["total"] >= 8 and (
                state["total"] >= state["checked"] + CHECK_EVERY
                or state["checked"] < 8
            )
            anomaly = None
            if should_check:
                state["checked"] = state["total"]
                tracker = state["tracker"]
                retained = RetainedSeries(points, state["base"], state["total"])
                tracker.advance(retained)
                anomaly = tracker.detect(retained)

        if anomaly is None:
            return None

        with _progress_lock:
            current = _progress.get(run_id)
            if current is None or anomaly["kind"] in current["seen"]:
                return None
            current["seen"].add(anomaly["kind"])
            latest_metrics = dict(current["metrics"])

        config = _enabled_config(config)
        if config is None:
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
        spawn_delivery("training.anomaly", payload, config["url"], config.get("secret", ""))
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
    result = deliver(
        event = "webhook.test",
        payload = payload,
        url = normalized_url,
        secret = secret,
        timeout = timeout,
        max_attempts = max_attempts,
    )
    result["url"] = normalized_url
    return result
