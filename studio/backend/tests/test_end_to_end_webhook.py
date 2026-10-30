# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json

from core.training import loss_anomaly, webhook_delivery, webhooks
from utils import webhook_settings
from .test_webhooks import (
    encrypted_store,
    fast_delivery,
    isolated_proxy_environment,
    settings_store,
    sink,
)


def _feed(run_id: str, values: list[float]) -> list[dict]:
    alerts = []
    for step, loss in enumerate(values):
        found = webhooks.observe_progress(run_id = run_id, step = step, loss = loss)
        if found is not None:
            alerts.append(found)
    return alerts


def _anomaly_payloads(sink) -> list[dict]:
    return [
        json.loads(request["body"])
        for request in sink.requests
        if json.loads(request["body"])["event"] == "training.anomaly"
    ]


def test_a_post_eviction_spike_is_delivered_to_a_loopback_sink_exactly_once(
    settings_store, encrypted_store, fast_delivery, isolated_proxy_environment, sink
):
    secret = "end-to-end-signing-secret"
    webhook_settings.set_webhook_settings(enabled = True, url = sink.url, secret = secret)
    bound = loss_anomaly.MAX_RETAINED_POINTS
    spike_index = bound + 500
    spike_index += (7 - spike_index % 10) % 10
    tail = 40
    values = [2.15 if step % 2 else 2.0 for step in range(spike_index + tail)]
    values[spike_index] = 9.0
    run_id = "run-end-to-end-spike"

    alerts = _feed(run_id, values)
    webhooks.wait_for_deliveries(5)

    state = webhooks._progress[run_id]
    assert state["total"] == spike_index + tail
    assert state["base"] > 0
    assert len(state["points"]) == bound
    assert state["checked"] > spike_index + 1
    assert len(alerts) == 1
    assert alerts[0]["kind"] == "spike"
    assert alerts[0]["step"] == spike_index

    payloads = _anomaly_payloads(sink)
    assert len(payloads) == 1
    assert payloads[0]["run"]["id"] == run_id
    assert payloads[0]["anomaly"]["kind"] == "spike"
    assert payloads[0]["anomaly"]["step"] == spike_index

    delivered = next(
        request
        for request in sink.requests
        if json.loads(request["body"])["event"] == "training.anomaly"
    )
    assert webhook_delivery.verify_signature(
        secret,
        int(delivered["headers"]["x-unsloth-timestamp"]),
        delivered["body"],
        delivered["headers"]["x-unsloth-signature-256"],
    )

    history = [item for item in webhooks.get_deliveries() if item["run_id"] == run_id]
    assert len(history) == 1
    assert history[0]["ok"] is True
    assert history[0]["signed"] is True
    assert history[0]["status_code"] == 200
    assert history[0]["attempts"] == 1
    assert history[0]["anomaly_kind"] == "spike"


def test_a_disabled_config_delivers_nothing_even_when_an_anomaly_fires(
    settings_store, encrypted_store, fast_delivery, isolated_proxy_environment, sink
):
    secret = "end-to-end-signing-secret"
    webhook_settings.set_webhook_settings(enabled = False, url = sink.url, secret = secret)
    run_id = "run-end-to-end-disabled"
    values = [2.0] * 7 + [9.0] + [2.0] * 11

    alerts = _feed(run_id, values)
    webhooks.wait_for_deliveries(1)

    assert len(alerts) == 1
    assert alerts[0]["kind"] == "spike"
    assert sink.requests == []
    assert not any(item["run_id"] == run_id for item in webhooks.get_deliveries())
