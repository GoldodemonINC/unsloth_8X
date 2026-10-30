# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

SIGNATURE_HEADER = "X-Unsloth-Signature-256"
EVENT_HEADER = "X-Unsloth-Event"
DELIVERY_HEADER = "X-Unsloth-Delivery"
TIMESTAMP_HEADER = "X-Unsloth-Timestamp"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
