# SPDX-License-Identifier: NRC-1
# Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-1.md

"""Benchmarks: keep the eval report a finished run actually produced.

The eval engine already scores a baseline and a candidate and writes
``auto_eval_report.json``. What it did not have was a place to remember the
result once the job record was gone, so every comparison was re-run. This module
persists one small record per run under ``outputs/labs-benchmarks``. The
artifact's own folder is never modified: a benchmark is evidence *about* a
checkpoint, and rewriting the checkpoint's manifest would make a read-only
result look like part of the model.

Records are keyed by a random id, never by a user-supplied filename, and every
path is checked against the Studio output folder before it is read.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Optional

_MAX_RECORDS = 500
_ID = re.compile(r"^[a-f0-9]{16,32}$")
_DIRNAME = "labs-benchmarks"


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------


def outputs() -> Path:
    from utils.paths.storage_roots import outputs_root

    return outputs_root()


def root() -> Path:
    return outputs() / _DIRNAME


def runs_root() -> Path:
    return root() / "runs"


def records_root() -> Path:
    return root() / "records"


def create_run_dir() -> Path:
    """A fresh directory for one eval job's reports, outside every artifact."""
    runs = runs_root()
    runs.mkdir(parents=True, exist_ok=True)
    target = runs / uuid.uuid4().hex
    target.mkdir()
    return target


def _within(base: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(base.resolve())
    except (OSError, ValueError):
        return False
    return True


def _read(path: Path) -> Optional[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def _scores(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    scores: dict[str, float] = {}
    for key, raw in value.items():
        if isinstance(raw, (int, float)):
            scores[str(key)] = float(raw)
    return scores


def _type_name(path: str) -> str:
    from core.labs import artifacts

    try:
        return str(artifacts.find(path)["name"])
    except Exception:  # noqa: BLE001 - a Hub id / bare path has no artifact record
        return Path(path.rstrip("/\\")).name or path


def save_record(
    *,
    job_id: str,
    artifact: str,
    baseline: str,
    report: dict[str, Any],
    suites: Iterable[str],
    limit: Optional[int],
    batch_size: int,
    duration_seconds: Optional[float],
    output_dir: str,
) -> dict[str, Any]:
    """Persist one completed report and return the record the page lists."""
    candidate_scores = _scores(report.get("candidate_scores"))
    baseline_scores = _scores(report.get("baseline_scores"))
    if not candidate_scores and not baseline_scores:
        raise ValueError("The eval report has no scores to record.")
    comparison = report.get("comparison")
    if not isinstance(comparison, dict):
        comparison = {}
    record = {
        "id": uuid.uuid4().hex,
        "job_id": job_id,
        "artifact": str(artifact),
        "artifact_name": _type_name(artifact),
        "baseline": str(baseline),
        "baseline_name": _type_name(baseline),
        "suites": [str(suite) for suite in suites],
        "limit": limit,
        "batch_size": int(batch_size),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_seconds": duration_seconds,
        "candidate_scores": candidate_scores,
        "baseline_scores": baseline_scores,
        "comparison": {
            "entries": comparison.get("entries") or [],
            "aggregate_delta": comparison.get("aggregate_delta"),
            "verdict": comparison.get("verdict") or report.get("verdict"),
            "regressions": comparison.get("regressions") or [],
            "improved": comparison.get("improved") or [],
            "flat": comparison.get("flat") or [],
        },
        "verdict": report.get("verdict") or comparison.get("verdict") or "unavailable",
        "errors": list(report.get("errors") or []),
        "warnings": list(report.get("warnings") or []),
        "report_dir": str(output_dir),
    }
    directory = records_root()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{record['id']}.json"
    target.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    _prune(directory)
    return record


def _prune(directory: Path) -> None:
    try:
        files = sorted(
            (path for path in directory.glob("*.json")),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return
    for stale in files[_MAX_RECORDS:]:
        try:
            stale.unlink()
        except OSError:
            continue


def list_records(
    *, limit: int = 50, artifact: Optional[str] = None
) -> list[dict[str, Any]]:
    """Newest first, optionally only the records for one checkpoint."""
    directory = records_root()
    if not directory.is_dir():
        return []
    records: list[dict[str, Any]] = []
    try:
        files = sorted(
            directory.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return []
    wanted = str(artifact) if artifact else None
    for path in files:
        if len(records) >= max(1, min(int(limit), 200)):
            break
        record = _read(path)
        if record is None or not isinstance(record.get("id"), str):
            continue
        if wanted is not None and str(record.get("artifact")) != wanted:
            continue
        record.setdefault("created_at", "")
        records.append(record)
    return records


def delete_record(record_id: str) -> bool:
    if not _ID.match(str(record_id)):
        raise ValueError(f"Invalid benchmark record id: {record_id!r}")
    path = records_root() / f"{record_id}.json"
    if not path.is_file():
        return False
    path.unlink()
    return True


def find_record(record_id: str) -> Optional[dict[str, Any]]:
    if not _ID.match(str(record_id)):
        raise ValueError(f"Invalid benchmark record id: {record_id!r}")
    return _read(records_root() / f"{record_id}.json")
