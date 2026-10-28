# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cheap, read-only dataset checks before a long training run starts.

A run that dies at step 500 because row 4,201 is truncated costs hours of GPU
time. This module walks the rows the trainer would read and reports the four
failure modes that actually stop it: unparseable rows, rows with no text field,
exact duplicates, and sequences far longer than any context this run can hold.

Nothing is loaded into memory beyond the row being checked and a set of row
digests, so a multi-GB JSONL file is scanned a line at a time. The report is a
description, never a rewrite: a check that silently "fixed" a dataset would
make the manifest lie about what trained.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Optional

SUPPORTED_SUFFIXES = (".jsonl", ".ndjson", ".json", ".csv", ".parquet")

# A record is conversational when any of these hold the text the trainer reads.
TEXT_FIELDS = (
    "text",
    "content",
    "instruction",
    "prompt",
    "question",
    "input",
    "output",
    "response",
    "completion",
    "messages",
    "conversations",
)

DEFAULT_MAX_ROWS = 200_000
DEFAULT_LONG_ROW_CHARS = 32_768
MAX_SAMPLE_ISSUES = 5


class _IssueLog:
    """Counts every issue while keeping only the first few for the report."""

    def __init__(self, limit: int = MAX_SAMPLE_ISSUES) -> None:
        self.count = 0
        self.samples: list[dict[str, Any]] = []
        self._limit = limit

    def add(self, sample: dict[str, Any]) -> None:
        self.count += 1
        if len(self.samples) < self._limit:
            self.samples.append(sample)


def _clip(text: str, limit: int = 240) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _text_from(value: Any) -> Optional[str]:
    """The trainable text inside one field, or None when it carries none."""
    if isinstance(value, str):
        return value if value.strip() else None
    if isinstance(value, dict):
        for key in ("content", "text", "value"):
            found = _text_from(value.get(key))
            if found:
                return found
        return None
    if isinstance(value, list):
        parts = [found for item in value if (found := _text_from(item))]
        return "\n".join(parts) if parts else None
    return None


def _row_text(row: Any) -> tuple[Optional[str], Optional[str]]:
    """(text, field) for one record. A bare string is its own text."""
    if isinstance(row, str):
        return (row, "text") if row.strip() else (None, None)
    if not isinstance(row, dict):
        return (None, None)
    for field in TEXT_FIELDS:
        if field not in row:
            continue
        found = _text_from(row[field])
        if found:
            return found, field
    return None, None


def _row_digest(row: Any) -> Optional[str]:
    if isinstance(row, str):
        payload = row
    else:
        try:
            payload = json.dumps(row, sort_keys = True, separators = (",", ":"), default = str)
        except (TypeError, ValueError):
            payload = repr(row)
    if not payload:
        return None
    return hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()


def _iter_jsonl(path: Path, invalid: _IssueLog) -> Iterator[Any]:
    with path.open("r", encoding = "utf-8-sig", errors = "replace") as stream:
        for number, line in enumerate(stream, start = 1):
            text = line.strip()
            if not text:
                continue
            try:
                yield json.loads(text)
            except json.JSONDecodeError as error:
                invalid.add(
                    {
                        "file": path.name,
                        "line": number,
                        "error": _clip(f"{error.msg} at column {error.colno}"),
                    }
                )


def _iter_json(path: Path, invalid: _IssueLog) -> Iterator[Any]:
    try:
        payload = json.loads(path.read_text(encoding = "utf-8-sig", errors = "replace"))
    except (OSError, json.JSONDecodeError) as error:
        invalid.add(
            {
                "file": path.name,
                "line": 0,
                "error": _clip(f"the file is not valid JSON: {error}"),
            }
        )
        return
    rows: Any = payload
    if isinstance(payload, dict):
        rows = next(
            (
                payload[key]
                for key in ("data", "rows", "examples", "records")
                if isinstance(payload.get(key), list)
            ),
            [payload],
        )
    if not isinstance(rows, list):
        invalid.add(
            {
                "file": path.name,
                "line": 0,
                "error": (
                    "expected a JSON array of rows, or an object holding one under "
                    "data/rows/examples/records"
                ),
            }
        )
        return
    yield from rows


def _iter_csv(path: Path) -> Iterator[Any]:
    with path.open("r", encoding = "utf-8-sig", errors = "replace", newline = "") as stream:
        yield from csv.DictReader(stream)


def _iter_parquet(path: Path) -> Iterator[Any]:
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415 - optional dependency, only parquet needs it
    except ImportError as error:  # pragma: no cover - depends on the host image
        raise ValueError(
            "Parquet datasets need pyarrow in the Studio environment. "
            "Convert the file to JSONL or install pyarrow, then run the check again."
        ) from error
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size = 1024):
        yield from batch.to_pylist()


def _iter_file(path: Path, invalid: _IssueLog) -> Iterator[Any]:
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        yield from _iter_jsonl(path, invalid)
    elif suffix == ".json":
        yield from _iter_json(path, invalid)
    elif suffix == ".csv":
        yield from _iter_csv(path)
    elif suffix == ".parquet":
        yield from _iter_parquet(path)
    else:
        raise ValueError(f"Unsupported dataset file {path.name!r}. Use JSONL, JSON, CSV or Parquet.")


def _collect_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(
                f"{path.name!r} is not a dataset file this check can read. "
                "Use JSONL, JSON, CSV or Parquet."
            )
        return [path]
    files = sorted(
        child
        for child in path.rglob("*")
        if child.is_file()
        and child.suffix.lower() in SUPPORTED_SUFFIXES
        and not child.name.startswith(".")
    )
    if not files:
        raise ValueError(
            f"No dataset files under {path}. Looked for {', '.join(SUPPORTED_SUFFIXES)}."
        )
    return files


def scan(
    source: str | Path,
    *,
    max_rows: int = DEFAULT_MAX_ROWS,
    long_row_chars: int = DEFAULT_LONG_ROW_CHARS,
) -> dict[str, Any]:
    """Scan *source* without writing anything, and describe what would break a run."""
    if max_rows < 1:
        raise ValueError("max_rows must be at least 1.")
    if long_row_chars < 1:
        raise ValueError("long_row_chars must be at least 1.")

    path = Path(source).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if not path.is_file() and not path.is_dir():
        raise ValueError(f"Dataset path is neither a file nor a folder: {path}")

    files = _collect_files(path)
    invalid = _IssueLog()
    missing_text = _IssueLog()
    duplicates = _IssueLog()
    long_rows = _IssueLog()
    empty_rows = 0
    rows = 0
    checked = 0
    truncated = False
    text_fields: dict[str, int] = {}
    seen: set[str] = set()
    warnings: list[str] = []

    for file in files:
        if truncated:
            break
        try:
            for row in _iter_file(file, invalid):
                rows += 1
                if checked >= max_rows:
                    truncated = True
                    break
                checked += 1
                text, field = _row_text(row)
                if not text:
                    empty = not isinstance(row, dict) or all(
                        value in (None, "", [], {}) for value in row.values()
                    )
                    if empty:
                        empty_rows += 1
                        continue
                    missing_text.add(
                        {
                            "file": file.name,
                            "row": checked,
                            "keys": sorted(row)[:8] if isinstance(row, dict) else [],
                        }
                    )
                    continue
                text_fields[field or "text"] = text_fields.get(field or "text", 0) + 1
                digest = _row_digest(row)
                if digest is not None:
                    if digest in seen:
                        duplicates.add(
                            {
                                "file": file.name,
                                "row": checked,
                                "preview": _clip(text, 120),
                            }
                        )
                    else:
                        seen.add(digest)
                if len(text) > long_row_chars:
                    long_rows.add(
                        {
                            "file": file.name,
                            "row": checked,
                            "chars": len(text),
                            "preview": _clip(text, 120),
                        }
                    )
        except ValueError:
            raise
        except OSError as error:
            invalid.add({"file": file.name, "line": 0, "error": _clip(str(error))})

    if truncated:
        warnings.append(
            f"Stopped after {checked:,} rows; the dataset holds more. "
            "Raise the row limit to scan all of it."
        )
    if duplicates.count:
        warnings.append("Exact duplicate rows repeat in the file.")
    if long_rows.count:
        warnings.append(
            f"Rows longer than {long_row_chars:,} characters are cut by the tokenizer, "
            "so their tail is never trained on."
        )

    # Duplicates and long rows are quality warnings, not parse failures; the run
    # will still start and train. Only rows the trainer cannot read block it.
    blocking = invalid.count + missing_text.count
    return {
        "path": str(path),
        "files": len(files),
        "file_names": [file.name for file in files[:20]],
        "format": path.suffix.lower().lstrip(".") if path.is_file() else "folder",
        "rows": rows,
        "checked_rows": checked,
        "truncated": truncated,
        "invalid_rows": invalid.count,
        "empty_rows": empty_rows,
        "missing_text_rows": missing_text.count,
        "duplicate_rows": duplicates.count,
        "long_rows": long_rows.count,
        "blocking_rows": blocking,
        "text_field": max(text_fields, key = text_fields.get) if text_fields else None,
        "long_row_chars": long_row_chars,
        "samples": {
            "invalid": invalid.samples,
            "missing_text": missing_text.samples,
            "duplicates": duplicates.samples,
            "long_rows": long_rows.samples,
        },
        "warnings": warnings,
        "ok": blocking == 0,
    }
