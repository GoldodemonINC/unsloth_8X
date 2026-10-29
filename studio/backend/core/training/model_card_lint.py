# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

CHECK_IDS = ("base_model", "license", "dataset", "evaluation")
CARD_NAMES = ("README.md", "README.MD", "readme.md", "MODEL_CARD.md", "model_card.md")
MAX_CARD_BYTES = 2 * 1024 * 1024

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_BASE_MODEL_BODY = re.compile(
    r"(?im)^\s*[-*]?\s*(?:base\s*model|base_model)\s*[:\-]\s*(\S.*)$"
)
_LICENSE_BODY = re.compile(
    r"(?im)^\s*[-*]?\s*(?:license|licence|licensing)\s*[:\-]\s*(\S.*)$"
)
_DATASET_BODY = re.compile(r"(?im)^\s*[-*]?\s*(?:dataset|datasets|training data)\s*[:\-]\s*(\S.*)$")
_METRIC_WORD = re.compile(
    r"(?i)\b(?:score|accuracy|acc|f1|loss|perplexity|mmlu|hellaswag|arc|gsm8k|"
    r"win\s*rate|pass@\d+|bleu|rouge|eval|benchmark)\b"
)
_NUMBER = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?%?")
_MODEL_INDEX_VALUE = re.compile(r"(?im)^\s*value\s*:\s*[-+]?\d")


def find_model_card(path: Path) -> Optional[Path]:
    if path.is_file():
        return path if path.name.lower() in {name.lower() for name in CARD_NAMES} else None
    if not path.is_dir():
        return None
    for name in CARD_NAMES:
        candidate = path / name
        if candidate.is_file():
            return candidate
    return None


def read_model_card(path: Path) -> tuple[Optional[str], Optional[str], Optional[str]]:
    card = find_model_card(path)
    if card is None:
        return None, None, "No README.md or MODEL_CARD.md was found for this run."
    try:
        size = card.stat().st_size
    except OSError as error:
        return None, str(card), f"The model card could not be inspected: {error}"
    if size > MAX_CARD_BYTES:
        return None, str(card), "The model card is larger than 2 MB and was not read."
    try:
        text = card.read_text(encoding = "utf-8-sig", errors = "replace")
    except OSError as error:
        return None, str(card), f"The model card could not be read: {error}"
    return text, str(card), None


def lint_model_card(text: str) -> dict:
    front, body = _split_front_matter(text or "")
    checks = [
        _check_base_model(front, body),
        _check_license(front, body),
        _check_dataset(front, body),
        _check_evaluation(front, body),
    ]
    missing = [check["id"] for check in checks if check["status"] == "missing"]
    return {
        "ok": not missing,
        "missing": missing,
        "checks": checks,
    }


def lint_model_card_path(path: Path) -> dict:
    text, source, error = read_model_card(path)
    if text is None:
        return {
            "card_found": False,
            "source_path": source,
            "reason": error,
            "ok": False,
            "missing": list(CHECK_IDS),
            "checks": [
                {"id": check_id, "status": "missing", "evidence": None}
                for check_id in CHECK_IDS
            ],
        }
    report = lint_model_card(text)
    return {
        "card_found": True,
        "source_path": source,
        "reason": None,
        **report,
    }


def _split_front_matter(text: str) -> tuple[str, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return "", text
    for index in range(1, len(lines)):
        if lines[index].strip() in {"---", "..."}:
            return "\n".join(lines[1:index]), "\n".join(lines[index + 1 :])
    return "", text


def _front_field(front: str, key: str) -> Optional[str]:
    pattern = re.compile(rf"(?im)^\s*{re.escape(key)}\s*:\s*(.*)$")
    match = pattern.search(front)
    if not match:
        return None
    value = match.group(1).strip().strip("\"'")
    if value:
        return value
    following = front[match.end() :].splitlines()
    for line in following:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("-"):
            return stripped[1:].strip().strip("\"'")
        return None
    return None


def _evidence(value: str, limit: int = 180) -> str:
    compact = " ".join(value.split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def _result(check_id: str, evidence: Optional[str]) -> dict:
    if evidence:
        return {"id": check_id, "status": "present", "evidence": _evidence(evidence)}
    return {"id": check_id, "status": "missing", "evidence": None}


def _heading_evidence(body: str, pattern: str) -> Optional[str]:
    for heading, section in _sections(body):
        if re.search(pattern, heading):
            first = next((line.strip() for line in section.splitlines() if line.strip()), "")
            if first:
                return f"{heading}: {first}"
    return None


def _check_base_model(front: str, body: str) -> dict:
    value = _front_field(front, "base_model") or _front_field(front, "base model")
    if value:
        return _result("base_model", f"base_model: {value}")
    match = _BASE_MODEL_BODY.search(body)
    if match:
        return _result("base_model", match.group(1))
    return _result(
        "base_model",
        _heading_evidence(body, r"(?i)\bbase\s*model\b"),
    )


def _check_license(front: str, body: str) -> dict:
    value = (
        _front_field(front, "license")
        or _front_field(front, "license_name")
        or _front_field(front, "licence")
    )
    if value:
        return _result("license", f"license: {value}")
    match = _LICENSE_BODY.search(body)
    if match:
        return _result("license", match.group(1))
    return _result(
        "license",
        _heading_evidence(body, r"(?i)\b(?:license|licence|licensing)\b"),
    )


def _check_dataset(front: str, body: str) -> dict:
    front_value = _front_field(front, "datasets") or _front_field(front, "dataset")
    if front_value:
        return _result("dataset", f"datasets: {front_value}")
    for heading, section in _sections(body):
        if re.search(r"(?i)\b(?:dataset|datasets|data)\b", heading):
            first = next((line.strip() for line in section.splitlines() if line.strip()), "")
            if first:
                return _result("dataset", f"{heading}: {first}")
    match = _DATASET_BODY.search(body)
    return _result("dataset", match.group(1) if match else None)


def _check_evaluation(front: str, body: str) -> dict:
    if re.search(r"(?im)^\s*model-index\s*:", front) and _MODEL_INDEX_VALUE.search(front):
        return _result("evaluation", "model-index results are present in the card metadata")
    for heading, section in _sections(body):
        if re.search(r"(?i)\b(?:evaluation|eval results|results|benchmark|benchmarks|metrics)\b", heading):
            if _NUMBER.search(section) and _METRIC_WORD.search(section):
                first = next((line.strip() for line in section.splitlines() if line.strip()), "")
                return _result("evaluation", f"{heading}: {first}")
    return _result("evaluation", None)


def _sections(body: str) -> list[tuple[str, str]]:
    matches = list(_HEADING.finditer(body))
    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        sections.append((match.group(2).strip(), body[start:end].strip()))
    return sections
