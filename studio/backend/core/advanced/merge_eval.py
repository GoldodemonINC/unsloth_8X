# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Structural evaluation of a merged model, on a CPU, in a minute or two.

A merge can fail in ways no benchmark catches because it never gets that far:
a shard the index points at is missing, a tensor is NaN, a scale collapsed to
zero, the tokenizer was not copied. Those are cheap to detect and expensive to
discover after a 3-hour quantize.

This tier never loads the full model. It opens the shard headers, checks the
index against the files, samples tensors for NaN/Inf and degenerate scales, and
validates the tokenizer. Verdicts use one vocabulary everywhere:

    structurally-valid    loadable and internally consistent
    structurally-broken   file/index/tensor defect that will fail or corrupt a load
    unloadable            nothing loadable at the path (not even an index)
    pareto-positive       benchmark tier only
    within-noise          benchmark tier only
    regressed             benchmark tier only

The benchmark verdicts are not produced here, and this module says so in its
result rather than implying that structural validity is quality.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import tensor_io

# Enough tensors to catch a bad shard without reading the checkpoint.
_MAX_SAMPLED = 24


def _weight_files(root: Path) -> list[Path]:
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding = "utf-8"))
            mapping = index.get("weight_map") or {}
            names = sorted({str(name) for name in mapping.values()})
            files = [root / name for name in names]
            if files and all(file.is_file() for file in files):
                return files
            return files
        except (OSError, ValueError):
            pass
    return sorted(root.glob("*.safetensors"))


def _tensor_stats(value) -> dict[str, Any]:
    """Cheap statistics for one sampled tensor, without keeping it in memory."""
    import numpy as np

    array = tensor_io.to_float32(value).reshape(-1)
    if array.size == 0:
        return {"numel": 0, "finite": True, "mean_abs": None, "max_abs": None, "zeros": True}
    finite = bool(np.isfinite(array).all())
    mean_abs = float(np.mean(np.abs(array))) if finite else None
    max_abs = float(np.max(np.abs(array))) if finite else None
    return {
        "numel": int(array.size),
        "finite": finite,
        "mean_abs": mean_abs,
        "max_abs": max_abs,
        "all_zero": bool(finite and max_abs == 0.0),
    }


def structural_eval(source: str | Path, *, sample_tensors: int = _MAX_SAMPLED) -> dict[str, Any]:
    """Check that a merged folder is loadable and internally sane.

    Returns ``verdict`` plus ``problems`` (fatal) and ``warnings`` (worth seeing
    but not fatal). Never raises for a broken checkpoint: a broken result is the
    answer, not an exception.
    """
    root = Path(source).expanduser()
    problems: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}

    if root.is_file() and root.suffix == ".safetensors":
        root = root.parent
    if not root.is_dir():
        return _result("unloadable", [f"{root} does not exist."], [], {})

    config_path = root / "config.json"
    config: dict[str, Any] = {}
    if not config_path.is_file():
        problems.append(
            "config.json is missing; a loader cannot build the model from these weights."
        )
    else:
        try:
            config = json.loads(config_path.read_text(encoding = "utf-8"))
            checks["architecture"] = config.get("model_type") or config.get("architectures")
        except (OSError, ValueError) as error:
            problems.append(f"config.json is not readable JSON: {error}")

    index_path = root / "model.safetensors.index.json"
    weight_map: dict[str, str] = {}
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding = "utf-8"))
            weight_map = dict(index.get("weight_map") or {})
            checks["index_tensors"] = len(weight_map)
        except (OSError, ValueError) as error:
            problems.append(f"model.safetensors.index.json is not readable: {error}")
        missing = sorted(
            {
                shard
                for shard in weight_map.values()
                if not (root / str(shard)).is_file()
            }
        )
        if missing:
            problems.append(
                "The index points at shards that do not exist: "
                + ", ".join(missing[:4])
                + (" …" if len(missing) > 4 else "")
            )
        declared = (index.get("metadata") or {}).get("total_size") if index_path.is_file() else None
        if declared is not None:
            on_disk = sum(
                (root / shard).stat().st_size
                for shard in set(weight_map.values())
                if (root / shard).is_file()
            )
            checks["declared_total_bytes"] = declared
            checks["on_disk_shard_bytes"] = on_disk
            if on_disk and declared > on_disk * 1.05:
                warnings.append(
                    f"The index declares {declared} bytes but the shards total {on_disk}; "
                    "the checkpoint may be truncated."
                )

    files = [file for file in _weight_files(root) if file.is_file()]
    if not files:
        # An index that names shards which are not there is a truncated checkpoint,
        # not an empty folder: say which defect it is.
        if weight_map:
            return _result("structurally-broken", problems, warnings, checks)
        return _result(
            "unloadable",
            problems + ["No .safetensors weights found."],
            warnings,
            checks,
        )
    checks["shards"] = len(files)

    sampled = 0
    degenerate: list[str] = []
    nan_tensors: list[str] = []
    stats_seen: list[dict[str, Any]] = []
    for file in files:
        if sampled >= sample_tensors:
            break
        try:
            handle = tensor_io.open_file(file)
        except Exception as error:  # noqa: BLE001 - an unreadable shard is the finding
            problems.append(f"Could not open {file.name}: {error}")
            continue
        with handle as opened:
            keys = list(opened.keys())
            if not keys:
                problems.append(f"{file.name} contains no tensors.")
                continue
            # Spread the sample across the shard rather than reading only its head.
            step = max(1, len(keys) // max(1, sample_tensors // max(1, len(files)) or 1))
            for key in keys[::step][: max(1, sample_tensors - sampled)]:
                try:
                    value = tensor_io.load_tensor(opened, key)
                except (TypeError, ValueError) as error:
                    warnings.append(f"Could not read {key}: {error}")
                    continue
                if not tensor_io.is_floating(value):
                    continue
                stats = _tensor_stats(value)
                sampled += 1
                if not stats["finite"]:
                    nan_tensors.append(key)
                elif stats["all_zero"]:
                    degenerate.append(key)
                elif stats["mean_abs"] is not None and stats["mean_abs"] < 1e-12:
                    degenerate.append(key)
                stats_seen.append({"tensor": key, **stats})
                del value
    checks["sampled_tensors"] = sampled
    if nan_tensors:
        problems.append(
            f"{len(nan_tensors)} sampled tensor(s) contain NaN/Inf: "
            + ", ".join(nan_tensors[:5])
            + ". Blending or alignment produced non-finite weights."
        )
    if degenerate:
        warnings.append(
            f"{len(degenerate)} sampled tensor(s) are all zero or vanishingly small: "
            + ", ".join(degenerate[:5])
            + ". Alignment padding can do this; check the merge report's skipped tensors."
        )

    tokenizer_files = [
        name
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
        if (root / name).is_file()
    ]
    checks["tokenizer_files"] = tokenizer_files
    if not tokenizer_files:
        problems.append(
            "No tokenizer files were copied into the merged folder. It will fall back to "
            "the base model's tokenizer, which may not match the merged vocabulary."
        )

    verdict = "structurally-broken" if problems else "structurally-valid"
    return _result(verdict, problems, warnings, checks, samples = stats_seen[:5])


def _result(
    verdict: str,
    problems: Iterable[str],
    warnings: Iterable[str],
    checks: dict[str, Any],
    *,
    samples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "verdict": verdict,
        "structurally_valid": verdict == "structurally-valid",
        "problems": list(problems),
        "warnings": list(warnings),
        "checks": checks,
        "samples": samples or [],
        "tier": "structural",
        "scope": (
            "CPU structural tier: loadability, index/files agreement, NaN/scale sanity and "
            "tokenizer presence. It says nothing about task quality — run an eval suite for that."
        ),
        "benchmark_verdicts": ["pareto-positive", "within-noise", "regressed"],
    }


def verdict_label(score_delta: float | None, *, noise_floor: float = 0.015) -> str:
    """The benchmark-side vocabulary, for callers that have a measured delta."""
    if score_delta is None:
        return "within-noise"
    if not math.isfinite(score_delta):
        return "regressed"
    if score_delta <= -abs(noise_floor):
        return "regressed"
    if score_delta >= abs(noise_floor):
        return "pareto-positive"
    return "within-noise"


def completion_gate(report: Mapping[str, Any]) -> dict[str, str]:
    """Turn a structural report into the keep-or-discard call a merge needs.

    A structurally broken folder should be thrown away rather than quantized:
    the defect surfaces as a load failure, and reaching it costs the merge plus
    the export that followed it. Warnings are not defects, so they land on
    ``review`` — the model loads, but the notes name something worth reading
    before it is taken as the result.

    This is still the structural tier. It never means "better": a loadable
    merge can be worse than its parents, and only a benchmark can say that.
    """
    verdict = str(report.get("verdict") or "unloadable")
    problems = list(report.get("problems") or [])
    warnings = list(report.get("warnings") or [])
    if verdict in ("structurally-broken", "unloadable"):
        first = problems[0] if problems else "The folder did not load."
        return {
            "decision": "discard",
            "reason": (
                f"{first} Discard it, or re-run the merge into a clean folder; "
                "quantizing it will only move the failure later."
            ),
        }
    if warnings:
        return {
            "decision": "review",
            "reason": (
                "It loads, but the check raised "
                f"{len(warnings)} warning(s): {warnings[0]} Safe to keep if that is expected."
            ),
        }
    return {
        "decision": "keep",
        "reason": (
            "Loadable and internally consistent. This says nothing about quality: "
            "compare it against its parents on your own prompts before choosing it."
        ),
    }
