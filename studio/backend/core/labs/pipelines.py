# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pipelines: plan a chain of stages, and say what it will cost before it runs.

This is deliberately a planner. Every stage here is an engine the Studio already
has -- merge, scale, train, quantize, benchmark -- and running them back to back
needs the durability the current job registry does not have (jobs live in
memory, leases are taken on the request thread, a restart loses the run). What
this module does is the part that can be right today: compose the stages'
config-only projections into one chain, price the whole thing, and refuse a
chain that cannot work before anything is downloaded.

The arithmetic is the same arithmetic the Scaler plans with: parameter counts
from ``config.json``, fp16 at two bytes per parameter, quantization by the bytes
per parameter each GGUF level actually writes. Nothing here loads weights.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional

from core.advanced import eval_harness
from core.advanced.scaling import (
    CAP_ADDED_PARAMS,
    GROWTH_FLOOR,
    METHODS,
    _base_params,
    _project,
)

STAGE_KINDS = ("merge", "scale", "train", "quantize", "benchmark")
STAGE_LABELS = {
    "merge": "Merge",
    "scale": "Scale",
    "train": "Train",
    "quantize": "Quantize",
    "benchmark": "Benchmark",
}
_MAX_STAGES = 20
_PIPELINE_ID = re.compile(r"^[a-z0-9]{8,32}$")
_PIPELINES_DIRNAME = "labs-pipelines"

# Bytes per parameter each GGUF outtype writes. f16 is the unquantized cast;
# the Q levels are the ones the export path offers.
QUANT_BYTES_PER_PARAM: dict[str, float] = {
    "f16": 2.0,
    "q8_0": 1.06,
    "q6_k": 0.82,
    "q5_k_m": 0.71,
    "q4_k_m": 0.58,
    "q2_k": 0.33,
}

# The token figure the expansion module cites for a 7B model reaching scratch
# loss (Stacking Your Transformers, NeurIPS 2024). A run far below it is not a
# verdict about growth; the plan says so rather than letting the number pass.
_RECOVERY_TOKENS_REFERENCE = 194_000_000_000

_DEFAULT_TRAIN = {"steps": 60, "batch_size": 1, "sequence_length": 2048}


class PipelineError(ValueError):
    """A pipeline that cannot be validated at all (as opposed to a blocked stage)."""


def _format_bytes(value: float) -> str:
    if value >= 1e12:
        return f"{value / 1e12:.2f} TB"
    if value >= 1e9:
        return f"{value / 1e9:.2f} GB"
    if value >= 1e6:
        return f"{value / 1e6:.0f} MB"
    return f"{int(value)} B"


def _format_params(value: Optional[float]) -> str:
    if not value:
        return "unknown"
    if value >= 1e9:
        return f"{value / 1e9:.2f}B"
    if value >= 1e6:
        return f"{value / 1e6:.1f}M"
    return f"{int(value)}"


def _local_dir(source: Any) -> Optional[Path]:
    """A local checkpoint folder, or None for a Hub id (which is not fetched to plan)."""
    if not isinstance(source, str) or not source.strip():
        return None
    path = Path(source).expanduser()
    try:
        return path if path.is_dir() else None
    except OSError:
        return None


def _config_of(source: Any) -> Optional[dict[str, Any]]:
    directory = _local_dir(source)
    if directory is None:
        return None
    try:
        value = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _int_field(stage: Mapping[str, Any], key: str, default: int, *, minimum: int) -> int:
    raw = stage.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise PipelineError(f"{key} must be a number.")
    value = int(raw)
    if value < minimum:
        raise PipelineError(f"{key} must be at least {minimum}.")
    return value


def _float_field(stage: Mapping[str, Any], key: str) -> Optional[float]:
    raw = stage.get(key)
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise PipelineError(f"{key} must be a number.")
    return float(raw)


def _stage(kind: str, index: int) -> dict[str, Any]:
    return {
        "index": index,
        "kind": kind,
        "label": STAGE_LABELS[kind],
        "params_before": None,
        "params_after": None,
        "output_bytes": 0,
        "warnings": [],
        "blocked": False,
        "reason": None,
        "details": {},
    }


def _blocked(stage: dict[str, Any], reason: str, current: Optional[int]) -> Optional[int]:
    """Mark the stage blocked and keep the chain's parameter count where it was."""
    stage["blocked"] = True
    stage["reason"] = reason
    stage["params_after"] = current
    return current


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def _plan_merge(stage: dict[str, Any], spec: Mapping[str, Any], current: Optional[int]) -> Optional[int]:
    parents = spec.get("parents") or []
    if not isinstance(parents, list) or len(parents) < 2:
        return _blocked(stage, "A merge stage needs at least two parent checkpoints.", current)
    configs: list[Optional[dict[str, Any]]] = []
    for parent in parents:
        if not isinstance(parent, str) or not parent.strip():
            return _blocked(
                stage, "Every merge parent must be a checkpoint path or Hub id.", current
            )
        configs.append(_config_of(parent))
    known = [config for config in configs if config]
    if len(known) != len(configs):
        stage["warnings"].append(
            "At least one parent is not a local folder, so its parameter count and "
            "tokenizer cannot be checked here. The Merge studio validates both before it runs."
        )
    if len(known) >= 2:
        model_types = {str(config.get("model_type")) for config in known}
        vocab_sizes = {config.get("vocab_size") for config in known}
        if len(model_types) > 1:
            return _blocked(
                stage,
                "Parents disagree on model_type "
                f"({', '.join(sorted(value for value in model_types if value))}). "
                "A merge of different architectures is not a merge.",
                current,
            )
        if len(vocab_sizes) > 1:
            return _blocked(
                stage,
                "Parents disagree on vocab_size. Their tokenizers are not the same, "
                "so one merged tokenizer would misread at least one parent.",
                current,
            )
        stage["details"]["vocab_size"] = next(iter(vocab_sizes))
        stage["details"]["model_type"] = next(iter(model_types))
    params = [value for value in (_base_params(config) for config in known) if value]
    merged = max(params) if params else current
    stage["details"]["parents"] = [str(parent) for parent in parents]
    stage["params_after"] = merged
    if merged:
        # fp16 output; the merge itself streams.
        stage["output_bytes"] = int(merged * 2)
    return merged


def _plan_scale(stage: dict[str, Any], spec: Mapping[str, Any], current: Optional[int]) -> Optional[int]:
    method = str(spec.get("method") or "depth")
    if method not in METHODS:
        return _blocked(stage, f"method must be one of {', '.join(METHODS)}.", current)
    if not current:
        return _blocked(
            stage,
            "Scaling needs the checkpoint's parameter count, and the source is not a "
            "local folder with a readable config.json.",
            current,
        )

    added = _float_field(spec, "added_params")
    target = _float_field(spec, "target_params")
    factor = _float_field(spec, "factor")
    if added is not None and target is not None:
        raise PipelineError("A scale stage takes added_params, target_params or factor, not two.")
    if factor is not None:
        if factor <= 1:
            return _blocked(stage, "factor must be greater than 1.", current)
        goal = float(current) * factor
    elif target is not None:
        goal = target
    elif added is not None:
        goal = float(current) + added
    else:
        goal = float(current) * GROWTH_FLOOR

    floor = float(current) * GROWTH_FLOOR
    if goal < floor:
        return _blocked(
            stage,
            f"{_format_params(goal)} is {goal / current:.2f}x this checkpoint. Growth below "
            f"{GROWTH_FLOOR:g}x is refused: no growth study reports a gain below it, and the run "
            f"costs the same as a real one. Target {_format_params(floor)} or more.",
            current,
        )
    if goal - current > CAP_ADDED_PARAMS:
        return _blocked(
            stage,
            f"Adding {_format_params(goal - current)} is past the {_format_params(CAP_ADDED_PARAMS)} "
            "ceiling the Scaler enforces. Split the growth into two runs, or target less.",
            current,
        )

    config = _config_of(spec.get("source")) or _config_of(spec.get("path"))
    projection = None
    if config:
        try:
            projection = _project(
                config,
                base_params=int(current),
                target_params=float(goal),
                method=method,
            )
        except Exception:  # noqa: BLE001 - a projection is an estimate, not a gate
            projection = None
    reached = int(projection["projected_params"]) if projection else int(goal)
    stage["params_after"] = reached
    stage["output_bytes"] = int(reached * 2)
    stage["details"] = {
        "method": method,
        "target_params": int(goal),
        "growth_factor": float(reached) / float(current) if current else None,
        "projection": projection,
    }
    stage["warnings"].append(
        "Expansion is exact at step 0 and inert after it: the new capacity only counts "
        "once the following train stage has used it."
    )
    return reached


def _plan_train(stage: dict[str, Any], spec: Mapping[str, Any], current: Optional[int]) -> Optional[int]:
    steps = _int_field(spec, "steps", _DEFAULT_TRAIN["steps"], minimum=1)
    batch = _int_field(spec, "batch_size", _DEFAULT_TRAIN["batch_size"], minimum=1)
    sequence = _int_field(
        spec, "sequence_length", _DEFAULT_TRAIN["sequence_length"], minimum=128
    )
    tokens = steps * batch * sequence
    stage["details"] = {
        "method": str(spec.get("method") or "CPT"),
        "steps": steps,
        "batch_size": batch,
        "sequence_length": sequence,
        "tokens": tokens,
    }
    if current:
        stage["output_bytes"] = int(current * 2)
    share = tokens / _RECOVERY_TOKENS_REFERENCE
    if share < 0.05:
        stage["warnings"].append(
            f"{tokens / 1e9:.3f}B tokens is {share * 100:.2f}% of the ~194B-token recovery "
            "datapoint this Studio cites. Treat a benchmark after this stage as "
            "under-trained, not as a verdict about the new capacity."
        )
    if "full" in str(spec.get("method") or "").lower():
        stage["warnings"].append(
            "A full-parameter run holds optimizer state (~16 bytes per parameter) in "
            "addition to the checkpoint on disk."
        )
    if not spec.get("dataset"):
        stage["warnings"].append(
            "No dataset is named. The Training studio will need one before this stage can start."
        )
    return current


def _plan_quantize(stage: dict[str, Any], spec: Mapping[str, Any], current: Optional[int]) -> Optional[int]:
    level = str(spec.get("level") or "q4_k_m").lower()
    if level not in QUANT_BYTES_PER_PARAM:
        return _blocked(
            stage,
            f"Unknown quantization {level!r}. Choose one of "
            f"{', '.join(QUANT_BYTES_PER_PARAM)}.",
            current,
        )
    stage["details"] = {"level": level}
    if not current:
        stage["warnings"].append(
            "The parameter count before this stage is unknown, so the output size cannot be estimated."
        )
        return current
    stage["output_bytes"] = int(current * QUANT_BYTES_PER_PARAM[level])
    stage["params_after"] = current
    if level == "q2_k":
        stage["warnings"].append(
            "Q2_K is a deployment size, not a quality candidate. Benchmark it for "
            "throughput and perplexity, not as an alternative to Q4."
        )
    return current


def _plan_benchmark(stage: dict[str, Any], spec: Mapping[str, Any], current: Optional[int]) -> Optional[int]:
    baseline = spec.get("baseline")
    if not baseline:
        stage["warnings"].append(
            "No baseline is pinned. A score without the base model's score on the same "
            "suite cannot say whether anything improved."
        )
    available = eval_harness.harness_available()
    stage["details"] = {
        "baseline": baseline,
        "suites": list(spec.get("suites") or []),
        "harness": available,
    }
    if not available.get("available"):
        stage["warnings"].append(str(available.get("install_hint") or "lm-eval is not installed."))
    if current:
        stage["output_bytes"] = int(current * 0.02)
    stage["params_after"] = current
    return current


# ---------------------------------------------------------------------------
# The chain
# ---------------------------------------------------------------------------


def preflight(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Price and validate a pipeline spec. Never loads weights, never starts work."""
    if not isinstance(spec, Mapping):
        raise PipelineError("A pipeline must be an object.")
    raw_stages = spec.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise PipelineError("A pipeline needs at least one stage.")
    if len(raw_stages) > _MAX_STAGES:
        raise PipelineError(f"A pipeline takes at most {_MAX_STAGES} stages.")

    source = spec.get("source")
    config = _config_of(source)
    current = _base_params(config) if config else None
    stages: list[dict[str, Any]] = []
    warnings: list[str] = []
    blocked_reason: Optional[str] = None
    quantized_at: Optional[int] = None

    for index, raw in enumerate(raw_stages):
        if not isinstance(raw, Mapping):
            raise PipelineError(f"Stage {index + 1} must be an object.")
        kind = str(raw.get("kind") or "")
        if kind not in STAGE_KINDS:
            raise PipelineError(
                f"Stage {index + 1} has kind {kind!r}; expected one of {', '.join(STAGE_KINDS)}."
            )
        if quantized_at is not None and kind not in ("benchmark",):
            raise PipelineError(
                f"Stage {index + 1} is {kind}, after the pipeline was quantized at stage "
                f"{quantized_at + 1}. A GGUF file is an output, not an input to another stage; "
                "put training and scaling before Quantize."
            )
        stage = _stage(kind, index)
        stage["params_before"] = current
        effective = dict(raw)
        if kind == "scale" and not effective.get("source"):
            effective["source"] = source
        before = current
        if kind == "merge":
            current = _plan_merge(stage, effective, current)
        elif kind == "scale":
            current = _plan_scale(stage, effective, current)
        elif kind == "train":
            current = _plan_train(stage, effective, current)
        elif kind == "quantize":
            current = _plan_quantize(stage, effective, current)
            quantized_at = index
        else:
            current = _plan_benchmark(stage, effective, current)
        stage["params_before"] = before
        stage["params_after"] = current
        for warning in stage["warnings"]:
            warnings.append(f"Stage {index + 1} ({stage['label']}): {warning}")
        if stage["blocked"] and blocked_reason is None:
            blocked_reason = f"Stage {index + 1} ({stage['label']}): {stage['reason']}"
        stages.append(stage)

    total_output = sum(int(stage["output_bytes"]) for stage in stages)
    fp16_bytes = int(current * 2) if current else 0
    peak = max([fp16_bytes] + [int(stage["output_bytes"]) for stage in stages])
    if current and current > CAP_ADDED_PARAMS:
        warnings.append(
            "This chain ends above the Scaler's growth ceiling. A later scale stage on the "
            "result will be refused."
        )
    if source and _local_dir(source) is None:
        warnings.append(
            "The source is not a local folder, so the parameter count is unknown and every "
            "size below is a lower bound. Download it first for an exact plan."
        )

    return {
        "source": source,
        "stages": stages,
        "final_params": current,
        "total_output_bytes": total_output,
        "total_output_label": _format_bytes(total_output),
        "peak_stage_bytes": peak,
        "peak_stage_label": _format_bytes(peak),
        "warnings": warnings,
        "blocked": blocked_reason is not None,
        "blocked_reason": blocked_reason,
        "notes": [
            "Estimates are config arithmetic: fp16 is two bytes per parameter, and each "
            "quantization level is the bytes per parameter it writes.",
            "Planning is not running. This release plans and validates a chain; it does not "
            "start the stages.",
        ],
    }


# ---------------------------------------------------------------------------
# Saved pipelines
# ---------------------------------------------------------------------------


def root() -> Path:
    from utils.paths.storage_roots import outputs_root

    return outputs_root() / _PIPELINES_DIRNAME


def _safe_id(pipeline_id: str) -> str:
    value = str(pipeline_id or "").strip().lower()
    if not _PIPELINE_ID.match(value):
        raise PipelineError(f"{pipeline_id!r} is not a pipeline id.")
    return value


def save_pipeline(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate, then write a pipeline spec. Returns the stored record."""
    report = preflight(spec)
    name = str(spec.get("name") or "").strip()
    if len(name) > 80:
        raise PipelineError("A pipeline name is at most 80 characters.")
    directory = root()
    directory.mkdir(parents = True, exist_ok = True)
    pipeline_id = uuid.uuid4().hex[:12]
    record = {
        "id": pipeline_id,
        "name": name or f"Pipeline {pipeline_id[:4]}",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "spec": {
            "name": name,
            "source": spec.get("source"),
            "stages": list(spec.get("stages") or []),
        },
        "plan": {
            "final_params": report["final_params"],
            "total_output_bytes": report["total_output_bytes"],
            "blocked": report["blocked"],
            "blocked_reason": report["blocked_reason"],
            "stage_kinds": [stage["kind"] for stage in report["stages"]],
        },
    }
    target = directory / f"{pipeline_id}.json"
    partial = target.with_suffix(".json.partial")
    partial.write_text(json.dumps(record, indent = 2), encoding = "utf-8")
    partial.replace(target)
    return record


def list_pipelines(limit: int = 50) -> list[dict[str, Any]]:
    directory = root()
    try:
        files = sorted(directory.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse = True)
    except OSError:
        return []
    records: list[dict[str, Any]] = []
    for path in files[: max(1, min(int(limit), 100))]:
        try:
            value = json.loads(path.read_text(encoding = "utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(value, dict) and value.get("id"):
            records.append(value)
    return records


def delete_pipeline(pipeline_id: str) -> bool:
    directory = root()
    target = directory / f"{_safe_id(pipeline_id)}.json"
    if not target.is_file():
        return False
    try:
        target.unlink()
    except OSError:
        return False
    return True
