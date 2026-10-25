# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CMT: capacity matched to the data, then only the new blocks are trained.

CMT is an experiment, not a method the stable trainer offers. One job does the
whole thing:

1. Every dataset is split into its own train and held-out part, so each one has
   the evaluation the user asked for instead of one shared split.
2. The base checkpoint grows by a number of parameters derived from the total
   dataset size. CMT deliberately bypasses the Scaler's 2x research floor: the
   growth here is sized to the data, not to a scaling study. The run says so.
3. Only the inserted blocks train. The inherited weights are frozen, which is
   what makes the added 1M-700M parameters the thing being trained.
4. Each dataset's held-out text is scored as its own task through training, and
   the result reports the per-dataset delta from the step-0 expanded model.

Two honesty rules matter here:

* The engine inserts whole transformer blocks. If the dataset-derived target is
  smaller than one block, the real growth is one block and the plan says that
  number instead of pretending it can add a fraction of a layer.
* The token estimate reads local files only and assumes four characters per
  token. It is a sizing heuristic, not a tokenizer count; the run reports the
  tokens the collator actually saw.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

MIN_ADDED_PARAMS = 1_000_000
MAX_ADDED_PARAMS = 700_000_000
MAX_DATASETS = 8
MIN_DATASETS = 2
DEFAULT_EVAL_FRACTION = 0.10
MAX_TRAIN_ROWS = 200_000
EVAL_TEXTS_PER_DATASET = 256
_SAMPLE_ROWS = 64
_SUPPORTED_SUFFIXES = (".jsonl", ".json", ".parquet", ".csv", ".txt")
_HUB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9._-]+$")

_SLUG = re.compile(r"[^a-z0-9_-]+")


# ---------------------------------------------------------------------------
# Request normalisation
# ---------------------------------------------------------------------------


def _number(value: Any, field: str, *, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a number.") from error
    if not math.isfinite(parsed) or parsed < minimum or parsed > maximum:
        raise ValueError(f"{field} must be between {minimum:g} and {maximum:g}.")
    return parsed


def _integer(value: Any, field: str, *, minimum: int, maximum: int, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a whole number.") from error
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{field} must be between {minimum} and {maximum}.")
    return parsed


def _normalise(spec: Mapping[str, Any]) -> dict[str, Any]:
    source = str(spec.get("source") or "").strip()
    if not source:
        raise ValueError("CMT needs a base checkpoint or Hub id.")
    mode = str(spec.get("mode") or "cpt").strip().lower()
    if mode not in ("cpt", "finetune"):
        raise ValueError("mode must be 'cpt' or 'finetune'.")

    raw_datasets = spec.get("datasets")
    if not isinstance(raw_datasets, list) or len(raw_datasets) < MIN_DATASETS:
        raise ValueError(f"CMT needs at least {MIN_DATASETS} datasets.")
    if len(raw_datasets) > MAX_DATASETS:
        raise ValueError(f"CMT takes at most {MAX_DATASETS} datasets.")

    datasets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_datasets, start=1):
        if not isinstance(raw, Mapping):
            raise ValueError(f"Dataset {index} is not an object.")
        path = str(raw.get("path") or "").strip()
        repo_id = str(raw.get("repo_id") or "").strip()
        if path and repo_id:
            raise ValueError(
                f"Dataset {index} names both a path and a Hub id; give it one of the two."
            )
        if not path and not repo_id:
            raise ValueError(f"Dataset {index} needs a path or a Hub dataset id.")
        expanded = ""
        if path:
            expanded = str(Path(path).expanduser())
            suffix = Path(expanded).suffix.lower()
            if suffix not in _SUPPORTED_SUFFIXES:
                raise ValueError(
                    f"Dataset {index} has unsupported type {suffix or '(none)'}. "
                    f"CMT reads {', '.join(_SUPPORTED_SUFFIXES)} or a Hub id."
                )
            if not Path(expanded).is_file():
                raise ValueError(f"Dataset {index} does not exist: {path}")
        elif not _HUB_ID.match(repo_id):
            raise ValueError(
                f"Dataset {index} is not a Hub id: {repo_id!r}. It needs to look like 'owner/name'."
            )
        subset = str(raw.get("subset") or "").strip() or None
        split = str(raw.get("split") or "").strip() or None
        identity = expanded or f"hf:{repo_id}:{subset or ''}:{split or ''}"
        if identity in seen:
            raise ValueError(
                f"Dataset {index} repeats {path or repo_id}; each entry must be a different dataset."
            )
        seen.add(identity)
        eval_fraction = _number(
            raw.get("eval_fraction", DEFAULT_EVAL_FRACTION),
            f"Dataset {index} eval fraction",
            minimum=0.0,
            maximum=0.5,
        )
        datasets.append(
            {
                "path": expanded,
                "repo_id": repo_id or None,
                "subset": subset,
                "split": split,
                "name": (
                    str(raw.get("name") or (Path(expanded).stem if expanded else repo_id.split("/")[-1]))
                    .strip()
                    or f"dataset-{index}"
                ),
                "text_column": str(raw.get("text_column") or "").strip() or None,
                "eval_fraction": eval_fraction,
            }
        )

    added_override = spec.get("added_params")
    added: Optional[int] = None
    if added_override not in (None, ""):
        added = int(
            _number(
                added_override,
                "added_params",
                minimum=MIN_ADDED_PARAMS,
                maximum=MAX_ADDED_PARAMS,
            )
        )

    return {
        "source": source,
        "mode": mode,
        "datasets": datasets,
        "added_params": added,
        "method": "depth",
        "insert_strategy": str(spec.get("insert_strategy") or "even"),
        "max_steps": _integer(spec.get("max_steps"), "max_steps", minimum=1, maximum=100_000, default=100),
        "batch_size": _integer(spec.get("batch_size"), "batch_size", minimum=1, maximum=512, default=1),
        "gradient_accumulation": _integer(
            spec.get("gradient_accumulation"), "gradient_accumulation", minimum=1, maximum=512, default=8
        ),
        "max_seq_length": _integer(
            spec.get("max_seq_length"), "max_seq_length", minimum=128, maximum=131_072, default=2048
        ),
        "learning_rate": _number(
            spec.get("learning_rate", 2e-4), "learning_rate", minimum=1e-8, maximum=10.0
        ),
        "eval_every": _integer(spec.get("eval_every"), "eval_every", minimum=1, maximum=10_000, default=20),
        "patience": _integer(spec.get("patience"), "patience", minimum=1, maximum=100, default=4),
        "seed": _integer(spec.get("seed"), "seed", minimum=0, maximum=2**31 - 1, default=3407),
        "max_train_rows": _integer(
            spec.get("max_train_rows"), "max_train_rows", minimum=100, maximum=2_000_000, default=MAX_TRAIN_ROWS
        ),
    }


# ---------------------------------------------------------------------------
# Local dataset inspection (planning must stay cheap)
# ---------------------------------------------------------------------------


def _row_text(value: Any) -> str:
    """Best-effort text from one row, without a tokenizer or the datasets package."""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("text", "content", "document", "raw", "completion", "response", "output"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate
        for key in ("messages", "conversations"):
            candidate = value.get(key)
            if isinstance(candidate, list):
                pieces = []
                for turn in candidate:
                    if isinstance(turn, Mapping):
                        role = str(turn.get("role") or turn.get("from") or "")
                        content = turn.get("content") if "content" in turn else turn.get("value")
                        if isinstance(content, list):
                            content = " ".join(
                                str(part.get("text") or part) if isinstance(part, Mapping) else str(part)
                                for part in content
                            )
                        pieces.append(f"{role}: {content}")
                if pieces:
                    return "\n".join(pieces)
        strings = [item for item in value.values() if isinstance(item, str) and item.strip()]
        if strings:
            return "\n".join(strings)
    return ""


def _sample_jsonl(path: Path) -> tuple[int, int, int, list[str]]:
    rows = 0
    chars = 0
    sampled = 0
    notes: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if not line.strip():
                continue
            rows += 1
            if sampled < _SAMPLE_ROWS:
                try:
                    payload = json.loads(line)
                except ValueError:
                    payload = line
                chars += len(_row_text(payload) or str(payload))
                sampled += 1
    if rows == 0:
        notes.append("The file has no non-empty rows.")
    return rows, chars, sampled, notes


def _sample_json(path: Path) -> tuple[int, int, int, list[str]]:
    if path.stat().st_size > 64 * 1024 * 1024:
        return -1, 0, 0, ["The JSON file is over 64 MB, so rows are estimated from its size."]
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except ValueError:
        return -1, 0, 0, ["The JSON could not be parsed while planning; the run will report the real count."]
    rows = payload if isinstance(payload, list) else payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return -1, 0, 0, ["The JSON is not a row list; rows are estimated from its size."]
    sample = rows[:_SAMPLE_ROWS]
    chars = sum(len(_row_text(row)) for row in sample)
    return len(rows), chars, len(sample), []


def _sample_parquet(path: Path) -> tuple[int, int, int, list[str]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError:
        return -1, 0, 0, ["pyarrow is not installed, so row counts come from the file size."]
    try:
        handle = parquet.ParquetFile(path)
        rows = int(handle.metadata.num_rows)
        batch = next(handle.iter_batches(batch_size=_SAMPLE_ROWS), None)
    except Exception:  # noqa: BLE001 - a broken parquet is reported by the run, not hidden
        return -1, 0, 0, ["The parquet metadata could not be read; rows are estimated from its size."]
    if batch is None:
        return rows, 0, 0, ["The parquet file has no batches."]
    values = batch.to_pylist()
    chars = sum(len(_row_text(row)) for row in values)
    return rows, chars, len(values), []


def _sample_delimited(path: Path, *, header: bool) -> tuple[int, int, int, list[str]]:
    import csv

    rows = 0
    chars = 0
    sampled = 0
    with path.open("r", encoding="utf-8", errors="replace", newline="") as stream:
        reader = csv.DictReader(stream) if header else None
        if reader is not None:
            for row in reader:
                rows += 1
                if sampled < _SAMPLE_ROWS:
                    chars += len(_row_text(row))
                    sampled += 1
        else:
            for line in stream:
                if not line.strip():
                    continue
                rows += 1
                if sampled < _SAMPLE_ROWS:
                    chars += len(line)
                    sampled += 1
    return rows, chars, sampled, []


def _inspect(entry: Mapping[str, Any]) -> dict[str, Any]:
    if entry.get("repo_id"):
        return {
            "path": "",
            "repo_id": entry["repo_id"],
            "subset": entry.get("subset"),
            "split": entry.get("split"),
            "name": str(entry.get("name") or entry["repo_id"]),
            "text_column": entry.get("text_column"),
            "eval_fraction": float(entry.get("eval_fraction") or 0.0),
            "bytes": 0,
            "rows": 0,
            "eval_rows": 0,
            "train_rows": 0,
            "tokens_estimate": 0,
            "notes": [
                "A Hub dataset is not downloaded while planning. Its rows, and the growth "
                "they justify, are measured when the run starts."
            ],
        }
    path = Path(str(entry["path"]))
    suffix = path.suffix.lower()
    notes: list[str] = []
    if suffix in (".jsonl", ".json"):
        rows, chars, sampled, notes = (
            _sample_jsonl(path) if suffix == ".jsonl" else _sample_json(path)
        )
    elif suffix == ".parquet":
        rows, chars, sampled, notes = _sample_parquet(path)
    elif suffix == ".csv":
        rows, chars, sampled, notes = _sample_delimited(path, header=True)
    else:
        rows, chars, sampled, notes = _sample_delimited(path, header=False)

    size = path.stat().st_size
    if rows > 0 and sampled > 0:
        tokens = int(round((chars / 4.0) * (rows / sampled)))
    else:
        tokens = int(round(size / 4.0))
        if rows < 0:
            notes.append("Rows and tokens are estimated from the file size; the run reports the real values.")
    tokens = max(1, tokens)
    if rows < 0:
        rows = 0 if size == 0 else -1
    eval_fraction = float(entry.get("eval_fraction") or 0.0)
    eval_rows = int(round(max(0, rows) * eval_fraction)) if rows > 0 else 0
    train_rows = max(0, rows - eval_rows) if rows > 0 else 0
    return {
        "path": str(path),
        "repo_id": None,
        "subset": entry.get("subset"),
        "split": entry.get("split"),
        "name": str(entry.get("name") or path.stem),
        "text_column": entry.get("text_column"),
        "eval_fraction": eval_fraction,
        "bytes": size,
        "rows": rows,
        "eval_rows": eval_rows,
        "train_rows": train_rows,
        "tokens_estimate": tokens,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


def _clamp_added(value: float) -> int:
    return int(max(MIN_ADDED_PARAMS, min(MAX_ADDED_PARAMS, round(value))))


def _added_from_tokens(tokens: float) -> int:
    """The sizing rule CMT ships with, in parameters.

    ``(1.7 * megatokens + 12) / 10`` megaparameters: 20M tokens asks for 4.6M,
    1B asks for 182M, and the 700M cap is reached around 4.0B tokens. It is a
    heuristic, and the UI labels it as one.
    """
    megatokens = max(0.0, float(tokens)) / 1e6
    return _clamp_added(((1.7 * megatokens + 12.0) / 10.0) * 1e6)


def _added_from_rows(rows: float) -> int:
    """The fallback rule: one million parameters per ten thousand rows."""
    return _clamp_added((max(0.0, float(rows)) / 10_000.0) * 1e6)


def sizing(rows: float = 0, tokens: float = 0) -> dict[str, Any]:
    """The growth a dataset justifies, in the one rule CMT and auto-scale share.

    Tokens win when they are known, rows are the fallback, and an empty dataset
    still gets the floor rather than nothing.
    """
    rows = max(0.0, float(rows or 0))
    tokens = max(0.0, float(tokens or 0))
    from_tokens = _added_from_tokens(tokens) if tokens > 0 else None
    from_rows = _added_from_rows(rows) if rows > 0 else None
    if from_tokens is not None:
        chosen, rule = from_tokens, "tokens"
    elif from_rows is not None:
        chosen, rule = from_rows, "rows"
    else:
        chosen, rule = MIN_ADDED_PARAMS, "floor"
    return {
        "rule": rule,
        "rows": int(rows),
        "tokens": int(tokens),
        "added_params": int(chosen),
        "added_label": _format_params(chosen),
        "suggested_from_tokens": from_tokens,
        "suggested_from_rows": from_rows,
        "min_added_params": MIN_ADDED_PARAMS,
        "max_added_params": MAX_ADDED_PARAMS,
        "note": (
            "added = clamp(((1.7 * megatokens + 12) / 10) * 1M, 1M, 700M), with a "
            "fallback of 1M per 10k rows. The engine inserts whole blocks, so the real "
            "growth is a multiple of one block."
        ),
    }


def inspect_file(path: str) -> dict[str, Any]:
    """Rows and an estimated token count for one local dataset file."""
    expanded = str(Path(path).expanduser())
    if not Path(expanded).is_file():
        raise ValueError(f"Dataset does not exist: {path}")
    return _inspect(
        {
            "path": expanded,
            "name": Path(expanded).stem,
            "text_column": None,
            "eval_fraction": 0.0,
        }
    )


def load_cpu_model(source: str, token: Optional[str]) -> tuple[Any, str]:
    """The checkpoint loader both Labs stages use, so they agree on dtype and device."""
    return _load_model(source, token)


def load_tokenizer(source: str, token: Optional[str]) -> Any:
    """The tokenizer loader both Labs stages use."""
    return _load_tokenizer(source, token)


def _format_params(value: float) -> str:
    if value >= 1e9:
        return f"{value / 1e9:.2f}B"
    if value >= 1e6:
        return f"{value / 1e6:.1f}M"
    if value >= 1e3:
        return f"{value / 1e3:.1f}K"
    return f"{int(value)}"


def _dependencies() -> dict[str, Any]:
    import importlib.util

    required = ("torch", "transformers", "datasets")
    missing = []
    for name in required:
        try:
            if importlib.util.find_spec(name) is None:
                missing.append(name)
        except (ImportError, ValueError):
            missing.append(name)
    return {
        "available": not missing,
        "missing": missing,
        "install_hint": "pip install torch transformers datasets",
    }


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def plan(spec: Mapping[str, Any], *, token: Optional[str] = None) -> dict[str, Any]:
    """Inspect the datasets and the checkpoint without loading a weight."""
    from core.advanced import scaling

    request = _normalise(spec)
    datasets = [_inspect(entry) for entry in request["datasets"]]
    total_rows = sum(max(0, entry["rows"]) for entry in datasets)
    total_tokens = sum(entry["tokens_estimate"] for entry in datasets)
    total_train_tokens = sum(
        int(round(entry["tokens_estimate"] * max(0.0, 1.0 - entry["eval_fraction"])))
        for entry in datasets
    )
    total_eval_rows = sum(entry["eval_rows"] for entry in datasets)

    suggested_from_tokens = _added_from_tokens(total_tokens)
    suggested_from_rows = _added_from_rows(total_rows)
    added = request["added_params"] or suggested_from_tokens

    try:
        scaled = scaling.plan(
            request["source"],
            added_params = float(added),
            method = request["method"],
            token = token,
        )
    except ValueError as error:
        raise ValueError(str(error)) from error

    projection = scaled.get("projection") or {}
    actual_added = int(projection.get("added_params") or scaled.get("added_params") or added)
    actual_target = int(projection.get("projected_params") or scaled.get("target_params") or 0)
    base_params = int(scaled.get("base_params") or 0)
    per_layer = int(projection.get("params_per_layer") or 0)
    new_layers = max(
        0,
        int(projection.get("new_num_layers") or 0) - int(projection.get("current_num_layers") or 0),
    )

    warnings: list[str] = []
    for entry in datasets:
        warnings.extend(f"{entry['name']}: {note}" for note in entry["notes"])

    if scaled.get("target_below_floor"):
        warnings.append(
            f"The request is {actual_target / max(1, base_params):.2f}x the base. CMT bypasses the "
            "Scaler's 2x floor on purpose; the growth is sized to the data, not to a scaling study."
        )
    if per_layer and actual_added > added * 1.5:
        warnings.append(
            f"The engine inserts whole blocks, so the real growth is {_format_params(actual_added)} "
            f"(one block at this width is ~{_format_params(per_layer)}), not {_format_params(added)}. "
            "Plan for the larger number."
        )
    if total_train_tokens < 1_000_000_000:
        warnings.append(
            f"{_format_params(total_train_tokens)} training tokens is far below the ~194B-token "
            "recovery datapoint the expansion module cites; a gain here is unproven, not promised."
        )
    if total_eval_rows == 0:
        warnings.append(
            "No dataset has a held-out split, so CMT cannot report a per-dataset delta. "
            "Set an eval fraction on at least one dataset."
        )

    blocked = bool(scaled.get("blocked"))
    blocked_reason = str(scaled.get("blocked_reason") or "") or None
    engine = scaled.get("engine") or {}
    if not engine.get("available"):
        blocked = True
        blocked_reason = str(engine.get("error") or "The expansion engine cannot load on this host.")
    if not projection or not actual_target or actual_target <= base_params:
        blocked = True
        blocked_reason = (
            blocked_reason
            or "The expansion engine could not project a larger checkpoint for this model."
        )
    deps = _dependencies()
    if not deps["available"]:
        blocked = True
        blocked_reason = (
            f"CMT needs {', '.join(deps['missing'])}; install them with `{deps['install_hint']}`."
        )
    if per_layer and per_layer > MAX_ADDED_PARAMS:
        blocked = True
        blocked_reason = (
            f"One block on this checkpoint is ~{_format_params(per_layer)}, already above CMT's "
            f"{_format_params(MAX_ADDED_PARAMS)} ceiling. CMT cannot add less than one block."
        )

    tokens_per_step = max(
        1,
        request["batch_size"] * request["gradient_accumulation"] * request["max_seq_length"],
    )
    suggested_steps = max(1, min(2000, math.ceil(max(1, total_train_tokens) / tokens_per_step)))

    return {
        "source": request["source"],
        "mode": request["mode"],
        "added_source": "override" if request["added_params"] else "tokens",
        "datasets": datasets,
        "totals": {
            "datasets": len(datasets),
            "rows": total_rows,
            "train_rows": sum(max(0, entry["train_rows"]) for entry in datasets),
            "eval_rows": total_eval_rows,
            "tokens_estimate": total_tokens,
            "train_tokens_estimate": total_train_tokens,
            "train_tokens_label": _format_params(total_train_tokens),
        },
        "sizing": {
            "rule": "tokens",
            "added_params": added,
            "added_label": _format_params(added),
            "suggested_from_tokens": suggested_from_tokens,
            "suggested_from_rows": suggested_from_rows,
            "min_added_params": MIN_ADDED_PARAMS,
            "max_added_params": MAX_ADDED_PARAMS,
            "note": (
                "added = clamp(((1.7 * megatokens + 12) / 10) * 1M, 1M, 700M). "
                "The rows fallback is 1M per 10k rows. Override either number."
            ),
        },
        "expansion": {
            "base_params": base_params,
            "base_label": _format_params(base_params),
            "target_params": actual_target,
            "target_label": _format_params(actual_target),
            "added_params": actual_added,
            "added_label": _format_params(actual_added),
            "growth_factor": float(projection.get("growth_factor") or 0.0),
            "params_per_layer": per_layer,
            "new_layers": new_layers,
            "floor_bypassed": bool(scaled.get("target_below_floor")),
            "growth_floor": scaled.get("growth_floor"),
            "supported": bool(projection),
            "notes": list(scaled.get("notes") or []),
        },
        "training": {
            "mode": request["mode"],
            "method": "new-blocks-only",
            "max_steps": request["max_steps"],
            "suggested_steps": suggested_steps,
            "batch_size": request["batch_size"],
            "gradient_accumulation": request["gradient_accumulation"],
            "max_seq_length": request["max_seq_length"],
            "learning_rate": request["learning_rate"],
            "eval_every": request["eval_every"],
            "patience": request["patience"],
            "seed": request["seed"],
            "train_tokens_per_step": tokens_per_step,
            "note": (
                "The inherited weights stay frozen; only the inserted blocks are optimised. "
                "A held-out perplexity task per dataset is scored during training."
            ),
        },
        "dependencies": deps,
        "warnings": warnings,
        "blocked": blocked,
        "blocked_reason": blocked_reason,
        "runnable": not blocked,
        "notes": [
            "CMT reads local JSONL, JSON, Parquet, CSV and TXT files in this version.",
            "Token counts are estimated at four characters per token while planning; training reports the real count.",
        ],
    }


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def _slug(value: str) -> str:
    cleaned = _SLUG.sub("-", value.strip().lower()).strip("-")
    return cleaned[:48] or "dataset"


def _render_row(
    row: Mapping[str, Any],
    *,
    text_column: Optional[str],
    tokenizer: Any,
    mode: str,
) -> str:
    if text_column and isinstance(row.get(text_column), str):
        return str(row[text_column])
    for key in ("text", "content", "document", "raw"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    messages = row.get("messages") or row.get("conversations")
    if isinstance(messages, list):
        normalised = []
        for turn in messages:
            if not isinstance(turn, Mapping):
                continue
            role = str(turn.get("role") or turn.get("from") or "user")
            content = turn.get("content") if "content" in turn else turn.get("value")
            if isinstance(content, list):
                content = " ".join(
                    str(part.get("text") or part) if isinstance(part, Mapping) else str(part)
                    for part in content
                )
            normalised.append({"role": role, "content": str(content or "")})
        if normalised and tokenizer is not None and mode == "finetune":
            try:
                return str(
                    tokenizer.apply_chat_template(
                        normalised,
                        tokenize = False,
                        add_generation_prompt = False,
                    )
                )
            except Exception:  # noqa: BLE001 - a model without a chat template falls back to text
                pass
        return "\n".join(f"{turn['role']}: {turn['content']}" for turn in normalised)
    return _row_text(row)


def _load_rows(entry: Mapping[str, Any]) -> Any:
    from datasets import load_dataset

    if entry.get("repo_id"):
        repo_id = str(entry["repo_id"])
        split = str(entry.get("split") or "train")
        try:
            return load_dataset(
                repo_id,
                name = entry.get("subset") or None,
                split = split,
            )
        except Exception as error:  # noqa: BLE001 - the loader's message names the split
            raise ValueError(
                f"{repo_id}: could not load split {split!r}: {error}"
            ) from error

    path = str(entry["path"])
    suffix = Path(path).suffix.lower()
    if suffix in (".jsonl", ".json"):
        return load_dataset("json", data_files = path, split = "train")
    if suffix == ".parquet":
        return load_dataset("parquet", data_files = path, split = "train")
    if suffix == ".csv":
        return load_dataset("csv", data_files = path, split = "train")
    return load_dataset("text", data_files = path, split = "train")


def _materialise(
    entries: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    mode: str,
    seed: int,
    max_rows: int,
    output_dir: Path,
) -> tuple[Any, dict[str, list[str]], list[dict[str, Any]]]:
    from datasets import concatenate_datasets

    train_parts = []
    eval_texts: dict[str, list[str]] = {}
    records: list[dict[str, Any]] = []
    eval_dir = output_dir / "eval"
    eval_dir.mkdir(parents = True, exist_ok = True)

    for entry in entries:
        raw = _load_rows(entry)
        text_column = entry.get("text_column")
        if text_column and text_column not in raw.column_names:
            raise ValueError(
                f"{entry['name']}: {text_column!r} is not a column; it has {', '.join(raw.column_names[:8])}."
            )
        columns = list(raw.column_names)
        formatted = raw.map(
            lambda row: {
                "text": _render_row(
                    row, text_column = text_column, tokenizer = tokenizer, mode = mode
                )
            },
            remove_columns = columns,
            desc = f"Formatting {entry['name']}",
        )
        formatted = formatted.filter(lambda row: bool(str(row["text"]).strip()))
        if len(formatted) == 0:
            raise ValueError(f"{entry['name']}: no readable text after formatting.")

        eval_fraction = float(entry.get("eval_fraction") or 0.0)
        train = formatted
        held_out = None
        if eval_fraction > 0 and len(formatted) >= 10:
            split = formatted.train_test_split(test_size = eval_fraction, seed = seed)
            train = split["train"]
            held_out = split["test"]
        elif eval_fraction > 0:
            raise ValueError(
                f"{entry['name']}: {len(formatted)} rows is too few for an eval split; add rows or set eval_fraction to 0."
            )

        if len(train) > max_rows:
            train = train.shuffle(seed = seed).select(range(max_rows))
        else:
            train = train.shuffle(seed = seed)
        train_parts.append(train)

        name = _slug(str(entry["name"]))
        texts: list[str] = []
        if held_out is not None:
            texts = [str(value) for value in held_out["text"]][:EVAL_TEXTS_PER_DATASET]
            target = eval_dir / f"{name}.jsonl"
            with target.open("w", encoding = "utf-8") as stream:
                for text in texts:
                    stream.write(json.dumps({"text": text}, ensure_ascii = False) + "\n")
            eval_texts[name] = texts
        sampled = [str(value) for value in train["text"]][:512]
        sampled_chars = sum(len(text) for text in sampled)
        tokens_estimate = (
            int(round((sampled_chars / 4.0) * (len(train) / len(sampled))))
            if sampled
            else 0
        )
        records.append(
            {
                "name": name,
                "path": entry["path"],
                "repo_id": entry.get("repo_id"),
                "rows": len(formatted),
                "train_rows": len(train),
                "eval_rows": len(texts),
                "tokens_estimate": tokens_estimate,
                "eval_file": str(eval_dir / f"{name}.jsonl") if texts else None,
            }
        )

    combined = concatenate_datasets(train_parts).shuffle(seed = seed)
    return combined, eval_texts, records


def _load_tokenizer(source: str, token: Optional[str]) -> Any:
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(source, token = token)
    except Exception as error:  # noqa: BLE001 - the message is the diagnosis
        raise ValueError(f"Could not load the tokenizer for {source}: {error}") from error
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_model(source: str, token: Optional[str]) -> tuple[Any, str]:
    import torch
    from transformers import AutoModelForCausalLM

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if getattr(torch.cuda, "is_bf16_supported", lambda: False)() else torch.float16
        device = "cuda"
    else:
        dtype = torch.float32
        device = "cpu"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            source,
            torch_dtype = dtype,
            low_cpu_mem_usage = True,
            token = token,
        )
    except Exception as error:  # noqa: BLE001
        raise ValueError(f"Could not load {source}: {error}") from error
    return model, device


def _eval_suite(eval_texts: Mapping[str, list[str]], *, max_length: int) -> dict[str, Any]:
    """One held-out perplexity task per dataset, all pointing the same way up."""
    from unsloth.models.scaling import heldout_perplexity_suite

    suite: dict[str, Any] = {}
    for name, texts in eval_texts.items():
        if not texts:
            continue
        tasks = heldout_perplexity_suite(texts, max_length = min(int(max_length), 512))
        if tasks:
            suite[name] = next(iter(tasks.values()))
    return suite


def run(
    spec: Mapping[str, Any],
    *,
    output_dir: str,
    token: Optional[str] = None,
    progress: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    """Expand, freeze the base, train the inserted blocks, and score each dataset."""
    from core.advanced import run_manifest, scaling

    def stage(name: str, message: str, **extra: Any) -> None:
        if progress is not None:
            progress({"stage": name, "message": message, **extra})

    report = plan(spec, token = token)
    if not report["runnable"]:
        raise ValueError(report.get("blocked_reason") or "This CMT plan cannot run.")
    request = _normalise(spec)
    out = Path(output_dir).expanduser()
    out.mkdir(parents = True, exist_ok = True)

    stage("tokenizer", "Loading the tokenizer...")
    tokenizer = _load_tokenizer(request["source"], token)

    stage("datasets", "Formatting and splitting each dataset...")
    train_dataset, eval_texts, dataset_records = _materialise(
        request["datasets"],
        tokenizer = tokenizer,
        mode = request["mode"],
        seed = request["seed"],
        max_rows = request["max_train_rows"],
        output_dir = out,
    )
    if not eval_texts:
        raise ValueError("No dataset produced a held-out split, so there is nothing to evaluate.")

    # The plan sized the growth from file sizes, which undercounts a Hub dataset it never
    # downloaded. Now that the real rows exist, re-derive the growth from them and say so;
    # an explicit override is the user's number and stays untouched.
    actual_tokens = sum(int(record.get("tokens_estimate") or 0) for record in dataset_records)
    if request["added_params"] is None and actual_tokens > 0:
        recomputed = _added_from_tokens(actual_tokens)
        try:
            re_scaled = scaling.plan(
                request["source"],
                added_params = float(recomputed),
                method = "depth",
                token = token,
            )
        except ValueError:
            re_scaled = None
        projection = (re_scaled or {}).get("projection") or {}
        target = int(projection.get("projected_params") or 0)
        added_real = int(projection.get("added_params") or 0)
        if target > int(report["expansion"]["base_params"]) and added_real > 0:
            report["expansion"].update(
                {
                    "target_params": target,
                    "target_label": _format_params(target),
                    "added_params": added_real,
                    "added_label": _format_params(added_real),
                    "growth_factor": float(projection.get("growth_factor") or 0.0),
                    "new_layers": max(
                        0,
                        int(projection.get("new_num_layers") or 0)
                        - int(projection.get("current_num_layers") or 0),
                    ),
                }
            )
            report["sizing"].update(
                {
                    "rule": "tokens-actual",
                    "added_params": recomputed,
                    "added_label": _format_params(recomputed),
                }
            )
            stage(
                "sizing",
                f"The datasets hold about {_format_params(actual_tokens)} tokens, so the model "
                f"grows by about {_format_params(added_real)} parameters.",
            )

    stage("loading", "Loading the base checkpoint on CPU...")
    model, device = _load_model(request["source"], token)

    stage("expanding", f"Growing by ~{report['expansion']['added_label']}...")
    engine = scaling.load_engine()
    model, expansion = engine.expand_model(
        model,
        target_params = float(report["expansion"]["target_params"]),
        method = "depth",
        insert_strategy = request["insert_strategy"],
        force_small_growth = True,
    )
    inserted = list(expansion.get("inserted_at") or [])
    if not inserted:
        raise ValueError("The expansion inserted no blocks, so there is nothing for CMT to train.")

    expanded_dir = out / "expanded"
    stage("saving-expanded", "Saving the expanded checkpoint...")
    model.save_pretrained(str(expanded_dir))
    tokenizer.save_pretrained(str(expanded_dir))

    if device == "cuda":
        stage("moving", "Moving the expanded model to the accelerator...")
        model.to(device)

    suite = _eval_suite(eval_texts, max_length = request["max_seq_length"])
    if not suite:
        raise ValueError("The held-out evaluation suite is empty.")

    training = request
    stage("training", "Training only the inserted blocks...")
    result = scaling.cpt_train(
        model,
        tokenizer,
        train_dataset,
        eval_suite = suite,
        base_scores = None,
        max_steps = int(training["max_steps"]),
        eval_every = int(training["eval_every"]),
        patience = int(training["patience"]),
        learning_rate = float(training["learning_rate"]),
        new_layer_lr_multiplier = 3.0,
        use_lora = False,
        train_new_layers_only = True,
        save_final = True,
        per_device_batch_size = int(training["batch_size"]),
        gradient_accumulation_steps = int(training["gradient_accumulation"]),
        max_seq_length = int(training["max_seq_length"]),
        gradient_checkpointing = device == "cuda",
        output_dir = str(out / "train"),
        new_layer_indices = inserted,
        seed = int(training["seed"]),
        report_to = "none",
    )

    per_dataset: dict[str, Any] = {}
    base_scores = dict(result.get("base_scores") or {})
    best_scores = dict(result.get("best_scores") or {})
    for name in eval_texts:
        base = base_scores.get(name)
        best = best_scores.get(name)
        per_dataset[name] = {
            "name": name,
            "eval_rows": len(eval_texts[name]),
            "base_score": base,
            "best_score": best,
            "delta": (best - base) if isinstance(base, (int, float)) and isinstance(best, (int, float)) else None,
        }

    manifest = run_manifest.build_manifest(
        "cmt",
        config = {
            "mode": request["mode"],
            "method": "new-blocks-only",
            "added_params": report["expansion"]["added_params"],
            "target_params": report["expansion"]["target_params"],
            "base_params": report["expansion"]["base_params"],
            "floor_bypassed": report["expansion"]["floor_bypassed"],
            "sizing_rule": report["sizing"]["rule"],
            "dataset_tokens": actual_tokens,
            "max_steps": request["max_steps"],
            "batch_size": request["batch_size"],
            "gradient_accumulation": request["gradient_accumulation"],
            "max_seq_length": request["max_seq_length"],
            "learning_rate": request["learning_rate"],
            "seed": request["seed"],
            "datasets": [
                {
                    "name": record["name"],
                    "rows": record["rows"],
                    "train_rows": record["train_rows"],
                    "eval_rows": record["eval_rows"],
                }
                for record in dataset_records
            ],
        },
        seeds = {"seed": request["seed"]},
        inputs = {
            "source": run_manifest.checkpoint_digest(request["source"]),
            "datasets": [
                {
                    "path": record["path"],
                    "sha256": run_manifest.sampled_file_digest(record["path"]),
                    "size_bytes": Path(record["path"]).stat().st_size,
                }
                for record in dataset_records
            ],
        },
        artifacts = {
            "output_dir": str(out),
            "expanded_dir": str(expanded_dir),
            "final_checkpoint": result.get("final_checkpoint"),
            "best_checkpoint": result.get("best_checkpoint"),
            "new_layers": len(inserted),
            "params": report["expansion"]["target_params"],
            "per_dataset": per_dataset,
        },
    )
    manifest_path = run_manifest.write_manifest(out, manifest)

    stage("completed", "CMT finished; per-dataset scores are in the result.", done = 1, total = 1)
    return {
        "output_dir": str(out),
        "manifest_path": str(manifest_path),
        "expanded_dir": str(expanded_dir),
        "mode": request["mode"],
        "base_params": report["expansion"]["base_params"],
        "final_params": int(expansion.get("final_params") or 0),
        "added_params": int(expansion.get("added_params") or 0),
        "growth_factor": float(expansion.get("growth_factor") or 0.0),
        "new_layers": len(inserted),
        "inserted_at": inserted,
        "datasets": dataset_records,
        "per_dataset": per_dataset,
        "dataset_tokens": actual_tokens,
        "sizing_rule": report["sizing"]["rule"],
        "training": {
            "steps_run": result.get("steps_run"),
            "tokens_seen": result.get("tokens_seen"),
            "gpu_hours": result.get("gpu_hours"),
            "peak_mem_gb": result.get("peak_mem_gb"),
            "recovery_step": result.get("recovery_step"),
            "new_layer_grad_nonzero": result.get("new_layer_grad_nonzero"),
            "final_checkpoint": result.get("final_checkpoint"),
            "best_checkpoint": result.get("best_checkpoint"),
        },
        "warnings": report["warnings"],
    }
