# SPDX-License-Identifier: NRC-2
# Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-2.md

"""Scale before training: grow a checkpoint by the size its data justifies.

CMT and this stage answer the same first question -- how much capacity does this
dataset pay for? -- and then part ways:

* CMT freezes everything it inherited and trains only the blocks it inserted.
* Auto-scale only grows the checkpoint. The method the user picked (LoRA, QLoRA,
  full, CPT) then trains that larger model normally.

The sizing rule is shared with CMT instead of restated here, so the 1M-700M band
and the rows fallback cannot drift apart. Nothing in this module trains.

Two limits are the engine's, and the plan says so rather than hiding them:

* Expansion inserts whole transformer blocks, so the real growth is a multiple of
  one block at that width, and may exceed the number the data asked for.
* A growth below the Scaler's 2x research floor is forced here on purpose. The
  manifest records ``floor_bypassed: true``; no other stage gets that flag.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from features.cmt import cmt


def _override(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("added_params must be a number.") from error
    if parsed < cmt.MIN_ADDED_PARAMS or parsed > cmt.MAX_ADDED_PARAMS:
        raise ValueError(
            f"added_params must be between {cmt._format_params(cmt.MIN_ADDED_PARAMS)} and "
            f"{cmt._format_params(cmt.MAX_ADDED_PARAMS)}."
        )
    return int(parsed)


def _datasets(spec: Mapping[str, Any]) -> list[str]:
    raw = spec.get("datasets")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("datasets must be a list of local files.")
    paths: list[str] = []
    for index, entry in enumerate(raw, start = 1):
        value = entry.get("path") if isinstance(entry, Mapping) else entry
        path = str(value or "").strip()
        if not path:
            raise ValueError(f"Dataset {index} has no path.")
        if str(Path(path).expanduser()) in paths:
            raise ValueError(f"Dataset {index} repeats {path}.")
        paths.append(str(Path(path).expanduser()))
    return paths


def _hint(spec: Mapping[str, Any], key: str) -> int:
    value = spec.get(key)
    if value in (None, ""):
        return 0
    try:
        parsed = int(float(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key} must be a whole number.") from error
    if parsed < 0:
        raise ValueError(f"{key} cannot be negative.")
    return parsed


def plan(spec: Mapping[str, Any], *, token: Optional[str] = None) -> dict[str, Any]:
    """Size the growth, and check the engine can produce it, without loading a weight."""
    from shared import scaling

    model = str(spec.get("model") or "").strip()
    if not model:
        raise ValueError("A model is required: a checkpoint path or a Hub id.")

    paths = _datasets(spec)
    inspected = [cmt.inspect_file(path) for path in paths]
    rows = _hint(spec, "rows") or sum(
        int(entry["rows"]) for entry in inspected if int(entry["rows"]) > 0
    )
    tokens = _hint(spec, "tokens") or sum(int(entry["tokens_estimate"]) for entry in inspected)

    size = cmt.sizing(rows, tokens)
    added = _override(spec.get("added_params")) or int(size["added_params"])

    try:
        scaled = scaling.plan(
            model,
            added_params = float(added),
            method = str(spec.get("method") or "depth"),
            token = token,
        )
    except ValueError as error:
        raise ValueError(str(error)) from error

    projection = scaled.get("projection") or {}
    base_params = int(scaled.get("base_params") or 0)
    actual_added = int(projection.get("added_params") or scaled.get("added_params") or added)
    target_params = int(projection.get("projected_params") or scaled.get("target_params") or 0)
    per_layer = int(projection.get("params_per_layer") or 0)
    new_layers = max(
        0,
        int(projection.get("new_num_layers") or 0)
        - int(projection.get("current_num_layers") or 0),
    )

    warnings: list[str] = []
    for entry in inspected:
        warnings.extend(f"{entry['name']}: {note}" for note in entry["notes"])
    if rows == 0 and tokens == 0:
        warnings.append(
            "No dataset size was known when this was sized, so the growth is the 1M floor. "
            "Add the local dataset files, or give a row or token count, for a data-sized number."
        )
    if scaled.get("target_below_floor"):
        warnings.append(
            f"The request is {target_params / max(1, base_params):.2f}x the base. Auto-scale bypasses "
            "the Scaler's 2x floor on purpose: the growth answers the data, not a scaling study."
        )
    if per_layer and actual_added > added * 1.5:
        warnings.append(
            f"The engine inserts whole blocks, so the real growth is ~{cmt._format_params(actual_added)} "
            f"(one block at this width is ~{cmt._format_params(per_layer)}), not "
            f"~{cmt._format_params(added)}. Plan for the larger number."
        )

    engine = scaled.get("engine") or {}
    blocked = bool(scaled.get("blocked"))
    blocked_reason = str(scaled.get("blocked_reason") or "") or None
    if not engine.get("available"):
        blocked = True
        blocked_reason = str(engine.get("error") or "The expansion engine cannot load on this host.")
    if not projection or target_params <= base_params:
        blocked = True
        blocked_reason = blocked_reason or (
            "The expansion engine could not project a larger checkpoint for this model."
        )
    if per_layer and per_layer > cmt.MAX_ADDED_PARAMS:
        blocked = True
        blocked_reason = (
            f"One block on this checkpoint is ~{cmt._format_params(per_layer)}, already above the "
            f"{cmt._format_params(cmt.MAX_ADDED_PARAMS)} ceiling. The engine cannot add less than one block."
        )

    return {
        "model": model,
        "method": str(spec.get("method") or "depth"),
        "sizing": {**size, "added_params": added, "added_label": cmt._format_params(added),
                   "source": "override" if _override(spec.get("added_params")) else size["rule"]},
        "datasets": [
            {
                "path": entry["path"],
                "name": entry["name"],
                "rows": entry["rows"],
                "bytes": entry["bytes"],
                "tokens_estimate": entry["tokens_estimate"],
            }
            for entry in inspected
        ],
        "expansion": {
            "base_params": base_params,
            "base_label": cmt._format_params(base_params),
            "added_params": actual_added,
            "added_label": cmt._format_params(actual_added),
            "target_params": target_params,
            "target_label": cmt._format_params(target_params),
            "growth_factor": float(projection.get("growth_factor") or 0.0),
            "params_per_layer": per_layer,
            "new_layers": new_layers,
            # Below 2x is allowed now, but still outside the measured range: the flag
            # survives so the plan can say the growth is an experiment.
            "floor_bypassed": float(projection.get("growth_factor") or 0.0) < 2.0,
            "growth_floor": scaled.get("growth_floor"),
            "notes": list(scaled.get("notes") or []),
        },
        "warnings": warnings,
        "blocked": blocked,
        "blocked_reason": blocked_reason,
        "runnable": not blocked,
        "notes": [
            "This stage only grows the checkpoint; training is a separate run of the method you pick.",
            "Token counts are estimated at four characters per token; the training run reports the real count.",
        ],
    }


def run(
    spec: Mapping[str, Any],
    *,
    output_dir: str,
    token: Optional[str] = None,
    progress: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    """Expand the checkpoint and save it, so Training can use it as its model."""
    from shared import run_manifest, scaling

    def stage(name: str, message: str, **extra: Any) -> None:
        if progress is not None:
            progress({"stage": name, "message": message, **extra})

    report = plan(spec, token = token)
    if not report["runnable"]:
        raise ValueError(report.get("blocked_reason") or "This scale plan cannot run.")

    out = Path(output_dir).expanduser()
    out.mkdir(parents = True, exist_ok = True)
    checkpoint = out / "scaled"

    stage("loading", f"Loading {report['model']} on CPU...")
    model, device = cmt.load_cpu_model(report["model"], token)

    stage(
        "expanding",
        f"Growing {report['expansion']['base_label']} by ~{report['expansion']['added_label']}...",
    )
    engine = scaling.load_engine()
    model, expansion = engine.expand_model(
        model,
        target_params = float(report["expansion"]["target_params"]),
        method = report["method"],
        insert_strategy = str(spec.get("insert_strategy") or "even"),
        force_small_growth = True,
    )
    inserted = list(expansion.get("inserted_at") or [])
    if not inserted:
        raise ValueError("The expansion inserted no blocks, so there is nothing to save.")

    stage("saving", f"Saving the scaled checkpoint to {checkpoint}...")
    model.save_pretrained(str(checkpoint))
    tokenizer = cmt.load_tokenizer(report["model"], token)
    tokenizer.save_pretrained(str(checkpoint))

    final_params = int(expansion.get("final_params") or report["expansion"]["target_params"])
    manifest = run_manifest.build_manifest(
        "autoscale",
        config = {
            "method": report["method"],
            "sizing_rule": report["sizing"]["rule"],
            "added_params": report["sizing"]["added_params"],
            "target_params": report["expansion"]["target_params"],
            "base_params": report["expansion"]["base_params"],
            "floor_bypassed": report["expansion"]["floor_bypassed"],
            "rows": report["sizing"]["rows"],
            "tokens": report["sizing"]["tokens"],
        },
        inputs = {"source": run_manifest.checkpoint_digest(report["model"])},
        artifacts = {
            "output_dir": str(out),
            "checkpoint": str(checkpoint),
            "added_params": int(expansion.get("added_params") or 0),
            "params": final_params,
            "new_layers": len(inserted),
        },
    )
    manifest_path = run_manifest.write_manifest(out, manifest)

    stage("completed", "The scaled checkpoint is ready to train.", done = 1, total = 1)
    return {
        "output_dir": str(out),
        "checkpoint": str(checkpoint),
        "manifest_path": str(manifest_path),
        "base_params": report["expansion"]["base_params"],
        "added_params": int(expansion.get("added_params") or 0),
        "final_params": final_params,
        "growth_factor": float(expansion.get("growth_factor") or 0.0),
        "new_layers": len(inserted),
        "inserted_at": inserted,
        "device": device,
        "sizing": report["sizing"],
        "warnings": report["warnings"],
    }
