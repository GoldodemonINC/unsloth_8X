# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Glue between the Advanced Studio config and the training worker.

Three mechanics are applied inside a normal SFT run:

* **QAT** installs fake-quantization on the prepared model's Linear layers.
* **Layer-wise LR** splits the optimizer's param groups by structural zone.
* **Curriculum / blender** overrides the trainer's batch sampler so each step
  draws from the stream the cadence (or the ratios) names.

The Regression Guard (post-run eval) runs after the checkpoint is saved.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from . import blending, curriculum, layer_lr, qat as qat_engine


def install_qat(model, advanced: Mapping[str, Any] | None):
    """Apply the QAT config to a prepared model, returning a removable session."""
    config = (advanced or {}).get("qat") or {}
    if not config.get("enabled"):
        return None
    return qat_engine.apply_qat(model, qat_engine.validate_qat_config(config))


def _num_layers(model) -> int | None:
    config = getattr(model, "config", None)
    for name in ("num_hidden_layers", "n_layer", "num_layers"):
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        for name in ("num_hidden_layers", "n_layer", "num_layers"):
            value = getattr(text_config, name, None)
            if isinstance(value, int) and value > 0:
                return value
    return None


def patch_layer_lr(trainer, model, advanced: Mapping[str, Any] | None, base_lr: float) -> dict[str, Any] | None:
    """Wrap ``trainer.create_optimizer`` so each parameter gets its zone's LR.

    The scheduler is built from ``optimizer.param_groups`` after this returns,
    so base_lrs match the zone learning rates rather than the global one.
    """
    config = (advanced or {}).get("layer_lr") or {}
    if not config.get("enabled"):
        return None
    try:
        resolved = layer_lr.validate_layer_lr_config(base_lr, config)
    except ValueError:
        return None
    total_layers = _num_layers(model)
    original = getattr(trainer, "create_optimizer", None)
    if not callable(original):
        return None

    def _optimizer_factory(*args, **kwargs):
        optimizer = original(*args, **kwargs)
        name_by_param = {id(param): name for name, param in model.named_parameters()}
        # One group per (zone, kind, lr), not one per parameter: the LR scheduler
        # steps every group each iteration, so a per-parameter explosion (tens of
        # thousands on a 256-expert MoE) is pure overhead for identical math.
        groups: dict[tuple[str, str, float], dict[str, Any]] = {}
        for group in optimizer.param_groups:
            options = {key: value for key, value in group.items() if key != "params"}
            group_lr = float(group.get("lr", base_lr))
            group_initial = options.get("initial_lr")
            for parameter in group["params"]:
                name = name_by_param.get(id(parameter), "")
                zone, kind, multiplier = layer_lr.resolve_assignment(
                    name, resolved, total_layers
                )
                lr = group_lr * multiplier
                key = (zone, kind, round(lr, 12))
                merged = groups.get(key)
                if merged is None:
                    merged = dict(options)
                    merged["params"] = []
                    merged["lr"] = lr
                    if group_initial is not None:
                        merged["initial_lr"] = float(group_initial) * multiplier
                    merged["_zone"] = zone
                    merged["_kind"] = kind
                    merged["_multiplier"] = multiplier
                    groups[key] = merged
                merged["params"].append(parameter)
        optimizer.param_groups = list(groups.values())
        return optimizer

    trainer.create_optimizer = _optimizer_factory
    return {
        "enabled": True,
        "total_layers": total_layers,
        "zones": resolved.to_dict(),
        # Only the kinds that actually move, so a zone-only run reports none.
        "kinds": {
            kind: value
            for kind, value in resolved.kind_multipliers.items()
            if value != 1.0
        },
    }


class _IndexSampler:
    """Torch-compatible sampler over one concatenated, multi-stream dataset."""

    def __init__(self, indices: Sequence[int]):
        self._indices = list(indices)

    def __iter__(self):
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)


def _rows_for_step(
    layout: Sequence[Mapping[str, Any]],
    *,
    step: int,
    batch_size: int,
    plan_blend,
    plan_curriculum,
    cursors: dict[str, int],
    orders: dict[str, list[int]],
    seed: int,
):
    import random

    if plan_curriculum is not None:
        stage = plan_curriculum.stage_for_step(step)
        name = stage.dataset or stage.name
        targets = [name] * batch_size
    else:
        targets = plan_blend.batch_layout()

    selected: list[int] = []
    for name in targets:
        stream = next((item for item in layout if item["name"] == name), None)
        if stream is None:
            continue
        order = orders[name]
        cursor = cursors[name]
        if cursor >= len(order):
            random.Random(seed + step + 1).shuffle(order)
            cursor = 0
        selected.append(stream["start"] + order[cursor])
        cursors[name] = cursor + 1
    return selected


def build_stream_sampler(
    layout: Sequence[Mapping[str, Any]],
    advanced: Mapping[str, Any],
    *,
    batch_size: int,
    steps_per_epoch: int,
    seed: int = 3407,
):
    """Build a batch sampler that alternates streams per optimizer step."""
    import random

    blend_config = (advanced.get("blend") or {}) if advanced else {}
    curriculum_config = (advanced.get("curriculum") or {}) if advanced else {}
    plan_blend = None
    plan_curriculum = None
    if curriculum_config.get("enabled"):
        plan_curriculum = curriculum.build_curriculum_plan(
            curriculum_config.get("cadence"),
            total_steps = steps_per_epoch,
        )
    if blend_config.get("enabled"):
        plan_blend = blending.validate_blend_slots(
            blend_config.get("slots") or [], batch_size = batch_size
        )
    if plan_blend is None and plan_curriculum is None:
        return None

    # Ratio changes are events keyed by global step, so the batch stream is a pure
    # function of (seed, events, step). A run with no slider move has an empty log
    # and the base ratio applies at every step -- which is what makes a resume
    # after a checkpoint reproduce the stream exactly.
    schedule = (
        blending.BlendSchedule(base = plan_blend, batch_size = batch_size)
        if plan_blend is not None
        else None
    )

    orders: dict[str, list[int]] = {}
    cursors: dict[str, int] = {}
    for stream in layout:
        local = list(range(int(stream["length"])))
        random.Random(seed).shuffle(local)
        orders[stream["name"]] = local
        cursors[stream["name"]] = 0

    indices: list[int] = []
    for step in range(max(1, steps_per_epoch)):
        indices.extend(
            _rows_for_step(
                layout,
                step = step,
                batch_size = batch_size,
                plan_blend = plan_blend,
                plan_curriculum = plan_curriculum,
                cursors = cursors,
                orders = orders,
                seed = seed,
            )
        )
    return _IndexSampler(indices), {
        "blend": plan_blend.to_dict() if plan_blend else None,
        "blend_schedule": schedule.manifest_block() if schedule is not None else None,
        "curriculum": plan_curriculum.to_dict() if plan_curriculum else None,
        "steps_per_epoch": steps_per_epoch,
    }


def patch_stream_sampler(
    trainer,
    advanced: Mapping[str, Any] | None,
    *,
    batch_size: int,
    seed: int = 3407,
) -> dict[str, Any] | None:
    """Replace ``_get_train_sampler`` with the cadence/ratio sampler."""
    layout = (advanced or {}).get("_stream_layout")
    if not layout:
        return None
    train_dataset = getattr(trainer, "train_dataset", None)
    total_rows = len(train_dataset) if hasattr(train_dataset, "__len__") else 0
    if total_rows <= 0:
        return None
    steps_per_epoch = max(1, math.ceil(total_rows / max(1, batch_size)))
    built = build_stream_sampler(
        layout,
        advanced,
        batch_size = batch_size,
        steps_per_epoch = steps_per_epoch,
        seed = seed,
    )
    if built is None:
        return None
    sampler, report = built
    trainer._get_train_sampler = lambda *args, **kwargs: sampler
    return report


def auto_eval_config(advanced: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    config = (advanced or {}).get("auto_eval") or {}
    return config if config.get("enabled") else None


def run_auto_eval(
    *,
    advanced: Mapping[str, Any] | None,
    baseline_model: str,
    candidate_model: str,
    output_dir: str,
    event_queue,
    timeout: int = 3600,
) -> None:
    """Run the post-training comparison and publish the report event.

    Failures are reported as an ``auto_eval`` event with ``errors``; they never
    fail the training run that just succeeded.
    """
    config = auto_eval_config(advanced)
    if config is None:
        return
    from . import eval_harness, eval_runner

    event_queue.put(
        {
            "type": "status",
            "message": "Run complete. Starting automated evaluation…",
        }
    )
    try:
        plan = eval_harness.build_eval_plan(
            config.get("suites") or [],
            baseline_model = baseline_model,
            candidate_model = candidate_model,
            eval_dataset = config.get("eval_dataset"),
            limit = config.get("limit"),
            batch_size = int(config.get("batch_size") or 8),
        )
        report = eval_runner.run_plan(
            plan,
            output_dir = output_dir,
            timeout = timeout,
        )
        event_queue.put({"type": "auto_eval", "report": report})
        event_queue.put(
            {
                "type": "warning",
                "message": "Regression Guard finished: "
                + str(report.get("comparison", {}).get("verdict", "unknown")),
            }
        )
    except Exception as error:  # noqa: BLE001 - post-run, never fatal
        event_queue.put(
            {
                "type": "auto_eval",
                "report": {
                    "plan": config,
                    "comparison": {"verdict": "unavailable", "entries": []},
                    "errors": [str(error)],
                },
            }
        )
