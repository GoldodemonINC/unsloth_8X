# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Alternating curriculum co-training.

Instead of concatenating datasets (which makes the model forget the first one)
or statically mixing them (which blurs the boundary between domains), the
curriculum loader alternates *batch steps* between streams on a fixed cadence::

    reasoning x2, code x1, reasoning x2, code x1, ...

The schedule is a pure function of the global step, so a resumed run lands on
the same stream it would have without the interruption.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator, Sequence


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    steps: int
    dataset: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CurriculumPlan:
    stages: tuple[CurriculumStage, ...]
    cycle_steps: int
    notes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": [stage.to_dict() for stage in self.stages],
            "cycle_steps": self.cycle_steps,
            "notes": list(self.notes),
            "warnings": list(self.warnings),
        }

    def stage_for_step(self, step: int) -> CurriculumStage:
        if not self.stages:
            raise ValueError("Curriculum has no stages.")
        index = int(step) % self.cycle_steps
        cursor = 0
        for stage in self.stages:
            cursor += stage.steps
            if index < cursor:
                return stage
        return self.stages[-1]

    def stream_sequence(self, total_steps: int) -> list[str]:
        return [self.stage_for_step(step).name for step in range(max(0, int(total_steps)))]


def parse_cadence(cadence: str | Sequence[Any] | None) -> tuple[CurriculumStage, ...]:
    """Parse ``"reasoning:2,code:1"`` (or ``"2:1"``) into stages."""
    if cadence is None:
        raise ValueError("A curriculum cadence is required.")
    if isinstance(cadence, str):
        text = cadence.strip()
        if not text:
            raise ValueError("Curriculum cadence is empty.")
        stages: list[CurriculumStage] = []
        for index, part in enumerate(text.split(",")):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                name, raw_steps = part.rsplit(":", 1)
                name = name.strip()
            else:
                name, raw_steps = f"stage{index + 1}", part
            try:
                steps = int(raw_steps)
            except ValueError as error:
                raise ValueError(f"Invalid step count in curriculum cadence: {part!r}") from error
            if steps <= 0:
                raise ValueError("Every curriculum stage needs at least one step.")
            stages.append(CurriculumStage(name=name, steps=steps))
        if not stages:
            raise ValueError("Curriculum cadence parsed to no stages.")
        return tuple(stages)
    if isinstance(cadence, (list, tuple)):
        stages = []
        for index, entry in enumerate(cadence):
            if isinstance(entry, str):
                stages.extend(parse_cadence(entry))
                continue
            if not isinstance(entry, dict):
                raise ValueError("Curriculum stages must be strings or objects.")
            name = str(entry.get("name") or f"stage{index + 1}")
            steps = int(entry.get("steps", 0))
            if steps <= 0:
                raise ValueError(f"Curriculum stage {name!r} needs steps > 0.")
            stages.append(
                CurriculumStage(name=name, steps=steps, dataset=entry.get("dataset"))
            )
        if not stages:
            raise ValueError("Curriculum needs at least one stage.")
        return tuple(stages)
    raise ValueError("Curriculum cadence must be a string or a list of stages.")


def build_curriculum_plan(
    cadence: str | Sequence[Any] | None,
    *,
    total_steps: int | None = None,
    datasets: Sequence[str] | None = None,
) -> CurriculumPlan:
    stages = parse_cadence(cadence)
    cycle = sum(stage.steps for stage in stages)
    notes = [f"One full cycle spans {cycle} optimizer steps."]
    warnings: list[str] = []
    if datasets is not None:
        available = {str(name) for name in datasets}
        for stage in stages:
            if stage.name not in available and stage.dataset not in available:
                warnings.append(
                    f"Curriculum stage {stage.name!r} does not match a configured dataset stream."
                )
    if total_steps is not None and total_steps > 0 and total_steps % cycle:
        warnings.append(
            f"The run ends mid-cycle ({total_steps} steps is not a multiple of {cycle}); "
            "the trailing domain gets fewer updates."
        )
    if len(stages) == 1:
        warnings.append("A single-stage curriculum is equivalent to plain training.")
    return CurriculumPlan(
        stages=stages, cycle_steps=cycle, notes=tuple(notes), warnings=tuple(warnings)
    )


@dataclass
class CurriculumSampler:
    """Yields batch indices from the stage that owns each global step.

    ``streams`` maps a stage name to a finite sequence of row indices. Rows are
    cycled when a stage is visited more often than its source has rows, so a
    long run never runs dry; the cycling is documented in ``exhausted`` so the
    UI can warn about repetition.
    """

    plan: CurriculumPlan
    streams: dict[str, Sequence[int]]
    batch_size: int = 1
    seed: int = 3407
    _cursors: dict[str, int] = field(default_factory=dict, init=False)
    _order: dict[str, list[int]] = field(default_factory=dict, init=False)
    _epochs: dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        import random

        rng = random.Random(self.seed)
        for name, indices in self.streams.items():
            order = [int(index) for index in indices]
            rng.shuffle(order)
            if not order:
                raise ValueError(f"Dataset stream {name!r} is empty.")
            self._order[name] = order
            self._cursors[name] = 0
            self._epochs[name] = 0

    def _next_row(self, name: str) -> int:
        order = self._order[name]
        cursor = self._cursors[name]
        if cursor >= len(order):
            # Reshuffle every pass so repeated rows are not in a fixed cycle.
            import random

            random.Random(self.seed + self._epochs[name] + 1).shuffle(order)
            cursor = 0
            self._epochs[name] += 1
        self._cursors[name] = cursor + 1
        return order[cursor]

    def batch_for_step(self, step: int) -> list[int]:
        stage = self.plan.stage_for_step(step)
        name = stage.dataset or stage.name
        if name not in self._order:
            raise ValueError(f"No dataset stream registered for curriculum stage {name!r}.")
        return [self._next_row(name) for _ in range(self.batch_size)]

    def __iter__(self) -> Iterator[list[int]]:
        step = 0
        while True:
            yield self.batch_for_step(step)
            step += 1

    def epochs(self) -> dict[str, int]:
        return dict(self._epochs)

    def snapshot(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "cursors": dict(self._cursors),
            "epochs": dict(self._epochs),
        }
