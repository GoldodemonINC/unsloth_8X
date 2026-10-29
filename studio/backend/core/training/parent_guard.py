# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


@dataclass(frozen = True)
class ParentLossReference:
    run_id: str
    label: str
    points: tuple[tuple[float, float], ...]
    patience_steps: int
    min_delta: float = 0.0

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ParentLossReference":
        run_id = str(payload["run_id"]).strip()
        if not run_id:
            raise ValueError("parent run id is required")
        patience = _positive_int(payload.get("patience_steps"))
        if patience <= 0:
            raise ValueError("parent guard patience must be positive")
        delta = _finite_nonnegative(payload.get("min_delta", 0.0))
        raw_points = payload.get("points") or []
        points: list[tuple[float, float]] = []
        for raw in raw_points:
            try:
                step = float(raw[0])
                loss = float(raw[1])
            except (TypeError, ValueError, IndexError):
                continue
            if math.isfinite(step) and math.isfinite(loss):
                points.append((step, loss))
        points.sort(key = lambda point: point[0])
        if not points:
            raise ValueError("parent run has no recorded loss points")
        return cls(
            run_id = run_id,
            label = str(payload.get("label") or run_id),
            points = tuple(points),
            patience_steps = patience,
            min_delta = delta,
        )


@dataclass
class ParentLossGuard:
    reference: ParentLossReference
    parent_steps: tuple[float, ...] = field(init = False)
    last_improvement_step: Optional[float] = None
    last_parent_step: Optional[float] = None
    last_parent_loss: Optional[float] = None
    best_child_loss: Optional[float] = None
    signal: Optional[dict] = None

    def __post_init__(self) -> None:
        self.parent_steps = tuple(step for step, _loss in self.reference.points)

    def observe(self, step: Any, loss: Any) -> Optional[dict]:
        if self.signal is not None:
            return self.signal
        try:
            child_step = float(step)
            child_loss = float(loss)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(child_step) or not math.isfinite(child_loss):
            return None

        index = bisect.bisect_right(self.parent_steps, child_step) - 1
        if index < 0:
            return None
        parent_step, parent_loss = self.reference.points[index]
        self.last_parent_step = parent_step
        self.last_parent_loss = parent_loss

        if self.last_improvement_step is None:
            self.last_improvement_step = child_step

        if child_loss < parent_loss - self.reference.min_delta:
            self.last_improvement_step = child_step
            self.best_child_loss = (
                child_loss
                if self.best_child_loss is None
                else min(self.best_child_loss, child_loss)
            )
            return None

        elapsed = child_step - self.last_improvement_step
        if elapsed < self.reference.patience_steps:
            return None

        self.signal = {
            "parent_run_id": self.reference.run_id,
            "parent_label": self.reference.label,
            "step": child_step,
            "loss": child_loss,
            "parent_step": parent_step,
            "parent_loss": parent_loss,
            "patience_steps": self.reference.patience_steps,
            "min_delta": self.reference.min_delta,
            "elapsed_steps": elapsed,
            "best_child_loss": self.best_child_loss,
        }
        return self.signal

    @property
    def triggered(self) -> bool:
        return self.signal is not None


def _positive_int(value: Any) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(number) or number < 1:
        return 0
    return int(number)


def _finite_nonnegative(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("parent guard min_delta must be a number")
    if not math.isfinite(number) or number < 0:
        raise ValueError("parent guard min_delta must be finite and non-negative")
    return number


def parent_loss_points(metrics: dict[str, Any]) -> list[list[float]]:
    steps: Iterable[Any] = metrics.get("loss_step_history") or []
    losses: Iterable[Any] = metrics.get("loss_history") or []
    points: list[list[float]] = []
    for step, loss in zip(steps, losses):
        try:
            step_number = float(step)
            loss_number = float(loss)
        except (TypeError, ValueError):
            continue
        if step_number > 0 and math.isfinite(step_number) and math.isfinite(loss_number):
            points.append([step_number, loss_number])
    return points
