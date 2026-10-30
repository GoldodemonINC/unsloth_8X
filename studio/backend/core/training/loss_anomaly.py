# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import heapq
import math
from collections import deque
from typing import Iterable, NamedTuple, Optional

MAX_RETAINED_POINTS = 8192
BASELINE_WINDOW = 3600
RECENT_WINDOW = 3600
PLATEAU_WINDOW = 2600
SPIKE_BASELINE = 20
CHECK_EVERY = 10


class _FloorTerm(NamedTuple):
    window: str
    value: int
    required: int


def _floor_terms() -> list[_FloorTerm]:
    return [
        _FloorTerm("RECENT_WINDOW", RECENT_WINDOW, RECENT_WINDOW + CHECK_EVERY),
        _FloorTerm("BASELINE_WINDOW", BASELINE_WINDOW, 2 * BASELINE_WINDOW + CHECK_EVERY + 1),
        _FloorTerm("PLATEAU_WINDOW", PLATEAU_WINDOW, PLATEAU_WINDOW + CHECK_EVERY),
        _FloorTerm("SPIKE_BASELINE", SPIKE_BASELINE, SPIKE_BASELINE + CHECK_EVERY + 1),
    ]


def _worst_term() -> _FloorTerm:
    return max(_floor_terms(), key = lambda term: term.required)


def retained_points_floor() -> int:
    return _worst_term().required


def _retention_refusal() -> Optional[str]:
    worst = _worst_term()
    if worst.required <= MAX_RETAINED_POINTS:
        return None
    return (
        f"MAX_RETAINED_POINTS = {MAX_RETAINED_POINTS} is below the floor "
        f"{worst.required} required by {worst.window} = {worst.value}"
    )


if (refusal := _retention_refusal()) is not None:
    raise AssertionError(refusal)


def _median(values: list[float]) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 0:
        return (ordered[middle - 1] + ordered[middle]) / 2
    return ordered[middle]


class RetainedSeries:
    def __init__(self, points: list[tuple[float, float]], base: int, total: int):
        self.points = points
        self.base = base
        self.total = total

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(self.total)
            start = max(start, self.base)
            stop = max(stop, self.base)
            return self.points[start - self.base : stop - self.base : step]
        if index < 0:
            index += self.total
        if index < self.base:
            raise IndexError(index)
        return self.points[index - self.base]


class _PrefixMedian:
    def __init__(self):
        self.low: list[float] = []
        self.high: list[float] = []

    def add(self, value: float) -> None:
        if self.low and value <= -self.low[0]:
            heapq.heappush(self.low, -value)
        else:
            heapq.heappush(self.high, value)
        while len(self.low) > len(self.high) + 1:
            heapq.heappush(self.high, -heapq.heappop(self.low))
        while len(self.low) < len(self.high):
            heapq.heappush(self.low, -heapq.heappop(self.high))

    def median(self) -> float:
        if not self.low:
            return self.high[0] if self.high else math.nan
        if len(self.low) > len(self.high):
            return -self.low[0]
        if len(self.low) < len(self.high):
            return self.high[0]
        return (-self.low[0] + self.high[0]) / 2


class _WindowMedian:
    def __init__(self):
        self.low: list[tuple[float, int]] = []
        self.high: list[tuple[float, int]] = []
        self.side: dict[int, int] = {}
        self.low_count = 0
        self.high_count = 0

    def add(self, index: int, value: float) -> None:
        if self.low and value <= -self.low[0][0]:
            heapq.heappush(self.low, (-value, index))
            self.side[index] = 0
            self.low_count += 1
        else:
            heapq.heappush(self.high, (value, index))
            self.side[index] = 1
            self.high_count += 1
        self._rebalance()

    def remove(self, index: int) -> None:
        side = self.side.pop(index, None)
        if side is None:
            return
        if side == 0:
            self.low_count -= 1
        else:
            self.high_count -= 1
        self._rebalance()

    def _prune(self) -> None:
        while self.low and self.low[0][1] not in self.side:
            heapq.heappop(self.low)
        while self.high and self.high[0][1] not in self.side:
            heapq.heappop(self.high)

    def _rebalance(self) -> None:
        self._prune()
        while self.low_count > self.high_count + 1:
            value, index = heapq.heappop(self.low)
            self.low_count -= 1
            heapq.heappush(self.high, (-value, index))
            self.side[index] = 1
            self.high_count += 1
            self._prune()
        while self.low_count < self.high_count:
            value, index = heapq.heappop(self.high)
            self.high_count -= 1
            heapq.heappush(self.low, (-value, index))
            self.side[index] = 0
            self.low_count += 1
            self._prune()

    def median(self) -> float:
        self._prune()
        if not self.low:
            return self.high[0][0] if self.high else math.nan
        if self.low_count > self.high_count:
            return -self.low[0][0]
        if self.low_count < self.high_count:
            return self.high[0][0]
        return (-self.low[0][0] + self.high[0][0]) / 2

    def entries(self) -> int:
        return len(self.low) + len(self.high)

    def rebuild(self, points: "RetainedSeries", start: int, end: int) -> None:
        self.low.clear()
        self.high.clear()
        self.side.clear()
        self.low_count = 0
        self.high_count = 0
        for index in range(start, end):
            self.add(index, points[index][1])


class LossAnomalyTracker:
    def __init__(self):
        self.nan_step: Optional[float] = None
        self.spike_scanned = 0
        self.spike_worst: Optional[dict] = None
        self.prefix = _PrefixMedian()
        self.prefix_count = 0
        self.last = _WindowMedian()
        self.last_start = 0
        self.last_sum = 0.0
        self.last_sum_iy = 0.0
        self.last_sums_at = 0
        self.plateau = _WindowMedian()
        self.plateau_start = 0
        self.plateau_min: deque[tuple[int, float]] = deque()
        self.plateau_max: deque[tuple[int, float]] = deque()

    def advance(self, points: "RetainedSeries | list[tuple[float, float]]") -> None:
        if not isinstance(points, RetainedSeries):
            points = RetainedSeries(list(points), 0, len(points))
        total = len(points)
        if self.nan_step is not None:
            return

        third = min(total // 3, BASELINE_WINDOW)
        while self.prefix_count < third and self.prefix_count >= points.base:
            self.prefix.add(points[self.prefix_count][1])
            self.prefix_count += 1

        last_start = max(total - third, total - RECENT_WINDOW)
        plateau_length = min(total, max(10, min(total // 4, PLATEAU_WINDOW)))
        plateau_start = total - plateau_length

        for index in range(self.spike_scanned, total):
            value = points[index][1]
            if not math.isfinite(value):
                self.nan_step = points[index][0]
                return
            self.last.add(index, value)
            self.last_sum_iy += (index - self.last_start) * value
            self.last_sum += value
            self.plateau.add(index, value)
            while self.plateau_min and self.plateau_min[-1][1] >= value:
                self.plateau_min.pop()
            self.plateau_min.append((index, value))
            while self.plateau_max and self.plateau_max[-1][1] <= value:
                self.plateau_max.pop()
            self.plateau_max.append((index, value))

            if index >= 5:
                baseline = _median(
                    [previous for _step, previous in points[max(0, index - SPIKE_BASELINE) : index]]
                )
                if math.isfinite(baseline) and baseline > 0:
                    ratio = value / baseline
                    jump = value - baseline
                    if ratio >= 2.5 and jump >= 0.5:
                        if self.spike_worst is None or ratio > self.spike_worst["detail"]:
                            self.spike_worst = {
                                "kind": "spike",
                                "severity": "warning",
                                "step": points[index][0],
                                "value": value,
                                "detail": ratio,
                            }

        while self.last_start < last_start:
            value = points[self.last_start][1]
            self.last_sum_iy = self.last_sum_iy - self.last_sum + value
            self.last_sum -= value
            self.last.remove(self.last_start)
            self.last_start += 1
        while self.plateau_start < plateau_start:
            self.plateau.remove(self.plateau_start)
            self.plateau_start += 1
        while self.plateau_min and self.plateau_min[0][0] < self.plateau_start:
            self.plateau_min.popleft()
        while self.plateau_max and self.plateau_max[0][0] < self.plateau_start:
            self.plateau_max.popleft()

        if self.last.entries() > int(1.5 * (total - self.last_start)) + 512:
            self.last.rebuild(points, self.last_start, total)
            self._refresh_last_sums(points, total)
        elif total - self.last_sums_at >= 10000:
            self._refresh_last_sums(points, total)
        if self.plateau.entries() > int(1.25 * (total - self.plateau_start)) + 512:
            self.plateau.rebuild(points, self.plateau_start, total)
        self.spike_scanned = total

    def _refresh_last_sums(self, points: "RetainedSeries", total: int) -> None:
        self.last_sum = 0.0
        self.last_sum_iy = 0.0
        for offset, index in enumerate(range(self.last_start, total)):
            value = points[index][1]
            self.last_sum += value
            self.last_sum_iy += offset * value
        self.last_sums_at = total

    def detect(
        self, points: "RetainedSeries | list[tuple[float, float]]"
    ) -> Optional[dict]:
        if not isinstance(points, RetainedSeries):
            points = RetainedSeries(list(points), 0, len(points))
        total = len(points)
        if total < 8:
            return None
        if self.nan_step is not None:
            return {
                "kind": "nan",
                "severity": "critical",
                "step": self.nan_step,
                "value": None,
                "detail": 0.0,
            }

        if total >= 30:
            start_median = self.prefix.median()
            end_median = self.last.median()
            if (
                math.isfinite(start_median)
                and start_median > 0
                and end_median > start_median * 1.25
            ):
                count = total - self.last_start
                trend = self.last_sum_iy - (count - 1) / 2 * self.last_sum
                if trend > 0:
                    step, value = points[-1]
                    return {
                        "kind": "divergence",
                        "severity": "critical",
                        "step": step,
                        "value": value,
                        "detail": end_median / start_median,
                    }

        if total >= 12 and self.spike_worst is not None:
            return dict(self.spike_worst)

        if total >= 40:
            tail_median = self.plateau.median()
            tail_values_start = points[self.plateau_start][1]
            step, value = points[-1]
            spread = self.plateau_max[0][1] - self.plateau_min[0][1]
            improvement = (
                (tail_values_start - value) / tail_values_start if tail_values_start > 0 else 1.0
            )
            if (
                math.isfinite(tail_median)
                and tail_median > 0.1
                and spread / tail_median <= 0.03
                and improvement <= 0.01
            ):
                return {
                    "kind": "plateau",
                    "severity": "warning",
                    "step": step,
                    "value": value,
                    "detail": spread / tail_median,
                }

        return None


def detect_loss_anomaly(points: Iterable[tuple[float, float]]) -> Optional[dict]:
    series = list(points)
    if len(series) < 8:
        return None

    for step, value in series:
        if not math.isfinite(value):
            return {
                "kind": "nan",
                "severity": "critical",
                "step": step,
                "value": None,
                "detail": 0.0,
            }

    ordered = sorted(
        ((step, value) for step, value in series if math.isfinite(step)),
        key = lambda point: point[0],
    )
    if len(ordered) < 8:
        return None

    tracker = LossAnomalyTracker()
    retained = RetainedSeries(ordered, 0, len(ordered))
    tracker.advance(retained)
    return tracker.detect(retained)
