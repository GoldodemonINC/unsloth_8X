# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Dynamic dataset blending and ratio scheduling.

Studio used to accept one dataset (or an S3 prefix) per run. The blender lets a
run queue several streams -- reasoning, code, chat, domain text -- each with its
own percentage, and interleaves them *during* training. Nothing is pre-merged
to disk, so changing a slider changes the very next batch.

Interleaving uses smooth weighted round-robin over the per-batch integer
counts obtained from the largest-remainder method:

    ratios 60/40, batch 10  ->  [A,A,B,A,B,A,A,B,A,B]  (not AAAAAABBBB)

which keeps every optimizer step close to the requested mixture, including the
first one.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator, Mapping, Sequence


@dataclass(frozen=True)
class BlendSlot:
    name: str
    ratio: float
    dataset: str | None = None
    rows: int | None = None
    subset: str | None = None
    split: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BlendPlan:
    slots: tuple[BlendSlot, ...]
    ratios: dict[str, float]
    batch_size: int
    notes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "slots": [slot.to_dict() for slot in self.slots],
            "ratios": dict(self.ratios),
            "batch_size": self.batch_size,
            "notes": list(self.notes),
            "warnings": list(self.warnings),
        }

    def batch_layout(self) -> list[str]:
        """Dataset name for each sample slot in one batch."""
        return smooth_round_robin(
            [slot.name for slot in self.slots],
            [self.ratios[slot.name] for slot in self.slots],
            self.batch_size,
        )


def validate_blend_slots(
    slots: Sequence[Mapping[str, Any]] | None,
    *,
    batch_size: int = 1,
) -> BlendPlan:
    if batch_size <= 0:
        raise ValueError("Blend batch_size must be positive.")
    if not slots:
        raise ValueError("The dataset blender needs at least one slot.")
    if len(slots) > 16:
        raise ValueError("The dataset blender accepts at most 16 streams.")

    parsed: list[BlendSlot] = []
    seen: set[str] = set()
    for index, raw in enumerate(slots):
        if isinstance(raw, BlendSlot):
            name, ratio = raw.name, raw.ratio
        elif isinstance(raw, Mapping):
            name = str(raw.get("name") or f"dataset{index + 1}").strip()
            ratio = float(raw.get("ratio", 0.0))
        else:
            raise ValueError("Dataset slots must be objects.")
        if not name:
            raise ValueError("Every dataset slot needs a name.")
        if name in seen:
            raise ValueError(f"Duplicate dataset slot name {name!r}.")
        seen.add(name)
        if not math.isfinite(ratio) or ratio < 0:
            raise ValueError(f"Ratio for {name!r} must be a non-negative number.")
        parsed.append(
            BlendSlot(
                name=name,
                ratio=ratio,
                dataset=(raw.get("dataset") if isinstance(raw, Mapping) else None) or None,
                rows=(int(raw["rows"]) if isinstance(raw, Mapping) and raw.get("rows") is not None else None),
                subset=(raw.get("subset") if isinstance(raw, Mapping) else None) or None,
                split=(raw.get("split") if isinstance(raw, Mapping) else None) or None,
            )
        )

    total = sum(slot.ratio for slot in parsed)
    if total <= 0:
        raise ValueError("At least one dataset slot needs a positive ratio.")
    ratios = {slot.name: slot.ratio / total for slot in parsed}

    warnings: list[str] = []
    positive = [slot for slot in parsed if ratios[slot.name] > 0]
    if len(positive) == 1:
        warnings.append("Only one stream has weight; the blender is equivalent to a single dataset.")
    if any(ratio > 0 and ratio < 0.01 for ratio in ratios.values()):
        warnings.append("A stream is below 1%; with small batches it may appear only occasionally.")
    for slot in positive:
        if slot.rows is not None and slot.rows < batch_size:
            warnings.append(
                f"Stream {slot.name!r} has fewer rows ({slot.rows}) than one batch ({batch_size})."
            )
    notes = [f"Weights normalized from {total:g} to 1.0."]
    return BlendPlan(
        slots=tuple(parsed),
        ratios=ratios,
        batch_size=batch_size,
        notes=tuple(notes),
        warnings=tuple(warnings),
    )


def _largest_remainder(ratios: Sequence[float], total: int) -> list[int]:
    """Allocate `total` integer slots proportionally to `ratios`."""
    raw = [ratio * total for ratio in ratios]
    base = [int(math.floor(value)) for value in raw]
    remainder = total - sum(base)
    order = sorted(
        range(len(raw)), key=lambda index: (raw[index] - base[index]), reverse=True
    )
    for index in order[:remainder]:
        base[index] += 1
    return base


def smooth_round_robin(names: Sequence[str], ratios: Sequence[float], total: int) -> list[str]:
    """Weighted, evenly spread sequence of length `total`.

    Smooth WRR tracks a running credit per stream, so the output order is
    deterministic and no stream is starved for a whole batch.
    """
    if len(names) != len(ratios):
        raise ValueError("names and ratios must have equal length.")
    if total <= 0:
        return []
    if not names:
        raise ValueError("At least one stream is required.")
    positive = [(name, max(0.0, float(ratio))) for name, ratio in zip(names, ratios)]
    weight_sum = sum(weight for _, weight in positive)
    if weight_sum <= 0:
        raise ValueError("Ratios must sum to a positive number.")
    credits = {name: 0.0 for name, _ in positive}
    sequence: list[str] = []
    for _ in range(total):
        for name, weight in positive:
            credits[name] += weight / weight_sum
        chosen = max(positive, key=lambda item: credits[item[0]])[0]
        credits[chosen] -= 1.0
        sequence.append(chosen)
    return sequence


@dataclass
class BlendedSampler:
    """Iterate batches over several streams at once.

    Each stream's rows are shuffled once and recycled when exhausted, so a run
    with ``max_steps`` larger than a stream still makes progress. The plan's
    ratio is re-evaluated for every batch, which is what makes a live slider
    change take effect immediately.
    """

    plan: BlendPlan
    streams: Mapping[str, Sequence[int]]
    seed: int = 3407
    schedule: BlendSchedule | None = None
    _cursors: dict[str, int] = field(default_factory = dict, init=False)
    _orders: dict[str, list[int]] = field(default_factory = dict, init=False)
    _passes: dict[str, int] = field(default_factory = dict, init=False)

    def __post_init__(self) -> None:
        import random

        for slot in self.plan.slots:
            if self.plan.ratios[slot.name] <= 0:
                continue
            rows = self.streams.get(slot.name)
            if rows is None:
                raise ValueError(f"No rows registered for dataset stream {slot.name!r}.")
            order = [int(row) for row in rows]
            if not order:
                raise ValueError(f"Dataset stream {slot.name!r} is empty.")
            random.Random(self.seed).shuffle(order)
            self._orders[slot.name] = order
            self._cursors[slot.name] = 0
            self._passes[slot.name] = 0

    def _next_row(self, name: str) -> int:
        import random

        order = self._orders[name]
        cursor = self._cursors[name]
        if cursor >= len(order):
            random.Random(self.seed + self._passes[name] + 1).shuffle(order)
            cursor = 0
            self._passes[name] += 1
        self._cursors[name] = cursor + 1
        return order[cursor]

    def next_batch(self, step: int | None = None) -> list[int]:
        """One batch. With a schedule and a step, the layout follows the event log."""
        layout = self.layout_at(step) if step is not None else self.plan.batch_layout()
        return [self._next_row(name) for name in layout]

    def layout_at(self, step: int | None) -> list[str]:
        if self.schedule is None or step is None:
            return self.plan.batch_layout()
        return self.schedule.layout_at(step)

    def __iter__(self) -> Iterator[list[int]]:
        while True:
            yield self.next_batch()

    def passes(self) -> dict[str, int]:
        return dict(self._passes)

    def resume(self, state: Mapping[str, Any]) -> None:
        """Restore cursors and pass counts from :meth:`snapshot`.

        The orders are rebuilt rather than stored: they are a pure function of
        ``(seed, stream length, pass count)``, so replaying the shuffles gives
        back the exact sequence the interrupted run would have produced. This is
        what makes a blend-slider move before a checkpoint resume-safe.
        """
        import random

        cursors = state.get("cursors") or {}
        passes = state.get("passes") or {}
        for name in self._orders:
            rows = [int(row) for row in self.streams[name]]
            rounds = int(passes.get(name, 0))
            order = list(rows)
            random.Random(self.seed).shuffle(order)
            for index in range(rounds):
                random.Random(self.seed + index + 1).shuffle(order)
            self._orders[name] = order
            self._passes[name] = rounds
            self._cursors[name] = int(cursors.get(name, 0))
            if self._cursors[name] > len(order):
                self._cursors[name] = 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "cursors": dict(self._cursors),
            "passes": dict(self._passes),
        }


@dataclass(frozen=True)
class BlendEvent:
    """One ratio change, pinned to a global optimizer step.

    The slider is live, so a run's batch stream depended on *when* it moved.
    Recording the step makes the stream a pure function of
    ``(seed, events, step)``: replaying the events reproduces the stream without
    the UI being involved at all.
    """

    step: int
    ratios: dict[str, float]
    reason: str = "ratio change"
    recorded_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": int(self.step),
            "ratios": {name: float(value) for name, value in self.ratios.items()},
            "reason": self.reason,
            "recorded_at": float(self.recorded_at),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BlendEvent":
        return cls(
            step = int(payload.get("step", 0)),
            ratios = {
                str(name): float(value)
                for name, value in (payload.get("ratios") or {}).items()
            },
            reason = str(payload.get("reason") or "ratio change"),
            recorded_at = float(payload.get("recorded_at") or 0.0),
        )


def _normalise(ratios: Mapping[str, float]) -> dict[str, float]:
    cleaned = {
        str(name): float(value)
        for name, value in ratios.items()
    }
    if any(not math.isfinite(value) or value < 0 for value in cleaned.values()):
        raise ValueError("Blend ratios must be non-negative finite numbers.")
    total = sum(cleaned.values())
    if total <= 0:
        raise ValueError("At least one blend ratio must be positive.")
    return {name: value / total for name, value in cleaned.items()}


@dataclass
class BlendSchedule:
    """The blend's history, and the ratio that applies at any step.

    ``base`` is the ratio the run started with. Every change is an event keyed by
    global step; ``ratios_at(step)`` replays them in order, which is how a resumed
    run reproduces the stream exactly (and what the acceptance test asserts).
    """

    base: BlendPlan
    events: list[BlendEvent] = field(default_factory = list)
    batch_size: int = 1

    def __post_init__(self) -> None:
        self.events.sort(key = lambda event: event.step)

    # -- recording ----------------------------------------------------------

    def set_ratios(
        self,
        ratios: Mapping[str, float],
        *,
        step: int,
        reason: str = "ratio change",
    ) -> BlendEvent:
        """Record new *weights* at *step*; streams not named keep their base weight.

        A move at the same step replaces the earlier one, because the run only
        consumed batches at that step once. Use :meth:`set_ratio` for a single
        stream's share.
        """
        if step < 0:
            raise ValueError("Blend events are keyed by a non-negative global step.")
        unknown = [name for name in ratios if name not in self.base.ratios]
        if unknown:
            raise ValueError(
                f"Unknown dataset stream(s): {', '.join(unknown)}. "
                f"Known streams: {', '.join(self.base.ratios)}."
            )
        merged = dict(self.base.ratios)
        merged.update({str(name): float(value) for name, value in ratios.items()})
        event = BlendEvent(
            step = int(step),
            ratios = _normalise(merged),
            reason = reason,
            recorded_at = time.time(),
        )
        # A move at the same step supersedes the earlier one: the run only ever
        # consumed batches at that step once.
        self.events = [item for item in self.events if item.step != event.step]
        self.events.append(event)
        self.events.sort(key = lambda item: item.step)
        return event

    def set_ratio(self, name: str, value: float, *, step: int) -> BlendEvent:
        """Set one stream's *share* of the batch; the others rebalance.

        This is what the UI slider calls: "reasoning is 25%" means reasoning
        gets 25% and the remaining streams split the other 75% in proportion to
        the shares they had, not "reasoning's raw weight is 25".
        """
        if name not in self.base.ratios:
            raise ValueError(
                f"Unknown dataset stream {name!r}. Known streams: "
                f"{', '.join(self.base.ratios)}."
            )
        share = float(value)
        if not math.isfinite(share) or share < 0 or share > 1:
            raise ValueError("A stream's share must be between 0 and 1.")
        current = self.ratios_at(step)
        others = {key: ratio for key, ratio in current.items() if key != name}
        remaining = 1.0 - share
        total_others = sum(others.values())
        ratios = {name: share}
        for key, ratio in others.items():
            ratios[key] = (
                remaining * (ratio / total_others)
                if total_others > 0
                else remaining / max(1, len(others))
            )
        return self.set_ratios(ratios, step = step, reason = f"{name} -> {share:.0%}")

    # -- reading -----------------------------------------------------------

    def ratios_at(self, step: int) -> dict[str, float]:
        current = dict(self.base.ratios)
        for event in self.events:
            if event.step > step:
                break
            current = dict(event.ratios)
        return current

    def plan_at(self, step: int) -> BlendPlan:
        ratios = self.ratios_at(step)
        slots = tuple(
            BlendSlot(
                name = slot.name,
                ratio = ratios[slot.name],
                dataset = slot.dataset,
                rows = slot.rows,
                subset = slot.subset,
                split = slot.split,
            )
            for slot in self.base.slots
        )
        return BlendPlan(
            slots = slots,
            ratios = ratios,
            batch_size = self.batch_size or self.base.batch_size,
            notes = self.base.notes,
            warnings = self.base.warnings,
        )

    def layout_at(self, step: int) -> list[str]:
        return self.plan_at(step).batch_layout()

    def events_document(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self.events]

    def manifest_block(self) -> dict[str, Any]:
        """What the run manifest stores: base plan plus the event log."""
        return {
            "batch_size": self.batch_size or self.base.batch_size,
            "base_ratios": dict(self.base.ratios),
            "slots": [slot.to_dict() for slot in self.base.slots],
            "events": self.events_document(),
            "event_count": len(self.events),
        }

    @classmethod
    def from_manifest(
        cls, payload: Mapping[str, Any] | None, *, batch_size: int | None = None
    ) -> "BlendSchedule | None":
        """Rebuild a schedule from a manifest block, or None when there is none."""
        if not payload:
            return None
        slots = payload.get("slots") or []
        if not slots:
            return None
        plan = validate_blend_slots(
            slots, batch_size = int(batch_size or payload.get("batch_size") or 1)
        )
        schedule = cls(
            base = plan,
            events = [BlendEvent.from_dict(item) for item in payload.get("events") or []],
            batch_size = int(payload.get("batch_size") or plan.batch_size),
        )
        return schedule

    # -- checkpointing ------------------------------------------------------

    def snapshot(self, step: int, *, checkpoint: str | None = None) -> dict[str, Any]:
        """The record written at a checkpoint boundary.

        It is self-contained: a resume from it needs no in-memory history, only
        the events up to *step* and the step itself.
        """
        applied = [event for event in self.events if event.step <= step]
        pending = [event for event in self.events if event.step > step]
        return {
            "step": int(step),
            "checkpoint": checkpoint,
            "ratios": self.ratios_at(step),
            "applied_events": [event.to_dict() for event in applied],
            "pending_events": [event.to_dict() for event in pending],
            "recorded_at": time.time(),
        }


def scale_ratios(ratios: Mapping[str, float], factor: float) -> dict[str, float]:
    """Scale ratios for a mid-run schedule change, re-normalized to 1.0."""
    if factor <= 0:
        raise ValueError("Ratio scale factor must be positive.")
    scaled = {name: max(0.0, float(value) * factor) for name, value in ratios.items()}
    total = sum(scaled.values())
    if total <= 0:
        raise ValueError("Scaled ratios sum to zero.")
    return {name: value / total for name, value in scaled.items()}
