# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Blend schedule determinism (F4 / A4).

Moving the blend slider used to change the batch stream depending on *when* it
moved, so a resumed run diverged from the run it resumed. A move is now an event
keyed by global step; the schedule is the run's history and replaying it
reproduces the stream exactly.
"""

from __future__ import annotations

import pytest

from core.advanced import blending


def _plan(batch_size: int = 4) -> blending.BlendPlan:
    return blending.validate_blend_slots(
        [
            {"name": "reasoning", "ratio": 60},
            {"name": "code", "ratio": 40},
        ],
        batch_size = batch_size,
    )


def _sampler(plan: blending.BlendPlan, schedule: blending.BlendSchedule | None = None) -> blending.BlendedSampler:
    return blending.BlendedSampler(
        plan = plan,
        streams = {"reasoning": list(range(64)), "code": list(range(64))},
        seed = 3407,
        schedule = schedule,
    )


def test_ratios_at_replays_the_event_log():
    schedule = blending.BlendSchedule(base = _plan())
    schedule.set_ratio("reasoning", 0.25, step = 20)
    schedule.set_ratio("reasoning", 0.9, step = 60)

    assert schedule.ratios_at(0)["reasoning"] == pytest.approx(0.6)
    assert schedule.ratios_at(19)["reasoning"] == pytest.approx(0.6)
    assert schedule.ratios_at(20)["reasoning"] == pytest.approx(0.25)
    assert schedule.ratios_at(59)["reasoning"] == pytest.approx(0.25)
    assert schedule.ratios_at(60)["reasoning"] == pytest.approx(0.9)
    # The other stream takes the remainder, and the ratios sum to one.
    assert schedule.ratios_at(20)["code"] == pytest.approx(0.75)
    assert sum(schedule.ratios_at(60).values()) == pytest.approx(1.0)


def test_slider_move_before_a_checkpoint_still_resumes_identically():
    """The acceptance test: move, checkpoint, resume -> same batch stream."""
    schedule = blending.BlendSchedule(base = _plan())
    schedule.set_ratio("reasoning", 0.1, step = 40)

    uninterrupted = _sampler(_plan(), schedule)
    expected = [uninterrupted.next_batch(step) for step in range(120)]

    # Interrupted at step 60: resume from the checkpoint snapshot.
    interrupted = _sampler(_plan(), schedule)
    before = [interrupted.next_batch(step) for step in range(60)]
    snapshot = schedule.snapshot(60)
    state = interrupted.snapshot()

    resumed_schedule = blending.BlendSchedule(base = _plan())
    resumed_schedule.set_ratio("reasoning", 0.1, step = 40)
    resumed = _sampler(_plan(), resumed_schedule)
    resumed.resume(state)
    after = [resumed.next_batch(step) for step in range(60, 120)]

    assert before == expected[:60]
    assert after == expected[60:]
    assert snapshot["ratios"] == resumed_schedule.ratios_at(60)


def test_resume_without_the_event_log_would_diverge():
    """Guards the test above: the event log is what makes resume exact."""
    schedule = blending.BlendSchedule(base = _plan())
    schedule.set_ratio("reasoning", 0.1, step = 40)
    sampler = _sampler(_plan(), schedule)
    expected = [sampler.next_batch(step) for step in range(80)]

    naive = _sampler(_plan())
    naive.resume(_sampler(_plan(), schedule).snapshot())
    naive_stream = [naive.next_batch(step) for step in range(80)]
    # Without the schedule, steps after the move use the base ratio and differ.
    assert naive_stream != expected


def test_snapshot_separates_applied_and_pending_events():
    schedule = blending.BlendSchedule(base = _plan())
    schedule.set_ratio("reasoning", 0.3, step = 10)
    schedule.set_ratio("reasoning", 0.7, step = 30)
    snapshot = schedule.snapshot(20)
    assert [event["step"] for event in snapshot["applied_events"]] == [10]
    assert [event["step"] for event in snapshot["pending_events"]] == [30]
    assert snapshot["ratios"]["reasoning"] == pytest.approx(0.3)


def test_schedule_round_trips_through_the_manifest_block():
    schedule = blending.BlendSchedule(base = _plan(), batch_size = 4)
    schedule.set_ratio("code", 0.75, step = 12)
    block = schedule.manifest_block()
    restored = blending.BlendSchedule.from_manifest(block)
    assert restored is not None
    assert restored.ratios_at(11) == schedule.ratios_at(11)
    assert restored.ratios_at(12)["code"] == pytest.approx(0.75)
    assert restored.events_document() == schedule.events_document()


def test_unknown_stream_and_negative_ratio_are_rejected():
    schedule = blending.BlendSchedule(base = _plan())
    with pytest.raises(ValueError):
        schedule.set_ratios({"not-a-stream": 1}, step = 0)
    with pytest.raises(ValueError):
        schedule.set_ratios({"reasoning": -1}, step = 0)
    with pytest.raises(ValueError):
        schedule.set_ratios({"reasoning": 1}, step = -1)


def test_a_move_at_the_same_step_supersedes_the_earlier_one():
    schedule = blending.BlendSchedule(base = _plan())
    schedule.set_ratio("code", 0.1, step = 5)
    schedule.set_ratio("code", 0.9, step = 5)
    assert len(schedule.events) == 1
    assert schedule.ratios_at(5)["code"] == pytest.approx(0.9)
