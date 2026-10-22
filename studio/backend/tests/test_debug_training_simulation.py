# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Simulated training runs, the Debugging setting that turns them on.

A simulated run is a fake trainer behind the real backend: it must emit the same
event vocabulary a worker does, so the pump, the SSE stream, the DB row and stop
behave as usual. These pin the event sequence, the stopping paths, the setting's
coercion, and the route that starts one without touching a model or a GPU.
"""

from __future__ import annotations

import asyncio
import math
import os
import queue
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from core.training import debug_simulation as sim  # noqa: E402


def _import_or_skip(name):
    try:
        return __import__(name, fromlist = ["_"])
    except Exception as exc:  # missing optional deps, or an import that needs a GPU stack
        pytest.skip(f"{name} is not importable here: {exc}")


class _FakeStore:
    def __init__(self):
        self.values: dict = {}


@pytest.fixture
def store(monkeypatch):
    """An in-memory stand-in for the app-settings table."""
    studio_db = _import_or_skip("storage.studio_db")
    fake = _FakeStore()
    monkeypatch.setattr(
        studio_db,
        "get_app_setting",
        lambda key, default = None: fake.values.get(key, default),
    )
    monkeypatch.setattr(
        studio_db, "upsert_app_settings", lambda mapping: fake.values.update(mapping)
    )
    return fake


def _drain(events):
    out = []
    while True:
        try:
            out.append(events.get_nowait())
        except queue.Empty:
            return out


def _run_to_completion(total_steps = 8, output_dir = None):
    events: queue.Queue = queue.Queue()
    stop: queue.Queue = queue.Queue()
    proc = sim.SimulatedTrainingProcess(
        events,
        stop,
        job_id = "job_debug",
        total_steps = total_steps,
        step_seconds = 0.0,
        output_dir = output_dir,
    )
    proc.start()
    proc.join(timeout = 10)
    assert not proc.is_alive()
    return proc, _drain(events)


def test_the_setting_is_off_by_default_and_round_trips(store):
    assert sim.is_enabled() is False
    assert sim.set_enabled(True) is True
    assert sim.is_enabled() is True
    assert sim.set_enabled(False) is False
    assert sim.is_enabled() is False


@pytest.mark.parametrize("stored", [None, 0, 1, "", "0", "no", "garbage", [], {}])
def test_only_an_explicit_on_reads_as_on(store, stored):
    store.values[sim.SETTING_KEY] = stored
    expected = stored == 1
    assert sim.is_enabled() is expected


def test_a_settings_read_failure_reads_as_off(monkeypatch):
    studio_db = _import_or_skip("storage.studio_db")

    def boom(*_args, **_kwargs):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(studio_db, "get_app_setting", boom)
    assert sim.is_enabled() is False


def test_a_simulated_run_emits_the_worker_event_sequence(tmp_path):
    proc, events = _run_to_completion(output_dir = str(tmp_path / "run"))
    types = [event["type"] for event in events]

    assert types[0] == "status"
    assert "model_load_started" in types
    assert "model_load_completed" in types
    assert types.count("output_dir") == 1
    assert "eval_configured" in types
    assert types[-1] == "complete"

    steps = [event for event in events if event["type"] == "progress"]
    assert [event["step"] for event in steps] == list(range(1, 9))
    assert all(event["total_steps"] == 8 for event in steps)
    assert all(math.isfinite(event["loss"]) for event in steps)
    assert all(math.isfinite(event["learning_rate"]) for event in steps)
    # A chart that does not descend reads as noise; the fake loss must trend down.
    assert steps[0]["loss"] > steps[-1]["loss"]

    evals = [event["step"] for event in steps if event.get("eval_loss") is not None]
    assert evals == [2, 4, 6, 8]

    complete = events[-1]
    assert complete["status_message"].startswith("Training completed")
    assert complete["output_dir"] == proc._output_dir
    assert Path(proc._output_dir, "SIMULATED_RUN.txt").is_file()
    assert proc.exitcode == 0


def test_stopping_a_simulated_run_reports_a_cancel_not_a_failure():
    events: queue.Queue = queue.Queue()
    stop: queue.Queue = queue.Queue()
    proc = sim.SimulatedTrainingProcess(
        events,
        stop,
        job_id = "job_stop",
        total_steps = 500,
        step_seconds = 0.01,
    )
    proc.start()
    # Let the fake trainer take a few steps before the stop lands, so this is a
    # cancel in the middle of a run rather than one before the first event.
    time.sleep(0.1)
    stop.put({"type": "stop"})
    proc.join(timeout = 10)

    assert not proc.is_alive()
    tail = _drain(events)
    assert tail, "the stop must still produce a terminal event"
    assert tail[-1]["type"] == "complete"
    assert tail[-1]["status_message"] == "Training cancelled."


def test_terminate_endswith_a_terminal_event_and_no_leaked_thread():
    events: queue.Queue = queue.Queue()
    stop: queue.Queue = queue.Queue()
    proc = sim.SimulatedTrainingProcess(
        events,
        stop,
        job_id = "job_term",
        total_steps = 500,
        step_seconds = 0.01,
    )
    proc.start()
    proc.terminate()
    proc.join(timeout = 10)
    assert not proc.is_alive()
    assert _drain(events)[-1]["type"] == "complete"


def test_the_fake_process_looks_like_one_without_being_one():
    proc = sim.SimulatedTrainingProcess(queue.Queue(), queue.Queue(), job_id = "job_pid")
    # Never os.getpid(): force_terminate() walks the pid's children.
    assert proc.pid != os.getpid()
    assert proc.pid == sim._SIMULATED_PID
    assert proc.is_alive() is False
    proc.start()
    assert proc.is_alive() is True
    proc.kill()
    proc.join(timeout = 5)


def test_the_route_starts_a_simulated_run_without_the_real_kwargs(store, monkeypatch):
    routes_training = _import_or_skip("routes.training")

    class _Backend:
        current_job_id = ""

        def __init__(self):
            self.calls = []
            self.trainer = SimpleNamespace(
                training_progress = SimpleNamespace(error = None)
            )

        def is_training_active(self):
            return False

        def start_training(self, **kwargs):
            self.calls.append(kwargs)
            self.current_job_id = kwargs["job_id"]
            return True

    backend = _Backend()
    monkeypatch.setattr(routes_training, "get_training_backend", lambda: backend)

    response = asyncio.run(routes_training._start_simulated_training())

    assert response.status == "queued"
    assert response.job_id.startswith("job_")
    assert backend.calls == [
        {"job_id": response.job_id, "debug_simulated": True}
    ]


def test_the_route_refuses_to_stack_a_second_run(store, monkeypatch):
    routes_training = _import_or_skip("routes.training")

    class _Busy:
        current_job_id = "job_live"

        def is_training_active(self):
            return True

    monkeypatch.setattr(routes_training, "get_training_backend", lambda: _Busy())

    response = asyncio.run(routes_training._start_simulated_training())

    assert response.status == "error"
    assert response.job_id == "job_live"
    assert response.error == "Training already active"


def test_the_settings_endpoints_read_and_write_the_toggle(store):
    routes_settings = _import_or_skip("routes.settings")

    initial = routes_settings.get_debug_training_simulation()
    assert initial.enabled is False
    assert initial.default_enabled is False

    payload = routes_settings.DebugTrainingSimulationPayload(enabled = True)
    saved = routes_settings.update_debug_training_simulation(payload)
    assert saved.enabled is True
    assert routes_settings.get_debug_training_simulation().enabled is True
