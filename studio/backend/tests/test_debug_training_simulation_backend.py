# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The backend half of a simulated run: start, stream, stop, finalize.

``debug_simulation`` only fakes the trainer. This drives the real
``TrainingBackend`` so the spawn short-circuit, the event pump, the DB row and
the terminal finalize are the same code a real run uses.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))


def _import_or_skip(name):
    try:
        module = __import__(name, fromlist = ["_"])
    except Exception as exc:
        pytest.skip(f"{name} is not importable here: {exc}")
    return module


def _wait_for(predicate, timeout = 30.0, interval = 0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


@pytest.fixture
def backend():
    training = _import_or_skip("core.training.training")
    instance = training.TrainingBackend()
    yield instance
    try:
        instance.force_terminate()
    except Exception:
        pass


def test_a_simulated_run_goes_start_to_history_through_the_real_backend(backend):
    job_id = "job_sim_test"
    assert backend.start_training(job_id = job_id, debug_simulated = True) is True
    assert backend.current_job_id == job_id

    assert _wait_for(lambda: len(backend.step_history) >= 3, timeout = 30), (
        backend.trainer.training_progress
    )
    progress = backend.trainer.training_progress
    assert progress.is_training is True
    assert progress.total_steps == 60

    assert _wait_for(lambda: not backend.is_training_active(), timeout = 60)
    assert backend.step_history, "no metrics were recorded"
    assert backend.step_history[-1] >= len(backend.step_history)
    assert backend.loss_history and all(
        isinstance(value, float) for value in backend.loss_history
    )
    assert backend.eval_loss_history, "the run never reported an eval loss"
    progress = backend.trainer.training_progress
    assert progress.is_completed is True
    assert progress.status_message.startswith("Training completed")

    # The point of the fake run is the History row, so it has to be written and closed.
    # The pump finalizes it as it drains the terminal event, a moment after the
    # process exits, so wait for the row rather than for the run to go idle.
    from storage.studio_db import get_run

    def _run_is_finished():
        run = get_run(job_id)
        return run is not None and run["status"] != "running"

    assert _wait_for(_run_is_finished, timeout = 30), get_run(job_id)
    run = get_run(job_id)
    assert run is not None, "the simulated run never reached the history table"
    assert run["status"] == "completed", run["status"]
    assert run["model_name"] == "debug/simulated-model"
    assert run["final_step"] == 60


def test_stopping_a_simulated_run_ends_it_without_an_error(backend):
    """The Stop button has to land somewhere sane, not on a missing checkpoint."""
    assert backend.start_training(job_id = "job_sim_stop", debug_simulated = True) is True
    assert _wait_for(lambda: len(backend.step_history) >= 2, timeout = 30)

    backend.stop_training(expected_job_id = "job_sim_stop")

    assert _wait_for(lambda: not backend.is_training_active(), timeout = 60)
    progress = backend.trainer.training_progress
    assert progress.is_training is False
    assert progress.is_completed is False
    assert not progress.error, progress.error
    assert "stop" in progress.status_message.lower() or "cancel" in progress.status_message.lower()
