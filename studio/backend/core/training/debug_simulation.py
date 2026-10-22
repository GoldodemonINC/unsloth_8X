# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A simulated training run, for the Debugging setting.

A real run needs a GPU, a model download and a usable dataset. With this setting
on, ``/api/train/start`` skips all three and hands the backend a process-shaped
stand-in that reports the same events a training worker does, so the training
screens - start, the live chart, the metrics, stop, the history row - can be
exercised on a machine that cannot train at all.

Nothing here touches torch. It is a fake trainer: the numbers are invented, and
the only thing a run proves is that the surfaces around the trainer work.
"""

from __future__ import annotations

import math
import random
import threading
import time
from typing import Any, Callable, Optional

SETTING_KEY = "debug_simulate_training"
SIMULATED_MODEL_NAME = "debug/simulated-model"
SIMULATED_DATASET_NAME = "debug/simulated-dataset"
DEFAULT_TOTAL_STEPS = 60
DEFAULT_STEP_SECONDS = 0.12

_SIMULATED_PID = 2**31 - 1
_STOP_POLL_SECONDS = 0.05

def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False

def is_enabled() -> bool:
    """Whether simulated training is on. A settings read failure means off."""
    try:
        from storage.studio_db import get_app_setting
        from utils.account_context import OWNER, run_as

        stored = run_as(OWNER, get_app_setting, SETTING_KEY, None)
    except Exception:
        return False
    return _truthy(stored)

def set_enabled(enabled: bool) -> bool:
    from storage.studio_db import upsert_app_settings

    upsert_app_settings({SETTING_KEY: bool(enabled)})
    return bool(enabled)

def simulated_output_dir(job_id: str) -> str:
    """Where a simulated run puts its (fake) artifacts, so the UI has a path."""
    from utils.paths import outputs_root

    try:
        root = outputs_root()
    except Exception:
        root = None
    name = f"debug_simulated_{job_id}"
    return str((root / name) if root is not None else name)

def simulated_worker_config(job_id: str) -> dict[str, Any]:
    """The stand-in for ``_build_training_worker_config``.

    Only the keys the backend reads outside the worker matter: the model and
    dataset names go into the history row, ``output_dir`` into the run record.
    """
    return {
        "model_name": SIMULATED_MODEL_NAME,
        "training_type": "LoRA/QLoRA",
        "hf_dataset": SIMULATED_DATASET_NAME,
        "max_steps": DEFAULT_TOTAL_STEPS,
        "num_train_epochs": 1,
        "output_dir": simulated_output_dir(job_id),
        "hf_token": None,
        "debug_simulated": True,
    }

class SimulatedTrainingProcess:
    """A ``multiprocessing.Process`` stand-in that runs a fake trainer in a thread.

    The backend only needs ``start``, ``pid``, ``is_alive``, ``exitcode``,
    ``join``, ``terminate`` and ``kill``, plus the two queues it already passes a
    real worker, so every surface downstream (the pump, the watchdog, the DB
    finalize, stop, status, metrics and the SSE stream) runs unchanged.
    """

    def __init__(
        self,
        event_queue: Any,
        stop_queue: Any,
        job_id: str = "",
        total_steps: int = DEFAULT_TOTAL_STEPS,
        step_seconds: float = DEFAULT_STEP_SECONDS,
        output_dir: Optional[str] = None,
        on_event: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self.pid = _SIMULATED_PID
        self.exitcode: Optional[int] = None
        self._event_queue = event_queue
        self._stop_queue = stop_queue
        self._job_id = job_id
        self._total_steps = max(int(total_steps), 1)
        self._step_seconds = max(float(step_seconds), 0.0)
        self._output_dir = output_dir or simulated_output_dir(job_id)
        self._on_event = on_event
        self._terminate = threading.Event()
        self._started = False
        self._alive = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._alive = True
        self._thread = threading.Thread(
            target = self._run, name = "debug-simulated-trainer", daemon = True
        )
        self._thread.start()

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: Optional[float] = None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def terminate(self) -> None:
        """Ask the trainer to stop at the next step, like a worker's SIGTERM."""
        self._terminate.set()

    def kill(self) -> None:
        self._terminate.set()
        self._alive = False

    def _emit(self, event_type: str, **payload: Any) -> None:
        event = {"type": event_type, "ts": time.time(), **payload}
        try:
            self._event_queue.put(event)
        except Exception:

            self._alive = False
            return
        if self._on_event is not None:
            try:
                self._on_event(event)
            except Exception:
                pass

    def _stop_requested(self) -> bool:
        if self._terminate.is_set():
            return True
        try:
            message = self._stop_queue.get_nowait()
        except Exception:
            return False

        return bool(message) or message is not None

    def _run(self) -> None:
        try:
            self._write_artifacts()
            if not self._run_real_smoke():
                return

            self._emit("status", message = "Loading model (simulated)...")
            self._emit("model_load_started")
            time.sleep(min(self._step_seconds * 4, 0.6))
            self._emit("model_load_completed")
            self._emit(
                "status",
                message = "Simulated run: no weights are loaded and no GPU is used.",
            )
            self._emit("output_dir", output_dir = self._output_dir)
            self._emit("eval_configured")

            base_lr = 2e-4
            eval_every = max(self._total_steps // 4, 1)
            started = time.monotonic()

            for step in range(1, self._total_steps + 1):
                if self._stop_requested():
                    self._emit(
                        "complete",
                        status_message = "Training cancelled.",
                        output_dir = self._output_dir,
                    )
                    return

                progress = step / self._total_steps

                loss = 0.34 + 2.05 * math.exp(-3.1 * progress) + random.uniform(-0.05, 0.05)
                learning_rate = base_lr * max(1.0 - progress, 0.0)
                elapsed = time.monotonic() - started
                per_step = elapsed / step
                self._emit(
                    "progress",
                    step = step,
                    total_steps = self._total_steps,
                    loss = round(max(loss, 0.05), 4),
                    learning_rate = learning_rate,
                    grad_norm = round(abs(random.gauss(0.9, 0.25)), 4),
                    epoch = round(progress, 4),
                    elapsed_seconds = round(elapsed, 3),
                    eta_seconds = round(per_step * (self._total_steps - step), 3),
                    num_tokens = step * 2048,
                    eval_loss = (
                        round(max(loss - 0.08, 0.05), 4)
                        if step % eval_every == 0
                        else None
                    ),
                )
                if self._step_seconds:
                    time.sleep(self._step_seconds)

            self._emit(
                "complete",
                status_message = "Training completed! Saved the simulated adapter.",
                output_dir = self._output_dir,
            )
        except Exception:
            self._emit(
                "error",
                error = "The simulated run failed unexpectedly.",
                stack = "",
            )
        finally:
            self.exitcode = 0
            self._alive = False

    def _run_real_smoke(self) -> bool:
        """Exercise the shipped merge engine before faking a training curve.

        The simulator used to prove only that the UI around a trainer works. This
        runs the real merge over generated checkpoints, so a broken blend, index,
        manifest or policy path fails the debug run instead of hiding behind
        invented metrics.
        """
        from pathlib import Path

        try:
            from core.training import debug_fixture
        except ImportError:
            self._emit("status", message = "Debug smoke skipped: no fixture runner.")
            return True
        self._emit(
            "status", message = "Debug smoke: merging three real tiny checkpoints..."
        )
        try:
            smoke = debug_fixture.run_merge_smoke(Path(self._output_dir) / "debug-merge-smoke")
        except ImportError:
            self._emit(
                "status",
                message = "Debug smoke skipped: safetensors is not installed.",
            )
            return True
        except Exception as error:  # noqa: BLE001 - the point is to surface it
            import traceback

            self._emit(
                "error",
                error = f"Debug merge smoke failed: {error}",
                stack = traceback.format_exc(),
            )
            return False
        self._emit(
            "debug_smoke",
            ok = smoke["ok"],
            checks = smoke["checks"],
            workdir = smoke["workdir"],
        )
        if not smoke["ok"]:
            self._emit(
                "error",
                error = "Debug merge smoke failed for: " + ", ".join(smoke["failed"]),
                stack = "",
            )
            return False
        self._emit(
            "status",
            message = "Debug smoke passed: "
            + ", ".join(check["method"] for check in smoke["checks"])
            + " merges produced the expected weights.",
        )
        return True

    def _write_artifacts(self) -> None:
        """Create the output dir with a marker, so 'open folder' has something."""
        try:
            from pathlib import Path

            path = Path(self._output_dir)
            path.mkdir(parents = True, exist_ok = True)
            (path / "SIMULATED_RUN.txt").write_text(
                "This directory belongs to a simulated training run started from\n"
                "Settings -> Debugging. No model was loaded and no weights were\n"
                "trained; the metrics in Studio are generated, not measured.\n",
                encoding = "utf-8",
            )
        except Exception:
            pass
