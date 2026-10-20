# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Resource arbitration between merge, split, MoE, eval and training.

Training, merging, splitting and evaluation all want the same GPU, and any two
of them can fit individually while crashing together. Before this module, each
job started and found out.

The broker is deliberately small: every job declares what it needs (VRAM, RAM,
disk), asks for a lease, and either gets one, waits, or is refused with a reason
naming the short resource and the number that is short. Nothing is prefetched,
nothing is evicted, no job is silently throttled.

Two rules keep the reasons honest:

* A request that could never fit on an idle machine is **denied** ("this needs
  12 GB VRAM, the card has 8 GB"), not queued.
* CPU-only work is modelled in **RAM and disk**. A merge on a CPU host is
  streaming, so its peak is one tensor pair, not the whole model -- the broker
  estimates it that way rather than inventing a CUDA error.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

GB = 1024**3

# How much of the machine a job may claim in total, leaving room for the engine,
# the UI, and the OS staying responsive. 0.85 of VRAM, 0.75 of RAM, 0.90 of disk.
VRAM_HEADROOM = 0.85
RAM_HEADROOM = 0.75
DISK_HEADROOM = 0.90

# Per-stage cost model. ``vram``/``ram`` are multipliers on the checkpoint size
# (or on the largest single tensor, for streaming paths); ``disk`` is the space
# the output needs.
_COST_MODEL: dict[str, dict[str, Any]] = {
    # Streaming merge: one tensor pair resident, one output copy on disk.
    "merge": {"vram": 0.0, "ram_stream": 2.2, "disk": 1.05, "gpu": False},
    "moe": {"vram": 0.0, "ram_stream": 2.2, "disk": 1.10, "gpu": False},
    # Splitting reads one tensor and writes one bucket at a time.
    "split": {"vram": 0.0, "ram_stream": 2.2, "disk": 1.05, "gpu": False},
    # Structural merge-eval on CPU: both parents are memory-mapped, tiny working set.
    "merge_eval": {"vram": 0.0, "ram_stream": 3.0, "disk": 0.02, "gpu": False},
    # Benchmark harness loads one full model on the GPU.
    "eval": {"vram": 1.25, "ram": 0.35, "disk": 0.05, "gpu": True},
    "qat": {"vram": 1.30, "ram": 0.30, "disk": 0.10, "gpu": True},
}


class ResourceDenied(RuntimeError):
    """The request cannot be satisfied even on an idle machine."""


@dataclass(frozen=True)
class Need:
    vram_bytes: int = 0
    ram_bytes: int = 0
    disk_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "vram_bytes": self.vram_bytes,
            "ram_bytes": self.ram_bytes,
            "disk_bytes": self.disk_bytes,
        }

    def total(self) -> int:
        return self.vram_bytes + self.ram_bytes + self.disk_bytes


@dataclass
class Lease:
    job_id: str
    kind: str
    need: Need
    granted_at: float
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["need"] = self.need.to_dict()
        return payload


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "?"
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TB"


def checkpoint_bytes(source: str | Path | None) -> int:
    """Total size of a checkpoint's weight files, or 0 when it cannot be read."""
    if source is None:
        return 0
    root = Path(str(source)).expanduser()
    try:
        if root.is_file():
            return root.stat().st_size if root.suffix == ".safetensors" else 0
        if not root.is_dir():
            return 0
        files = sorted(root.glob("*.safetensors"))
        if files:
            return sum(file.stat().st_size for file in files)
    except OSError:
        return 0
    return 0


def largest_tensor_bytes(source: str | Path | None) -> int:
    """Size of the biggest stored tensor; the streaming paths peak here."""
    if source is None:
        return 0
    root = Path(str(source)).expanduser()
    index_path = root / "model.safetensors.index.json" if root.is_dir() else None
    best = 0
    if index_path is not None and index_path.is_file():
        try:
            import json  # noqa: PLC0415

            index = json.loads(index_path.read_text(encoding = "utf-8"))
            total = int((index.get("metadata") or {}).get("total_size") or 0)
            tensors = len(index.get("weight_map") or {}) or 1
            # No per-tensor sizes in the index: the mean tensor is the honest guess.
            return max(int(total / tensors), 1) if total else 0
        except (OSError, ValueError, TypeError):
            return 0
    return best


def probe_memory() -> dict[str, Any]:
    """Free/total VRAM, RAM and disk, from the splitter's existing probe."""
    from . import splitter  # noqa: PLC0415 - lazy, keeps this import cheap

    profile = splitter.probe_memory()
    return {
        "vram_free_bytes": profile.vram_free_bytes,
        "vram_total_bytes": profile.vram_total_bytes,
        "ram_free_bytes": profile.system_ram_free_bytes,
        "ram_total_bytes": profile.system_ram_total_bytes,
        "disk_free_bytes": profile.disk_free_bytes,
        "gpus": [gpu.to_dict() for gpu in profile.gpus],
    }


def estimate(
    kind: str,
    *,
    sources: Any = None,
    output_dir: str | Path | None = None,
    memory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Estimate what one job of *kind* needs, in bytes, with its reasoning.

    ``sources`` is a path or a list of paths. Streaming stages are sized from the
    largest tensor, not the whole checkpoint, because they never hold both
    parents at once.
    """
    model = _COST_MODEL.get(kind)
    if model is None:
        raise ValueError(f"Unknown job kind {kind!r}.")
    paths = _as_paths(sources)
    sizes = [checkpoint_bytes(path) for path in paths]
    total = sum(sizes) or max(sizes, default = 0)
    largest = max((largest_tensor_bytes(path) for path in paths), default = 0)
    profile = dict(memory or probe_memory())
    gpus = profile.get("gpus") or []

    notes: list[str] = []
    vram = int(total * float(model.get("vram", 0.0)))
    if model.get("ram_stream"):
        if largest:
            ram = int(largest * float(model["ram_stream"]))
            notes.append(
                f"Streaming: peaks at one tensor pair ({_format_bytes(largest)} x "
                f"{model['ram_stream']}), not the whole checkpoint."
            )
        else:
            ram = int(total * 0.25)
            notes.append("Tensor sizes were not readable; RAM is estimated from the model size.")
    else:
        ram = int(total * float(model.get("ram", 0.0)))
    disk = int(total * float(model.get("disk", 0.0)))
    if output_dir is not None and total == 0:
        disk = 0
    if model.get("gpu") and not gpus:
        notes.append(
            "This stage needs a GPU; the host reported none. It can be planned but not run here."
        )
    return {
        "kind": kind,
        "need": Need(vram_bytes = vram, ram_bytes = ram, disk_bytes = disk).to_dict(),
        "model_bytes": total,
        "largest_tensor_bytes": largest,
        "notes": notes,
        "probe": profile,
    }


def _as_paths(sources: Any) -> list[str]:
    if sources is None:
        return []
    if isinstance(sources, (str, Path)):
        return [str(sources)]
    if isinstance(sources, Mapping):
        return [str(value) for value in sources.values() if isinstance(value, (str, Path))]
    return [str(item) for item in sources if isinstance(item, (str, Path))]


class ResourceBroker:
    """Leases VRAM/RAM/disk to jobs, one at a time when they contend."""

    def __init__(self, probe: Callable[[], Mapping[str, Any]] | None = None):
        self._probe = probe or probe_memory
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._leases: dict[str, Lease] = {}
        self._queue: list[str] = []

    # -- inspection ---------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        profile = dict(self._probe())
        with self._lock:
            leases = [lease.to_dict() for lease in self._leases.values()]
            queued = list(self._queue)
        reserved = {
            key: sum(lease["need"][key] for lease in leases)
            for key in ("vram_bytes", "ram_bytes", "disk_bytes")
        }
        return {
            **profile,
            "limits": self._limits(profile),
            "reserved": reserved,
            "available": self._available(profile, reserved),
            "leases": leases,
            "queued": queued,
        }

    @staticmethod
    def _limits(profile: Mapping[str, Any]) -> dict[str, int]:
        return {
            "vram_bytes": int(int(profile.get("vram_free_bytes") or 0) * VRAM_HEADROOM),
            "ram_bytes": int(int(profile.get("ram_free_bytes") or 0) * RAM_HEADROOM),
            "disk_bytes": int(int(profile.get("disk_free_bytes") or 0) * DISK_HEADROOM),
        }

    def _available(
        self, profile: Mapping[str, Any], reserved: Mapping[str, int]
    ) -> dict[str, int]:
        limits = self._limits(profile)
        return {key: max(0, limits[key] - int(reserved.get(key, 0))) for key in limits}

    # -- acquiring ----------------------------------------------------------

    def _shortfalls(
        self, need: Need, available: Mapping[str, int], limits: Mapping[str, int]
    ) -> tuple[list[str], list[str]]:
        """Split the gaps into "wait for a lease to end" and "never fits"."""
        waited: list[str] = []
        impossible: list[str] = []
        for key, label in (
            ("vram_bytes", "GPU memory"),
            ("ram_bytes", "system RAM"),
            ("disk_bytes", "disk space"),
        ):
            wanted = int(getattr(need, key))
            if wanted <= 0:
                continue
            if wanted > int(limits.get(key, 0)):
                impossible.append(
                    f"needs {_format_bytes(wanted)} {label}; this machine offers at most "
                    f"{_format_bytes(limits.get(key, 0))}"
                )
            elif wanted > int(available.get(key, 0)):
                waited.append(
                    f"waiting for {_format_bytes(wanted - int(available[key]))} more {label}"
                )
        return waited, impossible

    def acquire(
        self,
        job_id: str,
        kind: str,
        need: Need | Mapping[str, Any] | None = None,
        *,
        sources: Any = None,
        wait_seconds: float = 0.0,
    ) -> dict[str, Any]:
        """Try to lease resources for *job_id*.

        Returns ``{"status": "granted" | "queued" | "denied", "reason": str, ...}``.
        ``queued`` means another lease is in the way; ``denied`` means the request
        cannot fit even with every lease released.
        """
        need = (
            need
            if isinstance(need, Need)
            else Need(
                vram_bytes = int((need or {}).get("vram_bytes", 0) or 0),
                ram_bytes = int((need or {}).get("ram_bytes", 0) or 0),
                disk_bytes = int((need or {}).get("disk_bytes", 0) or 0),
            )
        )
        if need.total() <= 0 and sources is not None:
            estimate_payload = estimate(kind, sources = sources)
            need = Need(**estimate_payload["need"])
        if need.total() <= 0:
            # Nothing to reserve (an empty or unreadable plan): nothing to block.
            lease = Lease(job_id = job_id, kind = kind, need = need, granted_at = time.time())
            with self._lock:
                self._leases[job_id] = lease
            return {"status": "granted", "lease": lease.to_dict(), "reason": ""}

        deadline = time.monotonic() + max(0.0, float(wait_seconds))
        with self._condition:
            while True:
                profile = dict(self._probe())
                reserved = {
                    key: sum(
                        int(getattr(lease.need, key)) for lease in self._leases.values()
                    )
                    for key in ("vram_bytes", "ram_bytes", "disk_bytes")
                }
                available = self._available(profile, reserved)
                limits = self._limits(profile)
                waited, impossible = self._shortfalls(need, available, limits)
                if impossible:
                    return {
                        "status": "denied",
                        "reason": (
                            f"{kind} {', '.join(impossible)}. Run it on a larger host "
                            "or a Colab GPU runtime."
                        ),
                        "need": need.to_dict(),
                        "available": available,
                        "limits": limits,
                    }
                if not waited:
                    lease = Lease(
                        job_id = job_id, kind = kind, need = need, granted_at = time.time()
                    )
                    self._leases[job_id] = lease
                    self._queue = [item for item in self._queue if item != job_id]
                    self._condition.notify_all()
                    return {
                        "status": "granted",
                        "lease": lease.to_dict(),
                        "reason": "",
                        "need": need.to_dict(),
                        "available": available,
                    }
                reason = (
                    f"{kind} is queued: {', '.join(waited)}"
                    + _busy_note(self._leases)
                )
                if job_id not in self._queue:
                    self._queue.append(job_id)
                if time.monotonic() >= deadline or wait_seconds <= 0:
                    return {
                        "status": "queued",
                        "reason": reason,
                        "need": need.to_dict(),
                        "available": available,
                        "limits": limits,
                        "leases": [lease.to_dict() for lease in self._leases.values()],
                    }
                self._condition.wait(timeout = max(0.01, deadline - time.monotonic()))

    def release(self, job_id: str) -> None:
        with self._condition:
            self._leases.pop(job_id, None)
            self._queue = [item for item in self._queue if item != job_id]
            self._condition.notify_all()

    def leases(self) -> list[dict[str, Any]]:
        with self._lock:
            return [lease.to_dict() for lease in self._leases.values()]


def _busy_note(leases: Mapping[str, Lease]) -> str:
    if not leases:
        return ""
    busy = ", ".join(sorted({lease.kind for lease in leases.values()}))
    return f" ({busy} holds the rest)."


BROKER = ResourceBroker()


def acquire_for_job(
    job_id: str,
    kind: str,
    *,
    sources: Any = None,
    output_dir: str | Path | None = None,
    wait_seconds: float = 0.0,
) -> dict[str, Any]:
    """Estimate then acquire, the way ``_start_job`` does it."""
    payload = estimate(kind, sources = sources, output_dir = output_dir)
    result = BROKER.acquire(
        job_id, kind, Need(**payload["need"]), wait_seconds = wait_seconds
    )
    result["estimate"] = payload
    return result
