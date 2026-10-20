# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Loading and writing policy for the merge / split / QAT run layer.

Merging a checkpoint from the Hub means running over files this process does not
control, and writing gigabytes to disk means assuming that write can be
interrupted. Both are treated here as policy, not as best effort:

* **Safetensors only by default.** A pickle (``.bin``, ``.pt``, ``.pth``,
  ``.ckpt``, ``.msgpack``) can execute code while being read, so a source that
  carries one is refused before any tensor is opened unless the caller opts in.
* **Remote code is opt-in.** A repo whose ``config.json`` declares
  ``auto_map`` / ``auto_model``, or that ships ``modeling_*.py`` / an
  ``.py`` module, needs ``trust_remote_code`` to load later. Merging itself only
  reads tensors, but the *output* inherits that config, so the merge is refused
  by default with a message naming the exact opt-in.
* **Atomic writes.** Every shard lands on ``<name>.partial`` and is renamed into
  place, so an interrupted run leaves no file that a loader would half-read.
* **Disk full is a first-class error.** ``ENOSPC``/``EDQUOT`` become
  :class:`DiskFullError` with the bytes that were still needed, and the partial
  file is removed.
* **Path traversal.** Artifact names are resolved under the run directory, and
  ``..``, absolute paths and Windows drive paths are rejected.

Nothing in this module imports torch, safetensors or transformers at import
time: it is the cheap gate that runs before any of them.
"""

from __future__ import annotations

import errno
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Mapping

# Formats a loader would unpickle. Kept explicit rather than a glob so a new
# format has to be added deliberately.
PICKLE_SUFFIXES = frozenset(
    {".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".msgpack", ".h5", ".pb"}
)
SAFETENSORS_SUFFIX = ".safetensors"
_WEIGHT_SUFFIXES = frozenset({SAFETENSORS_SUFFIX}) | PICKLE_SUFFIXES

# How many offending files to name in one message before summarising.
_NAMES_SHOWN = 4


class PolicyViolation(ValueError):
    """A source or destination the run layer will not touch."""


class RemoteCodeRejected(PolicyViolation):
    """The checkpoint needs ``trust_remote_code`` and the caller did not opt in."""


class DiskFullError(OSError):
    """A write ran out of space; the partial artifact has been removed."""

    def __init__(self, path: str | Path, needed_bytes: int, message: str):
        super().__init__(errno.ENOSPC, message)
        self.path = str(path)
        self.needed_bytes = int(needed_bytes)


@dataclass(frozen=True)
class LoadingPolicy:
    """What this run is allowed to read and where it may write."""

    allow_remote_code: bool = False
    allow_non_safetensors: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_POLICY = LoadingPolicy()


def policy_from_request(payload: Mapping[str, Any] | None) -> LoadingPolicy:
    """Build a policy from an API payload, keeping the safe defaults."""
    payload = payload or {}
    return LoadingPolicy(
        allow_remote_code = bool(payload.get("allow_remote_code", False)),
        allow_non_safetensors = bool(payload.get("allow_non_safetensors", False)),
    )


# ---------------------------------------------------------------------------
# Source inspection
# ---------------------------------------------------------------------------


def _checkpoint_root(source: str | Path) -> Path:
    path = Path(source).expanduser()
    if path.is_file() and path.suffix == SAFETENSORS_SUFFIX:
        return path.parent
    return path


def source_files(root: Path) -> list[Path]:
    """Files a checkpoint directory contributes (one level, like the loader)."""
    if not root.is_dir():
        return []
    return sorted(child for child in root.iterdir() if child.is_file())


def _summarise(names: Iterable[str]) -> str:
    names = sorted(set(names))
    shown = ", ".join(names[:_NAMES_SHOWN])
    if len(names) > _NAMES_SHOWN:
        shown += f" (+{len(names) - _NAMES_SHOWN} more)"
    return shown


def find_pickle_files(source: str | Path) -> list[str]:
    """Weight files a loader would unpickle."""
    return [
        file.name
        for file in source_files(_checkpoint_root(source))
        if file.suffix.lower() in PICKLE_SUFFIXES
    ]


def remote_code_entries(source: str | Path) -> dict[str, Any]:
    """Remote-code surfaces declared or shipped by a checkpoint.

    Returns ``{"declared": [...], "modules": [...], "config_keys": [...]}``.
    ``declared`` comes from ``auto_map`` in config.json, ``modules`` from the
    ``*.py`` files shipped beside the weights, and ``config_keys`` records a
    top-level ``auto_map`` style key so the message can name it.
    """
    root = _checkpoint_root(source)
    declared: list[str] = []
    config_keys: list[str] = []
    config_path = root / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding = "utf-8"))
        except (OSError, ValueError):
            config = {}
        if isinstance(config, dict):
            auto_map = config.get("auto_map")
            if isinstance(auto_map, dict):
                declared = [str(key) for key in auto_map]
                config_keys.append("auto_map")
            for key in ("auto_model", "custom_pipelines", "model_type_custom"):
                if key in config:
                    declared.append(key)
                    config_keys.append(key)
    modules = [
        file.name
        for file in source_files(root)
        if file.suffix.lower() == ".py" and file.name != "tokenization_placeholder.py"
    ]
    return {"declared": declared, "modules": modules, "config_keys": config_keys}


def requires_remote_code(source: str | Path) -> bool:
    entries = remote_code_entries(source)
    return bool(entries["declared"] or entries["modules"])


def inspect_source(
    source: str | Path, policy: LoadingPolicy = DEFAULT_POLICY
) -> dict[str, Any]:
    """Report what a source contains and what the policy would do with it.

    Never raises for a policy problem: this is the planning view, so the UI can
    show the finding before the user commits to a run.
    """
    root = _checkpoint_root(source)
    pickles = find_pickle_files(root)
    remote = remote_code_entries(root)
    needs_remote = bool(remote["declared"] or remote["modules"])
    safetensors = [
        file.name
        for file in source_files(root)
        if file.suffix.lower() == SAFETENSORS_SUFFIX
    ]
    warnings: list[str] = []
    if pickles:
        warnings.append(
            f"{len(pickles)} pickle weight file(s) present ({_summarise(pickles)}). "
            "Pickles can run code when read; convert to safetensors or allow them explicitly."
        )
    if needs_remote:
        warnings.append(
            "This checkpoint declares custom code"
            + (f" ({_summarise(remote['declared'])})" if remote["declared"] else "")
            + (f" and ships {_summarise(remote['modules'])}" if remote["modules"] else "")
            + "; loading it needs trust_remote_code."
        )
    if not safetensors and not pickles:
        warnings.append("No weight files found at this path.")
    return {
        "source": str(root),
        "safetensors": safetensors,
        "pickles": pickles,
        "remote_code": remote,
        "requires_remote_code": needs_remote,
        "policy": policy.to_dict(),
        "allowed": (
            (policy.allow_non_safetensors or not pickles)
            and (policy.allow_remote_code or not needs_remote)
        ),
        "warnings": warnings,
    }


def enforce_source_policy(
    source: str | Path, policy: LoadingPolicy = DEFAULT_POLICY, *, label: str = "checkpoint"
) -> dict[str, Any]:
    """Raise unless *source* is acceptable under *policy*; return the inspection."""
    report = inspect_source(source, policy)
    if report["pickles"] and not policy.allow_non_safetensors:
        raise PolicyViolation(
            f"{label} carries pickle weight files ({_summarise(report['pickles'])}). "
            "Studio reads safetensors by default because a pickle can execute code while "
            "loading. Convert the checkpoint to safetensors, or pass "
            "allow_non_safetensors=true to accept the risk."
        )
    if report["requires_remote_code"] and not policy.allow_remote_code:
        remote = report["remote_code"]
        keys = ", ".join(remote["config_keys"]) or None
        detail = f"config declares {keys}" if keys else "custom code"
        if remote["declared"]:
            detail += f" ({_summarise(remote['declared'])})"
        if remote["modules"]:
            detail += f"; ships {_summarise(remote['modules'])}"
        raise RemoteCodeRejected(
            f"{label} needs custom code ({detail}). Studio refuses to load remote code by "
            "default: it runs arbitrary Python from the checkpoint's repository. Re-run "
            "with allow_remote_code=true only if you trust the source; Studio will record "
            "the override in the run manifest."
        )
    return report


def remote_code_warning(source: str | Path) -> str | None:
    """The warning a run records when remote code was explicitly allowed."""
    remote = remote_code_entries(source)
    if not remote["declared"] and not remote["modules"]:
        return None
    detail = _summarise(remote["declared"] + remote["modules"])
    return (
        f"Remote code was explicitly allowed for this run ({detail}). "
        "Custom Python from the checkpoint repository executed with Studio's privileges."
    )


# ---------------------------------------------------------------------------
# Paths and atomic writes
# ---------------------------------------------------------------------------


def safe_child(root: str | Path, name: str) -> Path:
    """Resolve *name* inside *root*, refusing anything that escapes it."""
    text = str(name or "").strip()
    if not text:
        raise PolicyViolation("An artifact name is required.")
    if is_absolute_any_platform(text):
        raise PolicyViolation(f"Artifact name {text!r} must be relative to the run folder.")
    parts = PurePosixPath(text.replace("\\", "/")).parts
    if any(part in ("..", "") for part in parts):
        raise PolicyViolation(f"Artifact name {text!r} escapes the run folder.")
    root_path = Path(root).expanduser().resolve()
    candidate = (root_path / text).resolve()
    if candidate != root_path and root_path not in candidate.parents:
        raise PolicyViolation(f"Artifact name {text!r} escapes the run folder.")
    return candidate


def is_absolute_any_platform(text: str) -> bool:
    """Absolute under POSIX *or* Windows rules (a ``C:/x`` path on POSIX too)."""
    return Path(text).is_absolute() or PureWindowsPath(text).is_absolute()


def _enospc(error: OSError) -> bool:
    return error.errno in (errno.ENOSPC, errno.EDQUOT) or getattr(
        error, "winerror", None
    ) in (112, 39)


def ensure_free_space(directory: str | Path, needed_bytes: int) -> None:
    """Raise :class:`DiskFullError` when *needed_bytes* cannot fit."""
    needed = int(needed_bytes)
    if needed <= 0:
        return
    target = Path(directory).expanduser()
    probe = target
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        free = shutil.disk_usage(probe).free
    except OSError:
        return
    if free < needed:
        raise DiskFullError(
            target,
            needed,
            f"Not enough disk space in {target}: {format_bytes(needed)} needed, "
            f"{format_bytes(free)} free. Free space or merge onto another drive.",
        )


def format_bytes(value: int | None) -> str:
    if value is None:
        return "?"
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TB"


def atomic_write_bytes(path: str | Path, payload: bytes, *, needed_bytes: int = 0) -> Path:
    """Write *payload* to *path* through a temporary file in the same folder."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if needed_bytes:
        ensure_free_space(target.parent, needed_bytes)
    handle = tempfile.NamedTemporaryFile(
        dir = str(target.parent),
        prefix = f".{target.name}.",
        suffix = ".partial",
        delete = False,
    )
    temp_path = Path(handle.name)
    try:
        with handle as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, target)
    except OSError as error:
        _cleanup(temp_path)
        if _enospc(error):
            raise DiskFullError(
                target,
                needed_bytes or len(payload),
                f"Disk full while writing {target.name}: "
                f"{format_bytes(needed_bytes or len(payload))} could not be written. "
                "The partial file was removed, so nothing half-written is left behind.",
            ) from error
        raise
    except BaseException:
        _cleanup(temp_path)
        raise
    return target


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
    return atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    return atomic_write_text(path, json.dumps(payload, indent=2, default = str))


def atomic_save_tensors(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Write a safetensors shard atomically, translating a full disk.

    ``safetensors`` writes straight to the destination it is given, so the shard
    is written to ``<name>.partial`` in the same folder and renamed. A crash, a
    full disk or a keyboard interrupt therefore never leaves a half-shard that
    ``model.safetensors.index.json`` could point at.
    """
    from . import tensor_io  # noqa: PLC0415 - keeps this module importable without torch

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.partial")
    _cleanup(temp)
    size = sum(tensor_io.nbytes(value) for value in payload.values())
    try:
        tensor_io.save_tensors(dict(payload), str(temp))
        os.replace(temp, target)
    except OSError as error:
        _cleanup(temp)
        if _enospc(error):
            raise DiskFullError(
                target,
                size,
                f"Disk full while writing {target.name}: {format_bytes(size)} could not "
                "be written. The partial shard was removed, so the checkpoint has no "
                "half-written file. Free space or merge onto another drive.",
            ) from error
        raise
    except BaseException:
        _cleanup(temp)
        raise
    return target


def _cleanup(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass
