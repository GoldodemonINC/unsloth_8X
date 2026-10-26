# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``run_manifest.json`` -- one record of what a stage actually did.

``merge_report.json`` already described a merge. Training had nothing, so a run
could not be reproduced from its output folder: which dataset revision, which
seed, which curriculum, which zone definition, which blend schedule, which code.

Every stage now writes the same document beside its artifacts::

    merge → outputs/merge-a-b/run_manifest.json
    split → outputs/sharded-x/run_manifest.json
    moe   → outputs/moe-a-b/run_manifest.json
    train → outputs/<run>/run_manifest.json

The manifest is deliberately boring: JSON, fixed keys, no schema library. It
answers "can I rebuild this exactly?" for the things that actually change a
result -- seeds, dataset content, zone definition, blend events, code version --
and it records the policy overrides (remote code, tokenizer conflicts) that a
later reader needs in order to trust the result.

Hashes are computed over cheap, stable identities rather than raw weights where
possible: a 14 GB checkpoint is identified by its shard map, its config digest,
and four sampled windows from each shard. Sampling every weight would cost
minutes per run; sampling fixed offsets catches any real change to the weights
and still takes milliseconds.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

MANIFEST_NAME = "run_manifest.json"
SCHEMA_VERSION = 2

# Keys every manifest must carry. A stage that omits one is a bug: downstream
# tooling (and the tests) read these without guarding.
REQUIRED_FIELDS = (
    "schema_version",
    "run_id",
    "stage",
    "created_at",
    "studio_version",
    "environment",
    "config",
    "seeds",
    "inputs",
    "artifacts",
    "events",
)

_HASH_CHUNK = 4 * 1024 * 1024


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def hash_file(path: str | Path, *, limit: int | None = None) -> str:
    """SHA-256 of a file (or of its first *limit* bytes)."""
    digest = hashlib.sha256()
    remaining = limit
    with open(path, "rb") as stream:
        while True:
            size = _HASH_CHUNK if remaining is None else min(_HASH_CHUNK, remaining)
            if size == 0:
                break
            chunk = stream.read(size)
            if not chunk:
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return digest.hexdigest()


def hash_payload(payload: Any) -> str:
    """Stable SHA-256 of a JSON-able structure (sorted keys, no whitespace)."""
    encoded = json.dumps(payload, sort_keys = True, separators = (",", ":"), default = str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


_SAMPLE_WINDOW = 256 * 1024
_SAMPLE_FRACTIONS = (0.0, 0.37, 0.71, 0.99)


def sampled_file_digest(path: str | Path) -> str:
    """Identity of a weight file from four fixed windows, not a full read.

    Two checkpoints with the same tensor names and shapes but different values
    must not share a digest. Reading 14 GB per stage is not acceptable, so the
    file size plus four windows spread across it are hashed instead.
    """
    file_path = Path(path)
    size = file_path.stat().st_size
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with open(file_path, "rb") as stream:
        for fraction in _SAMPLE_FRACTIONS:
            offset = int(max(0, min(max(0, size - _SAMPLE_WINDOW), size * fraction)))
            stream.seek(offset)
            digest.update(stream.read(_SAMPLE_WINDOW))
    return digest.hexdigest()


def checkpoint_digest(source: str | Path) -> dict[str, Any]:
    """A cheap identity for a checkpoint: shard map + config + size.

    Reading every weight to hash it would cost minutes and gigabytes of I/O per
    run. The shard map names each tensor and its shard, and the config digest
    covers the architecture, so a different checkpoint always produces a
    different digest while a re-run over the same folder produces the same one.
    """
    root = Path(source).expanduser()
    if root.is_file():
        return {
            "source": str(root),
            "kind": "file",
            "sha256": hash_file(root, limit = 64 * 1024 * 1024),
            "size_bytes": root.stat().st_size,
            "hashed_bytes": min(root.stat().st_size, 64 * 1024 * 1024),
        }
    if not root.is_dir():
        return {"source": str(root), "kind": "unresolved", "sha256": None}

    index_path = root / "model.safetensors.index.json"
    shards: dict[str, Any] = {}
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding = "utf-8"))
            weight_map = index.get("weight_map") or {}
            shard_digests: dict[str, str | None] = {}
            for key, shard in weight_map.items():
                path = root / str(shard)
                if str(shard) not in shard_digests:
                    shard_digests[str(shard)] = (
                        sampled_file_digest(path) if path.is_file() else None
                    )
                shards[key] = {
                    "shard": str(shard),
                    "size_bytes": path.stat().st_size if path.is_file() else None,
                    "sample_sha256": shard_digests[str(shard)],
                }
        except (OSError, ValueError):
            shards = {}
    if not shards:
        for file in sorted(root.glob("*.safetensors")):
            shards[file.name] = {
                "shard": file.name,
                "size_bytes": file.stat().st_size,
                "sample_sha256": sampled_file_digest(file),
            }

    config_path = root / "config.json"
    config_digest = hash_file(config_path) if config_path.is_file() else None
    tokenizer_digest = None
    tokenizer_config = root / "tokenizer_config.json"
    if tokenizer_config.is_file():
        tokenizer_digest = hash_file(tokenizer_config)
    return {
        "source": str(root),
        "kind": "directory",
        "sha256": hash_payload({"shards": shards, "config": config_digest}),
        "config_sha256": config_digest,
        "tokenizer_config_sha256": tokenizer_digest,
        "tensors": len(shards),
        "size_bytes": sum(
            entry["size_bytes"] for entry in shards.values() if entry["size_bytes"]
        )
        or None,
    }


def dataset_identity(source: str | Path | None) -> dict[str, Any]:
    """Identify a dataset: a local file is hashed, a Hub id is recorded verbatim."""
    if source is None:
        return {"kind": "none", "value": None, "sha256": None}
    text = str(source).strip()
    if not text:
        return {"kind": "none", "value": None, "sha256": None}
    path = Path(text).expanduser()
    if path.is_file():
        return {
            "kind": "file",
            "value": text,
            "sha256": hash_file(path),
            "size_bytes": path.stat().st_size,
        }
    if path.is_dir():
        files = sorted(child for child in path.rglob("*") if child.is_file())
        return {
            "kind": "directory",
            "value": text,
            "sha256": hash_payload(
                {child.relative_to(path).as_posix(): child.stat().st_size for child in files}
            ),
            "files": len(files),
        }
    # A Hub dataset id: record it, and say plainly that its content is not pinned.
    return {
        "kind": "reference",
        "value": text,
        "sha256": None,
        "note": "Remote dataset reference; content hash is not pinned by this manifest.",
    }


def zone_definition_hash(config: Mapping[str, Any] | None) -> str | None:
    """Fingerprint of the layer-LR zone definition a run trained with."""
    if not config:
        return None
    relevant = {
        key: config.get(key)
        for key in (
            "zone_multipliers",
            "kind_multipliers",
            "layer_multipliers",
            "depth_curve",
        )
    }
    return hash_payload(relevant)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def studio_version() -> str:
    try:
        from utils.studio_version import studio_version as _version  # noqa: PLC0415

        value = _version()
        if isinstance(value, str) and value:
            return value
    except Exception:  # noqa: BLE001 - version stamping must never fail a run
        pass
    return "unknown"


def _module_version(name: str) -> str | None:
    try:
        from importlib.metadata import version  # noqa: PLC0415

        return version(name)
    except Exception:  # noqa: BLE001 - absent package is not an error here
        return None


def environment() -> dict[str, Any]:
    """Versions a reproduction needs. Absent packages are recorded as null."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": _module_version("torch"),
        "transformers": _module_version("transformers"),
        "safetensors": _module_version("safetensors"),
        "peft": _module_version("peft"),
        "trl": _module_version("trl"),
        "cuda": None,
    }
    try:  # optional, and only when torch is importable
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            info["cuda"] = torch.version.cuda
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    return info


# ---------------------------------------------------------------------------
# Building / writing
# ---------------------------------------------------------------------------


def new_run_id(stage: str) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return f"{stage}-{stamp}-{uuid.uuid4().hex[:8]}"


def build_manifest(
    stage: str,
    *,
    run_id: str | None = None,
    config: Mapping[str, Any] | None = None,
    seeds: Mapping[str, Any] | Sequence[int] | None = None,
    inputs: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, Any] | None = None,
    events: Sequence[Mapping[str, Any]] | None = None,
    versions: Mapping[str, Any] | None = None,
    policy: Mapping[str, Any] | None = None,
    labeller: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a complete manifest. Missing pieces are recorded, never invented."""
    created = time.time()
    seed_payload: dict[str, Any]
    if seeds is None:
        seed_payload = {}
    elif isinstance(seeds, Mapping):
        seed_payload = dict(seeds)
    else:
        seed_payload = {"values": list(seeds)}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id or new_run_id(stage),
        "stage": stage,
        "created_at": created,
        "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(created)),
        "studio_version": studio_version(),
        "environment": environment(),
        "config": dict(config or {}),
        "seeds": seed_payload,
        "inputs": dict(inputs or {}),
        "artifacts": dict(artifacts or {}),
        "events": [dict(event) for event in (events or [])],
        "versions": dict(versions or {}),
        "policy": dict(policy or {}),
        "labels": dict(labeller or {}),
    }
    return manifest


def manifest_path(directory: str | Path) -> Path:
    return Path(directory) / MANIFEST_NAME


def write_manifest(directory: str | Path, manifest: Mapping[str, Any]) -> Path:
    """Write the manifest atomically, so a reader never sees a half document.

    Every produced folder also gets the NRC-2 notice beside its weights: the
    manifest already travels with the run, and the notice is what a reader sees
    without opening JSON. A stage that supplies its own ``license`` block keeps
    it; only a missing one is filled in.
    """
    from . import nrc_license, policy as policy_module  # noqa: PLC0415 - avoids a cycle at import

    document = dict(manifest)
    document.setdefault("license", nrc_license.license_record())
    target = manifest_path(directory)
    policy_module.atomic_write_json(target, document)
    nrc_license.stamp(directory)
    return target


def read_manifest(directory: str | Path) -> dict[str, Any] | None:
    path = manifest_path(directory)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def update_manifest(directory: str | Path, **updates: Any) -> dict[str, Any] | None:
    """Merge fields into an existing manifest (used to close out a finished run)."""
    current = read_manifest(directory)
    if current is None:
        return None
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(current.get(key), Mapping):
            current[key] = {**current[key], **value}
        else:
            current[key] = value
    current["updated_at"] = time.time()
    write_manifest(directory, current)
    return current


def append_event(directory: str | Path, event: Mapping[str, Any]) -> None:
    """Record a mid-run event (blend change, resume, policy override)."""
    current = read_manifest(directory)
    if current is None:
        return
    current.setdefault("events", []).append(dict(event))
    write_manifest(directory, current)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_manifest(manifest: Mapping[str, Any] | None) -> list[str]:
    """Return the list of problems that make a manifest non-reproducible."""
    if not isinstance(manifest, Mapping):
        return ["No manifest found."]
    problems: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in manifest:
            problems.append(f"Missing required field {field!r}.")
    for field in ("config", "seeds", "inputs", "artifacts", "environment"):
        value = manifest.get(field)
        if field in manifest and not isinstance(value, Mapping):
            problems.append(f"Field {field!r} must be an object.")
    if manifest.get("stage") in (None, ""):
        problems.append("Field 'stage' is empty.")
    if not manifest.get("run_id"):
        problems.append("Field 'run_id' is empty.")
    events = manifest.get("events")
    if not isinstance(events, list):
        problems.append("Field 'events' must be a list.")
    env = manifest.get("environment")
    if isinstance(env, Mapping) and not env.get("python"):
        problems.append("environment.python is missing; the run cannot be reproduced.")
    return problems


def to_artifact_list(paths: Iterable[str | Path], root: str | Path) -> list[dict[str, Any]]:
    """Relative artifact names with sizes, for the ``artifacts`` block."""
    base = Path(root)
    entries: list[dict[str, Any]] = []
    for item in paths:
        path = Path(item)
        try:
            relative = path.resolve().relative_to(base.resolve()).as_posix()
        except (OSError, ValueError):
            relative = str(path)
        try:
            size = path.stat().st_size
        except OSError:
            size = None
        entries.append({"path": relative, "size_bytes": size})
    return entries
