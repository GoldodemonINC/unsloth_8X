# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from core.training import dataset_fingerprint

TOKENIZER_CHECK_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
)
ADAPTER_FILES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "model.safetensors",
    "pytorch_model.bin",
    "adapters.safetensors",
)
OPTIMIZER_FILES = ("optimizer.pt", "optimizer_state.safetensors", "optimizer.bin")
SCHEDULER_FILES = ("scheduler.pt", "scheduler_state.safetensors")
INDEX_FILES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")


def _step(path: Path) -> int:
    try:
        return int(path.name.removeprefix("checkpoint-"))
    except ValueError:
        return -1


def _present(root: Path, names: tuple[str, ...]) -> list[str]:
    return [name for name in names if (root / name).is_file() and (root / name).stat().st_size > 0]


def _has_adapter(root: Path) -> bool:
    if _present(root, ADAPTER_FILES):
        return True
    if (root / "adapters.safetensors").is_file():
        return True
    return any((root / name).is_file() for name in INDEX_FILES)


def inspect_checkpoint(checkpoint: Path) -> dict[str, Any]:
    tokenizer = dataset_fingerprint.fingerprint_checkpoint_tokenizer(checkpoint)
    return {
        "path": str(checkpoint),
        "step": _step(checkpoint) if checkpoint.name.startswith("checkpoint-") else None,
        "adapter": _has_adapter(checkpoint),
        "optimizer": bool(_present(checkpoint, OPTIMIZER_FILES)),
        "scheduler": bool(_present(checkpoint, SCHEDULER_FILES)),
        "trainer_state": (checkpoint / "trainer_state.json").is_file(),
        "tokenizer": tokenizer,
    }


def _candidate_checkpoints(output_dir: str) -> list[Path]:
    root = Path(output_dir)
    if not root.is_dir():
        return []
    checkpoints = [
        child
        for child in root.glob("checkpoint-*")
        if child.is_dir() and _step(child) >= 0
    ]
    return sorted(checkpoints, key=_step, reverse=True)


def latest_checkpoint(output_dir: Optional[str]) -> Optional[Path]:
    if not output_dir:
        return None
    candidates = _candidate_checkpoints(output_dir)
    if not candidates:
        root = Path(output_dir)
        return root if root.is_dir() else None
    return candidates[0]


def _blocker(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def compare_dataset(
    stored: Optional[Mapping[str, Any]], current: Optional[Mapping[str, Any]]
) -> dict[str, Any]:
    stored_sha = stored.get("sha256") if isinstance(stored, Mapping) else None
    current_sha = current.get("sha256") if isinstance(current, Mapping) else None
    if not isinstance(stored, Mapping) or not stored_sha:
        return {
            "status": "unavailable",
            "stored_sha256": None,
            "current_sha256": current_sha,
            "message": (
                stored.get("reason")
                if isinstance(stored, Mapping) and isinstance(stored.get("reason"), str)
                else "This run predates dataset fingerprinting, so its exact rows are not pinned."
            ),
        }
    if not isinstance(current, Mapping) or not current_sha:
        return {
            "status": "unavailable",
            "stored_sha256": stored_sha,
            "current_sha256": None,
            "message": (
                current.get("reason")
                if isinstance(current, Mapping) and isinstance(current.get("reason"), str)
                else "The dataset cannot be read now, so its current content is unknown."
            ),
        }
    if stored_sha != current_sha:
        return {
            "status": "changed",
            "stored_sha256": stored_sha,
            "current_sha256": current_sha,
            "message": (
                "The dataset content changed since this run started "
                f"({stored_sha[:12]} is not {current_sha[:12]})."
            ),
        }
    return {
        "status": "match",
        "stored_sha256": stored_sha,
        "current_sha256": current_sha,
        "message": "The dataset content matches the run's recorded fingerprint.",
    }


def compare_tokenizer(
    stored: Optional[Mapping[str, Any]], checkpoint: Optional[Mapping[str, Any]]
) -> dict[str, Any]:
    stored_hashes = stored.get("file_hashes") if isinstance(stored, Mapping) else None
    checkpoint_hashes = (
        checkpoint.get("file_hashes") if isinstance(checkpoint, Mapping) else None
    )
    if not isinstance(stored_hashes, dict) or not stored_hashes:
        return {
            "status": "unavailable",
            "message": (
                stored.get("reason")
                if isinstance(stored, Mapping) and isinstance(stored.get("reason"), str)
                else "This run did not record a tokenizer fingerprint."
            ),
        }
    if not isinstance(checkpoint_hashes, dict) or not checkpoint_hashes:
        return {
            "status": "unavailable",
            "message": "The checkpoint contains no tokenizer files to compare.",
        }
    common = sorted(set(stored_hashes) & set(checkpoint_hashes))
    if not common:
        return {
            "status": "unavailable",
            "message": "The checkpoint and the run have no tokenizer file in common.",
        }
    changed = [
        name for name in common if stored_hashes.get(name) != checkpoint_hashes.get(name)
    ]
    if changed:
        return {
            "status": "changed",
            "files": changed,
            "message": (
                "The checkpoint's tokenizer differs from the model's tokenizer: "
                + ", ".join(changed)
                + "."
            ),
        }
    return {
        "status": "match",
        "files": common,
        "message": "The checkpoint's tokenizer matches the model's tokenizer.",
    }


def _find_usable_checkpoint(output_dir: str) -> tuple[Optional[Path], list[dict[str, str]]]:
    fallback_blockers: list[dict[str, str]] = []
    for checkpoint in _candidate_checkpoints(output_dir):
        artifacts = inspect_checkpoint(checkpoint)
        blockers = []
        if not artifacts["trainer_state"]:
            blockers.append(
                _blocker(
                    "trainer_state_missing",
                    f"{checkpoint.name} has no trainer_state.json, so the optimizer step is unknown.",
                )
            )
        if not artifacts["adapter"]:
            blockers.append(
                _blocker(
                    "adapter_missing",
                    f"{checkpoint.name} has no adapter or model weights to continue from.",
                )
            )
        if not artifacts["optimizer"]:
            blockers.append(
                _blocker(
                    "optimizer_state_missing",
                    f"{checkpoint.name} has no optimizer state, so a resume would restart the optimizer.",
                )
            )
        if not blockers:
            return checkpoint, []
        if not fallback_blockers:
            fallback_blockers = blockers
    return None, fallback_blockers


def evaluate_resume(
    output_dir: Optional[str],
    stored_config: Mapping[str, Any],
    current_dataset_fingerprint: Optional[Mapping[str, Any]] = None,
    current_tokenizer_fingerprint: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    blockers: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    if not output_dir:
        return {
            "ok": False,
            "checkpoint": None,
            "blockers": [_blocker("output_dir_missing", "This run has no output directory.")],
            "warnings": warnings,
            "checks": {},
            "dataset": None,
            "tokenizer": None,
        }

    checkpoint, artifact_blockers = _find_usable_checkpoint(output_dir)
    if checkpoint is None:
        blockers.extend(artifact_blockers)
        if not blockers:
            blockers.append(
                _blocker(
                    "checkpoint_missing",
                    "No complete checkpoint remains in this run's output directory.",
                )
            )
    artifacts = inspect_checkpoint(checkpoint) if checkpoint else None

    stored_dataset = stored_config.get("dataset_fingerprint")
    dataset = compare_dataset(stored_dataset, current_dataset_fingerprint)
    if dataset["status"] == "changed":
        blockers.append(_blocker("dataset_changed", dataset["message"]))
    elif dataset["status"] == "unavailable":
        warnings.append(_blocker("dataset_unavailable", dataset["message"]))

    tokenizer = compare_tokenizer(
        stored_config.get("tokenizer_fingerprint"),
        artifacts.get("tokenizer") if artifacts else None,
    )
    if tokenizer["status"] == "changed":
        blockers.append(_blocker("tokenizer_changed", tokenizer["message"]))
    elif tokenizer["status"] == "unavailable":
        warnings.append(_blocker("tokenizer_unavailable", tokenizer["message"]))

    if artifacts is not None:
        if not artifacts["scheduler"]:
            warnings.append(
                _blocker(
                    "scheduler_state_missing",
                    "The checkpoint has no scheduler state; the LR schedule will restart.",
                )
            )
        if artifacts["tokenizer"].get("status") != "hashed":
            warnings.append(
                _blocker(
                    "checkpoint_tokenizer_missing",
                    "The checkpoint saved no tokenizer, so the model tokenizer will be reused.",
                )
            )

    checks = {
        "checkpoint": artifacts,
        "dataset": dataset,
        "tokenizer": tokenizer,
        "current_tokenizer": current_tokenizer_fingerprint,
    }
    return {
        "ok": not blockers,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "blockers": blockers,
        "warnings": warnings,
        "checks": checks,
        "dataset": dataset,
        "tokenizer": tokenizer,
    }
