# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional

SCHEMA = "dataset-files-v1"
_CHUNK = 4 * 1024 * 1024


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys = True, separators = (",", ":"), default = str
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _unavailable(source: Optional[str], reason: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "schema": SCHEMA,
        "source": source,
        "sha256": None,
        "reason": reason,
        "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _files_under(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(
        child
        for child in root.rglob("*")
        if child.is_file()
        and not child.name.startswith(".")
        and not child.name.endswith(".lock")
    )


def fingerprint_source(
    source: Optional[str],
    *,
    subset: Optional[str] = None,
    train_split: Optional[str] = None,
    slice_start: Optional[int] = None,
    slice_end: Optional[int] = None,
    streaming: bool = False,
) -> dict[str, Any]:
    if streaming:
        return _unavailable(
            source,
            "Streaming datasets are read from the Hub while training, so no local files exist to hash.",
        )
    text = str(source or "").strip()
    if not text:
        return _unavailable(None, "This run did not record a dataset path.")
    root = Path(text).expanduser()
    if not root.exists():
        return _unavailable(
            text,
            "The dataset path is not on disk, so its content cannot be hashed. "
            "A Hub dataset needs a local snapshot before it can be fingerprinted.",
        )
    try:
        files = _files_under(root)
        if not files:
            return _unavailable(text, "The dataset path contains no files to hash.")
        entries = []
        total_bytes = 0
        for file in files:
            size = file.stat().st_size
            total_bytes += size
            entries.append(
                {
                    "path": (
                        file.relative_to(root).as_posix()
                        if root.is_dir()
                        else file.name
                    ),
                    "size_bytes": size,
                    "sha256": _sha256_file(file),
                }
            )
    except OSError as error:
        return _unavailable(text, f"The dataset files could not be read: {error}")

    selection = {
        "subset": subset,
        "train_split": train_split,
        "slice_start": slice_start,
        "slice_end": slice_end,
    }
    return {
        "status": "hashed",
        "schema": SCHEMA,
        "source": text,
        "sha256": _payload_sha(
            {"files": entries, "selection": selection, "schema": SCHEMA}
        ),
        "files": len(entries),
        "bytes": total_bytes,
        "method": "file-bytes",
        "selection": selection,
        "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def dataset_source_from_config(config: dict[str, Any]) -> Optional[str]:
    if config.get("dataset_source") == "s3" or config.get("s3_dataset"):
        return None
    snapshot = config.get("dataset_snapshot_path")
    if isinstance(snapshot, str) and snapshot.strip():
        return snapshot
    local = config.get("dataset_local_path")
    if isinstance(local, str) and local.strip():
        return local
    local_datasets = config.get("local_datasets")
    if isinstance(local_datasets, list):
        first = next(
            (item for item in local_datasets if isinstance(item, str) and item.strip()),
            None,
        )
        if first:
            return first
    uploaded = config.get("uploaded_file")
    if isinstance(uploaded, str) and uploaded.strip():
        return uploaded
    hf_dataset = config.get("hf_dataset")
    if isinstance(hf_dataset, str) and hf_dataset.strip():
        return hf_dataset
    return None


def fingerprint_config(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("dataset_source") == "s3" or config.get("s3_dataset"):
        return _unavailable(
            None,
            "S3 datasets are streamed by the trainer and have no local file set recorded at start.",
        )
    source = dataset_source_from_config(config)
    slice_start = config.get("dataset_slice_start")
    slice_end = config.get("dataset_slice_end")
    return fingerprint_source(
        source,
        subset=config.get("subset"),
        train_split=config.get("train_split"),
        slice_start=slice_start if isinstance(slice_start, int) else None,
        slice_end=slice_end if isinstance(slice_end, int) else None,
        streaming=bool(config.get("dataset_streaming")),
    )


TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "chat_template.jinja",
    "chat_template.json",
)


def fingerprint_tokenizer(source: Optional[str]) -> dict[str, Any]:
    text = str(source or "").strip()
    if not text:
        return _unavailable(
            None, "This run did not record a local model path to read a tokenizer from."
        )
    root = Path(text).expanduser()
    if not root.is_dir():
        return _unavailable(
            text, "The model has no local snapshot here, so its tokenizer cannot be hashed."
        )
    files = [root / name for name in TOKENIZER_FILES if (root / name).is_file()]
    if not files:
        return _unavailable(
            text, "The model snapshot contains no tokenizer files to hash."
        )
    entries = []
    try:
        for file in files:
            entries.append(
                {
                    "path": file.name,
                    "size_bytes": file.stat().st_size,
                    "sha256": _sha256_file(file),
                }
            )
    except OSError as error:
        return _unavailable(text, f"The tokenizer files could not be read: {error}")
    return {
        "status": "hashed",
        "schema": "tokenizer-files-v1",
        "source": text,
        "sha256": _payload_sha({"files": entries, "schema": "tokenizer-files-v1"}),
        "files": len(entries),
        "file_hashes": {entry["path"]: entry["sha256"] for entry in entries},
        "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def fingerprint_checkpoint_tokenizer(checkpoint: Path) -> dict[str, Any]:
    return fingerprint_tokenizer(str(checkpoint))


def read_fingerprint(output_dir: Optional[str]) -> Optional[dict[str, Any]]:
    if not output_dir:
        return None
    path = Path(output_dir) / "run_manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding = "utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    inputs = manifest.get("inputs") if isinstance(manifest, dict) else None
    if not isinstance(inputs, dict):
        return None
    dataset = inputs.get("dataset")
    if not isinstance(dataset, dict):
        return None
    fingerprint = dataset.get("fingerprint")
    return fingerprint if isinstance(fingerprint, dict) else None
