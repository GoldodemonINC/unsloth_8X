# SPDX-License-Identifier: NRC-2
# Copyright 2026-present the X8 authors. See /LICENSE-NRC-2.md

"""Merge LoRA / PEFT adapters into one adapter.

The merge studio merges full checkpoints. Adapters are the other half of the
work: a user who trained two LoRA runs on the same base has two small folders
and no way to combine them, because a checkpoint merge reads ``config.json`` and
full weight files that an adapter does not have.

An adapter is not weights, it is a *delta*: ``B @ A * (alpha / r)`` added to the
frozen base. Merging therefore happens on the deltas, which is also what makes
the existing TIES and DARE code in :mod:`core.advanced.merge` reusable -- the
same blend that combines two fine-tunes combines their adapters, and this module
does not grow a second, subtly different implementation.

The merged delta is factorised back into a low-rank pair with an SVD. That is a
lossy step by construction: summing two rank-8 deltas needs rank 16 to be exact,
so the output rank defaults to the sum of the inputs' ranks (capped by the
matrix dimensions) and the report states what was kept. Weights are validated
before anything is written, so a bad adapter set fails with a message rather
than a half-written folder.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import merge as merge_engine

#: Methods that make sense on a delta. SLERP is excluded: it interpolates two
#: models along a sphere, and a delta is not a point on one.
ADAPTER_METHODS = ("linear", "ties", "dare")

_ADAPTER_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")
_LORA_A = re.compile(r"^(?P<module>.+)\.lora_A(?:\.[^.]+)?\.weight$")
_LORA_B = re.compile(r"^(?P<module>.+)\.lora_B(?:\.[^.]+)?\.weight$")
_REPORT_NAME = "x8_adapter_merge.json"


class AdapterMergeError(ValueError):
    """The adapter set cannot be merged; the message says which one and why."""


@dataclass(frozen = True)
class Adapter:
    path: Path
    config: dict[str, Any]
    tensors: dict[str, Any]

    @property
    def rank(self) -> int:
        value = self.config.get("r")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise AdapterMergeError(f"{self.path}: adapter config has no usable r.")
        return int(value)

    @property
    def scaling(self) -> float:
        alpha = self.config.get("lora_alpha")
        alpha = float(alpha) if isinstance(alpha, (int, float)) else float(self.rank)
        if self.config.get("use_rslora"):
            return alpha / math.sqrt(self.rank)
        return alpha / self.rank


def read_adapter(directory: str | Path) -> Adapter:
    """Read ``adapter_config.json`` and the weight file beside it."""
    path = Path(directory).expanduser()
    if not path.is_dir():
        raise AdapterMergeError(f"{path}: not a folder.")
    config_path = path / "adapter_config.json"
    if not config_path.is_file():
        raise AdapterMergeError(
            f"{path.name}: not an adapter (no adapter_config.json). A full checkpoint "
            "goes through the Merge studio instead."
        )
    try:
        config = json.loads(config_path.read_text(encoding = "utf-8"))
    except (OSError, ValueError) as error:
        raise AdapterMergeError(f"{path.name}: adapter_config.json is unreadable ({error}).") from error
    if not isinstance(config, dict):
        raise AdapterMergeError(f"{path.name}: adapter_config.json must be an object.")
    weights = next((path / name for name in _ADAPTER_WEIGHTS if (path / name).is_file()), None)
    if weights is None:
        raise AdapterMergeError(f"{path.name}: no adapter_model.safetensors weight file.")
    if weights.suffix != ".safetensors":
        raise AdapterMergeError(
            f"{path.name}: adapter_model.bin is a pickle; export the adapter as safetensors."
        )
    from safetensors.numpy import load_file

    try:
        tensors = dict(load_file(str(weights)))
    except Exception as error:  # noqa: BLE001 - re-raised with the folder named
        raise AdapterMergeError(f"{path.name}: cannot read {weights.name} ({error}).") from error
    adapter = Adapter(path = path, config = config, tensors = tensors)
    adapter.rank  # noqa: B018 - validates r now, not halfway through a merge
    return adapter


def _pairs(tensors: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    """module -> (A key, B key) for every complete LoRA pair."""
    a_keys: dict[str, str] = {}
    b_keys: dict[str, str] = {}
    for key in tensors:
        match = _LORA_A.match(str(key))
        if match:
            a_keys[match.group("module")] = str(key)
            continue
        match = _LORA_B.match(str(key))
        if match:
            b_keys[match.group("module")] = str(key)
    return {
        module: (a_keys[module], b_keys[module])
        for module in a_keys.keys() & b_keys.keys()
    }


def adapter_deltas(adapter: Adapter) -> dict[str, Any]:
    """``module -> B @ A * scaling``, the weight change the adapter applies."""
    import numpy as np

    deltas: dict[str, Any] = {}
    for module, (a_key, b_key) in _pairs(adapter.tensors).items():
        a = np.asarray(adapter.tensors[a_key], dtype = np.float32)
        b = np.asarray(adapter.tensors[b_key], dtype = np.float32)
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
            raise AdapterMergeError(
                f"{adapter.path.name}: {module} has lora_A {a.shape} and lora_B {b.shape}, "
                "which cannot be multiplied."
            )
        deltas[module] = (b @ a) * adapter.scaling
    if not deltas:
        raise AdapterMergeError(f"{adapter.path.name}: no LoRA A/B pairs in the weight file.")
    return deltas


def _factorize(delta, rank: int):
    """Split a weight delta into ``(A, B)`` with ``B @ A`` closest in Frobenius norm."""
    import numpy as np

    u, s, vt = np.linalg.svd(np.asarray(delta, dtype = np.float64), full_matrices = False)
    keep = int(min(max(1, rank), s.size))
    root = np.sqrt(np.maximum(s[:keep], 0.0))
    a = (root[:, None] * vt[:keep]).astype(np.float32)
    b = (u[:, :keep] * root[None, :]).astype(np.float32)
    return a, b


def merge_adapters(
    directories: Sequence[str | Path],
    output_dir: str | Path,
    *,
    method: str = "linear",
    weights: Sequence[float] | None = None,
    density: float = 0.5,
    seed: int = 3407,
    rank: int | None = None,
) -> dict[str, Any]:
    """Blend two or more adapters and write a standard PEFT adapter folder."""
    if len(directories) < 2:
        raise AdapterMergeError("Merging needs at least two adapters.")
    method = str(method or "linear").lower()
    if method not in ADAPTER_METHODS:
        raise AdapterMergeError(
            f"method {method!r} is not available for adapters; use "
            f"{', '.join(ADAPTER_METHODS)}."
        )
    if weights is not None and len(weights) != len(directories):
        raise AdapterMergeError("Give one weight per adapter, or none at all.")
    if not 0 < float(density) <= 1:
        raise AdapterMergeError("density is the fraction of entries TIES/DARE keep: 0 < density <= 1.")
    adapters = [read_adapter(directory) for directory in directories]
    bases = {str(adapter.config.get("base_model_name_or_path") or "") for adapter in adapters}
    bases.discard("")
    if len(bases) > 1:
        raise AdapterMergeError(
            "The adapters name different base models "
            f"({', '.join(sorted(bases))}); their deltas are not comparable."
        )

    deltas = [adapter_deltas(adapter) for adapter in adapters]
    modules = set(deltas[0])
    for index, blend in enumerate(deltas[1:], start = 2):
        modules &= set(blend)
    if not modules:
        raise AdapterMergeError("The adapters have no LoRA module in common.")
    skipped = sorted(set().union(*[set(blend) for blend in deltas]) - modules)

    output_rank = int(rank) if rank is not None else sum(adapter.rank for adapter in adapters)
    # An SVD cannot keep more components than the smallest delta dimension.
    available = min(min(blend[module].shape) for blend in deltas for module in modules)
    output_rank = max(1, min(output_rank, int(available)))

    merged_tensors: dict[str, Any] = {}
    for module in sorted(modules):
        try:
            blended = merge_engine.blend_tensors(
                [blend[module] for blend in deltas],
                method = method,
                weights = list(weights) if weights else None,
                density = density,
                seed = seed,
            )
        except ValueError as error:
            raise AdapterMergeError(f"{module}: {error}") from error
        a, b = _factorize(blended, output_rank)
        merged_tensors[f"{module}.lora_A.weight"] = a
        merged_tensors[f"{module}.lora_B.weight"] = b

    output = Path(output_dir).expanduser()
    output.mkdir(parents = True, exist_ok = True)
    from safetensors.numpy import save_file

    save_file(merged_tensors, str(output / "adapter_model.safetensors"))

    config = dict(adapters[0].config)
    config["r"] = output_rank
    # Scaling is already baked into the blended delta, so the config must not
    # apply it twice: alpha == r means the loader multiplies by exactly 1.
    config["lora_alpha"] = output_rank
    config["use_rslora"] = False
    base_model = adapters[0].config.get("base_model_name_or_path")
    if base_model:
        config["base_model_name_or_path"] = base_model
    else:
        config.pop("base_model_name_or_path", None)
    (output / "adapter_config.json").write_text(
        json.dumps(config, indent = 2) + "\n", encoding = "utf-8"
    )
    report = {
        "method": method,
        "adapters": [str(adapter.path) for adapter in adapters],
        "weights": list(weights) if weights else [1.0] * len(adapters),
        "modules_merged": len(modules),
        "modules_skipped": skipped,
        "output_rank": output_rank,
        "source_ranks": [adapter.rank for adapter in adapters],
        "output": str(output),
        "notes": [
            "The merged delta is factorised back to rank "
            f"{output_rank} from {sum(adapter.rank for adapter in adapters)} input ranks; "
            "the report's output_rank is what was kept.",
            "Scaling is baked into the delta, so the merged adapter's lora_alpha equals "
            "its rank.",
        ],
    }
    (output / _REPORT_NAME).write_text(
        json.dumps(report, indent = 2) + "\n", encoding = "utf-8"
    )
    return report
