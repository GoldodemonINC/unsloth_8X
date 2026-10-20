# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cross-architecture and cross-scale model merging.

Three layers live here:

1. **Inspection** -- read a checkpoint's ``config.json`` into a :class:`ModelSpec`
   (hidden size, layers, heads, FFN width, vocabulary).
2. **Alignment planning** -- decide how a pair of specs can meet in the middle:
   identical, same architecture at different scales, or different architectures.
   Weight-space blending needs equal shapes, so the plan says which dimension is
   cropped, padded, or projected; MMoE needs no shape agreement because each
   expert keeps its own dimensions behind a router projection.
3. **Blending** -- deterministic tensor math for SLERP, TIES and DARE, plus a
   streaming safetensors writer that never loads a whole checkpoint at once.

The alignment math here is deliberately deterministic and dependency-light:
numpy for planning and blending, safetensors imported lazily for file I/O.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import tensor_io

MERGE_METHODS = ("slerp", "ties", "dare", "linear")

_CONFIG_FILES = ("config.json",)
_WEIGHT_SUFFIXES = (".safetensors",)

# Dimension keys across the architectures Studio can load.
_HIDDEN_KEYS = ("hidden_size", "n_embd", "d_model", "model_dim", "hidden_dim")
_LAYER_KEYS = ("num_hidden_layers", "n_layer", "num_layers", "n_layers", "depth")
_HEAD_KEYS = ("num_attention_heads", "n_head", "num_heads", "attention_heads")
_FFN_KEYS = ("intermediate_size", "n_inner", "ffn_dim", "intermediate_dim")
_VOCAB_KEYS = ("vocab_size", "n_vocab", "vocabulary_size")
_ARCH_KEYS = ("architectures", "model_type", "architectures_name")


def _first_int(config: Mapping[str, Any], keys: Sequence[str]) -> int | None:
    for key in keys:
        value = config.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and int(value) > 0:
            return int(value)
    # Nested text_config (multimodal wrappers).
    nested = config.get("text_config")
    if isinstance(nested, Mapping):
        return _first_int(nested, keys)
    return None


def _architecture_name(config: Mapping[str, Any]) -> str:
    for key in ("model_type", "architecture"):
        value = config.get(key)
        if isinstance(value, str) and value:
            return value
    architectures = config.get("architectures")
    if isinstance(architectures, list) and architectures:
        return str(architectures[0])
    return "unknown"


@dataclass(frozen=True)
class ModelSpec:
    source: str
    architecture: str
    hidden_size: int | None
    num_layers: int | None
    num_heads: int | None
    intermediate_size: int | None
    vocab_size: int | None
    # Rough parameter count, used only for UI labels.
    parameter_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def head_dim(self) -> int | None:
        if self.hidden_size and self.num_heads:
            return max(1, self.hidden_size // self.num_heads)
        return None


def _config_from_dir(directory: Path) -> dict[str, Any] | None:
    for name in _CONFIG_FILES:
        candidate = directory / name
        if candidate.is_file():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
    return None


def _snapshot_dirs(directory: Path) -> list[Path]:
    """A HF cache snapshot directory contains one revision folder."""
    if _config_from_dir(directory) is not None:
        return [directory]
    snapshots = directory / "snapshots"
    if snapshots.is_dir():
        found: list[Path] = []
        for child in sorted(snapshots.iterdir()):
            if child.is_dir() and _config_from_dir(child) is not None:
                found.append(child)
        return found
    return [directory]


# Hub ids arrive as ``org/name`` or ``hf://org/name``; anything that resolves to
# an existing filesystem entry is treated as local first, so a relative path such
# as ``models/llama`` never gets mistaken for a repo.
_HF_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
_HF_PREFIXES = ("hf://", "https://huggingface.co/")
# Everything a merge needs from a snapshot: weights, config, tokenizer metadata.
_HF_SNAPSHOT_PATTERNS = (
    "*.json",
    "*.safetensors",
    "*.model",
    "*.txt",
    "*.py",
    "*.tiktoken",
)


def hf_repo_id(source: str) -> str | None:
    """Return the Hub repo id a source names, or None when it is a filesystem path."""
    text = (source or "").strip()
    for prefix in _HF_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix) :]
    text = text.strip("/")
    if text.count("/") != 1 or _HF_ID_RE.match(text) is None:
        return None
    if Path(text).expanduser().exists():
        return None
    return text


def _hf_config(repo_id: str, token: str | None = None) -> dict[str, Any]:
    """Fetch just config.json, so planning never downloads weights."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:  # pragma: no cover - depends on the install
        raise ValueError(
            "huggingface_hub is not installed in this Studio environment."
        ) from error
    try:
        downloaded = hf_hub_download(repo_id = repo_id, filename = "config.json", token = token)
        return json.loads(Path(downloaded).read_text(encoding = "utf-8"))
    except Exception as error:  # noqa: BLE001 - HTTP/auth errors are user-facing
        raise ValueError(
            f"Could not read config.json for {repo_id!r} from the Hugging Face Hub: {error}"
        ) from error


def resolve_checkpoint_source(source: str, *, token: str | None = None) -> str:
    """Return a local directory for a path or Hub repo, downloading a snapshot if needed.

    Planning only needs config.json (see :func:`probe_model_spec`); running a merge
    needs the weights, so this is the call that materialises the Hub snapshot.
    """
    path = Path(source).expanduser()
    if path.exists():
        return str(path)
    repo = hf_repo_id(source)
    if repo is None:
        raise ValueError(
            f"No checkpoint at {source!r}. Give a local directory, a single .safetensors "
            "file, or a Hugging Face repo id such as unsloth/Llama-3.2-1B."
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:  # pragma: no cover - depends on the install
        raise ValueError(
            "huggingface_hub is not installed in this Studio environment."
        ) from error
    try:
        return str(
            snapshot_download(
                repo_id = repo,
                token = token,
                allow_patterns = list(_HF_SNAPSHOT_PATTERNS),
                ignore_patterns = ["*.gguf", "*.pth", "*.bin"],
            )
        )
    except Exception as error:  # noqa: BLE001 - HTTP/auth errors are user-facing
        raise ValueError(
            f"Could not download {repo!r} from the Hugging Face Hub: {error}"
        ) from error


def probe_model_spec(source: str, *, token: str | None = None) -> ModelSpec:
    """Read a checkpoint's config into a spec, from a local path or the Hub."""
    path = Path(source).expanduser()
    label = str(path)
    config: dict[str, Any] | None = None
    if path.is_file() and path.name == "config.json":
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"Could not read {path}: {error}") from error
    elif path.is_dir():
        for candidate in _snapshot_dirs(path):
            config = _config_from_dir(candidate)
            if config is not None:
                break
    if config is None:
        repo = hf_repo_id(source)
        if repo is not None:
            config = _hf_config(repo, token)
            label = f"hf://{repo}"
    if config is None:
        raise ValueError(
            f"No config.json found for {source!r}. Give a local checkpoint directory "
            "or a Hugging Face repo id."
        )
    hidden = _first_int(config, _HIDDEN_KEYS)
    layers = _first_int(config, _LAYER_KEYS)
    heads = _first_int(config, _HEAD_KEYS)
    ffn = _first_int(config, _FFN_KEYS)
    vocab = _first_int(config, _VOCAB_KEYS)
    params = None
    if hidden and layers:
        # Attention + MLP rough estimate, deterministic and only for display.
        mlp = ffn or (4 * hidden)
        params = layers * (12 * hidden * hidden + 2 * hidden * mlp)
    return ModelSpec(
        source=label,
        architecture=_architecture_name(config),
        hidden_size=hidden,
        num_layers=layers,
        num_heads=heads,
        intermediate_size=ffn,
        vocab_size=vocab,
        parameter_count=params,
    )


def _family(spec: ModelSpec) -> str:
    """Group derived model types (llama, qwen2, ...) into a coarse family."""
    name = spec.architecture.lower()
    for family in ("llama", "qwen", "mistral", "gemma", "phi", "granite", "deepseek"):
        if family in name:
            return family
    return name


@dataclass(frozen=True)
class AlignmentPlan:
    mode: str  # identical | same_family | cross_scale | cross_architecture
    target_hidden: int | None
    target_layers: int | None
    target_intermediate: int | None
    target_vocab: int | None
    hidden_strategy: str  # keep | crop | pad | svd
    layer_strategy: str  # keep | crop | pad
    vocab_strategy: str  # keep | intersect | union_pad
    notes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    mmoe_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_alignment(a: ModelSpec, b: ModelSpec) -> AlignmentPlan:
    """Decide how two checkpoints can be blended or routed.

    Identical specs blend directly. Same-family different-scale merges crop or
    pad to the smaller shape (weight-space averaging cannot invent parameters).
    Different architectures are marked ``cross_architecture``: standard blending
    is still possible after projection, but the plan recommends MMoE because a
    projection between unrelated feature spaces destroys most of the signal.
    """
    notes: list[str] = []
    warnings: list[str] = []
    same_arch = a.architecture == b.architecture
    family_a, family_b = _family(a), _family(b)
    same_family = family_a == family_b and family_a != "unknown"

    dims = {
        "hidden": (a.hidden_size, b.hidden_size),
        "layers": (a.num_layers, b.num_layers),
        "ffn": (a.intermediate_size, b.intermediate_size),
        "vocab": (a.vocab_size, b.vocab_size),
    }
    for label, (left, right) in dims.items():
        if left and right and left != right:
            notes.append(f"{label}: {left} vs {right}")

    if same_arch and all(
        left == right for left, right in dims.values() if left is not None and right is not None
    ):
        mode = "identical"
        target_hidden = a.hidden_size or b.hidden_size
        target_layers = a.num_layers or b.num_layers
        target_ffn = a.intermediate_size or b.intermediate_size
        target_vocab = a.vocab_size or b.vocab_size
        hidden_strategy = layer_strategy = vocab_strategy = "keep"
    else:
        if same_arch or same_family:
            mode = "cross_scale"
        else:
            mode = "cross_architecture"
        # Meet at the smaller dimension: cropping preserves real weights, while
        # padding invents zeros that then dominate the average.
        target_hidden = _min_positive(a.hidden_size, b.hidden_size)
        target_layers = _min_positive(a.num_layers, b.num_layers)
        target_ffn = _min_positive(a.intermediate_size, b.intermediate_size)
        target_vocab = _min_positive(a.vocab_size, b.vocab_size)
        hidden_strategy = "crop" if target_hidden else "keep"
        layer_strategy = "crop" if target_layers else "keep"
        vocab_strategy = "intersect" if target_vocab else "keep"

    if mode == "cross_architecture":
        warnings.append(
            "Different architectures: blending projects unrelated feature spaces. "
            "MMoE keeps each model intact and routes tokens instead."
        )
    if a.num_heads and b.num_heads and a.num_heads != b.num_heads:
        notes.append(f"attention heads: {a.num_heads} vs {b.num_heads}")
    if a.vocab_size and b.vocab_size and a.vocab_size != b.vocab_size:
        warnings.append(
            "Vocabularies differ; the merged tokenizer must be rebuilt (union or intersection) "
            "before the output can tokenize both models' text."
        )
    if target_layers and a.num_layers and b.num_layers and target_layers < max(a.num_layers, b.num_layers):
        warnings.append(
            f"Layer count cropped to {target_layers}; the deeper model's remaining layers are dropped."
        )

    return AlignmentPlan(
        mode=mode,
        target_hidden=target_hidden,
        target_layers=target_layers,
        target_intermediate=target_ffn,
        target_vocab=target_vocab,
        hidden_strategy=hidden_strategy,
        layer_strategy=layer_strategy,
        vocab_strategy=vocab_strategy,
        notes=tuple(notes),
        warnings=tuple(warnings),
        mmoe_required=mode == "cross_architecture",
    )


def _min_positive(*values: int | None) -> int | None:
    present = [v for v in values if v]
    return min(present) if present else None


# ---------------------------------------------------------------------------
# Tensor alignment and blending
# ---------------------------------------------------------------------------


def _resize_axis(array, axis: int, target: int, *, pad_value: float = 0.0):
    """Crop or zero-pad one axis of a numpy array to `target` length."""
    import numpy as np

    current = array.shape[axis]
    if current == target:
        return array
    if current > target:
        start = (current - target) // 2
        return np.take(array, np.arange(start, start + target), axis=axis)
    pad_width = [(0, 0)] * array.ndim
    pad_width[axis] = (0, target - current)
    return np.pad(array, pad_width, constant_values=pad_value)


def align_tensor(array, target_shape: Sequence[int], *, method: str = "crop"):
    """Crop/pad a numpy array to `target_shape`.

    ``method='svd'`` truncates 2-D matrices by rank instead of cropping, which
    keeps the dominant directions of a cross-architecture projection.
    """
    import numpy as np

    arr = np.asarray(array)
    target = tuple(int(t) for t in target_shape)
    if arr.shape == target:
        return arr
    if method == "svd" and arr.ndim == 2 and arr.shape[1] != target[1]:
        rows, cols = min(arr.shape[0], target[0]), min(arr.shape[1], target[1])
        rank = max(1, min(rows, cols))
        try:
            u, s, vh = np.linalg.svd(arr.astype(np.float32), full_matrices=False)
            projected = (u[:, :rank] * s[:rank]) @ vh[:rank, :]
            result = np.zeros(target, dtype=np.float32)
            result[:rows, :cols] = projected[:rows, :cols]
            return result.astype(arr.dtype, copy=False)
        except np.linalg.LinAlgError:
            pass
    result = arr
    for axis, size in enumerate(target):
        if axis < result.ndim:
            result = _resize_axis(result, axis, size)
    if result.ndim < len(target):
        result = np.reshape(result, target)
    return result


def _slerp_pair(a, b, t: float):
    import numpy as np

    # SLERP is defined on the flattened vector, but the result has to come back in
    # the input's shape: writing back a flattened [n] where the model expects [n, m]
    # produced a checkpoint whose tensors no longer load.
    shape = np.shape(a)
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("SLERP needs two tensors of equal size after alignment.")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return ((1 - t) * a + t * b).reshape(shape)
    cosine = float(np.dot(a, b) / (na * nb))
    if abs(cosine) > 0.9995:
        return ((1 - t) * a + t * b).reshape(shape)
    if cosine < 0:
        # Antipodal: the shortest arc is undefined; fall back to linear.
        return ((1 - t) * a + t * b).reshape(shape)
    theta = math.acos(max(-1.0, min(1.0, cosine)))
    sin_theta = math.sin(theta)
    wa = math.sin((1 - t) * theta) / sin_theta
    wb = math.sin(t * theta) / sin_theta
    return (wa * a + wb * b).reshape(shape)


def blend_tensors(
    tensors: Sequence[Any],
    *,
    method: str = "linear",
    weights: Sequence[float] | None = None,
    density: float = 0.5,
    seed: int = 3407,
):
    """Blend aligned tensors with linear, SLERP, TIES or DARE."""
    import numpy as np

    if not tensors:
        raise ValueError("No tensors to blend.")
    arrays = [np.asarray(t, dtype=np.float32) for t in tensors]
    shape = arrays[0].shape
    if any(a.shape != shape for a in arrays):
        raise ValueError("All tensors must be aligned to the same shape before blending.")
    if method == "linear":
        return _linear_blend(arrays, weights)
    if method == "slerp":
        if len(arrays) == 2:
            return _slerp_pair(arrays[0], arrays[1], _normalized_weight(weights, 2)[1])
        result = arrays[0]
        for index in range(1, len(arrays)):
            t = 1.0 / (index + 1)
            result = _slerp_pair(result, arrays[index], t)
        return result
    if method == "ties":
        return _ties_blend(arrays, weights, density)
    if method == "dare":
        return _dare_blend(arrays, weights, density, seed)
    raise ValueError(f"Unsupported merge method {method!r}; expected one of {MERGE_METHODS}.")


def _normalized_weight(weights: Sequence[float] | None, count: int) -> list[float]:
    if weights is None:
        return [1.0 / count] * count
    if len(weights) != count:
        raise ValueError(f"Expected {count} weights, got {len(weights)}.")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Merge weights must sum to a positive number.")
    return [float(w) / total for w in weights]


def _linear_blend(arrays, weights):
    import numpy as np

    norm = _normalized_weight(weights, len(arrays))
    result = np.zeros(arrays[0].shape, dtype=np.float32)
    for array, weight in zip(arrays, norm):
        result += weight * array
    return result


def _ties_blend(arrays, weights, density: float):
    """TIES: trim small deltas, elect a sign, disjoint-mean the survivors."""
    import numpy as np

    norm = _normalized_weight(weights, len(arrays))
    stacked = np.stack(arrays, axis=0)
    reference = stacked.mean(axis=0)
    deltas = stacked - reference
    flat = deltas.reshape(deltas.shape[0], -1)
    count = flat.shape[1]
    keep = max(1, int(round(count * float(density))))
    if keep >= count:
        elected = np.sign(flat.sum(axis=0))
    else:
        threshold = np.partition(np.abs(flat), count - keep, axis=1)[:, count - keep][:, None]
        trimmed = np.where(np.abs(flat) >= threshold, flat, 0.0)
        elected = np.sign(trimmed.sum(axis=0))
    signed = np.where(np.sign(flat) == elected[None, :], flat, 0.0)
    # Weighted mean of the agreeing deltas; divide by the matching mass so a
    # rarely-agreeing direction is not silently attenuated to zero.
    mass = np.zeros(count, dtype=np.float32)
    for row, weight in zip(signed, norm):
        mass += (row != 0).astype(np.float32) * weight
    merged = (signed * np.asarray(norm)[:, None]).sum(axis=0) / np.maximum(mass, 1e-8)
    return (reference + merged.reshape(reference.shape)).astype(np.float32)


def _dare_blend(arrays, weights, density: float, seed: int):
    """DARE: randomly drop deltas with probability 1-p, rescale by 1/p."""
    import numpy as np

    norm = _normalized_weight(weights, len(arrays))
    reference = arrays[0]
    result = reference.astype(np.float32).copy()
    rng = np.random.default_rng(seed)
    scale = 1.0 / max(float(density), 1e-3)
    for index in range(1, len(arrays)):
        delta = arrays[index] - reference
        mask = (rng.random(delta.shape) < float(density)).astype(np.float32)
        result = result + norm[index] * delta * mask * scale
    return result


def merge_state_dicts(
    states: Sequence[Mapping[str, Any]],
    *,
    method: str = "linear",
    weights: Sequence[float] | None = None,
    density: float = 0.5,
    seed: int = 3407,
    alignment: AlignmentPlan | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Blend in-memory state dicts (numpy or torch tensors).

    Tensors that exist in only one model are copied through and reported.
    Returns ``(merged_state, report)``.
    """
    import numpy as np

    if not states:
        raise ValueError("No state dicts to merge.")
    common = set(states[0])
    for state in states[1:]:
        common &= set(state)
    merged: dict[str, Any] = {}
    blended: list[str] = []
    copied: list[str] = []
    skipped: list[str] = []

    for key in sorted(common):
        tensors = [state[key] for state in states]
        arrays = [_to_numpy(t) for t in tensors]
        if any(a.shape != arrays[0].shape for a in arrays):
            if alignment is None:
                skipped.append(key)
                merged[key] = tensors[0]
                continue
            arrays = [_align_for_key(a, key, alignment) for a in arrays]
            if any(a.shape != arrays[0].shape for a in arrays):
                skipped.append(key)
                merged[key] = tensors[0]
                continue
        try:
            merged[key] = blend_tensors(
                arrays, method=method, weights=weights, density=density, seed=seed
            )
            blended.append(key)
        except ValueError:
            skipped.append(key)
            merged[key] = tensors[0]

    for state in states:
        for key in state:
            if key not in common:
                merged.setdefault(key, state[key])
                copied.append(key)

    report = {
        "method": method,
        "blended_tensors": len(blended),
        "copied_tensors": len(set(copied)),
        "skipped_tensors": len(skipped),
        "skipped_keys": skipped[:50],
        "alignment": alignment.to_dict() if alignment else None,
    }
    return merged, report


def _to_numpy(tensor):
    """Float32 numpy view of a tensor, torch or numpy."""
    return tensor_io.to_float32(tensor)


def _align_for_key(array, key: str, plan: AlignmentPlan):
    import numpy as np

    arr = np.asarray(array)
    if arr.ndim == 1:
        target = plan.target_vocab if "embed" in key or "lm_head" in key else arr.shape[0]
        if target and arr.shape[0] != target:
            return _resize_axis(arr, 0, target)
        return arr
    if arr.ndim == 2:
        rows = plan.target_vocab if ("embed" in key or "lm_head" in key) else arr.shape[0]
        # [vocab, hidden] and [intermediate, hidden] both put hidden on axis 1.
        cols = plan.target_hidden or arr.shape[1]
        if arr.shape[0] < arr.shape[1] and plan.target_intermediate and "down" not in key:
            rows = plan.target_intermediate
        if rows and cols and (arr.shape[0] != rows or arr.shape[1] != cols):
            return align_tensor(arr, (rows, cols), method="crop")
    return arr


# ---------------------------------------------------------------------------
# Streaming safetensors I/O
# ---------------------------------------------------------------------------


def _weight_files(directory: Path) -> list[Path]:
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        try:
            mapping = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
            names = sorted({str(v) for v in mapping.values()})
            files = [directory / name for name in names]
            if all(f.is_file() for f in files):
                return files
        except (OSError, ValueError):
            pass
    return sorted(directory.glob("*.safetensors"))


def _open_state(directory: str) -> tuple[dict[str, tuple[Path, str]], dict[str, Any] | None]:
    root = Path(directory)
    if root.is_file() and root.suffix == ".safetensors":
        root = root.parent
    files = _weight_files(root)
    if not files:
        raise ValueError(f"No .safetensors weights found in {directory!r}.")
    entries: dict[str, tuple[Path, str]] = {}
    metadata: dict[str, Any] | None = None
    for file_path in files:
        with tensor_io.open_file(file_path) as handle:
            metadata = metadata or handle.metadata()
            for key in handle.keys():
                entries[key] = (file_path, key)
    return entries, metadata


def _read_tensor(entries, key):
    file_path, inner = entries[key]
    with tensor_io.open_file(file_path) as handle:
        return tensor_io.load_tensor(handle, inner)


def inspect_checkpoint(directory: str) -> dict[str, Any]:
    """Return the spec plus the tensor inventory of a local checkpoint."""
    spec = probe_model_spec(directory)
    try:
        entries, metadata = _open_state(directory)
    except (ValueError, OSError) as error:
        return {"spec": spec.to_dict(), "tensor_count": 0, "error": str(error)}
    return {
        "spec": spec.to_dict(),
        "tensor_count": len(entries),
        "metadata": metadata or {},
    }


def merge_checkpoints(
    model_a: str,
    model_b: str,
    output_dir: str,
    *,
    method: str = "linear",
    weight: float = 0.5,
    density: float = 0.5,
    seed: int = 3407,
    max_shard_bytes: int = 4 * 1024 * 1024 * 1024,
    token: str | None = None,
    policy: Any = None,
    allow_tokenizer_conflict: bool = False,
) -> dict[str, Any]:
    """Stream two safetensors checkpoints into an aligned merged model.

    Sources may be local directories or Hub repo ids (a snapshot is downloaded
    first). Only one tensor pair is resident at a time, so a merge on a CPU host
    peaks at one tensor pair rather than both parents; shards are written as
    they fill, each one atomically.

    Two gates run **before the output folder is created**, so a refused merge
    leaves nothing behind:

    * the loading policy (safetensors-only, remote code opt-in),
    * the tokenizer check (disagreeing bos/eos/pad refuses the merge).

    The returned report is written to ``merge_report.json`` and a
    ``run_manifest.json`` records the run.
    """
    from . import licenses  # noqa: PLC0415
    from . import policy as policy_module  # noqa: PLC0415 - avoids an import cycle
    from . import run_manifest, tokenizer_check  # noqa: PLC0415

    active_policy = policy or policy_module.DEFAULT_POLICY
    model_a = resolve_checkpoint_source(model_a, token = token)
    model_b = resolve_checkpoint_source(model_b, token = token)
    policy_report = [
        policy_module.enforce_source_policy(model_a, active_policy, label = "Model A"),
        policy_module.enforce_source_policy(model_b, active_policy, label = "Model B"),
    ]
    spec_a = probe_model_spec(model_a)
    spec_b = probe_model_spec(model_b)
    plan = plan_alignment(spec_a, spec_b)

    # The quiet failure: weights blend fine, the tokenizer does not.
    tokenizer_report = tokenizer_check.check_before_merge(
        [model_a, model_b], allow_conflict = allow_tokenizer_conflict
    )

    entries_a, _ = _open_state(model_a)
    entries_b, _ = _open_state(model_b)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    keys = sorted(set(entries_a) | set(entries_b))
    shard: dict[str, Any] = {}
    shard_bytes = 0
    shard_index = 1
    weight_map: dict[str, str] = {}
    blended = copied = skipped = 0
    total_bytes = 0

    def flush() -> None:
        nonlocal shard, shard_bytes, shard_index
        if not shard:
            return
        name = f"model-{shard_index:05d}.safetensors"
        # Atomic: a crash or a full disk never leaves a shard the index points at.
        policy_module.atomic_save_tensors(shard, out / name)
        for key in shard:
            weight_map[key] = name
        shard = {}
        shard_bytes = 0
        shard_index += 1

    for key in keys:
        in_a, in_b = key in entries_a, key in entries_b
        if in_a and in_b:
            left = _read_tensor(entries_a, key)
            right = _read_tensor(entries_b, key)
            # Blend in float32, then hand the result back in the dtype the pair
            # was stored in, so a bf16 merge stays bf16 on disk.
            left_array = tensor_io.to_float32(left)
            right_array = tensor_io.to_float32(right)
            if left_array.shape != right_array.shape:
                left_array = _align_for_key(left_array, key, plan)
                right_array = _align_for_key(right_array, key, plan)
            if left_array.shape == right_array.shape and tensor_io.is_floating(left):
                merged = tensor_io.cast_like(
                    blend_tensors(
                        [left_array, right_array],
                        method=method,
                        weights=[1.0 - weight, weight],
                        density=density,
                        seed=seed,
                    ),
                    left,
                )
                blended += 1
            else:
                # Integer bookkeeping tensors (positions, token ids) must not be averaged.
                merged = left
                skipped += 1
        else:
            merged = _read_tensor(entries_a if in_a else entries_b, key)
            copied += 1

        nbytes = tensor_io.nbytes(merged)
        if shard and shard_bytes + nbytes > max_shard_bytes:
            flush()
        shard[key] = merged
        shard_bytes += nbytes
        total_bytes += nbytes
    flush()

    index = {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}
    (out / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8"
    )
    # Carry the base config and tokenizer files through so the result loads.
    for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        for source in (model_a, model_b):
            candidate = Path(source) / name
            if candidate.is_file():
                try:
                    (out / name).write_bytes(candidate.read_bytes())
                except OSError:
                    pass
                break

    # The output inherits one parent's tokenizer files; confirm it really did.
    output_tokenizer = tokenizer_check.verify_output(out, tokenizer_report.fingerprints[0])

    report = {
        "status": "completed",
        "output_dir": str(out),
        "method": method,
        "weight": weight,
        "models": [spec_a.to_dict(), spec_b.to_dict()],
        "alignment": plan.to_dict(),
        "tokenizer": {
            **tokenizer_report.to_dict(),
            "summary": tokenizer_report.summary(),
            "override_used": bool(allow_tokenizer_conflict)
            and tokenizer_report.status == "conflict",
            "output": output_tokenizer,
        },
        "policy": {
            "loading": active_policy.to_dict(),
            "sources": [item["source"] for item in policy_report],
            "remote_code": [
                warning
                for warning in (
                    policy_module.remote_code_warning(model_a),
                    policy_module.remote_code_warning(model_b),
                )
                if warning
            ],
        },
        # What the merged result may be used for is a property of both parents.
        "licenses": licenses.to_report_block([model_a, model_b]),
        "metrics": {
            "blended_tensors": blended,
            "copied_tensors": copied,
            "skipped_tensors": skipped,
            "shards": shard_index - 1,
            "total_bytes": total_bytes,
        },
    }
    manifest = run_manifest.build_manifest(
        "merge",
        config={
            "method": method,
            "weight": weight,
            "density": density,
            "max_shard_bytes": max_shard_bytes,
            "alignment": plan.to_dict(),
        },
        seeds={"seed": seed},
        inputs={
            "model_a": run_manifest.checkpoint_digest(model_a),
            "model_b": run_manifest.checkpoint_digest(model_b),
        },
        artifacts={
            "output_dir": str(out),
            "shards": shard_index - 1,
            "total_bytes": total_bytes,
            "report": "merge_report.json",
        },
        events=[
            {
                "kind": "tokenizer_check",
                "step": 0,
                "status": tokenizer_report.status,
                "override": bool(allow_tokenizer_conflict),
            }
        ],
        versions={"tokenizer_check": 1, "merge_engine": 1},
        policy={"loading": active_policy.to_dict()},
        labeller={"output_tokenizer": output_tokenizer["status"]},
    )
    manifest["tokenizer"] = {
        "status": tokenizer_report.status,
        "conflicts": list(tokenizer_report.conflicts),
        "warnings": list(tokenizer_report.warnings),
        "override_used": bool(allow_tokenizer_conflict)
        and tokenizer_report.status == "conflict",
    }
    manifest["licenses"] = report["licenses"]
    run_manifest.write_manifest(out, manifest)
    report["run_manifest"] = str(out / run_manifest.MANIFEST_NAME)
    report["manifest_id"] = manifest["run_id"]
    policy_module.atomic_write_json(out / "merge_report.json", report)
    return report


# ---------------------------------------------------------------------------
# MMoE planning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoePlan:
    experts: tuple[dict[str, Any], ...]
    top_k: int
    router_hidden: int
    router_temperature: float
    shared_vocab: int | None
    notes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_moe_merge(
    specs: Sequence[ModelSpec],
    *,
    top_k: int = 2,
    router_hidden: int | None = None,
    temperature: float = 1.0,
    layer_range: tuple[int, int] | None = None,
) -> MoePlan:
    """Plan an MMoE fusion: each model becomes an FFN expert behind a router.

    Experts keep their native widths; the router projects each hidden state to
    its own input width, so cross-scale and cross-architecture pairs need no
    weight averaging. A shared vocabulary is required for token input, hence the
    warning when the models disagree.
    """
    if len(specs) < 2:
        raise ValueError("MMoE needs at least two models.")
    if top_k < 1 or top_k > len(specs):
        raise ValueError(f"top_k must be between 1 and the expert count ({len(specs)}).")
    if temperature <= 0:
        raise ValueError("Router temperature must be positive.")

    experts = tuple(
        {
            "index": index,
            "source": spec.source,
            "architecture": spec.architecture,
            "hidden_size": spec.hidden_size,
            "intermediate_size": spec.intermediate_size,
            "num_layers": spec.num_layers,
            "layer_range": list(layer_range) if layer_range else None,
        }
        for index, spec in enumerate(specs)
    )
    target_router = router_hidden or min(
        (spec.hidden_size for spec in specs if spec.hidden_size), default=1024
    )
    vocabs = {spec.vocab_size for spec in specs if spec.vocab_size}
    notes = ["Experts are frozen FFNs; only the router and optional adapters train."]
    warnings: list[str] = []
    if len(vocabs) > 1:
        warnings.append("Vocabularies differ; rebuild a union tokenizer before training the router.")
    if any(spec.hidden_size is None for spec in specs):
        warnings.append("One model had no readable hidden_size; the router width is a guess.")
    return MoePlan(
        experts=experts,
        top_k=top_k,
        router_hidden=int(target_router),
        router_temperature=float(temperature),
        shared_vocab=min(vocabs) if vocabs else None,
        notes=tuple(notes),
        warnings=tuple(warnings),
    )
