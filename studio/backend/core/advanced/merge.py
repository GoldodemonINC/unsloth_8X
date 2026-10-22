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
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import tensor_io

MERGE_METHODS = ("slerp", "ties", "dare", "linear")

_CONFIG_FILES = ("config.json",)
_WEIGHT_SUFFIXES = (".safetensors",)

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

_HF_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
_HF_PREFIXES = ("hf://", "https://huggingface.co/")

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

def resolve_checkpoint_source(
    source: str, *, token: str | None = None, allow_pickle: bool = False
) -> str:
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
                allow_patterns = list(_HF_SNAPSHOT_PATTERNS)
                + (["*.bin"] if allow_pickle else []),
                ignore_patterns = ["*.gguf", "*.pth"]
                + ([] if allow_pickle else ["*.bin"]),
            )
        )
    except Exception as error:  # noqa: BLE001 - HTTP/auth errors are user-facing
        raise ValueError(
            f"Could not download {repo!r} from the Hugging Face Hub: {error}"
        ) from error

def read_model_config(source: str, *, token: str | None = None) -> tuple[str, dict[str, Any]]:
    """Read a checkpoint's config.json, from a local path or the Hub.

    Planning only needs the config, so nothing here downloads weights. Returns the
    label to show in the UI alongside the parsed config.
    """
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
    return label, config

def probe_model_spec(source: str, *, token: str | None = None) -> ModelSpec:
    """Read a checkpoint's config into a spec, from a local path or the Hub."""
    label, config = read_model_config(source, token = token)
    hidden = _first_int(config, _HIDDEN_KEYS)
    layers = _first_int(config, _LAYER_KEYS)
    heads = _first_int(config, _HEAD_KEYS)
    ffn = _first_int(config, _FFN_KEYS)
    vocab = _first_int(config, _VOCAB_KEYS)
    params = None
    if hidden and layers:

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
    mode: str
    target_hidden: int | None
    target_layers: int | None
    target_intermediate: int | None
    target_vocab: int | None
    hidden_strategy: str
    layer_strategy: str
    vocab_strategy: str
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

def common_target_spec(specs: Sequence[ModelSpec]) -> ModelSpec:
    """The shape all parents meet at: the smallest dimension of any of them.

    A pair can use ``plan_alignment(specs[0], specs[1])`` because both sides are
    in that one plan. Three or more parents cannot: parent 2 and parent 3 would
    each be cropped to a different target, so their tensors could never be added
    to the same key. This derives one target for the whole group first, and the
    first parent is aligned to it too, which is also what makes a cross-scale
    pair blend instead of silently copying the larger parent.

    The architecture comes from the first parent: it decides whether a plan is
    cross-architecture (MMoE territory), not the shape.
    """
    first = specs[0]
    return ModelSpec(
        source=first.source,
        architecture=first.architecture,
        hidden_size=_min_positive(*(spec.hidden_size for spec in specs)),
        num_layers=_min_positive(*(spec.num_layers for spec in specs)),
        num_heads=_min_positive(*(spec.num_heads for spec in specs)),
        intermediate_size=_min_positive(*(spec.intermediate_size for spec in specs)),
        vocab_size=_min_positive(*(spec.vocab_size for spec in specs)),
        parameter_count=None,
    )

def choose_template_index(specs: Sequence[ModelSpec]) -> int:
    """The parent whose architecture the merged checkpoint keeps.

    Weight-space blending can only meet at one shape, and the smallest parent is
    the one whose dimensions the others can be cropped down to without inventing
    parameters. The template decides the copied config and tokenizer too, which
    is what keeps the written tensors and the written config consistent: a
    per-axis minimum can take a wide model's attention rows and pair them with a
    narrow model's `head_dim`, which no loader accepts. Ties keep the first.
    """

    def rank(index: int) -> tuple[float, float, float, float, int]:
        spec = specs[index]
        return (
            spec.hidden_size or math.inf,
            spec.num_layers or math.inf,
            spec.intermediate_size or math.inf,
            spec.vocab_size or math.inf,
            index,
        )

    return min(range(len(specs)), key = rank)


def _meet_strategy(target: int | None, current: int | None) -> str:
    if not target or not current or target == current:
        return "keep"
    return "crop" if current > target else "pad"


def plan_alignment_many(specs: Sequence[ModelSpec]) -> list[AlignmentPlan]:
    """One plan per parent, every one resolving to the template parent's shape."""
    if not specs:
        raise ValueError("A merge needs at least two checkpoints.")
    template = specs[choose_template_index(specs)]
    cross_scale = any(
        spec.hidden_size != template.hidden_size
        or spec.num_layers != template.num_layers
        or spec.intermediate_size != template.intermediate_size
        or spec.vocab_size != template.vocab_size
        for spec in specs
    )
    scale_warning = (
        "Cross-scale merge: the smallest parent's architecture is the target, so every wider "
        "tensor is cropped to it. The output is loadable and internally consistent, but the "
        "larger parent's extra dimensions are discarded -- evaluate quality before use."
    )
    plans: list[AlignmentPlan] = []
    for spec in specs:
        pair = plan_alignment(template, spec)
        plans.append(
            replace(
                pair,
                target_hidden = template.hidden_size or pair.target_hidden,
                target_layers = template.num_layers or pair.target_layers,
                target_intermediate = template.intermediate_size or pair.target_intermediate,
                target_vocab = template.vocab_size or pair.target_vocab,
                hidden_strategy = _meet_strategy(template.hidden_size, spec.hidden_size),
                layer_strategy = _meet_strategy(template.num_layers, spec.num_layers),
                vocab_strategy = _meet_strategy(template.vocab_size, spec.vocab_size),
                warnings = pair.warnings + ((scale_warning,) if cross_scale else ()),
            )
        )
    return plans

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

        norm = _normalized_weight(weights, len(arrays))
        result = arrays[0]
        taken = norm[0]
        for index in range(1, len(arrays)):
            taken += norm[index]
            t = norm[index] / taken if taken > 0 else 1.0 / (index + 1)
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

    mass = np.zeros(count, dtype=np.float32)
    for row, weight in zip(signed, norm):
        mass += (row != 0).astype(np.float32) * weight
    merged = (signed * np.asarray(norm)[:, None]).sum(axis=0) / np.maximum(mass, 1e-8)
    return (reference + merged.reshape(reference.shape)).astype(np.float32)

def dare_mask(flat_start: int, count: int, seed: int, density: float, salt: int = 0):
    """A deterministic DARE drop mask addressed by flat element index.

    The mask is a pure function of ``(seed, salt, index)`` rather than the next
    draw from a stateful generator, which is what lets the streaming path build
    the mask for the rows it is holding and get exactly the values the
    whole-tensor path would have produced for those same elements. ``salt`` is
    the parent index, so every parent still drops a different subset.
    """
    import numpy as np

    index = np.arange(int(flat_start), int(flat_start) + int(count), dtype = np.uint64)
    stream = np.uint64((int(seed) + int(salt) * 0x9E3779B1) & 0xFFFFFFFFFFFFFFFF)
    with np.errstate(over = "ignore"):
        state = index + stream * np.uint64(0x9E3779B97F4A7C15)
        state = (state ^ (state >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        state = (state ^ (state >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        state = state ^ (state >> np.uint64(31))
    uniform = (state >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)
    return (uniform < float(density)).astype(np.float32)

def _dare_blend(arrays, weights, density: float, seed: int):
    """DARE: randomly drop deltas with probability 1-p, rescale by 1/p."""
    import numpy as np

    norm = _normalized_weight(weights, len(arrays))
    reference = arrays[0]
    result = reference.astype(np.float32).copy()
    scale = 1.0 / max(float(density), 1e-3)
    count = int(reference.size)
    for index in range(1, len(arrays)):
        delta = arrays[index] - reference
        mask = dare_mask(0, count, seed, density, salt = index).reshape(delta.shape)
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

# FFN and attention projections carry different axes, and the shape alone cannot tell them
# apart: Qwen3's q_proj is [heads * head_dim, hidden] with more rows than hidden, which the old
# shape heuristic mistook for an up_proj and padded to the FFN width.
_FFN_DOWN_NAMES = ("down_proj", "ffn_down", "fc2", "w2")
_FFN_UP_NAMES = ("up_proj", "gate_proj", "ffn_up", "ffn_gate", "fc1", "w1", "w3")
_ATTN_OUT_NAMES = ("o_proj", "out_proj", "attn_output")
_ATTN_IN_NAMES = ("q_proj", "k_proj", "v_proj", "qkv_proj", "query", "key", "value")


def _aligned_shape(shape: Sequence[int], key: str, plan: AlignmentPlan) -> tuple[int, ...]:
    """The shape `_align_for_key` would produce, without touching any data.

    Keyed by projection name, not by which axis happens to equal the hidden size.
    """
    shape = tuple(int(dim) for dim in shape)
    lowered = key.lower()
    is_tied = "embed" in lowered or "lm_head" in lowered
    if len(shape) == 1:
        return (plan.target_vocab or shape[0],) if is_tied else shape
    if len(shape) != 2:
        return shape
    if is_tied:
        return (plan.target_vocab or shape[0], plan.target_hidden or shape[1])
    if "router" in lowered or ".gate." in lowered:
        return shape
    if any(name in lowered for name in _FFN_DOWN_NAMES):
        return (plan.target_hidden or shape[0], plan.target_intermediate or shape[1])
    if any(name in lowered for name in _FFN_UP_NAMES):
        return (plan.target_intermediate or shape[0], plan.target_hidden or shape[1])
    if any(name in lowered for name in _ATTN_OUT_NAMES):
        return (plan.target_hidden or shape[0], shape[1])
    if any(name in lowered for name in _ATTN_IN_NAMES):
        return (shape[0], plan.target_hidden or shape[1])
    return shape

def _align_for_key(array, key: str, plan: AlignmentPlan):
    import numpy as np

    arr = np.asarray(array)
    target = _aligned_shape(arr.shape, key, plan)
    if target != arr.shape:
        return align_tensor(arr, target, method="crop")
    return arr

def _indexed_files(directory: Path, index_name: str, suffix: str) -> list[Path]:
    index = directory / index_name
    if index.is_file():
        try:
            mapping = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
            names = sorted({str(v) for v in mapping.values()})
            files = [directory / name for name in names]
            if files and all(f.is_file() for f in files):
                return files
        except (OSError, ValueError):
            pass
    return sorted(directory.glob(f"*{suffix}"))

def _weight_files(directory: Path, *, include_pickle: bool = False) -> list[Path]:
    files = _indexed_files(directory, "model.safetensors.index.json", ".safetensors")
    if files or not include_pickle:
        return files
    return _indexed_files(directory, "pytorch_model.bin.index.json", ".bin")

_Entry = tuple[Path, str, tuple[int, ...], str]

_PICKLE_STATES: dict[str, dict[str, Any]] = {}

_TORCH_DTYPE_NAMES = {
    "float16": "F16",
    "half": "F16",
    "bfloat16": "BF16",
    "float32": "F32",
    "float": "F32",
    "float64": "F64",
    "double": "F64",
    "float8_e4m3fn": "F8_E4M3",
    "float8_e5m2": "F8_E5M2",
}

def _pickle_dtype_name(dtype: Any) -> str:
    """A safetensors-style dtype name for a torch dtype, so one predicate covers both."""
    name = str(dtype).rsplit(".", 1)[-1].lower()
    return _TORCH_DTYPE_NAMES.get(name, name.upper())

def _pickle_state(path: Path) -> dict[str, Any]:
    """A checkpoint's tensors, memory-mapped so rows are paged in on demand.

    ``mmap=True`` is what keeps a pickle on the same footing as safetensors: the
    unpickled tensors point at the file instead of copying it into RAM. Entries
    stay cached for the run because each row read would otherwise re-unpickle the
    whole dictionary.
    """
    torch = tensor_io.torch_module()
    if torch is None:
        raise ValueError(
            f"Cannot read {path.name}: pickle checkpoints need torch, and this "
            "environment does not have it. Convert the checkpoint to safetensors."
        )
    resolved = str(Path(path).resolve())
    state = _PICKLE_STATES.get(resolved)
    if state is not None:
        return state
    try:
        loaded = torch.load(resolved, map_location="cpu", mmap=True, weights_only=True)
    except TypeError:
        loaded = torch.load(resolved, map_location="cpu", weights_only=True)
    except Exception as error:  # noqa: BLE001 - corrupt or unsafe pickles are user-facing
        raise ValueError(f"Could not read {path}: {error}") from error
    if not isinstance(loaded, Mapping):
        raise ValueError(f"{path} does not hold a state dict.")
    state = {str(key): value for key, value in loaded.items() if hasattr(value, "shape")}
    _PICKLE_STATES[resolved] = state
    return state

def release_pickle_states() -> None:
    """Drop the cached mmaps so a finished merge does not pin the files."""
    _PICKLE_STATES.clear()

def _open_state(
    directory: str, *, allow_pickle: bool = False
) -> tuple[dict[str, _Entry], dict[str, Any] | None]:
    root = Path(directory)
    if root.is_file() and root.suffix in (".safetensors", ".bin"):
        root = root.parent
    files = _weight_files(root, include_pickle = allow_pickle)
    if not files:
        if not allow_pickle and _weight_files(root, include_pickle = True):
            raise ValueError(
                f"{directory!r} only carries pickle weights; enable "
                "allow_non_safetensors to read them."
            )
        raise ValueError(f"No .safetensors weights found in {directory!r}.")
    entries: dict[str, _Entry] = {}
    metadata: dict[str, Any] | None = None
    for file_path in files:
        if file_path.suffix == ".bin":
            for key, tensor in _pickle_state(file_path).items():
                entries[key] = (
                    file_path,
                    key,
                    tuple(int(dim) for dim in tensor.shape),
                    _pickle_dtype_name(tensor.dtype),
                )
            continue
        with tensor_io.open_file(file_path) as handle:
            metadata = metadata or handle.metadata()
            for key in handle.keys():
                entries[key] = (
                    file_path,
                    key,
                    tensor_io.shape_of(handle, key),
                    tensor_io.dtype_name(handle, key),
                )
    return entries, metadata

def _read_tensor(entries, key):
    file_path, inner, _, _ = entries[key]
    if file_path.suffix == ".bin":
        return _pickle_state(file_path)[inner]
    with tensor_io.open_file(file_path) as handle:
        return tensor_io.load_tensor(handle, inner)

def _disk_shape(entries, key: str) -> tuple[int, ...]:
    return entries[key][2]

def _read_slice(entries, key: str, start: int, stop: int):
    """Read rows ``start:stop`` of a tensor; safetensors and mmap both stay lazy."""
    file_path, inner, _, _ = entries[key]
    if file_path.suffix == ".bin":
        return _pickle_state(file_path)[inner][start:stop]
    with tensor_io.open_file(file_path) as handle:
        return handle.get_slice(inner)[start:stop]

CHUNK_BUDGET_BYTES = 32 * 1024 * 1024

def chunk_rows(shape: Sequence[int], parents: int) -> int:
    """Rows per streaming step so one chunk costs about CHUNK_BUDGET_BYTES."""
    row_elements = 1
    for dim in tuple(shape)[1:]:
        row_elements *= max(int(dim), 1)
    per_row = row_elements * 4 * max(int(parents), 1)
    rows = max(1, CHUNK_BUDGET_BYTES // max(per_row, 1))
    return min(rows, max(int(shape[0]), 1))

def _can_stream(
    entries,
    key: str,
    present: Sequence[int],
    target_shape: Sequence[int],
    method: str = "linear",
) -> bool:
    """Whether this key can be blended a chunk of rows at a time.

    Alignment must be the identity for every owner: cropping shifts rows and
    padding invents them, so a chunk index would no longer mean the same rows in
    every parent. Every parent matching the template shape exactly is that case.

    N-way SLERP is the one method that cannot stream, because its running result
    is the tensor the next pair's angle is measured against: the whole result has
    to exist before the fold can continue. Two-parent SLERP is elementwise once
    the global norms are known, and streams like the rest.
    """
    target = tuple(int(dim) for dim in target_shape)
    if not target or target[0] <= 0:
        return False
    if method == "slerp" and len(present) != 2:
        return False
    if any(_disk_shape(entries[index], key) != target for index in present):
        return False
    return all(
        tensor_io.is_float_dtype_name(entries[index][key][3]) for index in present
    )

def _row_windows(shape: Sequence[int], rows: int):
    """Every ``(start, stop)`` row window of a tensor, in order."""
    total = int(shape[0])
    for start in range(0, total, rows):
        yield start, min(start + rows, total)

def _stream_chunks(entries, key: str, present: Sequence[int], rows: int):
    """``(start, stop, stored slices)`` -- one row window per parent."""
    shape = _disk_shape(entries[present[0]], key)
    for start, stop in _row_windows(shape, rows):
        yield start, stop, [
            _read_slice(entries[index], key, start, stop) for index in present
        ]

def _stream_linear(entries, key, present, weights, shape, rows):
    """Linear blend of one tensor, one row chunk at a time.

    Same arithmetic as :func:`_linear_blend`, but a parent's tensor is never
    fully resident: each chunk is read, promoted, weighted and written into the
    output slice before the next chunk is touched.
    """
    output = None
    for start, stop, raw in _stream_chunks(entries, key, present, rows):

        if output is None:
            output = tensor_io.empty_like_shape(shape, raw[0])
        chunks = [tensor_io.to_float32(slice_) for slice_ in raw]
        output[start:stop] = tensor_io.cast_like(
            _linear_blend(chunks, weights), output
        )
    if output is None:  # pragma: no cover - zero-row tensors are not real
        raise ValueError(f"{key!r} has no rows to blend.")
    return output

def _slerp_coefficients(na: float, nb: float, dot: float, t: float):
    """``(wa, wb)`` for one arc, or None when the pair falls back to linear.

    The same decision tree as :func:`_slerp_pair`, with the norms handed in: the
    streaming caller accumulates them over the rows it has already read.
    """
    if na == 0 or nb == 0:
        return None
    cosine = float(dot / (na * nb))
    if abs(cosine) > 0.9995 or cosine < 0:
        return None
    theta = math.acos(max(-1.0, min(1.0, cosine)))
    sin_theta = math.sin(theta)
    if sin_theta == 0:
        return None
    return (math.sin((1 - t) * theta) / sin_theta, math.sin(t * theta) / sin_theta)

def _stream_slerp(entries, key, present, weights, shape, rows):
    """Two-parent SLERP, one row chunk at a time.

    The arc between two tensors is described by ``|a|``, ``|b|`` and ``a.b``, all
    of which accumulate over chunks. Once those three numbers exist the blend is
    elementwise, so the second pass only ever holds one chunk per parent.
    """
    import numpy as np

    dot = na = nb = 0.0
    for _start, _stop, raw in _stream_chunks(entries, key, present, rows):
        first = tensor_io.to_float32(raw[0]).reshape(-1)
        second = tensor_io.to_float32(raw[1]).reshape(-1)
        dot += float(np.dot(first, second))
        na += float(np.dot(first, first))
        nb += float(np.dot(second, second))
    t = _normalized_weight(weights, 2)[1]
    coefficients = _slerp_coefficients(math.sqrt(na), math.sqrt(nb), dot, t)
    output = None
    for start, stop, raw in _stream_chunks(entries, key, present, rows):
        if output is None:
            output = tensor_io.empty_like_shape(shape, raw[0])
        first = tensor_io.to_float32(raw[0])
        second = tensor_io.to_float32(raw[1])
        if coefficients is None:
            blended = (1 - t) * first + t * second
        else:
            blended = coefficients[0] * first + coefficients[1] * second
        output[start:stop] = tensor_io.cast_like(blended, output)
    return output

_TIES_BUCKETS = 1 << 16

def _magnitude_keys(values):
    """Order-preserving uint32 keys for ``|values|``: float bits compare like ints."""
    import numpy as np

    return np.abs(np.asarray(values, dtype = np.float32)).reshape(-1).view(np.uint32)

def _ties_trim_thresholds(entries, key, present, density, count: int, rows):
    """The exact keep-th largest ``|delta|`` per parent, in two streaming passes.

    :func:`_ties_blend` trims with ``np.partition(...)[:, count - keep]``, i.e. the
    keep-th largest magnitude over the whole tensor. Sampling a threshold instead
    would change which deltas survive and therefore the merged weights, so this
    finds the exact same element: a non-negative IEEE-754 float orders exactly
    like its integer bit pattern, so a two-level histogram over those 32 bits
    selects it with 65536 counters per level instead of a copy per parent.
    """
    import numpy as np

    keep = max(1, int(round(count * float(density))))
    if keep >= count:
        return None
    parents = len(present)
    high = np.zeros((parents, _TIES_BUCKETS), dtype = np.int64)
    for _start, _stop, raw in _stream_chunks(entries, key, present, rows):
        chunks = [tensor_io.to_float32(slice_) for slice_ in raw]
        reference = np.stack(chunks, axis = 0).mean(axis = 0)
        for index, chunk in enumerate(chunks):
            keys = _magnitude_keys(chunk - reference)
            high[index] += np.bincount(
                (keys >> np.uint32(16)).astype(np.int64), minlength = _TIES_BUCKETS
            )
    buckets: list[int] = []
    ranks: list[int] = []
    for index in range(parents):
        suffix = np.cumsum(high[index][::-1])[::-1]
        bucket = int(np.max(np.nonzero(suffix >= keep)[0]))
        above = int(suffix[bucket + 1]) if bucket + 1 < _TIES_BUCKETS else 0
        buckets.append(bucket)
        ranks.append(keep - above)

    low = np.zeros((parents, _TIES_BUCKETS), dtype = np.int64)
    for _start, _stop, raw in _stream_chunks(entries, key, present, rows):
        chunks = [tensor_io.to_float32(slice_) for slice_ in raw]
        reference = np.stack(chunks, axis = 0).mean(axis = 0)
        for index, chunk in enumerate(chunks):
            keys = _magnitude_keys(chunk - reference)
            in_bucket = np.asarray(keys >> np.uint32(16)).astype(np.int64) == buckets[index]
            if not bool(in_bucket.any()):
                continue
            low[index] += np.bincount(
                (keys[in_bucket] & np.uint32(0xFFFF)).astype(np.int64),
                minlength = _TIES_BUCKETS,
            )
    thresholds: list[float] = []
    for index in range(parents):
        suffix = np.cumsum(low[index][::-1])[::-1]
        low_bucket = int(np.max(np.nonzero(suffix >= ranks[index])[0]))
        bits = (buckets[index] << 16) | low_bucket
        thresholds.append(float(np.array([bits], dtype = np.uint32).view(np.float32)[0]))
    return thresholds

def _stream_ties(entries, key, present, weights, density, shape, rows):
    """TIES, one row chunk at a time, with the whole-tensor trim threshold.

    Reference, deltas, sign election and the disjoint weighted mean are all
    elementwise; only the trim threshold is global, and the first pass already
    found it exactly.
    """
    import numpy as np

    count = int(math.prod(int(dim) for dim in shape))
    thresholds = _ties_trim_thresholds(entries, key, present, density, count, rows)
    norm = np.asarray(_normalized_weight(weights, len(present)), dtype = np.float32)
    output = None
    for start, stop, raw in _stream_chunks(entries, key, present, rows):
        if output is None:
            output = tensor_io.empty_like_shape(shape, raw[0])
        chunks = [tensor_io.to_float32(slice_) for slice_ in raw]
        reference = np.stack(chunks, axis = 0).mean(axis = 0)
        flat = np.stack([chunk - reference for chunk in chunks], axis = 0).reshape(
            len(chunks), -1
        )
        if thresholds is None:
            trimmed = flat
        else:
            trimmed = np.where(
                np.stack(
                    [
                        np.abs(flat[index]) >= thresholds[index]
                        for index in range(len(chunks))
                    ],
                    axis = 0,
                ),
                flat,
                0.0,
            )
        elected = np.sign(trimmed.sum(axis = 0))
        signed = np.where(np.sign(flat) == elected[None, :], flat, 0.0)
        mass = np.zeros(flat.shape[1], dtype = np.float32)
        for row, weight in zip(signed, norm):
            mass += (row != 0).astype(np.float32) * weight
        merged = (signed * norm[:, None]).sum(axis = 0) / np.maximum(mass, 1e-8)
        blended = (reference + merged.reshape(reference.shape)).astype(np.float32)
        output[start:stop] = tensor_io.cast_like(blended, output)
    return output

def _stream_dare(entries, key, present, weights, density, seed, shape, rows):
    """DARE, one row chunk at a time, with a counter-addressed drop mask.

    The mask is keyed by flat element index (see :func:`dare_mask`), so a chunk
    gets exactly the mask the whole-tensor path would have drawn for it.
    """
    import numpy as np

    norm = _normalized_weight(weights, len(present))
    scale = 1.0 / max(float(density), 1e-3)
    row_elements = max(1, int(math.prod(int(dim) for dim in tuple(shape)[1:])))
    output = None
    for start, stop, raw in _stream_chunks(entries, key, present, rows):
        if output is None:
            output = tensor_io.empty_like_shape(shape, raw[0])
        reference = tensor_io.to_float32(raw[0])
        result = reference.astype(np.float32).copy()
        for offset in range(1, len(present)):
            delta = tensor_io.to_float32(raw[offset]) - reference
            mask = dare_mask(
                start * row_elements, int(delta.size), seed, density, salt = offset
            ).reshape(delta.shape)
            result = result + norm[offset] * delta * mask * scale
        output[start:stop] = tensor_io.cast_like(result, output)
    return output

def _stream_blend(entries, key, present, weights, *, method, density, seed):
    """Blend one tensor a row chunk at a time, or None when that is not possible.

    None is only N-way SLERP (see :func:`_can_stream`); the caller then takes the
    whole-tensor path for that key.
    """
    shape = _disk_shape(entries[present[0]], key)
    rows = chunk_rows(shape, len(present))
    if method == "slerp":
        if len(present) != 2:
            return None
        return _stream_slerp(entries, key, present, weights, shape, rows)
    if method == "ties":
        return _stream_ties(entries, key, present, weights, density, shape, rows)
    if method == "dare":
        return _stream_dare(entries, key, present, weights, density, seed, shape, rows)
    return _stream_linear(entries, key, present, weights, shape, rows)

def _merge_key(
    entries,
    key,
    present,
    target_shape,
    template_index,
    method,
    normalized,
    density,
    seed,
):
    """One key's merged tensor, plus how it was produced.

    Every owner is cropped or padded to the template's exact shape, so the result
    always matches the config and tokenizer copied from that same parent. The
    template tensor also fixes the output dtype. ``streamed`` and ``blended`` are
    both "averaged"; the split lets the report say how much stayed inside the RAM
    budget instead of materialising a float32 copy per parent.
    """
    weights = [normalized[index] for index in present]
    template_position = present.index(template_index)
    if len(present) > 1 and _can_stream(entries, key, present, target_shape, method):
        streamed = _stream_blend(
            entries,
            key,
            present,
            weights,
            method = method,
            density = density,
            seed = seed,
        )
        if streamed is not None:
            return streamed, "streamed"
    if len(present) > 1:
        tensors = [_read_tensor(entries[index], key) for index in present]
        template_tensor = tensors[template_position]
        try:
            aligned = [
                align_tensor(tensor_io.to_float32(tensor), target_shape)
                for tensor in tensors
            ]
        except (ValueError, TypeError):
            return template_tensor, "skipped"

        if all(array.shape == tuple(target_shape) for array in aligned) and tensor_io.is_floating(
            template_tensor
        ):
            return (
                tensor_io.cast_like(
                    blend_tensors(
                        aligned,
                        method = method,
                        weights = weights,
                        density = density,
                        seed = seed,
                    ),
                    template_tensor,
                ),
                "blended",
            )

        return template_tensor, "skipped"
    return _read_tensor(entries[present[0]], key), "copied"

def shard_budget_bytes(requested: int) -> int:
    """How much merged output may sit in RAM before a shard is flushed.

    The 4 GiB default is a disk-shard choice, not a memory one, and on an 8 GB
    CPU host it would buffer a whole checkpoint before writing anything. Cap it
    at a quarter of free RAM, with a floor so a flush loop never degenerates
    into one write per tensor.
    """
    try:
        from . import splitter  # noqa: PLC0415 - lazy, keeps this import cheap

        free = int(splitter.probe_memory().system_ram_free_bytes or 0)
    except Exception:  # noqa: BLE001 - a probe failure must not block a merge
        return int(requested)
    if free <= 0:
        return int(requested)
    return max(64 * 1024 * 1024, min(int(requested), int(free * 0.25)))

def inspect_checkpoint(directory: str) -> dict[str, Any]:
    """Return the spec plus the tensor inventory of a local checkpoint."""
    spec = probe_model_spec(directory)
    try:
        entries, metadata = _open_state(directory, allow_pickle = True)
    except (ValueError, OSError) as error:
        return {"spec": spec.to_dict(), "tensor_count": 0, "error": str(error)}
    return {
        "spec": spec.to_dict(),
        "tensor_count": len(entries),
        "metadata": metadata or {},
    }

MAX_MERGE_MODELS = 6

MAX_MERGE_MODELS_CPU = 3

def host_has_gpu() -> bool:
    """Whether a CUDA device is visible to this process.

    Merge itself runs on the CPU either way; this only decides the parent cap and
    the tier a job reports. Apple MPS reports False here, which is honest: an MPS
    host merges on the CPU like any other.
    """
    torch = tensor_io.torch_module()
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - a broken driver must not look like a GPU
        return False

def free_ram_bytes() -> int:
    """Free system RAM, or 0 when the probe cannot answer."""
    try:
        from . import splitter  # noqa: PLC0415 - lazy, keeps this import cheap

        return int(splitter.probe_memory().system_ram_free_bytes or 0)
    except Exception:  # noqa: BLE001 - a probe failure must not block planning
        return 0

def largest_tensor_bytes(directory: Path) -> int:
    """Bytes of the largest single tensor, the floor a streaming merge needs."""
    best = 0
    for file_path in _weight_files(directory, include_pickle = True):
        if file_path.suffix == ".bin":
            try:
                for tensor in _pickle_state(file_path).values():
                    best = max(best, tensor_io.nbytes(tensor))
            except (ValueError, OSError):
                continue
            continue
        try:
            with tensor_io.open_file(file_path) as handle:
                for key in handle.keys():
                    shape = tensor_io.shape_of(handle, key)
                    best = max(
                        best,
                        int(math.prod(shape)) * tensor_io.dtype_itemsize(handle, key),
                    )
        except (ValueError, OSError):
            continue
    return best

def checkpoint_weight_bytes(source: str) -> int:
    """Bytes of weight files at a source, or 0 when it is not a local path."""
    root = Path(source).expanduser()
    if root.is_file():
        try:
            return int(root.stat().st_size)
        except OSError:
            return 0
    if not root.is_dir():
        return 0
    total = 0
    for file_path in _weight_files(root, include_pickle = True):
        try:
            total += int(file_path.stat().st_size)
        except OSError:
            pass
    return total

def merge_model_limit(
    sources: Sequence[str] | None = None,
    *,
    has_gpu: bool | None = None,
    free_bytes: int | None = None,
) -> int:
    """How many parents this host can stream at once.

    A GPU host keeps the flat six. A CPU host is capped by RAM rather than by the
    absence of a GPU: the peak is one output tensor, one row chunk of about
    ``CHUNK_BUDGET_BYTES`` per parent, and the shard buffer, so a small checkpoint
    can merge six parents on the same laptop that can only manage two of a large
    one. With no sources to measure, three is the answer the page plans against.
    """
    if has_gpu is None:
        has_gpu = host_has_gpu()
    if has_gpu:
        return MAX_MERGE_MODELS
    if not sources:
        return MAX_MERGE_MODELS_CPU
    free = free_ram_bytes() if free_bytes is None else int(free_bytes)
    if free <= 0:
        return MAX_MERGE_MODELS_CPU
    largest = 0
    for source in sources:
        root = Path(source).expanduser()
        if root.is_file():
            root = root.parent
        if not root.is_dir():
            continue
        try:
            largest = max(largest, largest_tensor_bytes(root))
        except (ValueError, OSError):
            continue

    headroom = int(free * 0.6) - largest
    if headroom <= 0:
        return 2
    allowance = int(headroom // (CHUNK_BUDGET_BYTES * 2))
    return max(2, min(MAX_MERGE_MODELS, allowance))

def max_merge_models(has_gpu: bool | None = None, sources: Sequence[str] | None = None) -> int:
    """The parent cap for this host: six with a GPU, RAM-derived on a CPU host."""
    return merge_model_limit(sources, has_gpu = has_gpu)

def check_model_count(count: int, limit: int) -> None:
    if count < 2:
        raise ValueError("A merge needs at least two checkpoints.")
    if count <= limit:
        return
    if limit >= MAX_MERGE_MODELS:
        raise ValueError(
            f"At most {MAX_MERGE_MODELS} checkpoints can be merged at once; got {count}."
        )
    raise ValueError(
        f"This host can merge at most {limit} checkpoints per run (got {count}). "
        "The cap follows free RAM: every streaming parent holds one row chunk "
        "beside the output tensor. Merge them in two passes, free some memory, "
        "or run this on a GPU host for up to "
        f"{MAX_MERGE_MODELS}."
    )

def _verify_merged_against_template(
    out: Path, template: Mapping[str, _Entry], template_config: dict[str, Any] | None
) -> None:
    """Fail the merge unless the written inventory matches the template exactly.

    The config and tokenizer are copied from the template, so one missing, extra
    or reshaped tensor makes the folder load with random weights or not at all.
    Reading shard headers costs milliseconds and turns a silently bad merge into
    a named error before a report calls it completed.
    """
    written, _ = _open_state(str(out))
    expected = {key: entry[2] for key, entry in template.items()}
    actual = {key: entry[2] for key, entry in written.items()}
    if expected != actual:
        missing = sorted(set(expected) - set(actual))[:4]
        extra = sorted(set(actual) - set(expected))[:4]
        wrong = sorted(
            key for key in set(expected) & set(actual) if expected[key] != actual[key]
        )[:4]
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        if wrong:
            details.append("wrong shape " + ", ".join(wrong))
        raise ValueError(
            "The merged checkpoint does not match its template parent"
            + (f" ({'; '.join(details)})" if details else "")
            + "; refusing to report a checkpoint that would load with mismatched tensors."
        )
    if template_config is not None:
        try:
            written_config = json.loads((out / "config.json").read_text(encoding = "utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"The merged config.json is not readable: {error}") from error
        if written_config != template_config:
            raise ValueError(
                "The merged config.json does not match the template parent's config; "
                "the weights and the architecture would disagree."
            )


def merge_checkpoints_many(
    sources: Sequence[str],
    output_dir: str,
    *,
    method: str = "linear",
    weights: Sequence[float] | None = None,
    density: float = 0.5,
    seed: int = 3407,
    max_shard_bytes: int = 4 * 1024 * 1024 * 1024,
    token: str | None = None,
    policy: Any = None,
    allow_tokenizer_conflict: bool = False,
    progress: Any = None,
) -> dict[str, Any]:
    """Stream 2 to 6 safetensors checkpoints into one aligned merged model.

    Every source is aligned against the first one, so a N-way merge never has to
    agree on shapes pairwise. Every method reads equal-shaped tensors in row
    chunks and flushes shards against a RAM-derived budget, so a CPU host peaks at
    one output tensor plus one chunk per parent rather than every checkpoint at
    once. The parent cap follows the host's free RAM on a CPU and is six on a GPU.

    Weights are normalized over the sources that actually own a tensor, so a key
    missing from one parent reweights the survivors instead of being blended as
    if that parent had a zero tensor there.

    ``progress`` is called with ``{"stage", "done", "total", ...}`` dictionaries:
    the merge page polls that through the job record, the way training reports
    its own step.
    """
    from . import licenses  # noqa: PLC0415
    from . import policy as policy_module  # noqa: PLC0415 - avoids an import cycle
    from . import run_manifest, tokenizer_check  # noqa: PLC0415

    count = len(sources)
    has_gpu = host_has_gpu()
    release_pickle_states()
    normalized = _normalized_weight(weights, count)

    shard_buffer_bytes = shard_budget_bytes(max_shard_bytes)

    def report(stage: str, **extra: Any) -> None:
        if progress is None:
            return
        try:
            progress({"stage": stage, **extra})
        except Exception:  # noqa: BLE001 - a progress callback must never fail a merge
            pass

    active_policy = policy or policy_module.DEFAULT_POLICY
    report(stage = "resolving", done = 0, total = count,
           message = "Resolving checkpoints")
    resolved = [
        resolve_checkpoint_source(
            source, token = token, allow_pickle = active_policy.allow_non_safetensors
        )
        for source in sources
    ]
    policy_report = [
        policy_module.enforce_source_policy(resolved[i], active_policy, label = f"Model {i + 1}")
        for i in range(count)
    ]
    model_limit = max_merge_models(has_gpu, sources = resolved)
    check_model_count(count, model_limit)
    specs = [probe_model_spec(source) for source in resolved]

    plans = plan_alignment_many(specs)
    template_index = choose_template_index(specs)
    tokenizer_report = tokenizer_check.check_before_merge(
        resolved, allow_conflict = allow_tokenizer_conflict
    )

    report(stage = "scanning", done = 0, total = count,
           message = "Reading shard indexes")
    entries = [
        _open_state(source, allow_pickle = active_policy.allow_non_safetensors)[0]
        for source in resolved
    ]

    out = Path(output_dir)
    out.mkdir(parents = True, exist_ok = True)

    template_entries = entries[template_index]
    # The template's inventory, not the union: a key the template does not have has no shape
    # the copied config can describe, and a deeper parent's extra layers would otherwise be
    # written while the config declared fewer, leaving the loader to invent the missing ones.
    keys = sorted(template_entries)
    dropped_keys = sorted(set().union(*(set(entry) for entry in entries)) - set(keys))
    total_keys = max(len(keys), 1)
    shard: dict[str, Any] = {}
    shard_bytes = 0
    shard_index = 1
    weight_map: dict[str, str] = {}
    blended = copied = skipped = streamed = 0
    total_bytes = 0
    report_every = max(1, total_keys // 100)

    def flush() -> None:
        nonlocal shard, shard_bytes, shard_index
        if not shard:
            return
        name = f"model-{shard_index:05d}.safetensors"

        policy_module.atomic_save_tensors(shard, out / name)
        for key in shard:
            weight_map[key] = name
        shard = {}
        shard_bytes = 0
        shard_index += 1

    for position, key in enumerate(keys, start = 1):
        present = [index for index, entry in enumerate(entries) if key in entry]
        merged, status = _merge_key(
            entries,
            key,
            present,
            _disk_shape(template_entries, key),
            template_index,
            method,
            normalized,
            density,
            seed,
        )
        if status in ("streamed", "blended"):
            blended += 1
            if status == "streamed":
                streamed += 1
        elif status == "copied":
            copied += 1
        else:
            skipped += 1

        nbytes = tensor_io.nbytes(merged)
        if shard and shard_bytes + nbytes > shard_buffer_bytes:
            flush()
        shard[key] = merged
        shard_bytes += nbytes
        total_bytes += nbytes
        if position % report_every == 0 or position == total_keys:
            report(
                stage = "merging",
                done = position,
                total = total_keys,
                message = f"{position}/{total_keys} tensors",
                blended_tensors = blended,
                copied_tensors = copied,
                skipped_tensors = skipped,
                bytes_written = total_bytes,
                shards = shard_index - 1,
            )
    flush()

    report(stage = "finalizing", done = total_keys, total = total_keys,
           message = "Writing index, config and tokenizer")
    index = {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}
    (out / "model.safetensors.index.json").write_text(
        json.dumps(index, indent = 2), encoding = "utf-8"
    )

    # The template's own files: its config describes the tensor shapes just written, and its
    # tokenizer's ids index those embedding rows. Mixing in another parent's tokenizer would
    # decode the merged vocabulary with the wrong ids.
    template_source = Path(resolved[template_index])
    for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        candidate = template_source / name
        if candidate.is_file():
            try:
                (out / name).write_bytes(candidate.read_bytes())
            except OSError:
                pass

    _verify_merged_against_template(
        out, template_entries, _config_from_dir(template_source)
    )
    output_tokenizer = tokenizer_check.verify_output(
        out, tokenizer_report.fingerprints[template_index]
    )

    report_block = {
        "status": "completed",
        "output_dir": str(out),
        "method": method,
        "model_count": count,

        "tier": "gpu" if has_gpu else "cpu",
        "max_models": model_limit,
        "parent_cap_reason": "gpu" if has_gpu else "ram",

        "weight": normalized[1] if count == 2 else None,
        "weights": list(normalized),
        "models": [spec.to_dict() for spec in specs],
        "template": {
            "index": template_index,
            "source": specs[template_index].source,
            "reason": "smallest architecture; config, tokenizer and tensor shapes come from it",
        },
        "alignment": plans[1].to_dict() if count > 1 else plans[0].to_dict(),
        "alignments": [plan.to_dict() for plan in plans],
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
                    policy_module.remote_code_warning(source) for source in resolved
                )
                if warning
            ],
        },

        "licenses": licenses.to_report_block(resolved),
        "metrics": {
            "blended_tensors": blended,

            "streamed_tensors": streamed,
            "copied_tensors": copied,
            "skipped_tensors": skipped,
            "dropped_tensors": len(dropped_keys),
            "shards": shard_index - 1,
            "shard_buffer_bytes": shard_buffer_bytes,
            "total_bytes": total_bytes,
        },
        "dropped_keys": dropped_keys[:20],
    }
    manifest = run_manifest.build_manifest(
        "merge",
        config={
            "method": method,
            "model_count": count,
            "weights": list(normalized),
            "density": density,
            "max_shard_bytes": max_shard_bytes,
            "shard_buffer_bytes": shard_buffer_bytes,
            "max_models": model_limit,
            "template_index": template_index,
            "alignment": plans[1].to_dict() if count > 1 else None,
        },
        seeds={"seed": seed},
        inputs={
            (f"model_{index + 1}" if count > 2 else ("model_a" if index == 0 else "model_b")): run_manifest.checkpoint_digest(source)
            for index, source in enumerate(resolved)
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
    manifest["licenses"] = report_block["licenses"]
    run_manifest.write_manifest(out, manifest)
    report_block["run_manifest"] = str(out / run_manifest.MANIFEST_NAME)
    report_block["manifest_id"] = manifest["run_id"]
    policy_module.atomic_write_json(out / "merge_report.json", report_block)
    release_pickle_states()
    report(stage = "completed", done = total_keys, total = total_keys, message = "Merge complete")
    return report_block

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
    progress: Any = None,
) -> dict[str, Any]:
    """Two-parent merge, expressed as the general case with ``[1 - w, w]``."""
    return merge_checkpoints_many(
        [model_a, model_b],
        output_dir,
        method = method,
        weights = [1.0 - float(weight), float(weight)],
        density = density,
        seed = seed,
        max_shard_bytes = max_shard_bytes,
        token = token,
        policy = policy,
        allow_tokenizer_conflict = allow_tokenizer_conflict,
        progress = progress,
    )

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
