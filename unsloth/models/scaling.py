# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Model expansion ("upscaling") for continued pretraining.

The pipeline is **expand -> CPT -> eval**, and only the last step decides whether
any of it was worth the compute:

* :func:`expand_model` makes the architecture bigger and initializes it from the
  source checkpoint. At step 0 the expanded model computes the same function as
  the base, which :func:`assert_function_preserved` verifies by comparing logits.
  Growth is a warm start; the added capacity is inert until it is trained.
* :func:`cpt_train` runs continued pretraining against the base as an anchor and
  reports the *recovery step*: the first eval where every task is at or above the
  base and the mean is above it. That is the step where upscaling has paid for the
  growth dip, and it is the number the pipeline exists to produce.
* :func:`evaluate_pipeline` compares the trained model against the base and
  returns ``better`` / ``same`` / ``worse``. Unless a same-compute baseline is
  supplied the verdict is labelled ``"vs base only"``: expansion has not been
  shown to beat simply training the smaller model for the same compute, and this
  module will not imply that it has.

Two knobs grow the model:

* **depth** (default) copies a block, adds noise to the copy's input projections
  and zeroes its ``o_proj``/``down_proj``, so ``x + 0 * f(x) = x``. This is the
  best-evidenced axis (LLaMA Pro, ACL 2024; Stacking Your Transformers, NeurIPS
  2024: 7B reached the scratch loss in 194B rather than 300B tokens).
* **mlp_width** widens ``gate_proj``/``up_proj``/``down_proj`` only. Attention,
  norms, embeddings and the head keep their shapes, which is what keeps the
  checkpoint loadable; the new channels are copies of existing ones and their
  contribution through ``down_proj`` starts at zero, so the MLP output is
  unchanged at step 0.

Growth below 2x is refused by default (``force_small_growth=True`` overrides with
a warning): every growth study that reports a gain grows by 2x or more, and a 12%
increase measured against the same training budget is usually a loss.

Refused outright: quantized checkpoints (start from the fp16/bf16 original; the
scales encode the old matrix shapes), PEFT-wrapped models (expand first, adapt
after) and MoE checkpoints (packed expert tensors and ``moe_intermediate_size``
need a dedicated path, which this version does not have).

Prior art: Net2WiderNet/Net2Net (2015), LEMON (ICLR 2024), HyperCloning (2024),
LLaMA Pro block expansion (ACL 2024), Stacking Your Transformers (NeurIPS 2024),
``llm-grow`` (safetensors-level width expansion, verified exact-logit on
Qwen2.5-0.5B/Qwen3-0.6B) and Apple ``ml-hypercloning`` (Llama/Gemma/OPT).
"""

from __future__ import annotations

import copy
import json
import math
import struct
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

__all__ = [
    "GROWTH_FLOOR",
    "expand_model",
    "save_expanded_model",
    "assert_function_preserved",
    "refresh_shard_index",
    "cpt_train",
    "evaluate_pipeline",
    "eval_only",
    "default_eval_suite",
    "heldout_perplexity_suite",
]

# Below this factor the literature reports no capability gain, so it is a refusal
# rather than a warning: a 1.7B -> 1.9B run costs a GPU-hour and buys noise.
GROWTH_FLOOR = 2.0

# Dense decoder-only families whose blocks are `self_attn.{q,k,v,o}_proj` plus
# `mlp.{gate,up,down}_proj`. The structural probe below is the real gate; this list
# keeps a mislabelled checkpoint from reaching it.
_SUPPORTED_MODEL_TYPES = frozenset(
    {
        "llama",
        "qwen2",
        "qwen3",
        "mistral",
        "gemma",
        "gemma2",
        "phi3",
        "granite",
        "olmo",
        "olmo2",
        "olmo3",
        "smollm",
        "smollm3",
        "hunyuan",
        "helios",
    }
)

_MOE_CONFIG_KEYS = (
    "num_experts",
    "num_local_experts",
    "n_routed_experts",
    "moe_intermediate_size",
    "shared_expert_intermediate_size",
    "num_experts_per_tok",
)

_INIT_MODES = ("identity_noisy", "identity")
_METHODS = ("depth", "mlp_width", "depth+mlp_width")
_STRATEGIES = ("even", "middle", "tail", "explicit")

# New widths land on a multiple of this so the projections stay friendly to the
# fused kernels the QLoRA path uses.
_WIDTH_ALIGNMENT = 64

# safetensors dtype -> bytes per element, for the shard index total.
_ST_DTYPE_BYTES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _torch():
    import torch
    return torch


def _parameter_count(module: Any) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def _layers_owner(model: Any) -> tuple[Any, str, Any]:
    """``(parent, attribute, block list)`` for the decoder blocks."""
    for parent_path, attribute in (
        ("model", "layers"),
        ("base_model.model", "layers"),
        ("", "layers"),
    ):
        parent = model
        for part in [piece for piece in parent_path.split(".") if piece]:
            parent = getattr(parent, part, None)
            if parent is None:
                break
        if parent is None:
            continue
        blocks = getattr(parent, attribute, None)
        if blocks is not None and hasattr(blocks, "__len__") and len(blocks):
            return parent, attribute, blocks
    raise ValueError(
        "Could not find the decoder blocks on this model (looked for model.layers, "
        "base_model.model.layers and layers). Expansion supports dense decoder-only "
        "checkpoints with a standard block layout."
    )


def _iter_linears(module: Any) -> Iterable[tuple[str, Any]]:
    import torch
    for name, child in module.named_modules():
        if name and isinstance(child, torch.nn.Linear):
            yield name, child


def _leaf(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def _find_linear(block: Any, leaf: str) -> Any:
    for name, module in _iter_linears(block):
        if _leaf(name) == leaf:
            return module
    return None


def _check_block_layout(block: Any, *, method: str) -> None:
    """Refuse a block that is not the standard attention + SwiGLU MLP."""
    attention = getattr(block, "self_attn", None) or getattr(block, "attention", None)
    mlp = getattr(block, "mlp", None) or getattr(block, "feed_forward", None)
    if attention is None or mlp is None:
        raise ValueError(
            "Expansion needs a block with `self_attn` and `mlp`; this checkpoint's blocks "
            f"are {type(block).__name__} containing "
            f"{[name for name, _ in block.named_children()]}."
        )
    if method in ("depth", "depth+mlp_width"):
        for leaf in ("o_proj", "down_proj"):
            if _find_linear(block, leaf) is None:
                raise ValueError(
                    f"Block expansion zeroes `{leaf}` so the copied block returns its input "
                    f"untouched; this block has no linear named `{leaf}`."
                )
    if method in ("mlp_width", "depth+mlp_width"):
        for leaf in ("gate_proj", "up_proj", "down_proj"):
            if _find_linear(block, leaf) is None:
                raise ValueError(
                    f"MLP-width expansion needs `mlp.{leaf}`; this block has no linear "
                    f"named `{leaf}`."
                )


def _guard_expandable(model: Any) -> Any:
    """Refuse the three inputs that would produce a broken or silently wrong result."""
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("expand_model needs a model with a config (a HF transformer).")

    quantization = getattr(config, "quantization_config", None)
    loaded_low_precision = bool(
        getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False)
    )
    if quantization is not None or loaded_low_precision:
        raise ValueError("Cannot expand a quantized model; load the fp16/bf16 original first.")

    if hasattr(model, "peft_config") or hasattr(model, "peft_type"):
        raise ValueError(
            "expand_model must run on the plain HF model, not a PEFT/LoRA-wrapped one: the "
            "adapter was trained against the old shapes. Expand first, then adapt."
        )

    model_type = str(getattr(config, "model_type", "") or "")
    if any(hasattr(config, key) for key in _MOE_CONFIG_KEYS) or "moe" in model_type.lower():
        raise ValueError(
            "MoE expansion not supported in v1: expert MLPs are packed tensors with their own "
            "width keys (moe_intermediate_size), and widening them without widening the router "
            "changes what the router scores. Use a dense source checkpoint."
        )

    if model_type and model_type not in _SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Expansion is not implemented for model_type={model_type!r}; supported dense "
            f"families are {sorted(_SUPPORTED_MODEL_TYPES)}."
        )
    return config


# ---------------------------------------------------------------------------
# Depth expansion
# ---------------------------------------------------------------------------


def _insertion_positions(
    num_layers: int, num_new: int, strategy: str, layer_indices: Optional[Sequence[int]]
) -> list[int]:
    """Source block indices *after* which a new block is inserted."""
    if num_new < 1:
        raise ValueError("num_new_layers must be at least 1.")
    if strategy not in _STRATEGIES:
        raise ValueError(f"insert_strategy must be one of {_STRATEGIES}, got {strategy!r}.")
    if strategy == "explicit":
        if not layer_indices:
            raise ValueError("insert_strategy='explicit' needs layer_indices.")
        positions = [int(index) for index in layer_indices]
        if len(positions) != num_new:
            raise ValueError(
                f"layer_indices has {len(positions)} entries but num_new_layers is {num_new}; "
                "pass one source index per new block."
            )
        bad = [index for index in positions if not 0 <= index < num_layers]
        if bad:
            raise ValueError(
                f"layer_indices {bad} are outside the model's {num_layers} blocks "
                f"(valid range is 0..{num_layers - 1})."
            )
        return positions
    if strategy == "tail":
        return [num_layers - 1] * num_new
    if strategy == "middle":
        return [(num_layers - 1) // 2] * num_new
    # "even": one per internal gap, so the copies are not all adjacent.
    if num_layers <= 1:
        return [0] * num_new
    return [
        max(0, min(num_layers - 1, round(index * (num_layers - 1) / (num_new + 1))))
        for index in range(1, num_new + 1)
    ]


def _make_identity_block(source: Any, *, init: str, init_noise_scale: float) -> Any:
    """A copy of ``source`` whose residual contribution starts at exactly zero."""
    import torch

    block = copy.deepcopy(source)
    output_modules = {leaf: _find_linear(block, leaf) for leaf in ("o_proj", "down_proj")}
    hidden_size = None
    for leaf, module in output_modules.items():
        if module is None:
            raise ValueError(f"The copied block has no `{leaf}` to zero.")
        if leaf == "o_proj":
            hidden_size = int(module.weight.shape[1])
    if hidden_size is None:
        hidden_size = int(output_modules["down_proj"].weight.shape[0])

    output_ids = {id(module) for module in output_modules.values()}
    with torch.no_grad():
        if init == "identity_noisy" and init_noise_scale:
            # Noise only on the projections that read the residual stream. The block's
            # output is zeroed below, so this cannot change the function at step 0; it
            # does stop every copy from receiving an identical gradient later.
            for _, module in _iter_linears(block):
                if id(module) in output_ids or module.weight.shape[1] != hidden_size:
                    continue
                module.weight.add_(
                    torch.randn_like(module.weight) * (init_noise_scale * module.weight.std())
                )
        for module in output_modules.values():
            module.weight.zero_()
            if module.bias is not None:
                module.bias.zero_()
    return block


def _expand_depth(
    model: Any,
    *,
    num_new_layers: int,
    insert_strategy: str,
    layer_indices: Optional[Sequence[int]],
    init: str,
    init_noise_scale: float,
) -> dict[str, Any]:
    import torch.nn as nn

    parent, attribute, layers = _layers_owner(model)
    positions = _insertion_positions(len(layers), num_new_layers, insert_strategy, layer_indices)
    pending: dict[int, int] = {}
    for position in positions:
        pending[position] = pending.get(position, 0) + 1

    built = nn.ModuleList()
    inserted_at: list[int] = []
    for index, block in enumerate(layers):
        built.append(block)
        for _ in range(pending.get(index, 0)):
            inserted_at.append(len(built))
            built.append(_make_identity_block(block, init = init, init_noise_scale = init_noise_scale))
    setattr(parent, attribute, built)

    config = model.config
    config.num_hidden_layers = len(built)
    # Models with an explicit per-layer attention pattern (sliding vs full) keep that
    # pattern in a list; a copy inherits its source's type, so the list must too.
    layer_types = getattr(config, "layer_types", None)
    if isinstance(layer_types, list) and len(layer_types) == len(layers):
        expanded: list[Any] = []
        for index, entry in enumerate(layer_types):
            expanded.append(entry)
            expanded.extend([entry] * pending.get(index, 0))
        config.layer_types = expanded

    for index in inserted_at:
        built[index]._unsloth_expanded_layer = True
    return {"new_num_layers": len(built), "inserted_at": inserted_at}


# ---------------------------------------------------------------------------
# MLP-width expansion
# ---------------------------------------------------------------------------


def _align(value: float) -> int:
    return max(_WIDTH_ALIGNMENT, int(_WIDTH_ALIGNMENT * round(float(value) / _WIDTH_ALIGNMENT)))


def _width_delta_for_params(params: float, *, hidden_size: int, num_layers: int) -> int:
    """Intermediate channels that add roughly ``params`` across every block."""
    per_channel = 3 * hidden_size * max(1, num_layers)
    return _align(params / per_channel)


def _expand_mlp_width(
    model: Any, *, new_intermediate_size: int, init_noise_scale: float
) -> dict[str, Any]:
    import torch

    _, _, layers = _layers_owner(model)
    probe = layers[0].mlp if hasattr(layers[0], "mlp") else layers[0].feed_forward
    old_intermediate = int(probe.gate_proj.weight.shape[0])
    if int(new_intermediate_size) <= old_intermediate:
        raise ValueError(
            f"mlp_width_target must widen intermediate_size past {old_intermediate}; "
            f"got {new_intermediate_size}."
        )

    expanded_layers: list[int] = []
    with torch.no_grad():
        for index, block in enumerate(layers):
            mlp = block.mlp if hasattr(block, "mlp") else block.feed_forward
            for leaf, reads_hidden in (
                ("gate_proj", True),
                ("up_proj", True),
                ("down_proj", False),
            ):
                module = getattr(mlp, leaf)
                weight = module.weight
                if reads_hidden:
                    target = torch.zeros(
                        (int(new_intermediate_size), weight.shape[1]),
                        dtype = weight.dtype,
                        device = weight.device,
                    )
                    target[:old_intermediate].copy_(weight)
                    for row in range(old_intermediate, int(new_intermediate_size)):
                        donor = int(torch.randint(0, old_intermediate, (1,)).item())
                        copied = weight[donor].clone()
                        if init_noise_scale:
                            copied.add_(
                                torch.randn_like(copied) * (init_noise_scale * copied.std())
                            )
                        target[row].copy_(copied)
                else:
                    # down_proj.weight is [hidden, intermediate]: the new channels are
                    # its *columns*. Zeroing the rows would zero every output; zeroing
                    # the new columns is what leaves the MLP's output unchanged, and
                    # their gradient arrives from the copied gate/up rows, so they
                    # still learn on the first step.
                    target = torch.zeros(
                        (weight.shape[0], int(new_intermediate_size)),
                        dtype = weight.dtype,
                        device = weight.device,
                    )
                    target[:, :old_intermediate].copy_(weight)
                module.weight = torch.nn.Parameter(target, requires_grad = weight.requires_grad)
                # nn.Linear caches these and PEFT sizes its adapter from them, so a resized
                # weight alone leaves an adapter built at the old width and every forward
                # fails on the first backward-free pass.
                module.in_features = int(target.shape[1])
                module.out_features = int(target.shape[0])
                if module.bias is not None:
                    bias = module.bias
                    if reads_hidden:
                        new_bias = torch.zeros(
                            (int(new_intermediate_size),),
                            dtype = bias.dtype,
                            device = bias.device,
                        )
                        new_bias[:old_intermediate].copy_(bias)
                    else:
                        new_bias = bias.detach().clone()
                    module.bias = torch.nn.Parameter(new_bias, requires_grad = bias.requires_grad)
            expanded_layers.append(index)

    model.config.intermediate_size = int(new_intermediate_size)
    return {
        "new_intermediate_size": int(new_intermediate_size),
        "old_intermediate_size": old_intermediate,
        "added_channels": int(new_intermediate_size) - old_intermediate,
        "expanded_mlp_layers": len(expanded_layers),
    }


# ---------------------------------------------------------------------------
# Public: expand_model
# ---------------------------------------------------------------------------


def expand_model(
    model: Any,
    target_params: Optional[float] = None,
    num_new_layers: Optional[int] = None,
    method: str = "depth",
    insert_strategy: str = "even",
    layer_indices: Optional[Sequence[int]] = None,
    init: str = "identity_noisy",
    init_noise_scale: float = 0.2,
    mlp_width_target: Optional[int] = None,
    force_small_growth: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Grow ``model`` in place and report what changed.

    ``target_params`` is a parameter *count* (``3.4e9``, not ``"3.4B"``). It becomes
    whole blocks for ``method="depth"`` and intermediate channels for
    ``method="mlp_width"``; with ``method="depth+mlp_width"`` and no explicit
    ``mlp_width_target`` the gap is split between the two.

    The model is modified and returned; nothing is written to disk. At step 0 the
    returned model computes the same function as the input - check it with
    :func:`assert_function_preserved` before spending a GPU-hour on it.
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}.")
    if init not in _INIT_MODES:
        raise ValueError(f"init must be one of {_INIT_MODES}, got {init!r}.")
    if init_noise_scale < 0:
        raise ValueError("init_noise_scale must be >= 0.")
    if target_params is None and num_new_layers is None and mlp_width_target is None:
        raise ValueError("Give at least one of target_params, num_new_layers or mlp_width_target.")

    config = _guard_expandable(model)
    _, _, layers = _layers_owner(model)
    _check_block_layout(layers[0], method = method)

    base_params = _parameter_count(model)
    params_per_layer = _parameter_count(layers[0])
    probe = layers[0].mlp if hasattr(layers[0], "mlp") else layers[0].feed_forward
    hidden_size = int(probe.gate_proj.weight.shape[1])
    old_intermediate = int(probe.gate_proj.weight.shape[0])

    report: dict[str, Any] = {
        "base_params": base_params,
        "method_requested": method,
        "method_used": None,
        "new_num_layers": None,
        "inserted_at": [],
        "new_intermediate_size": None,
        "params_per_layer": params_per_layer,
    }
    wants_width = method in ("mlp_width", "depth+mlp_width")
    wants_depth = method in ("depth", "depth+mlp_width")

    new_intermediate = int(mlp_width_target) if mlp_width_target is not None else None
    if wants_width and new_intermediate is None:
        if target_params is None:
            raise ValueError("mlp_width_target is required when the method has no depth delta.")
        gap = max(0.0, float(target_params) - base_params)
        if method == "depth+mlp_width":
            gap /= 2.0
        new_intermediate = old_intermediate + _width_delta_for_params(
            gap, hidden_size = hidden_size, num_layers = len(layers)
        )

    if wants_depth and num_new_layers is None:
        if target_params is None:
            raise ValueError("num_new_layers or target_params is required for depth growth.")
        remaining = float(target_params) - base_params
        if method == "depth+mlp_width" and new_intermediate is not None:
            remaining -= (new_intermediate - old_intermediate) * 3 * hidden_size * len(layers)
        num_new_layers = max(1, int(round(remaining / max(1, params_per_layer))))

    if wants_width and new_intermediate is not None:
        report.update(
            _expand_mlp_width(
                model,
                new_intermediate_size = new_intermediate,
                init_noise_scale = init_noise_scale,
            )
        )
        report["method_used"] = "mlp_width"

    if wants_depth:
        report.update(
            _expand_depth(
                model,
                num_new_layers = int(num_new_layers),
                insert_strategy = insert_strategy,
                layer_indices = layer_indices,
                init = init,
                init_noise_scale = init_noise_scale,
            )
        )
        report["method_used"] = "depth+mlp_width" if wants_width else "depth"

    final_params = _parameter_count(model)
    growth = final_params / max(1, base_params)
    report["final_params"] = final_params
    report["added_params"] = final_params - base_params
    report["growth_factor"] = growth
    report["config_num_hidden_layers"] = getattr(config, "num_hidden_layers", None)
    report["config_intermediate_size"] = getattr(config, "intermediate_size", None)

    if growth < GROWTH_FLOOR:
        message = (
            f"Growth factor is {growth:.2f}x, below the {GROWTH_FLOOR:g}x floor: no growth study "
            "reports a capability gain below that, and this run costs the same compute as a real "
            "one. Ask for more parameters, or pass force_small_growth=True."
        )
        if not force_small_growth:
            raise ValueError(message)
        report["warning"] = (
            f"growth {growth:.2f}x is below the evidenced range; capability gain unlikely"
        )

    # cpt_train reads these to give the new blocks a higher learning rate.
    model._unsloth_new_layer_indices = list(report.get("inserted_at") or [])
    model._unsloth_scaling_report = report
    return model, report


# ---------------------------------------------------------------------------
# Public: function preservation + saving
# ---------------------------------------------------------------------------


def assert_function_preserved(
    base_model: Any,
    expanded_model: Any,
    sample_input_ids: Any,
    atol: Optional[float] = None,
    *,
    base_logits: Any = None,
) -> float:
    """Assert the expanded model produces the base model's logits, and return the max diff.

    ``atol`` defaults to 1e-4 for fp32 and 1e-2 otherwise. A checkpoint that loads is
    not evidence of anything; matching logits is.

    Depth growth is bit-exact (the new block contributes exactly zero) and reports 0.0.
    Width growth cannot be: the widened ``down_proj`` sums more terms, so float32
    reassociates the sum and the logits move by roughly 1e-5 of their own magnitude.
    The working tolerance is therefore ``atol + 1e-5 * max|base logits|`` - strict
    enough that a block which really does change the function (a diff of order the
    logit spread) still fails.

    On a model too large to hold twice, capture the logits before growing it and pass
    them as ``base_logits`` (with ``base_model=None``): expansion mutates the model in
    place, so the same object cannot serve as both sides of this check.
    """
    torch = _torch()
    if base_logits is None and base_model is None:
        raise ValueError("Pass base_model, or base_logits captured before the expansion.")
    if atol is None:
        dtype = next(expanded_model.parameters()).dtype
        atol = 1e-4 if dtype == torch.float32 else 1e-2

    device = next(expanded_model.parameters()).device
    expanded_was_training = expanded_model.training
    expanded_model.eval()

    base_was_training = None
    moved = False
    base_device = None
    if base_logits is None:
        base_device = next(base_model.parameters()).device
        base_was_training = base_model.training
        base_model.eval()
        moved = base_device != device
        if moved:
            base_model.to(device)
    try:
        ids = sample_input_ids.to(device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        with torch.no_grad():
            if base_logits is None:
                base_values = base_model(input_ids = ids).logits.float()
            else:
                base_values = base_logits.to(device).float()
            expanded_values = expanded_model(input_ids = ids).logits.float()
    finally:
        if moved:
            base_model.to(base_device)
        if base_was_training is not None:
            base_model.train(base_was_training)
        expanded_model.train(expanded_was_training)

    max_diff = float((base_values - expanded_values).abs().max().item())
    scale = float(base_values.abs().max().item())
    allowed = atol + 1e-5 * scale
    if not math.isfinite(max_diff) or max_diff > allowed:
        raise AssertionError(
            f"Expanded model is not function-preserving: max |delta logit| = {max_diff:.3e} "
            f"> allowed {allowed:.3e} (atol {atol:g}, logit scale {scale:.3g}). The new "
            "blocks and channels must contribute zero at step 0."
        )
    return max_diff


def _shard_header(path: Path) -> dict[str, Any]:
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
    return {key: value for key, value in header.items() if key != "__metadata__"}


def refresh_shard_index(out_dir: str | Path) -> Optional[dict[str, Any]]:
    """Rebuild ``model.safetensors.index.json`` from the shards that are on disk.

    ``save_pretrained`` writes an index, but an expansion changes tensor byte sizes,
    and every reader (vLLM, llama.cpp's converter, ``from_pretrained``) trusts the
    index's ``weight_map`` and ``total_size``. Rebuilding both from the shard headers
    is what keeps "it loads here" true everywhere else.
    """
    root = Path(out_dir)
    index_path = root / "model.safetensors.index.json"
    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        return None
    if len(shards) == 1 and not index_path.is_file():
        return None

    weight_map: dict[str, str] = {}
    total_size = 0
    for shard in shards:
        for key, entry in _shard_header(shard).items():
            weight_map[key] = shard.name
            count = 1
            for dimension in entry.get("shape") or []:
                count *= int(dimension)
            total_size += count * _ST_DTYPE_BYTES.get(str(entry.get("dtype")), 4)

    payload = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    index_path.write_text(json.dumps(payload, indent = 2), encoding = "utf-8")
    return {
        "shards": [shard.name for shard in shards],
        "total_size": total_size,
        "tensors": len(weight_map),
    }


def save_expanded_model(
    model: Any,
    tokenizer: Any,
    out_dir: str | Path,
    *,
    max_shard_size: str = "5GB",
    verify: bool = True,
    sample_input_ids: Any = None,
    atol: Optional[float] = None,
) -> dict[str, Any]:
    """Save the expanded checkpoint, rebuild its shard index, and verify the reload.

    Returns the verification report. With ``verify=True`` the saved directory is
    reloaded from disk and its logits compared with the in-memory model, so a silent
    save bug fails here instead of after 2000 CPT steps.
    """
    if getattr(model, "peft_config", None) is not None:
        raise ValueError(
            "save_expanded_model must save the plain expanded model, before any adapter is "
            "injected: an injected model's state dict carries adapter keys that a plain "
            "from_pretrained reload drops, so the checkpoint would not be what was verified. "
            "Save the expanded base first, then train and save the adapter on its own."
        )

    root = Path(out_dir).expanduser()
    root.mkdir(parents = True, exist_ok = True)
    model.save_pretrained(str(root), max_shard_size = max_shard_size)
    if tokenizer is not None:
        tokenizer.save_pretrained(str(root))
    report: dict[str, Any] = {
        "output_dir": str(root),
        "index": refresh_shard_index(root),
    }

    if not verify:
        report["verified"] = False
        return report

    if sample_input_ids is None and tokenizer is not None:
        encoded = tokenizer(
            "Unsloth function-preservation check: the expanded model must reproduce "
            "the base model's logits at step 0.",
            return_tensors = "pt",
        )
        sample_input_ids = encoded["input_ids"]
    if sample_input_ids is None:
        report["verified"] = False
        report["note"] = "No sample_input_ids and no tokenizer; the reload was not verified."
        return report

    from transformers import AutoModelForCausalLM

    try:
        reloaded = AutoModelForCausalLM.from_pretrained(
            str(root), dtype = "auto", low_cpu_mem_usage = True
        )
    except Exception as error:  # noqa: BLE001 - surface the real loader error
        raise RuntimeError(
            f"The expanded checkpoint at {root} could not be reloaded: {error}"
        ) from error

    report["verified"] = True
    report["max_logit_diff"] = assert_function_preserved(
        model, reloaded, sample_input_ids, atol = atol
    )
    report["verified_atol"] = atol
    del reloaded
    return report


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


def _score_model(
    model: Any, tokenizer: Any, eval_suite: Mapping[str, Callable[[Any, Any], float]]
) -> dict[str, float]:
    """Run every task and restore the model's mode; each score is 0..1, higher is better."""
    was_training = model.training
    use_cache = getattr(model.config, "use_cache", None)
    model.eval()
    if use_cache is not None:
        model.config.use_cache = True
    scores: dict[str, float] = {}
    try:
        with _torch().no_grad():
            for name, task in eval_suite.items():
                scores[name] = float(task(model, tokenizer))
    finally:
        if use_cache is not None:
            model.config.use_cache = use_cache
        model.train(was_training)
    return scores


def _normalise(text: str) -> str:
    return "".join(character for character in str(text).strip().lower() if character.isalnum())


def _greedy_accuracy(
    model: Any,
    tokenizer: Any,
    pairs: Sequence[tuple[str, str]],
    *,
    max_new_tokens: int = 8,
) -> float:
    import torch

    if tokenizer is None:
        raise ValueError("The default eval suite needs a tokenizer.")
    if not pairs:
        return 0.0
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    correct = 0
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for prompt, expected in pairs:
                encoded = tokenizer(prompt, return_tensors = "pt")
                encoded = {key: value.to(model.device) for key, value in encoded.items()}
                output = model.generate(
                    **encoded,
                    max_new_tokens = max_new_tokens,
                    do_sample = False,
                    pad_token_id = pad_id,
                )
                generated = tokenizer.decode(
                    output[0][encoded["input_ids"].shape[1] :], skip_special_tokens = True
                )
                if _normalise(expected) and _normalise(expected) in _normalise(generated):
                    correct += 1
    finally:
        model.train(was_training)
    return correct / len(pairs)


_BUILTIN_QA = (
    ("Answer with one word. The capital of France is", "paris"),
    ("Answer with one word. The capital of Japan is", "tokyo"),
    ("Answer with one word. The capital of Italy is", "rome"),
    ("Answer with one word. The opposite of hot is", "cold"),
    ("Answer with one word. The opposite of up is", "down"),
    ("Answer with one word. Water freezes at zero degrees", "celsius"),
    ("Answer with one word. The colour of grass is", "green"),
    ("Answer with one word. Two plus two equals", "four"),
    ("Answer with one word. The first month of the year is", "january"),
    ("Answer with one word. A baby dog is called a", "puppy"),
    ("Answer with one word. The largest planet is", "jupiter"),
    ("Answer with one word. A shape with three sides is a", "triangle"),
)

_BUILTIN_ARITHMETIC = (
    ("7 + 5 =", "12"),
    ("9 - 4 =", "5"),
    ("6 * 3 =", "18"),
    ("20 / 4 =", "5"),
    ("13 + 8 =", "21"),
    ("15 - 9 =", "6"),
    ("4 * 4 =", "16"),
    ("18 / 3 =", "6"),
)


def default_eval_suite() -> dict[str, Callable[[Any, Any], float]]:
    """A two-task suite that needs no dataset: built-in QA and arithmetic accuracy.

    Weak on purpose - a smoke signal that training moved the model at all, not a
    benchmark. For a real verdict pass your own held-out tasks. Every task returns a
    0..1 score where higher is better, which is what :func:`evaluate_pipeline`
    compares in points.
    """

    def qa(model: Any, tokenizer: Any) -> float:
        return _greedy_accuracy(model, tokenizer, _BUILTIN_QA, max_new_tokens = 6)

    def arithmetic(model: Any, tokenizer: Any) -> float:
        return _greedy_accuracy(model, tokenizer, _BUILTIN_ARITHMETIC, max_new_tokens = 4)

    return {"builtin_qa": qa, "builtin_arithmetic": arithmetic}


def heldout_perplexity_suite(
    texts: Sequence[str],
    *,
    max_length: int = 512,
    stride: int = 256,
) -> dict[str, Callable[[Any, Any], float]]:
    """A held-out token-likelihood task over ``texts`` (higher is better, 0..1).

    Returns ``exp(-mean_nll)`` rather than perplexity so every task in a suite shares
    the "higher is better" direction the verdict math assumes.
    """
    corpus = [str(text) for text in texts if str(text).strip()]
    if not corpus:
        raise ValueError("heldout_perplexity_suite needs at least one non-empty text.")

    def token_likelihood(model: Any, tokenizer: Any) -> float:
        import torch

        if tokenizer is None:
            raise ValueError("The perplexity task needs a tokenizer.")
        total_nll = 0.0
        total_tokens = 0
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for text in corpus:
                    ids = tokenizer(text, return_tensors = "pt", truncation = False)["input_ids"]
                    ids = ids[:, :max_length].to(model.device)
                    if ids.shape[1] < 2:
                        continue
                    logits = model(input_ids = ids).logits.float()
                    shift_logits = logits[:, :-1, :]
                    shift_labels = ids[:, 1:]
                    loss = torch.nn.functional.cross_entropy(
                        shift_logits.reshape(-1, shift_logits.shape[-1]),
                        shift_labels.reshape(-1),
                        reduction = "sum",
                    )
                    total_nll += float(loss.item())
                    total_tokens += int(shift_labels.numel())
        finally:
            model.train(was_training)
        if total_tokens == 0:
            return 0.0
        return math.exp(-total_nll / total_tokens)

    return {"heldout_token_likelihood": token_likelihood}


def eval_only(
    model: Any,
    tokenizer: Any,
    eval_suite: Optional[Mapping[str, Callable[[Any, Any], float]]] = None,
) -> dict[str, float]:
    """Score one model on a suite, without a comparison. Useful for the base anchor."""
    return _score_model(model, tokenizer, eval_suite or default_eval_suite())


def evaluate_pipeline(
    base_model: Any,
    candidate_model: Any,
    eval_suite: Optional[Mapping[str, Callable[[Any, Any], float]]] = None,
    tokenizer: Any = None,
    *,
    base_scores: Optional[Mapping[str, float]] = None,
    same_compute_baseline: Any = None,
    same_compute_baseline_scores: Optional[Mapping[str, float]] = None,
    improved_tasks: int = 2,
    improvement_points: float = 1.0,
    regression_points: float = 1.0,
    same_tolerance_points: float = 0.5,
    gpu_hours_expanded: Optional[float] = None,
    gpu_hours_baseline: Optional[float] = None,
) -> dict[str, Any]:
    """Compare a trained model with the base and return a verdict.

    Scores are 0..1 and compared in points (``score * 100``):

    * ``better`` - at least ``improved_tasks`` tasks improve by ``improvement_points``,
      no task regresses by more than ``regression_points``, and the mean improves.
      A run of fewer tasks needs only as many as it has, so a single-task suite can
      still clear the bar instead of being stuck at ``same`` forever.
    * ``same`` - the mean is within ``same_tolerance_points`` either way, or it
      improved by less than the ``better`` bar.
    * ``worse`` - the mean regresses by more than ``same_tolerance_points``.

    ``comparison`` is ``"vs base only"`` unless a same-compute baseline is supplied.
    That baseline - the base trained for the same tokens or GPU-hours without
    expansion - is the comparison the growth literature is missing, so the label
    never implies it was made.
    """
    if eval_suite is None and tokenizer is None:
        raise ValueError(
            "evaluate_pipeline needs a tokenizer for the default suite, or your own "
            "eval_suite of {task_name: callable(model, tokenizer) -> float}."
        )
    suite = eval_suite or default_eval_suite()
    base = dict(base_scores or _score_model(base_model, tokenizer, suite))
    candidate = _score_model(candidate_model, tokenizer, suite)

    tasks = sorted(set(base) | set(candidate))
    per_task = []
    deltas: dict[str, float] = {}
    for name in tasks:
        before = float(base.get(name, 0.0))
        after = float(candidate.get(name, 0.0))
        delta = (after - before) * 100.0
        deltas[name] = delta
        per_task.append({"task": name, "base": before, "candidate": after, "delta_points": delta})

    mean_base = sum(base.values()) / max(1, len(base))
    mean_candidate = sum(candidate.values()) / max(1, len(candidate))
    mean_delta = (mean_candidate - mean_base) * 100.0

    improved = [name for name, delta in deltas.items() if delta >= improvement_points]
    regressed = [name for name, delta in deltas.items() if delta <= -regression_points]
    required = min(improved_tasks, max(1, len(tasks)))
    if len(improved) >= required and not regressed and mean_delta > 0:
        verdict = "better"
        reason = (
            f"{len(improved)} task(s) improved by at least {improvement_points:g} pt with no "
            f"regression over {regression_points:g} pt."
        )
    elif abs(mean_delta) <= same_tolerance_points:
        verdict = "same"
        reason = (
            f"Mean change {mean_delta:+.2f} pt is within the {same_tolerance_points:g} pt band."
        )
    elif mean_delta < -same_tolerance_points:
        verdict = "worse"
        reason = f"Mean regressed {mean_delta:+.2f} pt."
    else:
        verdict = "same"
        reason = (
            f"Mean improved {mean_delta:+.2f} pt, but the bar is {required} task(s) up "
            f"{improvement_points:g} pt with no regression over {regression_points:g} pt."
        )

    result: dict[str, Any] = {
        "verdict": verdict,
        "comparison": "vs base only",
        "reason": reason,
        "per_task": per_task,
        "mean_base": mean_base,
        "mean_candidate": mean_candidate,
        "mean_delta_points": mean_delta,
        "improved_tasks": improved,
        "regressed_tasks": regressed,
        "gpu_hours_expanded": gpu_hours_expanded,
        "gpu_hours_baseline": None,
        "thresholds": {
            "improved_tasks": improved_tasks,
            "improvement_points": improvement_points,
            "regression_points": regression_points,
            "same_tolerance_points": same_tolerance_points,
        },
        "notes": [
            "Verification is 'vs base only': expansion has not been shown to beat training the "
            "smaller model for the same compute unless the same_compute_baseline is run.",
        ],
    }

    if same_compute_baseline is not None or same_compute_baseline_scores is not None:
        baseline = dict(
            same_compute_baseline_scores or _score_model(same_compute_baseline, tokenizer, suite)
        )
        head_to_head = []
        for name in tasks:
            line = float(baseline.get(name, 0.0))
            head_to_head.append(
                {
                    "task": name,
                    "baseline": line,
                    "candidate": float(candidate.get(name, 0.0)),
                    "delta_points": (float(candidate.get(name, 0.0)) - line) * 100.0,
                }
            )
        baseline_mean = sum(baseline.values()) / max(1, len(baseline))
        result["comparison"] = "vs same-compute baseline"
        result["same_compute"] = {
            "per_task": head_to_head,
            "mean_baseline": baseline_mean,
            "mean_delta_points": (mean_candidate - baseline_mean) * 100.0,
            "gpu_hours_baseline": gpu_hours_baseline,
        }
        result["gpu_hours_baseline"] = gpu_hours_baseline
        result["notes"] = [
            "Same-compute baseline supplied: the head-to-head delta is the comparison that "
            "matters, and the 'vs base' verdict is reported separately.",
        ]
    return result


# `evaluate_pipeline.eval_only(...)` is in the shipped example, so the attribute exists.
evaluate_pipeline.eval_only = eval_only  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# CPT
# ---------------------------------------------------------------------------


class _CptCollator:
    """Tokenizes ``text`` rows, pads to the longest row, and ignores pad in the loss."""

    def __init__(
        self,
        tokenizer: Any,
        max_seq_length: int,
        text_column: str = "text",
    ) -> None:
        if tokenizer is None:
            raise ValueError("cpt_train needs a tokenizer for a text dataset.")
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.text_column = text_column

    def __call__(self, features: list[Any]) -> Any:
        texts = []
        for feature in features:
            if isinstance(feature, Mapping):
                if self.text_column not in feature:
                    raise ValueError(
                        f"The dataset has no {self.text_column!r} column; pass text_column= "
                        f"(columns: {sorted(feature)})"
                    )
                texts.append(feature[self.text_column])
            else:
                texts.append(feature)
        batch = self.tokenizer(
            [str(text) for text in texts],
            padding = True,
            truncation = True,
            max_length = self.max_seq_length,
            return_tensors = "pt",
        )
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


class _TokenCountingCollator:
    """Wraps a collator so ``tokens_seen`` is measured, not estimated from steps."""

    def __init__(self, inner: Callable[[list[Any]], Any]) -> None:
        self.inner = inner
        self.tokens_seen = 0

    def __call__(self, features: list[Any]) -> Any:
        batch = self.inner(features)
        mask = batch.get("attention_mask")
        if mask is not None:
            self.tokens_seen += int(mask.sum().item())
        elif "input_ids" in batch:
            self.tokens_seen += int(batch["input_ids"].numel())
        return batch


def _first_row(dataset: Any) -> Any:
    try:
        return dataset[0]
    except Exception:  # noqa: BLE001 - DatasetDict is indexed by split, not row
        for split in ("train", "training"):
            candidate = getattr(dataset, split, None)
            if candidate is not None:
                try:
                    return candidate[0]
                except Exception:  # noqa: BLE001
                    return None
    return None


def _layer_index_from_name(name: str) -> Optional[int]:
    marker = ".layers."
    position = name.find(marker)
    if position < 0:
        return None
    head = name[position + len(marker) :].split(".", 1)[0]
    return int(head) if head.isdigit() else None


def _resolve_suite(eval_suite: Any) -> Mapping[str, Callable[[Any, Any], float]]:
    if eval_suite is None:
        return default_eval_suite()
    if callable(eval_suite) and not isinstance(eval_suite, Mapping):
        raise ValueError(
            "eval_suite must be a mapping of {task_name: callable(model, tokenizer) -> float}."
        )
    return eval_suite


def _make_recovery_callback(
    *,
    model: Any,
    tokenizer: Any,
    suite: Mapping[str, Callable[[Any, Any], float]],
    anchor: Mapping[str, float],
    eval_every: int,
    patience: int,
    best_dir: Path,
    tokens: _TokenCountingCollator,
    new_layer_indices: Sequence[int],
):
    """Score the suite during training, keep the best mean, and report the recovery step."""
    from transformers import TrainerCallback

    class _RecoveryCallback(TrainerCallback):
        def __init__(self) -> None:
            self.anchor = dict(anchor)
            self.history: list[dict[str, Any]] = []
            self.recovery_step: Optional[int] = None
            self.best_mean: float = -math.inf
            self.best_scores: Optional[dict[str, float]] = None
            self.new_layer_grad_nonzero: Optional[bool] = None
            self.evaluations = 0
            self._stale = 0
            self._checked_grad = False

        def evaluate(self, target: Any, step: int) -> dict[str, float]:
            scores = _score_model(target, tokenizer, suite)
            mean = sum(scores.values()) / max(1, len(scores))
            anchor_mean = sum(self.anchor.values()) / max(1, len(self.anchor))
            recovered = (
                all(scores.get(name, 0.0) >= value for name, value in self.anchor.items())
                and mean > anchor_mean
            )
            if recovered and self.recovery_step is None:
                self.recovery_step = int(step)

            improved = mean > self.best_mean
            if improved:
                self.best_mean = mean
                self.best_scores = dict(scores)
                self._stale = 0
                target.save_pretrained(str(best_dir))
            else:
                self._stale += 1

            self.history.append(
                {
                    "step": int(step),
                    "scores": scores,
                    "mean": mean,
                    "recovered": bool(recovered),
                    "best": bool(improved),
                    "tokens_seen": tokens.tokens_seen,
                }
            )
            self.evaluations += 1
            return scores

        def _check_new_layer_grads(self, target: Any) -> None:
            """A LoRA B=0 init delays A's gradient by a step; verify it, don't assume it."""
            if not new_layer_indices:
                return
            touched = 0
            for name, parameter in target.named_parameters():
                if parameter.grad is None:
                    continue
                index = _layer_index_from_name(name)
                if (
                    index is not None
                    and index in new_layer_indices
                    and bool(parameter.grad.abs().sum() > 0)
                ):
                    touched += 1
            self.new_layer_grad_nonzero = touched > 0

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            # Gradients are live here; on_step_end runs after zero_grad and sees none.
            if not self._checked_grad:
                self._checked_grad = True
                self._check_new_layer_grads(kwargs.get("model") or model)
            return control

        def on_step_end(self, args, state, control, **kwargs):
            target = kwargs.get("model") or model
            if state.global_step % eval_every == 0:
                self.evaluate(target, state.global_step)
                if self._stale >= patience:
                    control.should_training_stop = True
            return control

        def on_train_end(self, args, state, control, **kwargs):
            target = kwargs.get("model") or model
            if not self.history or self.history[-1]["step"] != state.global_step:
                self.evaluate(target, state.global_step)
            return control

    return _RecoveryCallback()


def cpt_train(
    model: Any,
    tokenizer: Any,
    cpt_dataset: Any,
    *,
    eval_suite: Any = None,
    base_scores: Optional[Mapping[str, float]] = None,
    base_model: Any = None,
    max_steps: int = 2000,
    eval_every: int = 100,
    patience: int = 5,
    learning_rate: float = 2e-4,
    new_layer_lr_multiplier: float = 3.0,
    use_lora: bool = True,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    per_device_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    max_seq_length: int = 2048,
    warmup_ratio: float = 0.02,
    weight_decay: float = 0.0,
    optim: str = "adamw_torch",
    bf16: Optional[bool] = None,
    fp16: Optional[bool] = None,
    gradient_checkpointing: bool = True,
    output_dir: Optional[str] = None,
    text_column: str = "text",
    new_layer_indices: Optional[Sequence[int]] = None,
    seed: int = 3407,
    logging_steps: int = 10,
    dataloader_num_workers: int = 0,
    report_to: str = "none",
) -> dict[str, Any]:
    """Continued pretraining that reports the recovery step and what it cost.

    The model trains against the base as an anchor: the suite is scored before the
    first step (numerically the base, because expansion preserves the function and a
    fresh LoRA adapter starts at zero delta), then every ``eval_every`` steps.
    ``recovery_step`` is the first eval where every task is at or above the anchor and
    the mean is above it - the point where the new capacity has paid for the growth
    dip. Training stops after ``patience`` evals without a better mean, and the best
    scoring checkpoint comes back, not the last one.

    New blocks get ``new_layer_lr_multiplier`` times the base learning rate; the
    inherited ones keep it, because they are already good. With ``use_lora`` the
    adapter covers every linear layer including the new ones.

    ``gpu_hours`` is the wall clock of the ``train()`` call, which is GPU hours on a
    GPU host and CPU hours on the CPU path; ``peak_mem_gb`` is 0 off CUDA.
    """
    import torch
    from transformers import Trainer, TrainingArguments

    suite = _resolve_suite(eval_suite)
    if eval_every < 1:
        raise ValueError("eval_every must be at least 1.")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1.")

    if tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    indices = (
        list(new_layer_indices)
        if new_layer_indices is not None
        else list(getattr(model, "_unsloth_new_layer_indices", []) or [])
    )

    # Anchor. With no explicit base, score the step-0 model: expansion is
    # function-preserving, so this *is* the base, without holding two models.
    if base_scores is not None:
        anchor = dict(base_scores)
        anchor_source = "provided"
    elif base_model is not None:
        anchor = _score_model(base_model, tokenizer, suite)
        anchor_source = "base_model"
    else:
        anchor = _score_model(model, tokenizer, suite)
        anchor_source = "step-0-expanded"

    if use_lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(
            model,
            LoraConfig(
                r = lora_rank,
                lora_alpha = lora_alpha,
                lora_dropout = lora_dropout,
                bias = "none",
                target_modules = "all-linear",
                task_type = "CAUSAL_LM",
            ),
        )
        if gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    else:
        model.requires_grad_(True)

    new_params, old_params = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        index = _layer_index_from_name(name)
        if index is not None and index in indices:
            new_params.append(parameter)
        else:
            old_params.append(parameter)
    groups = []
    if new_params:
        groups.append(
            {
                "params": new_params,
                "lr": learning_rate * new_layer_lr_multiplier,
                "weight_decay": weight_decay,
            }
        )
    if old_params:
        groups.append({"params": old_params, "lr": learning_rate, "weight_decay": weight_decay})
    if not groups:
        raise ValueError("No trainable parameters: check use_lora and the model's requires_grad.")

    first = _first_row(cpt_dataset)
    pre_tokenised = isinstance(first, Mapping) and "input_ids" in first
    if pre_tokenised:
        from transformers import DataCollatorForLanguageModeling
        inner = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = False)
    else:
        inner = _CptCollator(tokenizer, max_seq_length, text_column)
    tokens = _TokenCountingCollator(inner)

    use_bf16 = bf16
    if use_bf16 is None:
        use_bf16 = bool(
            torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        )
    use_fp16 = fp16
    if use_fp16 is None:
        use_fp16 = bool(torch.cuda.is_available() and not use_bf16)

    run_dir = Path(output_dir or f"./cpt_run_{int(time.time())}").expanduser()
    run_dir.mkdir(parents = True, exist_ok = True)
    best_dir = run_dir / "best"

    # warmup_ratio left TrainingArguments in transformers 5; the steps form is read by both.
    warmup_steps = (
        min(int(max_steps) - 1, max(1, math.ceil(warmup_ratio * int(max_steps))))
        if warmup_ratio > 0
        else 0
    )

    recovery = _make_recovery_callback(
        model = model,
        tokenizer = tokenizer,
        suite = suite,
        anchor = anchor,
        eval_every = eval_every,
        patience = patience,
        best_dir = best_dir,
        tokens = tokens,
        new_layer_indices = indices,
    )

    trainer = Trainer(
        model = model,
        args = TrainingArguments(
            output_dir = str(run_dir),
            max_steps = int(max_steps),
            per_device_train_batch_size = per_device_batch_size,
            gradient_accumulation_steps = gradient_accumulation_steps,
            learning_rate = learning_rate,
            lr_scheduler_type = "cosine",
            warmup_steps = warmup_steps,
            weight_decay = weight_decay,
            optim = optim,
            bf16 = use_bf16,
            fp16 = use_fp16,
            gradient_checkpointing = gradient_checkpointing,
            logging_steps = logging_steps,
            save_strategy = "no",
            report_to = report_to,
            seed = seed,
            data_seed = seed,
            dataloader_num_workers = dataloader_num_workers,
            remove_unused_columns = False,
        ),
        train_dataset = cpt_dataset,
        data_collator = tokens,
        callbacks = [recovery],
    )
    callback = recovery

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    trainer.train()
    wall_hours = (time.monotonic() - started) / 3600.0

    if not callback.history or callback.history[-1]["step"] != trainer.state.global_step:
        callback.evaluate(trainer.model, trainer.state.global_step)

    peak_memory = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    best_scores = callback.best_scores
    if best_scores is None and callback.history:
        best_scores = callback.history[-1]["scores"]

    return {
        "best_checkpoint": str(best_dir) if best_dir.exists() else None,
        "recovery_step": callback.recovery_step,
        "base_scores": anchor,
        "base_scores_source": anchor_source,
        "best_scores": best_scores or dict(anchor),
        "best_mean": callback.best_mean if callback.best_mean > -math.inf else None,
        "tokens_seen": tokens.tokens_seen,
        "gpu_hours": wall_hours,
        "peak_mem_gb": peak_memory,
        "steps_run": int(trainer.state.global_step),
        "evaluations": callback.evaluations,
        "history": callback.history,
        "new_layer_indices": indices,
        "new_layer_grad_nonzero": callback.new_layer_grad_nonzero,
        "output_dir": str(run_dir),
    }
