# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the CPU CoRA training path.

The trainer module cannot be imported here (trl + unsloth + torch), and the host running these
tests has no accelerator anyway, so the wiring is checked at source level the way the other
trainer tests in this suite do it. The optimizer normalization is exercised directly.
"""

import ast
import re
from pathlib import Path

from core.training.training import (
    cpu_fp32_weights_fit,
    normalize_training_optimizer_for_device,
)

_BACKEND = Path(__file__).resolve().parent.parent
_TRAINER = _BACKEND / "core" / "training" / "trainer.py"
_TRAINING = _BACKEND / "core" / "training" / "training.py"
_WORKER = _BACKEND / "core" / "training" / "worker.py"
_MODELS = _BACKEND / "models" / "training.py"
_ROUTES = _BACKEND / "routes" / "training.py"


def _src(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _func_src(path: Path, name: str) -> str:
    source = _src(path)
    node = next(
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    )
    return ast.get_source_segment(source, node)


def test_advanced_settings_reach_the_worker_config():
    """QAT and layer-wise LR were dropped here, so the CoRA switches were cosmetic."""
    body = _func_src(_TRAINING, "_build_training_worker_config")
    assert '"advanced": values.get("advanced")' in body


def test_cpu_trainer_uses_processing_class():
    """Upstream TRL removed SFTTrainer(tokenizer=...); the CPU path runs it unpatched."""
    source = _src(_TRAINER)
    assert re.search(
        r'trainer_kwargs = \{\s*"model": self\.model,\s*"processing_class": sft_tokenizer,.*?"args": SFTConfig',
        source,
        re.DOTALL,
    )


def test_cpu_optimizer_falls_back_to_torch_adamw():
    assert (
        normalize_training_optimizer_for_device(
            "paged_adamw_8bit", device_backend="cpu"
        )
        == "adamw_torch"
    )
    assert (
        normalize_training_optimizer_for_device("adamw_8bit", device_backend="cpu")
        == "adamw_torch"
    )
    # Every other device keeps the recipe's optimizer.
    assert (
        normalize_training_optimizer_for_device(
            "paged_adamw_8bit", device_backend="cuda"
        )
        == "paged_adamw_8bit"
    )
    assert (
        normalize_training_optimizer_for_device("adamw_torch", device_backend="cpu")
        == "adamw_torch"
    )


def test_cpu_fp32_fit_leaves_headroom_for_activations():
    gb = 1024**3
    # Unknown sizes keep the old fp32 path rather than refusing to load.
    assert cpu_fp32_weights_fit(None, 3 * gb)
    assert cpu_fp32_weights_fit(2 * gb, None)
    # A 1.7B model is ~6.9 GB in fp32: it fits a roomy host but not an 8 GB one.
    assert not cpu_fp32_weights_fit(6_900_000_000, 4 * gb)
    assert cpu_fp32_weights_fit(6_900_000_000, 12 * gb)


def test_cpu_loader_falls_back_to_bf16_when_fp32_cannot_fit():
    body = _func_src(_TRAINER, "_load_cpu_model")
    assert "estimate_fp16_model_size_bytes" in body
    assert "cpu_fp32_weights_fit" in body
    assert "torch.bfloat16" in body


def test_worker_config_carries_the_training_preset():
    body = _func_src(_TRAINING, "_build_training_worker_config")
    assert '"training_preset": values.get("training_preset")' in body


def test_route_forwards_the_preset_to_the_worker():
    source = _src(_ROUTES)
    assert '"training_preset": request.training_preset' in source


def test_request_accepts_the_two_cpu_presets():
    source = _src(_MODELS)
    assert 'training_preset: Optional[Literal["cora", "qlora"]]' in source


def test_worker_hands_the_preset_to_the_trainer():
    source = _src(_WORKER)
    assert 'trainer.training_preset = config.get("training_preset")' in source


def test_cpu_load_is_gated_to_cora_and_qlora():
    body = _func_src(_TRAINER, "load_model")
    assert "cpu_training_host()" in body
    # Full finetuning, plain LoRA, CPT and RL need Unsloth's GPU loaders; one of those reaching
    # this path would crash, so the gate names both recipes it accepts and refuses the rest.
    assert 'not in ("cora", "qlora")' in body
    assert "self._load_cpu_model(" in body


def test_cpu_loader_uses_plain_transformers():
    body = _func_src(_TRAINER, "_load_cpu_model")
    assert "AutoModelForCausalLM.from_pretrained" in body
    assert "AutoTokenizer.from_pretrained" in body
    assert "low_cpu_mem_usage = True" in body
    assert "self.load_in_4bit = False" in body
    assert "self._cpu_training = True" in body
    # No Unsloth loader or kernel patch on this path.
    assert "FastLanguageModel" not in body
    assert "get_peft_model" not in body


def test_cpu_lora_goes_through_peft():
    body = _func_src(_TRAINER, "_prepare_cpu_lora")
    assert "from peft import LoraConfig, get_peft_model" in body
    assert 'task_type = "CAUSAL_LM"' in body
    assert "lora_alpha = lora_alpha" in body


def test_cpu_training_config_turns_off_gpu_only_settings():
    body = _func_src(_TRAINER, "_train_worker")
    assert '"use_cpu": True' in body
    assert '"fp16": False' in body
    assert '"bf16": False' in body
    assert '"optim": "adamw_torch"' in body
    assert '"packing": False' in body
    assert '"dataloader_num_workers": 0' in body


def test_completion_masking_falls_back_on_cpu():
    body = _func_src(_TRAINER, "_train_worker")
    assert "train_on_responses_enabled and getattr(self, \"_cpu_training\", False)" in body


def test_module_import_survives_a_host_without_unsloth():
    """The guarded import is what lets this module load on the CPU hosts the path exists for."""
    source = _src(_TRAINER)
    assert (
        "from unsloth import FastLanguageModel, FastVisionModel, is_bfloat16_supported"
        in source
    )
    assert "except Exception as _unsloth_train_exc" in source
    assert "_UNSLOTH_TRAIN_IMPORT_ERROR" in source
