# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Policy Alignment (DPO / GRPO) training entrypoint.

Runs in the training worker subprocess, like the SFT path. It reuses the
Unsloth model loader and PEFT setup, then hands the dataset to TRL's
``DPOTrainer`` or ``GRPOTrainer``. Events use the same schema as SFT training so
the existing Studio progress UI works unchanged.
"""

from __future__ import annotations

import inspect
import math
import os
import queue as queue_module
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

from . import rl as rl_math


def _send_status(event_queue, message: str) -> None:
    event_queue.put({"type": "status", "message": message, "ts": time.time()})


def _send_error(event_queue, message: str) -> None:
    event_queue.put(
        {
            "type": "error",
            "error": message,
            "stack": traceback.format_exc(limit = 12),
            "ts": time.time(),
        }
    )


def _filter_kwargs(factory, values: Mapping[str, Any]) -> dict[str, Any]:
    """Drop config keys this TRL version does not accept."""
    try:
        parameters = inspect.signature(factory.__init__).parameters
    except (TypeError, ValueError):
        return dict(values)
    if any(parameter.kind == parameter.VAR_KEYWORD for parameter in parameters.values()):
        return dict(values)
    return {key: value for key, value in values.items() if key in parameters}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _load_rl_dataset(config: Mapping[str, Any]):
    from datasets import load_dataset

    hf_token = (config.get("hf_token") or "").strip() or None
    local_datasets = list(config.get("local_datasets") or [])
    dataset_name = config.get("hf_dataset")

    if local_datasets:
        files = [
            path
            if os.path.isabs(path)
            else str(Path(os.environ.get("UNSLOTH_DATASETS_DIR", ".")) / path)
            for path in local_datasets
        ]
        suffix = Path(files[0]).suffix.lower()
        loader = {
            ".json": "json",
            ".jsonl": "json",
            ".csv": "csv",
            ".parquet": "parquet",
        }.get(suffix)
        if loader is None:
            raise ValueError(f"Unsupported RL dataset format: {files[0]}")
        return load_dataset(loader, data_files = files, split = "train")
    if dataset_name:
        return load_dataset(
            dataset_name,
            name = config.get("subset"),
            split = config.get("train_split") or "train",
            token = hf_token,
        )
    raise ValueError("Policy Alignment needs a local or Hugging Face dataset.")


def _prepare_dpo_dataset(dataset) -> Any:
    columns = set(dataset.column_names)
    missing = [name for name in ("prompt", "chosen", "rejected") if name not in columns]
    if missing:
        raise ValueError(
            "DPO needs prompt, chosen, and rejected columns; missing: " + ", ".join(missing)
        )
    keep = [
        name
        for name in ("prompt", "chosen", "rejected", "system", "images")
        if name in columns
    ]
    return dataset.select_columns(keep)


def _prepare_grpo_dataset(dataset) -> Any:
    if "prompt" not in dataset.column_names:
        raise ValueError("GRPO needs a prompt column.")
    return dataset


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


def _make_reward_function(rl_config: Mapping[str, Any]):
    preset = str(rl_config.get("reward_preset") or "exact_match")
    weights = rl_config.get("reward_weights") or None

    def _reward(completions, **kwargs):
        references = (
            kwargs.get("reference")
            or kwargs.get("references")
            or kwargs.get("answer")
            or kwargs.get("chosen")
        )
        if isinstance(references, str):
            references = [references] * len(completions)
        return rl_math.score_completions(
            [str(completion) for completion in completions],
            references,
            preset = preset,
            weights = weights,
        )

    _reward.__name__ = f"unsloth_{preset}_reward"
    return _reward


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class _ProgressCallback:
    """TRL/transformers callback factory (imported lazily)."""

    @staticmethod
    def build(event_queue, stop_event: threading.Event, is_grpo: bool):
        from transformers import TrainerCallback

        class _Callback(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                _send_status(event_queue, "Policy alignment training in progress...")
                event_queue.put(
                    {
                        "type": "progress",
                        "step": 0,
                        "epoch": 0,
                        "loss": None,
                        "learning_rate": args.learning_rate,
                        "total_steps": state.max_steps,
                        "elapsed_seconds": 0.0,
                        "eta_seconds": None,
                        "ts": time.time(),
                    }
                )

            def on_log(self, args, state, control, logs = None, **kwargs):
                if not logs:
                    return
                loss = logs.get("loss")
                if loss is None:
                    loss = logs.get("rewards/accuracies")
                event_queue.put(
                    {
                        "type": "progress",
                        "step": int(state.global_step),
                        "epoch": round(float(state.epoch or 0.0), 4),
                        "loss": float(loss) if isinstance(loss, (int, float)) else None,
                        "learning_rate": logs.get("learning_rate"),
                        "total_steps": int(state.max_steps),
                        "elapsed_seconds": 0.0,
                        "eta_seconds": None,
                        "grad_norm": logs.get("grad_norm"),
                        "ts": time.time(),
                    }
                )

            def on_step_end(self, args, state, control, **kwargs):
                if stop_event.is_set():
                    control.should_training_stop = True

        return _Callback()


def _start_stop_poller(stop_queue, stop_event: threading.Event) -> None:
    def poll() -> None:
        while not stop_event.is_set():
            try:
                message = stop_queue.get(timeout = 1.0)
            except queue_module.Empty:
                continue
            except (EOFError, OSError, ValueError):
                return
            if isinstance(message, dict) and message.get("type") == "stop":
                stop_event.set()
                return

    threading.Thread(target = poll, daemon = True, name = "rl-stop-poller").start()


def _build_trainer(trainer_class, values: dict[str, Any]):
    """Instantiate a TRL trainer across the tokenizer -> processing_class rename."""
    try:
        parameters = inspect.signature(trainer_class.__init__).parameters
        accepts_kwargs = any(
            parameter.kind == parameter.VAR_KEYWORD for parameter in parameters.values()
        )
    except (TypeError, ValueError):
        parameters, accepts_kwargs = {}, True
    if not accepts_kwargs and "processing_class" not in parameters:
        values["tokenizer"] = values.pop("processing_class")
    return trainer_class(**values)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def run_policy_alignment(event_queue, stop_queue, config: Mapping[str, Any]) -> None:
    """Run a DPO or GRPO pass and save the adapter to the run's output dir."""
    start = time.time()
    try:
        import torch
        from unsloth import FastLanguageModel
        from trl import DPOConfig, DPOTrainer, GRPOConfig, GRPOTrainer

        advanced = config.get("advanced") or {}
        raw_rl = dict(advanced.get("rl") or {})
        method = str(raw_rl.pop("method", "dpo")).lower()
        rl_config = rl_math.validate_rl_config(method, raw_rl)

        dataset = _load_rl_dataset(config)
        dataset = (
            _prepare_dpo_dataset(dataset) if method == "dpo" else _prepare_grpo_dataset(dataset)
        )
        rl_math.validate_rl_config(
            method,
            raw_rl,
            dataset_columns = dataset.column_names,
            num_rows = len(dataset),
        )
        for warning in rl_math.validate_rl_config(
            method,
            raw_rl,
            dataset_columns = dataset.column_names,
            num_rows = len(dataset),
        ).get("warnings", []):
            event_queue.put({"type": "warning", "message": warning, "ts": time.time()})

        model_name = config.get("model_name")
        _send_status(event_queue, f"Loading {model_name} for {method.upper()}...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name,
            max_seq_length = int(config.get("max_seq_length") or 2048),
            dtype = None,
            load_in_4bit = bool(config.get("load_in_4bit", True)),
            token = (config.get("hf_token") or "").strip() or None,
            trust_remote_code = bool(config.get("trust_remote_code", False)),
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r = int(config.get("lora_r") or 16),
            target_modules = config.get("target_modules") or None,
            lora_alpha = int(config.get("lora_alpha") or 16),
            lora_dropout = float(config.get("lora_dropout") or 0.0),
            bias = "none",
            use_gradient_checkpointing = config.get("gradient_checkpointing", "unsloth"),
            random_state = int(config.get("random_seed") or 3407),
            use_rslora = bool(config.get("use_rslora", False)),
            use_dora = bool(config.get("use_dora", False)),
            loftq_config = None,
        )
        FastLanguageModel.for_training(model)

        output_dir = str(
            Path(config.get("output_dir") or "rl_output").expanduser()
        )
        Path(output_dir).mkdir(parents = True, exist_ok = True)

        max_steps = int(config.get("max_steps") or 0)
        common = {
            "output_dir": output_dir,
            "learning_rate": float(rl_config["learning_rate"]),
            "per_device_train_batch_size": int(config.get("batch_size") or 1),
            "gradient_accumulation_steps": int(config.get("gradient_accumulation_steps") or 4),
            "num_train_epochs": float(config.get("num_epochs") or 1),
            "max_steps": max_steps if max_steps > 0 else -1,
            "logging_steps": 1,
            "save_steps": max(1, int(config.get("save_steps") or 50)),
            "save_total_limit": 2,
            "seed": int(config.get("random_seed") or 3407),
            "report_to": ["wandb"] if config.get("enable_wandb") else "none",
            "bf16": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
            "fp16": bool(torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
            "gradient_checkpointing": True,
            "remove_unused_columns": False,
            "disable_tqdm": True,
        }

        stop_event = threading.Event()
        _start_stop_poller(stop_queue, stop_event)
        callbacks = [_ProgressCallback.build(event_queue, stop_event, method == "grpo")]

        max_seq = int(config.get("max_seq_length") or 2048)
        if method == "dpo":
            dpo_values = {
                **common,
                "beta": float(rl_config["beta"]),
                "label_smoothing": float(rl_config["label_smoothing"]),
                "max_length": max_seq,
                "max_prompt_length": max(128, max_seq // 2),
            }
            args = DPOConfig(**_filter_kwargs(DPOConfig, dpo_values))
            trainer = _build_trainer(
                DPOTrainer,
                {
                    "model": model,
                    "ref_model": None,  # PEFT: the adapter-disabled policy is the reference
                    "args": args,
                    "train_dataset": dataset,
                    "processing_class": tokenizer,
                    "callbacks": callbacks,
                },
            )
        else:
            grpo_values = {
                **common,
                "num_generations": int(rl_config["num_generations"]),
                "max_completion_length": int(rl_config["max_completion_length"]),
                "max_prompt_length": some_prompt_length(max_seq),
                "temperature": float(rl_config["temperature"]),
                "top_p": float(rl_config["top_p"]),
                "beta": float(rl_config["kl_coefficient"]),
            }
            args = GRPOConfig(**_filter_kwargs(GRPOConfig, grpo_values))
            trainer = _build_trainer(
                GRPOTrainer,
                {
                    "model": model,
                    "args": args,
                    "train_dataset": dataset,
                    "reward_funcs": [_make_reward_function(raw_rl)],
                    "processing_class": tokenizer,
                    "callbacks": callbacks,
                },
            )

        _send_status(event_queue, f"Starting {method.upper()} training...")
        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        event_queue.put(
            {
                "type": "complete",
                "output_dir": output_dir,
                "status_message": f"{method.upper()} policy alignment completed",
                "elapsed_seconds": round(time.time() - start, 2),
                "ts": time.time(),
            }
        )
    except Exception as error:  # noqa: BLE001 - reported to the UI
        _send_error(event_queue, f"Policy Alignment failed: {error}")


def some_prompt_length(max_seq: int) -> int:
    return max(128, max_seq // 2)
