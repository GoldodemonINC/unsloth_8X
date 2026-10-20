# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Request validation for the Dataset panel's second (evaluation) dataset."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

pytest.importorskip("pydantic")

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))


def _load_training_models():
    # Loaded by path so the request schema can be exercised without the models package
    # pulling in the auth stack through models/__init__.py.
    spec = importlib.util.spec_from_file_location(
        "_training_request_models", _BACKEND_DIR / "models" / "training.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TrainingStartRequest = _load_training_models().TrainingStartRequest


def _request(**overrides):
    base = {
        "model_name": "unsloth/Qwen3-1.7B",
        "training_type": "LoRA/QLoRA",
        "hf_dataset": "unsloth/alpaca-cleaned",
        "format_type": "auto",
        "train_split": "train",
        "eval_steps": 0.1,
    }
    base.update(overrides)
    return TrainingStartRequest(**base)


def test_a_second_dataset_is_carried_with_its_subset():
    request = _request(
        eval_hf_dataset="  tatsu-lab/alpaca_eval  ",
        eval_dataset_subset="default",
        eval_split="test",
    )
    assert request.eval_hf_dataset == "tatsu-lab/alpaca_eval"
    assert request.eval_dataset_subset == "default"
    assert request.eval_split == "test"


def test_a_second_dataset_needs_evaluation_enabled():
    """A selection that is never read is the failure nobody notices until the end."""
    with pytest.raises(ValueError, match = "eval_steps above 0"):
        _request(eval_hf_dataset="tatsu-lab/alpaca_eval", eval_steps = 0)
    with pytest.raises(ValueError, match = "eval_steps above 0"):
        _request(eval_hf_dataset="tatsu-lab/alpaca_eval", eval_steps = float("inf"))


def test_the_training_dataset_cannot_be_its_own_eval_dataset():
    with pytest.raises(ValueError, match = "leave it empty"):
        _request(eval_hf_dataset="unsloth/alpaca-cleaned")


def test_a_malformed_eval_repo_id_is_refused():
    with pytest.raises(ValueError, match = "eval_hf_dataset"):
        _request(eval_hf_dataset="not a/repo id")


def test_a_plain_split_name_is_still_required_for_streaming():
    # The eval split now belongs to the second dataset, but it is still a split
    # name: slice syntax cannot be streamed.
    with pytest.raises(ValueError, match = "plain split name"):
        _request(
            dataset_streaming = True,
            max_steps = 10,
            eval_hf_dataset="tatsu-lab/alpaca_eval",
            eval_split="train[:100]",
        )
