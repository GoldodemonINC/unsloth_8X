# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the Policy Alignment math."""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.advanced import rl


def test_dpo_loss_prefers_chosen_side():
    good = rl.dpo_loss([0.0], [-2.0], [0.0], [0.0], beta=0.1)
    bad = rl.dpo_loss([-2.0], [0.0], [0.0], [0.0], beta=0.1)
    assert good["loss"] < bad["loss"]
    assert good["reward_margin"] > 0
    assert good["accuracy"] == 1.0
    assert bad["accuracy"] == 0.0
    # At a zero margin the loss is log(2).
    neutral = rl.dpo_loss([0.0], [0.0], [0.0], [0.0], beta=0.1)
    assert neutral["loss"] == pytest.approx(math.log(2.0), rel=1e-6)


def test_dpo_label_smoothing_and_validation():
    smoothed = rl.dpo_loss([1.0], [0.0], [0.0], [0.0], beta=0.1, label_smoothing=0.1)
    plain = rl.dpo_loss([1.0], [0.0], [0.0], [0.0], beta=0.1, label_smoothing=0.0)
    assert smoothed["loss"] > plain["loss"]
    with pytest.raises(ValueError):
        rl.dpo_loss([0.0], [0.0], [0.0], [0.0], beta=0)
    with pytest.raises(ValueError):
        rl.dpo_loss([0.0, 1.0], [0.0], [0.0], [0.0])


def test_sequence_logprobs_ignores_masked_tokens():
    # Two positions, vocab 3; label -100 must contribute nothing.
    logits = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]], dtype=np.float64)
    labels = np.array([[0, 1, -100]], dtype=np.int64)
    token_logps, totals = rl.sequence_logprobs_from_logits(logits, labels)
    assert totals.shape == (1,)
    # First predicted token is 1 with uniform logits -> -log(3); second is ignored.
    assert totals[0] == pytest.approx(-math.log(3.0), rel=1e-6)


def test_grpo_advantages_are_group_relative():
    advantages = rl.grpo_advantages([[1.0, 3.0, 2.0, 2.0]])
    assert advantages.shape == (1, 4)
    assert np.isclose(advantages.mean(), 0.0, atol=1e-8)
    assert advantages[0, 1] > advantages[0, 0]
    flat = rl.grpo_advantages([0.0, 0.0, 0.0, 0.0], group_size=4)
    assert np.allclose(flat, 0.0)
    with pytest.raises(ValueError):
        rl.grpo_advantages([1.0, 2.0, 3.0], group_size=2)


def test_grpo_clip_limits_update():
    result = rl.grpo_policy_loss(
        [1.0, -1.0],
        [0.0, 0.0],
        [1.0, -1.0],
        clip_ratio=0.2,
    )
    assert result["clip_fraction"] >= 0.0
    assert 0.0 <= result["mean_ratio"]
    # A zero advantage means no policy gradient signal.
    zero = rl.grpo_policy_loss([0.5], [0.0], [0.0])
    assert zero["loss"] == pytest.approx(0.0, abs=1e-12)


def test_reward_presets():
    assert rl.score_completions(["42"], ["42"], preset="exact_match") == [1.0]
    assert rl.score_completions(["the answer is 42"], ["42"], preset="contains") == [1.0]
    assert rl.score_completions(["<answer>42</answer>"], preset="format") == [1.0]
    assert rl.score_completions(["no tags"], preset="format") == [0.0]
    reasoning = "<reasoning>think</reasoning><answer>42</answer>"
    assert rl.score_completions([reasoning], preset="reasoning_tags") == [1.0]
    weighted = rl.score_completions(
        ["42"], ["42"], weights={"exact_match": 0.75, "length": 0.25}
    )
    assert 0.0 <= weighted[0] <= 1.0
    with pytest.raises(ValueError):
        rl.score_completions(["x"], preset="nope")


def test_validate_rl_config():
    dpo = rl.validate_rl_config(
        "dpo",
        {"beta": 0.1},
        dataset_columns=["prompt", "chosen", "rejected"],
        num_rows=100,
    )
    assert dpo["method"] == "dpo"
    assert dpo["warnings"] == []

    with pytest.raises(ValueError):
        rl.validate_rl_config(
            "dpo", {}, dataset_columns=["prompt", "chosen"], num_rows=100
        )
    with pytest.raises(ValueError):
        rl.validate_rl_config("grpo", {"num_generations": 1})
    with pytest.raises(ValueError):
        rl.validate_rl_config("ppo", {})

    grpo = rl.validate_rl_config(
        "grpo",
        {"num_generations": 8, "reward_preset": "format"},
        dataset_columns=["prompt"],
        num_rows=32,
    )
    assert grpo["num_generations"] == 8
    assert grpo["warnings"] == []
