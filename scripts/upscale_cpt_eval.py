# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Upscale -> CPT -> eval, end to end, on a GPU host.

Expansion is function-preserving at step 0, so the expanded model starts out
computing exactly what the base computed; everything the pipeline claims is
earned by the continued pretraining and measured by the eval. Run this on a
32 GB card or larger: a 3x expansion of a 1.7B model needs room for the weights,
their optimizer state and the activations.

Two things this script is careful about, because they are easy to get wrong:

* The base logits are captured *before* ``expand_model``, which mutates the model
  in place. Passing the same object to ``assert_function_preserved`` as both
  sides would compare it with itself and always "pass".
* ``cpt_train`` saves its best checkpoint as a LoRA adapter when ``use_lora`` is
  on, so it is reloaded with ``PeftModel.from_pretrained``, not
  ``FastLanguageModel.from_pretrained``.

Usage:
    python scripts/upscale_cpt_eval.py [--model Qwen/Qwen2.5-1.5B] [--target 3.4e9]
"""

from __future__ import annotations

import argparse

import torch
from datasets import load_dataset
from peft import PeftModel

from unsloth import (
    FastLanguageModel,
    assert_function_preserved,
    cpt_train,
    evaluate_pipeline,
    expand_model,
    save_expanded_model,
)

# A task is callable(model, tokenizer) -> float in 0..1, higher is better.
# The default suite (built-in QA + arithmetic) needs no data but is only a smoke
# signal; pass your own held-out tasks for a verdict worth quoting.
EVAL_SUITE = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default = "Qwen/Qwen2.5-1.5B")
    parser.add_argument("--target", type = float, default = 3.4e9)
    parser.add_argument("--max-steps", type = int, default = 2000)
    parser.add_argument("--out", default = "./expanded-cpt")
    args = parser.parse_args()

    # 4-bit would be refused: the quantisation scales encode the old matrix shapes.
    model, tokenizer = FastLanguageModel.from_pretrained(
        args.model,
        max_seq_length = 2048,
        load_in_4bit = False,
        dtype = None,
    )

    probe = tokenizer(
        "Unsloth function-preservation check: the expanded model must reproduce "
        "the base model's logits at step 0.",
        return_tensors = "pt",
    ).input_ids.to(model.device)
    with torch.no_grad():
        base_logits = model(input_ids = probe).logits.float().clone()

    print("scoring the base ...")
    base_scores = evaluate_pipeline.eval_only(model, tokenizer, EVAL_SUITE)
    print("base:", base_scores)

    print("expanding ...")
    model, report = expand_model(
        model,
        target_params = args.target,
        method = "depth+mlp_width",
        insert_strategy = "even",
    )
    print(
        {
            key: report[key]
            for key in (
                "method_used",
                "base_params",
                "final_params",
                "added_params",
                "growth_factor",
                "new_num_layers",
                "inserted_at",
                "new_intermediate_size",
            )
        }
    )
    assert report["growth_factor"] >= 2.0, report

    diff = assert_function_preserved(None, model, probe, base_logits = base_logits)
    print(f"step-0 max |delta logit| (in memory): {diff:.3e}")

    saved = save_expanded_model(
        model,
        tokenizer,
        f"{args.out}-checkpoint",
        sample_input_ids = probe,
    )
    print(f"step-0 max |delta logit| (after reload): {saved['max_logit_diff']:.3e}")

    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split = "train[:5000]")
    result = cpt_train(
        model,
        tokenizer,
        dataset,
        eval_suite = EVAL_SUITE,
        base_scores = base_scores,
        max_steps = args.max_steps,
        eval_every = 100,
        patience = 5,
    )
    print(
        {
            "recovery_step": result["recovery_step"],
            "gpu_hours": round(result["gpu_hours"], 3),
            "tokens_seen": result["tokens_seen"],
            "peak_mem_gb": round(result["peak_mem_gb"], 2),
            "best_checkpoint": result["best_checkpoint"],
        }
    )

    # The best checkpoint is a LoRA adapter: wrap the expanded base with it.
    cpt_model = PeftModel.from_pretrained(model, result["best_checkpoint"])
    verdict = evaluate_pipeline(
        None,  # base scores are cached; None keeps the base off the GPU
        cpt_model,
        EVAL_SUITE,
        tokenizer,
        base_scores = base_scores,
        gpu_hours_expanded = result["gpu_hours"],
    )
    print(verdict["verdict"], "-", verdict["comparison"], "-", verdict["reason"])
    for row in verdict["per_task"]:
        print(
            f"  {row['task']}: {row['base']:.4f} -> {row['candidate']:.4f} "
            f"({row['delta_points']:+.2f} pt)"
        )

    # The comparison the literature is missing: the base trained for the same
    # tokens. Run it on a second card (or after this one) and pass it back in as
    # same_compute_baseline=..., gpu_hours_baseline=... to get a head-to-head
    # verdict instead of "vs base only".


if __name__ == "__main__":
    main()
