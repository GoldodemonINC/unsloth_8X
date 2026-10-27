# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Runner for the post-run evaluation harness.

The harness shells out to ``lm_eval`` (lm-evaluation-harness) with the same
settings for the baseline and the candidate. Each invocation writes its own
results JSON; parsing is done here so the route stays thin and the parsing is
unit-testable without the dependency installed.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from . import eval_harness as harness
from . import eval_tasks

# Preference order when one task reports several metrics.
_METRIC_PREFERENCE = (
    "acc_norm",
    "acc",
    "exact_match",
    "pass@1",
    "prompt_level_strict_acc",
    "inst_level_strict_acc",
    "word_perplexity",
    "perplexity",
)


def parse_results_file(path: str | Path) -> dict[str, float]:
    """Extract ``{task: score}`` from an lm-eval results JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    results = payload.get("results") or payload
    scores: dict[str, float] = {}
    if not isinstance(results, Mapping):
        raise ValueError("lm-eval results file has no results object.")
    for task, metrics in results.items():
        if not isinstance(metrics, Mapping):
            continue
        value = None
        for metric in _METRIC_PREFERENCE:
            if metric in metrics and isinstance(metrics[metric], (int, float)):
                value = float(metrics[metric])
                break
        if value is None:
            for key, candidate in metrics.items():
                if (
                    isinstance(candidate, (int, float))
                    and not key.endswith("_stderr")
                    and "alias" not in key
                ):
                    value = float(candidate)
                    break
        if value is not None:
            scores[str(task)] = value
    if not scores:
        raise ValueError("lm-eval results contained no numeric task scores.")
    return scores


def _default_limit(suite: str, requested: int | None) -> int | None:
    if requested is not None:
        return requested
    # A post-run check must finish in minutes, not hours. Keep the limit
    # conservative for the heavy suites unless the user opted into a full run.
    # A custom task is deliberately unlimited: it is the user's own set, usually
    # small, and scoring 200 of 2000 samples would report a task score that the
    # task did not earn.
    if suite == "custom" or suite.startswith(eval_tasks.SUITE_PREFIX):
        return None
    return 200


def run_suite(
    suite: str,
    model: str,
    *,
    limit: int | None = None,
    batch_size: int = 8,
    output_dir: str | Path,
    timeout: int = 3600,
    on_status: Callable[[str], None] | None = None,
    tasks_dir: str | Path | None = None,
) -> dict[str, float]:
    """Run one suite and return its parsed scores."""
    if not harness.harness_available()["available"]:
        raise RuntimeError(
            "lm-evaluation-harness is not installed. Install it with `pip install lm-eval` "
            "to run the post-run evaluation."
        )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    effective_limit = _default_limit(suite, limit)
    task = eval_tasks.resolve_suite(suite, tasks_dir)
    task_name = None
    include_path = None
    if task is not None:
        # A custom task is materialised beside this run's results: the scored
        # samples are the validated ones, not whatever the file says now.
        written = eval_tasks.write_task_files(task, out_dir / "tasks")
        task_name = task.lm_eval_name
        include_path = str(Path(written["config"]).parent)
    command = harness.build_lm_eval_command(
        suite,
        model,
        limit=effective_limit,
        batch_size=batch_size,
        output_dir=str(out_dir / f"{suite}.json"),
        task_name=task_name,
        include_path=include_path,
    )
    if on_status:
        on_status(f"Evaluating {suite} on {model} (limit={effective_limit})…")
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-6:]
        raise RuntimeError(
            f"lm-eval failed for {suite} (exit {completed.returncode}): "
            + " | ".join(tail)
        )
    return parse_results_file(out_dir / f"{suite}.json")


def run_plan(
    plan: harness.EvalPlan,
    *,
    output_dir: str | Path,
    timeout: int = 3600,
    on_status: Callable[[str], None] | None = None,
    tasks_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Score baseline and candidate for every suite in the plan, then compare.

    A failure on one suite does not throw away the others: it is recorded in
    ``errors`` and the comparison uses whatever pairs completed.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    baseline: dict[str, float] = {}
    candidate: dict[str, float] = {}
    errors: list[str] = []
    with tempfile.TemporaryDirectory(dir=str(out)) as scratch:
        for suite in plan.suites:
            for model, sink in (
                (plan.baseline_model, baseline),
                (plan.candidate_model, candidate),
            ):
                try:
                    scores = run_suite(
                        suite,
                        model,
                        limit=plan.limit,
                        batch_size=plan.batch_size,
                        output_dir=Path(scratch) / suite / ("baseline" if model == plan.baseline_model else "candidate"),
                        timeout=timeout,
                        on_status=on_status,
                        tasks_dir=tasks_dir,
                    )
                    sink.update(scores)
                except Exception as error:  # noqa: BLE001 - reported, not swallowed
                    errors.append(f"{suite} / {model}: {error}")

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_name": Path(plan.candidate_model).name,
        "plan": plan.to_dict(),
        "baseline_scores": baseline,
        "candidate_scores": candidate,
        "warnings": list(plan.warnings),
        "errors": errors,
    }
    try:
        comparison = harness.compute_deltas(baseline, candidate)
        report["comparison"] = comparison
        report["verdict"] = comparison["verdict"]
    except ValueError as error:
        report["comparison"] = {
            "entries": [],
            "aggregate_delta": 0.0,
            "regressions": [],
            "improved": [],
            "flat": [],
            "verdict": "unavailable",
        }
        report["errors"].append(str(error))

    (out / "auto_eval_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (out / "auto_eval_report.md").write_text(
        harness.render_markdown_report(report), encoding="utf-8"
    )
    return report


def python_executable() -> str:
    return sys.executable
