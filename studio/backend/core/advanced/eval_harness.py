# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Post-run automated evaluation harness.

When a run finishes, the harness scores the new checkpoint against a held-out
benchmark suite, compares it with the baseline model, and reports a quantitative
delta per benchmark plus one aggregate score. The point is to catch a run that
lowered loss while regressing instruction-following or reasoning -- something a
training-loss curve cannot show.

Benchmarks are executed by `lm-evaluation-harness` when it is installed (the
standard, reproducible path). When it is not, the plan says so explicitly rather
than silently reporting zeroes; the UI surfaces that as "install lm-eval".
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import eval_tasks

# Registry keys match lm-eval task names where one exists.
SUITES: dict[str, dict[str, Any]] = {
    "mmlu": {
        "label": "MMLU",
        "kind": "multiple_choice",
        "shots": 5,
        "metric": "acc",
        "description": "Broad academic knowledge across 57 subjects.",
    },
    "gsm8k": {
        "label": "GSM8K",
        "kind": "generative",
        "shots": 5,
        "metric": "exact_match",
        "description": "Grade-school math word problems.",
    },
    "arc_challenge": {
        "label": "ARC-Challenge",
        "kind": "multiple_choice",
        "shots": 25,
        "metric": "acc_norm",
        "description": "Science reasoning questions.",
    },
    "hellaswag": {
        "label": "HellaSwag",
        "kind": "multiple_choice",
        "shots": 10,
        "metric": "acc_norm",
        "description": "Commonsense sentence completion.",
    },
    "truthfulqa": {
        "label": "TruthfulQA",
        "kind": "multiple_choice",
        "shots": 0,
        "metric": "acc",
        "description": "Resistance to common misconceptions.",
    },
    "humaneval": {
        "label": "HumanEval",
        "kind": "code",
        "shots": 0,
        "metric": "pass@1",
        "description": "Python function synthesis.",
    },
    "ifeval": {
        "label": "IFEval",
        "kind": "instruction",
        "shots": 0,
        "metric": "prompt_level_strict_acc",
        "description": "Verifiable instruction following.",
    },
    "custom": {
        "label": "Custom held-out set",
        "kind": "custom",
        "shots": 0,
        "metric": "perplexity",
        "description": "Perplexity on a local held-out dataset.",
    },
}


def list_suites() -> list[dict[str, Any]]:
    return [{"id": key, **value} for key, value in SUITES.items()]


def harness_available() -> dict[str, Any]:
    """Where lm-eval would run from, without importing it."""
    candidates = {
        "python_module": _module_available("lm_eval"),
        "cli": shutil.which("lm_eval") is not None,
    }
    return {
        "available": any(candidates.values()),
        "lm_eval_module": candidates["python_module"],
        "lm_eval_cli": candidates["cli"],
        "install_hint": "pip install lm-eval",
    }


def _module_available(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


# The distribution name. lm-evaluation-harness renamed its PyPI package to
# lm-eval while keeping the import name ``lm_eval``, so the install name and the
# import name disagree and both have to be written down somewhere.
HARNESS_PACKAGE = "lm-eval"

# An operator cannot remove a button, but can turn off what it does.
DISABLE_ENV = "UNSLOTH_DISABLE_LM_EVAL_INSTALL"


class HarnessInstallError(RuntimeError):
    """The install ran and did not work. The message carries pip's own output tail."""


def install_disabled() -> bool:
    value = os.environ.get(DISABLE_ENV, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def install_plan() -> dict[str, Any]:
    """The command that would install lm-eval here, without running it.

    The interpreter is named explicitly because ``pip`` from PATH can belong to a
    different environment: it would report success while the module stayed
    invisible to this process. When pip is missing from the interpreter, uv is
    asked to install *into* it for the same reason.
    """
    has_pip = _module_available("pip")
    command = (
        [sys.executable, "-m", "pip", "install", HARNESS_PACKAGE]
        if has_pip
        else [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            HARNESS_PACKAGE,
        ]
    )
    return {
        "package": HARNESS_PACKAGE,
        "command": command,
        "python_executable": sys.executable,
        "has_pip": has_pip,
        "has_uv": shutil.which("uv") is not None,
        "available": harness_available()["available"],
        "disabled": install_disabled(),
        "install_hint": harness_available()["install_hint"],
    }


def install(*, timeout: float = 900.0) -> dict[str, Any]:
    """Install lm-evaluation-harness into the interpreter Studio is running on.

    This is the action behind the "pip install lm-eval" hint, so it runs that
    command and nothing more: dependencies are left to pip, because pinning them
    here would go stale against whatever torch this machine already has, and
    lm-eval is only imported when a benchmark actually runs.

    A failure raises instead of returning a flag: the caller has to be able to
    tell "installed" from "pip said something", and a silent False reads like a
    no-op in the UI.
    """
    if install_disabled():
        raise HarnessInstallError(
            f"Installing from Studio is turned off by {DISABLE_ENV}. Install "
            "lm-eval in this environment from a terminal, then check again."
        )
    plan = install_plan()
    try:
        completed = subprocess.run(
            plan["command"],
            capture_output = True,
            text = True,
            timeout = timeout,
        )
    except FileNotFoundError as error:
        raise HarnessInstallError(
            f"Could not run {plan['command'][0]}: {error}. Install lm-eval from a "
            "terminal instead."
        ) from error
    except subprocess.TimeoutExpired as error:
        raise HarnessInstallError(
            f"The install did not finish within {int(timeout)}s and was stopped. "
            "A resolver this slow usually means a large download; run the same "
            "command in a terminal to see it finish."
        ) from error

    output = _tail(f"{completed.stdout or ''}\n{completed.stderr or ''}")
    if completed.returncode != 0:
        raise HarnessInstallError(
            f"`{' '.join(plan['command'])}` exited {completed.returncode}. {output}"
        )

    # find_spec remembers that it looked and found nothing, so without this the
    # check immediately after a successful install still reports the module missing.
    import importlib

    importlib.invalidate_caches()
    available = harness_available()
    return {
        "installed": True,
        "available": available["available"],
        "harness": available,
        "command": plan["command"],
        "output": output,
        # pip exits 0 for "already satisfied" too, and it can be satisfied in a
        # different environment; saying which case this is beats a green tick.
        "note": (
            "lm_eval imports in this environment now; the next benchmark uses it "
            "without a restart."
            if available["available"]
            else "pip exited 0 but lm_eval still does not import here. It may have "
            "installed into another environment; run the command in a terminal "
            "from Studio's Python to see where it lands."
        ),
    }


def _tail(text: str, *, limit: int = 2000) -> str:
    """The end of pip's output, which is where the reason for a failure is."""
    stripped = text.strip()
    return stripped if len(stripped) <= limit else "…" + stripped[-limit:]


@dataclass(frozen=True)
class EvalPlan:
    suites: tuple[str, ...]
    baseline_model: str
    candidate_model: str
    eval_dataset: str | None
    limit: int | None
    batch_size: int
    notes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["suites"] = list(self.suites)
        return payload


def build_eval_plan(
    suites: Sequence[str],
    *,
    baseline_model: str,
    candidate_model: str,
    eval_dataset: str | None = None,
    limit: int | None = None,
    batch_size: int = 8,
    tasks_dir: str | Path | None = None,
) -> EvalPlan:
    if not suites:
        raise ValueError("Select at least one benchmark suite.")
    # A custom task is a suite too: it is validated the same way and scored by
    # the same lm-eval path, so the plan does not need a second code path.
    known = set(SUITES) | eval_tasks.known_suite_ids(tasks_dir)
    unknown = [suite for suite in suites if suite not in known]
    if unknown:
        raise ValueError(f"Unknown benchmark suites: {', '.join(unknown)}.")
    if batch_size <= 0:
        raise ValueError("Eval batch_size must be positive.")
    if limit is not None and limit <= 0:
        raise ValueError("Eval limit must be positive or null.")
    if "custom" in suites and not eval_dataset:
        raise ValueError("The custom suite needs a local held-out dataset path.")
    warnings: list[str] = []
    if limit is None:
        warnings.append(
            "No example limit set: full suites can take hours on a single GPU. "
            "Use a limit for a quick post-run check."
        )
    if not harness_available()["available"]:
        warnings.append(
            "lm-evaluation-harness is not installed in this environment; "
            "install it with `pip install lm-eval` to run the selected suites."
        )
    notes = ["Baseline and candidate are scored with the same settings."]
    return EvalPlan(
        suites=tuple(suites),
        baseline_model=baseline_model,
        candidate_model=candidate_model,
        eval_dataset=eval_dataset,
        limit=limit,
        batch_size=batch_size,
        notes=tuple(notes),
        warnings=tuple(warnings),
    )


def build_lm_eval_command(
    suite: str,
    model: str,
    *,
    limit: int | None = None,
    batch_size: int = 8,
    output_dir: str | None = None,
    task_name: str | None = None,
    include_path: str | None = None,
) -> list[str]:
    """Command line for one suite; kept separate so it is testable.

    ``task_name``/``include_path`` carry a custom task: the task is run by name
    from the generated include directory, so lm-eval loads the same task schema
    it uses for its own suites.
    """
    if task_name is None and suite not in SUITES:
        raise ValueError(f"Unknown suite {suite!r}.")
    command = [
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={model}",
        "--tasks",
        task_name or suite,
        "--batch_size",
        str(batch_size),
    ]
    if include_path:
        command.extend(["--include_path", str(include_path)])
    if limit is not None:
        command.extend(["--limit", str(limit)])
    if output_dir:
        command.extend(["--output_path", output_dir])
    return command


def compute_deltas(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    regression_threshold: float = 0.02,
    noise_floor: float = 0.015,
    higher_is_better: bool = True,
) -> dict[str, Any]:
    """Compare two score maps (suite -> metric value in [0, 1] or percent).

    ``noise_floor`` is the honesty knob: a delta smaller than it is labelled
    *within noise* rather than reported as an improvement. On a 1.5B model 1.5
    points is roughly the run-to-run spread, so calling +0.5 a win would be a
    claim the data cannot support.
    """
    if not baseline:
        raise ValueError("Baseline scores are empty.")
    entries = []
    for suite, base_value in baseline.items():
        if suite not in candidate:
            continue
        base = float(base_value)
        cand = float(candidate[suite])
        delta = cand - base
        signed = delta if higher_is_better else -delta
        status = (
            "regressed"
            if signed < -abs(regression_threshold)
            else "improved"
            if signed > abs(regression_threshold)
            else "flat"
        )
        within_noise = abs(delta) <= abs(noise_floor)
        entries.append(
            {
                "suite": suite,
                "label": SUITES.get(suite, {}).get("label", suite),
                "baseline": base,
                "candidate": cand,
                "delta": delta,
                "relative": (delta / base) if base else None,
                "status": status,
                "within_noise": within_noise,
                "noise_note": (
                    f"within noise (±{abs(noise_floor) * 100:.1f} pts)"
                    if within_noise
                    else None
                ),
            }
        )
    if not entries:
        raise ValueError("No overlapping suites between baseline and candidate.")
    aggregate = sum(entry["delta"] for entry in entries) / len(entries)
    regressions = [entry["suite"] for entry in entries if entry["status"] == "regressed"]
    improvements = [e["suite"] for e in entries if e["status"] == "improved"]
    if regressions:
        verdict = "regressed"
    elif improvements:
        verdict = "pareto-positive"
    else:
        verdict = "within-noise"
    return {
        "entries": entries,
        "aggregate_delta": aggregate,
        "noise_floor": abs(noise_floor),
        "regressions": regressions,
        "improved": improvements,
        "flat": [e["suite"] for e in entries if e["status"] == "flat"],
        "within_noise": [e["suite"] for e in entries if e["within_noise"]],
        # ``verdict`` keeps the historical vocabulary (regressed / ok) for callers
        # that gate on it; ``honest_verdict`` uses the full label set.
        "verdict": "regressed" if regressions else "ok",
        "honest_verdict": verdict,
    }


def render_markdown_report(report: Mapping[str, Any]) -> str:
    """Render a delta report as Markdown for the run detail and notifications."""
    lines: list[str] = []
    title = report.get("run_name") or report.get("candidate_model") or "Evaluation report"
    lines.append(f"# Regression Guard: {title}")
    lines.append("")
    generated = report.get("generated_at") or datetime.now(timezone.utc).isoformat()
    lines.append(f"_Generated {generated}_")
    lines.append("")
    lines.append("| Benchmark | Baseline | Candidate | Delta | Verdict |")
    lines.append("|---|---:|---:|---:|---|")
    for entry in report.get("comparison", {}).get("entries", []):
        delta = float(entry.get("delta", 0.0))
        note = f" · {entry['noise_note']}" if entry.get("noise_note") else ""
        lines.append(
            f"| {entry.get('label', entry.get('suite'))} | "
            f"{_fmt(entry.get('baseline'))} | {_fmt(entry.get('candidate'))} | "
            f"{delta:+.4f} | {_icon(entry.get('status'))} {entry.get('status')}{note} |"
        )
    comparison = report.get("comparison", {})
    lines.append("")
    floor = comparison.get("noise_floor")
    floor_note = f", noise floor ±{float(floor) * 100:.1f} pts" if floor else ""
    lines.append(
        f"**Aggregate delta: {float(comparison.get('aggregate_delta', 0.0)):+.4f}**  "
        f"(verdict: {comparison.get('honest_verdict') or comparison.get('verdict', 'unknown')}"
        f"{floor_note})"
    )
    if report.get("warnings"):
        lines.append("")
        lines.append("## Warnings")
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
    if report.get("errors"):
        lines.append("")
        lines.append("## Errors")
        for error in report["errors"]:
            lines.append(f"- {error}")
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "—"


def _icon(status: Any) -> str:
    return {"improved": "▲", "regressed": "▼", "flat": "•"}.get(str(status), "•")


def empty_report(plan: EvalPlan, *, errors: Sequence[str] = ()) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan": plan.to_dict(),
        "baseline_scores": {},
        "candidate_scores": {},
        "comparison": {
            "entries": [],
            "aggregate_delta": 0.0,
            "regressions": [],
            "improved": [],
            "flat": [],
            "verdict": "unavailable" if errors else "pending",
        },
        "warnings": list(plan.warnings),
        "errors": list(errors),
    }
