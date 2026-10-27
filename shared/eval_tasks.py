# SPDX-License-Identifier: NRC-2
# Copyright 2026-present the X8 authors. See /LICENSE-NRC-2.md

"""User-defined evaluation tasks.

The harness ships a fixed set of suites (MMLU, GSM8K, and the rest). That is
enough to answer "did this fine-tune get worse at general ability", and not
enough to answer "did it get worse at *our* task" -- which is the question a
domain fine-tune actually cares about.

A custom task is a file the user drops in a directory:

    ~/.unsloth/eval-tasks/support-triage.jsonl      one sample per line
    ~/.unsloth/eval-tasks/ticket-severity.json      metadata + samples

Generative samples carry a `prompt` and an `answer`; multiple-choice samples
carry a `prompt`, a `choices` list and an `answer` (an index or the exact text
of the right choice). Nothing here loads a model or imports a scoring library:
the tasks are normalised, validated, and rendered into the task config that
`lm_eval` already knows how to run, so a custom task is scored by exactly the
same code path as a built-in one. That matters for the comparison report -- two
different scorers would make the delta meaningless.

Task ids are namespaced ``custom:<slug>`` so a local task can never shadow a
built-in suite, and the generated lm-eval name is prefixed ``x8_local_`` so it
cannot shadow an upstream task either.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SUITE_PREFIX = "custom:"
LM_EVAL_PREFIX = "x8_local_"

#: Where tasks are read from when no directory is given.
DEFAULT_DIRECTORY = Path.home() / ".unsloth" / "eval-tasks"
DIRECTORY_ENV = "UNSLOTH_EVAL_TASKS_DIR"

KINDS = ("generative", "multiple_choice")

#: metric -> (lm-eval output_type, higher_is_better). Only metrics lm-eval
#: scores natively are offered; inventing one would mean scoring differently
#: from the built-in suites.
METRICS: dict[str, tuple[str, bool]] = {
    "acc": ("multiple_choice", True),
    "acc_norm": ("multiple_choice", True),
    "exact_match": ("generate_until", True),
}

_PROMPT_KEYS = ("prompt", "question", "input", "text")
_ANSWER_KEYS = ("answer", "target", "completion", "label")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


class EvalTaskError(ValueError):
    """A task file is present but cannot be used; the message names the file."""


@dataclass(frozen = True)
class EvalTask:
    slug: str
    label: str
    kind: str
    metric: str
    samples: tuple[dict[str, Any], ...]
    description: str = ""
    source: str = ""

    @property
    def suite_id(self) -> str:
        """The id the plan and the UI use, e.g. ``custom:support-triage``."""
        return f"{SUITE_PREFIX}{self.slug}"

    @property
    def lm_eval_name(self) -> str:
        return f"{LM_EVAL_PREFIX}{self.slug}"

    def suite_entry(self) -> dict[str, Any]:
        """The same shape ``eval_harness.list_suites`` returns."""
        return {
            "id": self.suite_id,
            "label": self.label,
            "kind": self.kind,
            "shots": 0,
            "metric": self.metric,
            "description": self.description,
            "custom": True,
            "samples": len(self.samples),
            "source": self.source,
        }


def tasks_directory(directory: str | Path | None = None) -> Path:
    """Resolve the task directory: explicit, then env, then the default."""
    raw = directory or os.environ.get(DIRECTORY_ENV) or DEFAULT_DIRECTORY
    return Path(raw).expanduser()


def slugify(text: str) -> str:
    slug = _SLUG_RE.sub("-", str(text).strip().lower()).strip("-")
    return slug or "task"


def _first(record: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def _normalise_samples(raw: Iterable[Any], kind: str, *, source: str) -> tuple[dict[str, Any], ...]:
    samples: list[dict[str, Any]] = []
    for position, record in enumerate(raw, start = 1):
        if not isinstance(record, Mapping):
            raise EvalTaskError(f"{source}: sample {position} is not an object.")
        prompt = _first(record, _PROMPT_KEYS)
        if prompt is None:
            raise EvalTaskError(
                f"{source}: sample {position} has no prompt "
                f"(one of {', '.join(_PROMPT_KEYS)})."
            )
        if kind == "multiple_choice":
            choices = record.get("choices")
            if not isinstance(choices, (list, tuple)) or len(choices) < 2:
                raise EvalTaskError(
                    f"{source}: sample {position} needs a choices list with at "
                    "least two entries."
                )
            answer = record.get("answer")
            index = _choice_index(answer, choices)
            if index is None:
                raise EvalTaskError(
                    f"{source}: sample {position} answer {answer!r} is neither an "
                    "index nor one of its choices."
                )
            samples.append(
                {
                    "prompt": str(prompt),
                    "choices": [str(choice) for choice in choices],
                    "answer": index,
                }
            )
        else:
            answer = _first(record, _ANSWER_KEYS)
            if answer is None:
                raise EvalTaskError(
                    f"{source}: sample {position} has no answer "
                    f"(one of {', '.join(_ANSWER_KEYS)})."
                )
            samples.append({"prompt": str(prompt), "answer": str(answer)})
    if not samples:
        raise EvalTaskError(f"{source}: the task has no samples.")
    return tuple(samples)


def _choice_index(answer: Any, choices: Sequence[Any]) -> int | None:
    """Accept an index, a numeric string, or the exact text of a choice.

    Text matching is exact-after-strip on purpose: fuzzy matching would quietly
    score a sample against a choice the task author did not name.
    """
    if isinstance(answer, bool):
        return None
    if isinstance(answer, (int, float)) and float(answer).is_integer():
        index = int(answer)
        return index if 0 <= index < len(choices) else None
    if isinstance(answer, str):
        stripped = answer.strip()
        if stripped.isdigit():
            index = int(stripped)
            return index if 0 <= index < len(choices) else None
        for index, choice in enumerate(choices):
            if str(choice).strip() == stripped:
                return index
    return None


def _infer_kind(records: Sequence[Any]) -> str:
    """A task is multiple-choice when every sample offers choices."""
    if records and all(isinstance(r, Mapping) and isinstance(r.get("choices"), (list, tuple)) for r in records):
        return "multiple_choice"
    return "generative"


def _default_metric(kind: str) -> str:
    return "acc" if kind == "multiple_choice" else "exact_match"


def _build_task(
    *,
    name: str,
    raw_samples: Sequence[Any],
    source: str,
    kind: str | None = None,
    metric: str | None = None,
    label: str | None = None,
    description: str = "",
) -> EvalTask:
    resolved_kind = kind or _infer_kind(raw_samples)
    if resolved_kind not in KINDS:
        raise EvalTaskError(
            f"{source}: kind {resolved_kind!r} is not one of {', '.join(KINDS)}."
        )
    resolved_metric = metric or _default_metric(resolved_kind)
    if resolved_metric not in METRICS:
        raise EvalTaskError(
            f"{source}: metric {resolved_metric!r} is not one of "
            f"{', '.join(sorted(METRICS))}."
        )
    output_type, _ = METRICS[resolved_metric]
    if output_type == "multiple_choice" and resolved_kind != "multiple_choice":
        raise EvalTaskError(
            f"{source}: metric {resolved_metric!r} needs kind 'multiple_choice'."
        )
    if output_type == "generate_until" and resolved_kind != "generative":
        raise EvalTaskError(
            f"{source}: metric {resolved_metric!r} needs kind 'generative'."
        )
    samples = _normalise_samples(raw_samples, resolved_kind, source = source)
    return EvalTask(
        slug = slugify(name),
        label = label or name,
        kind = resolved_kind,
        metric = resolved_metric,
        samples = samples,
        description = description,
        source = source,
    )


def load_task(path: str | Path) -> EvalTask:
    """Read one ``.json`` or ``.jsonl`` task file."""
    file = Path(path)
    source = file.name
    try:
        text = file.read_text(encoding = "utf-8")
    except OSError as error:
        raise EvalTaskError(f"{source}: cannot read the file ({error}).") from error
    if file.suffix.lower() == ".jsonl":
        records: list[Any] = []
        metadata: dict[str, Any] = {}
        for line_number, line in enumerate(text.splitlines(), start = 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise EvalTaskError(
                    f"{source}: line {line_number} is not valid JSON ({error.msg})."
                ) from error
            if isinstance(record, Mapping) and "__task__" in record:
                # An optional first line can rename and describe the task.
                metadata = dict(record["__task__"] or {})
                continue
            records.append(record)
        name = metadata.get("name") or file.stem
        return _build_task(
            name = name,
            raw_samples = records,
            source = source,
            kind = metadata.get("kind"),
            metric = metadata.get("metric"),
            label = metadata.get("label"),
            description = metadata.get("description", ""),
        )
    if file.suffix.lower() == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise EvalTaskError(f"{source}: not valid JSON ({error.msg}).") from error
        if not isinstance(payload, Mapping):
            raise EvalTaskError(f"{source}: expected an object with a samples list.")
        raw_samples = payload.get("samples")
        if not isinstance(raw_samples, (list, tuple)):
            raise EvalTaskError(f"{source}: expected a samples list.")
        name = payload.get("name") or payload.get("label") or file.stem
        return _build_task(
            name = name,
            raw_samples = raw_samples,
            source = source,
            kind = payload.get("kind"),
            metric = payload.get("metric"),
            label = payload.get("label"),
            description = payload.get("description", ""),
        )
    raise EvalTaskError(f"{source}: task files are .json or .jsonl, not {file.suffix or 'extensionless'}.")


def load_tasks(
    directory: str | Path | None = None,
) -> tuple[dict[str, EvalTask], list[str]]:
    """Discover every task in *directory*.

    Returns the tasks that loaded and the problems with the ones that did not,
    so one broken file cannot hide the rest of the folder.
    """
    root = tasks_directory(directory)
    tasks: dict[str, EvalTask] = {}
    problems: list[str] = []
    if not root.is_dir():
        return tasks, problems
    for file in sorted(root.iterdir()):
        if file.suffix.lower() not in (".json", ".jsonl") or not file.is_file():
            continue
        try:
            task = load_task(file)
        except EvalTaskError as error:
            problems.append(str(error))
            continue
        # First file to claim a slug wins; the clash is reported rather than
        # silently overwriting, because the scorer would otherwise change the
        # meaning of a task the user already ran.
        if task.slug in tasks:
            problems.append(
                f"{file.name}: slug {task.slug!r} is already used by "
                f"{tasks[task.slug].source}; rename the task."
            )
            continue
        tasks[task.slug] = task
    return tasks, problems


def resolve_suite(
    suite: str,
    directory: str | Path | None = None,
) -> EvalTask | None:
    """Return the custom task for ``custom:<slug>``, or None for built-ins."""
    if not str(suite).startswith(SUITE_PREFIX):
        return None
    slug = slugify(str(suite)[len(SUITE_PREFIX) :])
    tasks, _ = load_tasks(directory)
    return tasks.get(slug)


def known_suite_ids(directory: str | Path | None = None) -> set[str]:
    tasks, _ = load_tasks(directory)
    return {task.suite_id for task in tasks.values()}


def list_custom_suites(
    directory: str | Path | None = None,
) -> dict[str, Any]:
    """Everything the UI needs to show the folder: tasks, path, problems."""
    root = tasks_directory(directory)
    tasks, problems = load_tasks(directory)
    return {
        "directory": str(root),
        "exists": root.is_dir(),
        "tasks": [task.suite_entry() for task in tasks.values()],
        "errors": problems,
        "formats": {
            "jsonl": "one sample per line; optional first line {\\\"__task__\\\": {\\\"name\\\": ...}}",
            "generative": "{prompt, answer}",
            "multiple_choice": "{prompt, choices: [...], answer: index or exact choice text}",
            "metrics": sorted(METRICS),
        },
    }


def write_task_files(task: EvalTask, include_path: str | Path) -> dict[str, str]:
    """Materialise *task* as the JSONL dataset + YAML config lm-eval runs.

    The dataset is rewritten from the normalised samples rather than pointing at
    the user's file, so a task that was edited between planning and scoring
    cannot score a different set of samples than the one that was validated.
    """
    include = Path(include_path)
    include.mkdir(parents = True, exist_ok = True)
    data_path = include / f"{task.lm_eval_name}.jsonl"
    data_path.write_text(
        "\n".join(json.dumps(sample, ensure_ascii = False) for sample in task.samples) + "\n",
        encoding = "utf-8",
    )
    config_path = include / f"{task.lm_eval_name}.yaml"
    config_path.write_text(render_lm_eval_yaml(task, data_path), encoding = "utf-8")
    return {"data": str(data_path), "config": str(config_path)}


def render_lm_eval_yaml(task: EvalTask, data_path: Path) -> str:
    """The task config, as lm-eval's YAML schema wants it."""
    output_type, higher_is_better = METRICS[task.metric]
    lines = [
        f"task: {task.lm_eval_name}",
        "dataset_path: json",
        "dataset_kwargs:",
        "  data_files:",
        f"    test: {Path(data_path).as_posix()}",
        "test_split: test",
        'doc_to_text: "{{prompt}}"',
        'doc_to_target: "{{answer}}"',
        "num_fewshot: 0",
        f"output_type: {output_type}",
    ]
    if output_type == "multiple_choice":
        lines.append('doc_to_choice: "{{choices}}"')
    else:
        lines.extend(
            [
                "generation_kwargs:",
                "  until:",
                '    - "\\n\\n"',
                "  do_sample: false",
                "  temperature: 0.0",
                "  max_gen_toks: 64",
            ]
        )
    lines.extend(
        [
            "metric_list:",
            f"  - metric: {task.metric}",
            "    aggregation: mean",
            f"    higher_is_better: {'true' if higher_is_better else 'false'}",
            "metadata:",
            f"  version: 1.0",
        ]
    )
    return "\n".join(lines) + "\n"
