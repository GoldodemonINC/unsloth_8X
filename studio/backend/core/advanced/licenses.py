# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""License audit for merged models.

Merging two checkpoints produces a model whose licence is a function of both
parents, and that is exactly the fact a merge report used to omit. The audit
reads the licence a checkpoint declares (``config.json`` fields and the YAML
front matter of a model card) and states what the merged result inherits.

It does not interpret law. It reports:

* the licence each parent declares, or that it declares none,
* whether the parents disagree (the output then inherits the stricter one, and a
  human has to decide),
* flags for the common non-commercial and research-only markers, and for
  licences whose terms are not in the file (``other``/``custom``).

Nothing here fetches the network: a Hub repo is audited from its local snapshot,
so the audit costs nothing at merge time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

# Markers worth surfacing before someone ships a merged model.
_NON_COMMERCIAL = ("non-commercial", "noncommercial", "nc", "cc-by-nc", "research only")
_COPYLEFT = ("gpl", "agpl", "lgpl", "cc-by-sa", "openrail")
_PERMISSIVE = ("apache", "mit", "bsd", "cc-by-4.0", "llama", "gemma", "qwen")

_README_NAMES = ("README.md", "README.MD", "readme.md", "MODEL_CARD.md")


def _checkpoint_root(source: str | Path) -> Path:
    path = Path(source).expanduser()
    if path.is_file():
        return path.parent
    if (path / "snapshots").is_dir():
        for child in sorted((path / "snapshots").iterdir()):
            if child.is_dir() and (child / "config.json").is_file():
                return child
    return path


def _readme_front_matter(path: Path) -> dict[str, str]:
    """Parse the simple ``key: value`` pairs at the top of a model card."""
    try:
        text = path.read_text(encoding = "utf-8", errors = "replace")
    except OSError:
        return {}
    if not text.startswith("---"):
        return {}
    fields: dict[str, str] = {}
    for line in text.splitlines()[1:]:
        if line.strip() == "---":
            break
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        value = value.strip().strip("'\"")
        if value and value not in ("|", ">"):
            fields[key.strip()] = value
    return fields


def _classify(licence: str | None) -> str:
    if not licence:
        return "undeclared"
    lowered = licence.lower()
    if any(marker in lowered for marker in _NON_COMMERCIAL):
        return "non-commercial"
    if any(marker in lowered for marker in _COPYLEFT):
        return "copyleft"
    if any(marker in lowered for marker in _PERMISSIVE):
        return "permissive"
    if lowered in ("other", "custom", "unknown"):
        return "unclear"
    return "other"


def checkpoint_license(source: str | Path) -> dict[str, Any]:
    """What one checkpoint declares about its licence."""
    root = _checkpoint_root(source)
    licence: str | None = None
    origin = None
    config_path = root / "config.json"
    if config_path.is_file():
        try:
            import json  # noqa: PLC0415

            config = json.loads(config_path.read_text(encoding = "utf-8"))
        except (OSError, ValueError):
            config = {}
        if isinstance(config, dict):
            for key in ("license", "license_name", "licence"):
                value = config.get(key)
                if isinstance(value, str) and value.strip():
                    licence, origin = value.strip(), f"config.json:{key}"
                    break
    if licence is None:
        for name in _README_NAMES:
            candidate = root / name
            if not candidate.is_file():
                continue
            fields = _readme_front_matter(candidate)
            for key in ("license", "license_name", "licence"):
                value = fields.get(key)
                if value:
                    licence, origin = value.strip(), f"{name}:{key}"
                    break
            if licence is not None:
                break
    return {
        "source": str(root),
        "license": licence,
        "license_origin": origin,
        "category": _classify(licence),
        "flags": _flags(licence),
    }


def _flags(licence: str | None) -> list[str]:
    if not licence:
        return [
            "No licence is declared. Redistributing a merge of this checkpoint has no "
            "stated terms; check the model card or the source repository."
        ]
    lowered = licence.lower()
    flags: list[str] = []
    if any(marker in lowered for marker in _NON_COMMERCIAL):
        flags.append("Non-commercial licence: the merged result inherits it.")
    if "research only" in lowered or "research-only" in lowered:
        flags.append("Research-only licence: the merged result inherits the restriction.")
    if any(marker in lowered for marker in _COPYLEFT):
        flags.append("Copyleft-style licence: derivative terms may apply to the merge.")
    if lowered in ("other", "custom", "unknown"):
        flags.append("Licence terms are not in the checkpoint; read the model card.")
    return flags


def audit(sources: Iterable[str | Path], *, stage: str = "merge") -> dict[str, Any]:
    """Audit every parent and state what the output inherits.

    ``effective`` is the strictest declared category: a non-commercial parent
    makes the merge non-commercial. ``conflict`` means the parents disagree, so
    the caller must decide rather than assume.
    """
    entries = [checkpoint_license(source) for source in sources]
    declared = [entry["license"] for entry in entries if entry["license"]]
    categories = {entry["category"] for entry in entries}
    if "non-commercial" in categories:
        effective = "non-commercial"
    elif "copyleft" in categories:
        effective = "copyleft"
    elif "unclear" in categories:
        effective = "unclear"
    elif "undeclared" in categories:
        effective = "undeclared"
    elif declared:
        effective = "permissive"
    else:
        effective = "undeclared"

    notes: list[str] = []
    distinct = {licence.lower() for licence in declared}
    # Two permissive licences are compatible; a restrictive one beside a different
    # licence is the case a human has to settle.
    restrictive = any(
        category in ("non-commercial", "copyleft", "unclear") for category in categories
    )
    conflict = len(distinct) > 1 and restrictive
    if conflict:
        notes.append(
            "The parents declare different licences ("
            + ", ".join(sorted({licence for licence in declared}))
            + f"); the {stage} inherits the stricter one. Confirm the terms before "
            "distributing the result."
        )
    if effective == "non-commercial":
        notes.append(
            "At least one parent is non-commercial, so the merged model is "
            "non-commercial regardless of the other parent."
        )
    if not declared:
        notes.append(
            "No parent declares a licence. The merge is usable locally, but you have no "
            "stated terms for redistributing it."
        )
    return {
        "stage": stage,
        "effective": effective,
        "conflict": conflict,
        "parents": entries,
        "declared": sorted({licence for licence in declared}),
        "notes": notes,
    }


def to_report_block(sources: Iterable[str | Path], *, stage: str = "merge") -> dict[str, Any]:
    """The block written into ``merge_report.json`` and the run manifest."""
    return audit(sources, stage = stage)
