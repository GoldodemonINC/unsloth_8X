# SPDX-License-Identifier: NRC-3
# Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-3.md

"""The NRC-3 notice that travels with a produced checkpoint.

A finished merge or expansion is a folder that leaves this machine, and by then
nothing in it says what the terms are. ``run_manifest.json`` already records the
seeds, inputs and code version; the same write now records the license and drops
a plain-text copy beside the weights, so a reader who never opens the manifest
still sees both what is permitted and which upstream licenses came along.

Only the X8 original additions and the run record are covered. The weights,
tokenizer and dataset keep whatever license their source had, and the notice
says so rather than implying the whole folder was relicensed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

LICENSE_NAME = "NRC-3"
LICENSE_VERSION = "1.0"
LICENSE_TITLE = "Non-Rebranding Commons"
LICENSE_FILE = "LICENSE-NRC-3.txt"
LICENSE_DOCUMENT = "LICENSE-NRC-3.md"

NOTICE_TEXT = """NRC-3 License 1.0 (Non-Rebranding Commons)
X8 / Unsloth Studio

This folder was produced by X8. The NRC-3 terms cover the X8 original
additions and this run record. Weights, tokenizers and datasets inherited from
a source checkpoint or dataset keep that source's license; nothing here
relicenses them.

You may:
  - use, modify and self-host this freely, including commercially and inside a
    company, with no fee and no permission;
  - share it, including modified copies, if you keep this notice and the file
    headers and say what you changed;
  - publish your changed source, teach with it, write about it and mirror it
    for research.

You may not, without written permission:
  - present a rebranded copy as your own product, under a different name, logo
    or interface, or under the project's name with a qualifier that suggests it
    is an official version;
  - run it as a hosted service for other people, whether or not you charge for
    it;
  - distribute packaged builds to the public at scale, or include it in a
    product or service you sell;
  - strip this notice, a file header, or the license block in
    run_manifest.json;
  - call a copy official or endorsed, or attribute results to the project that
    it did not produce.

The full terms are in LICENSE-NRC-3.md at the project root. This notice is a
summary; where the two differ, the document governs.
"""


def notice_text() -> str:
    """The plain-text notice written beside every produced checkpoint."""
    return NOTICE_TEXT


def license_record() -> dict[str, Any]:
    """The ``license`` block stored in ``run_manifest.json``."""
    return {
        "name": LICENSE_NAME,
        "version": LICENSE_VERSION,
        "title": LICENSE_TITLE,
        "file": LICENSE_FILE,
        "document": LICENSE_DOCUMENT,
        "applies_to": "X8 original additions and this run record",
        "upstream_unchanged": True,
        "note": (
            "Weights, tokenizers and datasets inherited from a source keep that "
            "source's license."
        ),
    }


def stamp(directory: str | Path, record: Mapping[str, Any] | None = None) -> Path | None:
    """Write the notice into *directory*; a read-only folder is not an error.

    The manifest is the record of truth and has already been written by the time
    this runs, so a failed notice must never fail the stage that produced it.
    """
    path = Path(directory) / LICENSE_FILE
    try:
        path.write_text(notice_text(), encoding = "utf-8")
    except OSError:
        return None
    return path
