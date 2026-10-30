# SPDX-License-Identifier: NRC-3
# Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-3.md

"""Where the features read and write.

Studio resolves these through its account workspace, which knows about
multi-user roots, per-account isolation and symlink escapes. That machinery is
part of the app and is not here. What is here is the part that decides *which*
directory: the same environment variables, in the same priority order, so a
feature run beside Studio lands in the folders Studio lists.

If a path looks wrong, it is because Studio's answer also depends on its stored
settings; these functions deliberately do not, so they stay importable with no
database and no running server.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def studio_root() -> Path:
    """UNSLOTH_STUDIO_HOME, then STUDIO_HOME, then UNSLOTH_HOME/studio, then ~/.unsloth/studio."""
    override = (os.environ.get("UNSLOTH_STUDIO_HOME") or "").strip()
    if not override:
        override = (os.environ.get("STUDIO_HOME") or "").strip()
    if override:
        return Path(override).expanduser()
    home = (os.environ.get("UNSLOTH_HOME") or "").strip()
    if home:
        return Path(home).expanduser() / "studio"
    return Path.home() / ".unsloth" / "studio"


def outputs_root() -> Path:
    """Trained and merged checkpoints, which is also what the Library scans."""
    return studio_root() / "outputs"


def exports_root() -> Path:
    """GGUF and other exported files."""
    return studio_root() / "exports"


@dataclass(frozen = True)
class HuggingFaceCachePaths:
    """The two Hugging Face folders cleanup offers, in the same shape Studio uses."""

    cache_home: Path
    hub_cache: Path


def get_hf_cache_paths() -> HuggingFaceCachePaths:
    """HF_HOME / HF_HUB_CACHE when set, otherwise the default cache layout."""
    cache_home = (os.environ.get("HF_HOME") or "").strip()
    hub_cache = (
        os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE") or ""
    ).strip()
    home = Path(cache_home).expanduser() if cache_home else Path.home() / ".cache" / "huggingface"
    hub = Path(hub_cache).expanduser() if hub_cache else home / "hub"
    return HuggingFaceCachePaths(cache_home = home, hub_cache = hub)
