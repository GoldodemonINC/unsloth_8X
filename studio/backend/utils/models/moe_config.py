# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Whether a model's config describes a mixture-of-experts model.

Split out of the models route so it can be tested against plain objects: the route
imports the whole inference stack, and a config reader does not need any of it.
"""

from __future__ import annotations

from typing import Optional, Tuple

EXPERT_COUNT_ATTRS = ("num_experts", "n_routed_experts", "num_local_experts")
TOP_K_ATTRS = ("num_experts_per_tok", "top_k", "num_selected_experts")


def _positive_int(value: object) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    return None


def moe_info(config: object) -> Tuple[bool, Optional[int], Optional[int]]:
    """(is_moe, experts, top_k) from a loaded or raw config.

    Every MoE family spells these differently and some nest the text config under a
    vision wrapper, so read the same three names a load resolves, at both levels. A
    missing count is not MoE: guessing from an architecture name would offer an
    expert scope for a dense model.
    """
    if config is None:
        return False, None, None
    candidates = [config]
    nested = getattr(config, "text_config", None)
    if nested is not None:
        candidates.append(nested)
    for candidate in candidates:
        count = None
        for attr in EXPERT_COUNT_ATTRS:
            count = _positive_int(getattr(candidate, attr, None))
            if count is not None:
                break
        if count is None:
            continue
        top_k = None
        for attr in TOP_K_ATTRS:
            top_k = _positive_int(getattr(candidate, attr, None))
            if top_k is not None:
                break
        return True, count, top_k
    return False, None, None
