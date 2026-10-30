# SPDX-License-Identifier: NRC-3
# Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-3.md

"""Labs: the experimental-features switch and the registry behind it.

The switch is installation-wide, stored in ``app_settings`` beside the other
Studio settings, not in browser storage: the server has to refuse a Labs route
itself, or the switch would only hide a button. :func:`feature_enabled` is the
single question every Labs route asks, and :func:`require` raises the error the
route turns into a 403.

Every experiment is one entry in :data:`FEATURES`. A feature id that is not in
that tuple cannot be turned on, which is what keeps a stale browser or an old
stored value from enabling a feature that no longer exists.
"""

from __future__ import annotations

import threading
from typing import Any, Mapping, Optional

SETTING_KEY = "labs_state"

# alpha: the surface can change or break between updates.
# beta: the behaviour is settled, the polish is not.
FEATURES: tuple[dict[str, Any], ...] = (
    {
        "id": "artifacts",
        "label": "Artifacts studio",
        "description": (
            "Every checkpoint this Studio wrote, with the run that produced it and "
            "the next steps that checkpoint actually allows."
        ),
        "status": "alpha",
        "studio": True,
    },
    {
        "id": "pipelines",
        "label": "Pipelines",
        "description": (
            "Chain merge, scale, train, quantize and benchmark into one plan. The "
            "planner validates the chain and prices it before any compute runs."
        ),
        "status": "alpha",
        "studio": True,
    },
    {
        "id": "benchmarks",
        "label": "Benchmark studio",
        "description": (
            "Score any produced checkpoint against a baseline on the eval suites, "
            "and keep the report beside the run so two checkpoints can be compared later."
        ),
        "status": "alpha",
        "studio": True,
    },
    {
        "id": "lineage",
        "label": "Run lineage",
        "description": (
            "Trace a produced checkpoint back through the runs and merges that made "
            "it, using the run_manifest.json every stage already writes."
        ),
        "status": "alpha",
        "studio": True,
    },
    {
        "id": "cmt",
        "label": "CMT",
        "description": (
            "Capacity matched to the data: split 2+ datasets, grow the model by the "
            "size their content justifies, freeze the base, and train only the new blocks."
        ),
        "status": "alpha",
        "studio": True,
    },
    {
        "id": "housekeeping",
        "label": "Cleanup studio",
        "description": (
            "What the outputs folder and the Hugging Face cache are using, with a "
            "guarded delete for the folders that are safe to remove."
        ),
        "status": "beta",
        "studio": True,
    },
    {
        "id": "auto-scale",
        "label": "Scale before training",
        "description": (
            "Grow a checkpoint by the 1M-700M its training data justifies, then train "
            "the method you chose on the larger model. CMT trains only what it grows; "
            "this stage grows and leaves the training to you."
        ),
        "status": "alpha",
        "studio": False,
    },
    {
        "id": "manifest-viewer",
        "label": "Run manifest viewer",
        "description": (
            "Read the run_manifest.json a stage wrote, so a checkpoint can be traced "
            "back to its dataset, seeds and settings."
        ),
        "status": "beta",
        "studio": False,
    },
    {
        "id": "scale-from-artifact",
        "label": "Scale from an artifact",
        "description": (
            "Send an Artifacts row straight to the Scaler instead of copying its path "
            "between pages."
        ),
        "status": "beta",
        "studio": False,
    },
    {
        "id": "training-continuation",
        "label": "Continue training from an artifact",
        "description": (
            "Make a finished checkpoint the Training model in one click, so a second "
            "round starts from what the first round actually produced."
        ),
        "status": "beta",
        "studio": False,
    },
)

FEATURE_IDS: tuple[str, ...] = tuple(feature["id"] for feature in FEATURES)
_STATE_LOCK = threading.Lock()


class LabsDisabled(RuntimeError):
    """A Labs route reached while its experiment is off."""

    def __init__(self, feature: str, message: str):
        super().__init__(message)
        self.feature = feature


def _normalise(stored: Any) -> dict[str, Any]:
    """The stored value, with every unknown or malformed part dropped."""
    enabled = False
    raw_features: Mapping[str, Any] = {}
    if isinstance(stored, Mapping):
        enabled = bool(stored.get("enabled"))
        candidate = stored.get("features")
        if isinstance(candidate, Mapping):
            raw_features = candidate
    features = {feature_id: True for feature_id in FEATURE_IDS}
    for feature_id in FEATURE_IDS:
        value = raw_features.get(feature_id)
        if isinstance(value, bool):
            features[feature_id] = value
    return {"enabled": enabled, "features": features}


def read_state() -> dict[str, Any]:
    """The stored switch. A settings read failure means off, never on."""
    try:
        from storage.studio_db import get_app_setting
        from utils.account_context import OWNER, run_as

        stored = run_as(OWNER, get_app_setting, SETTING_KEY, None)
    except Exception:  # noqa: BLE001 - an unreadable setting must not enable anything
        stored = None
    return _normalise(stored)


def is_enabled() -> bool:
    """Whether Labs is on at all, before any per-feature answer."""
    return bool(read_state()["enabled"])


def feature_enabled(feature_id: str) -> bool:
    """Whether *feature_id* may run: Labs on and that feature not switched off."""
    if feature_id not in FEATURE_IDS:
        raise ValueError(f"Unknown Labs feature {feature_id!r}.")
    state = read_state()
    return bool(state["enabled"]) and bool(state["features"].get(feature_id, True))


def set_state(
    *,
    enabled: Optional[bool] = None,
    features: Optional[Mapping[str, bool]] = None,
) -> dict[str, Any]:
    """Turn Labs or one experiment on/off, then return the saved state."""
    if features:
        unknown = sorted(set(features) - set(FEATURE_IDS))
        if unknown:
            raise ValueError(f"Unknown Labs feature(s): {', '.join(unknown)}.")
        for feature_id, value in features.items():
            if not isinstance(value, bool):
                raise ValueError(f"Labs feature {feature_id!r} needs a true/false value.")
    with _STATE_LOCK:
        state = read_state()
        if enabled is not None:
            state["enabled"] = bool(enabled)
        if features:
            state["features"].update({key: bool(value) for key, value in features.items()})
        try:
            from storage.studio_db import upsert_app_settings

            upsert_app_settings({SETTING_KEY: state})
        except Exception as error:  # noqa: BLE001 - the caller reports a failed save
            raise RuntimeError(f"Could not save the Labs switch: {error}") from error
        return state


def describe_state() -> dict[str, Any]:
    """The switch plus every registry entry, with its own enabled flag."""
    state = read_state()
    return {
        "enabled": bool(state["enabled"]),
        "features": [
            {**feature, "enabled": bool(state["features"].get(feature["id"], True))}
            for feature in FEATURES
        ],
    }


def _disabled_message(feature_id: str, label: str) -> str:
    if not is_enabled():
        return (
            f"Labs is off, so {label} is not available. Turn on Settings → Labs "
            "to use it."
        )
    return (
        f"The {label} experiment is switched off. Turn it on in Settings → Labs "
        "to use it."
    )


def require(feature_id: str) -> None:
    """Raise :class:`LabsDisabled` unless *feature_id* is on."""
    if feature_id not in FEATURE_IDS:
        raise ValueError(f"Unknown Labs feature {feature_id!r}.")
    if feature_enabled(feature_id):
        return
    label = next(
        (str(feature["label"]) for feature in FEATURES if feature["id"] == feature_id),
        feature_id,
    )
    raise LabsDisabled(feature_id, _disabled_message(feature_id, label))
