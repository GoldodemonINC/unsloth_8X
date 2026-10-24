# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Labs API: the experimental-features switch, the Artifacts studio and the
Pipelines planner.

Every route past the switch asks :func:`core.labs.require` first, so turning an
experiment off closes its endpoint as well as its page. That is the whole point
of a server-side flag: a hidden button is not a gate.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict

from auth import policy
from auth.authentication import get_current_subject
from core import labs
from core.labs import artifacts, pipelines

router = APIRouter()


async def _require_owner(current_subject: str = Depends(get_current_subject)) -> None:
    await policy.require_owner()


class LabsFeature(BaseModel):
    id: str
    label: str
    description: str
    status: str
    studio: bool
    enabled: bool


class LabsState(BaseModel):
    enabled: bool
    features: list[LabsFeature]
    note: str


class LabsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: Optional[bool] = None
    features: Optional[dict[str, bool]] = None


class PipelinePlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec: dict[str, Any]


class PipelineSaveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec: dict[str, Any]


_LABS_NOTE = (
    "Labs experiments are unfinished. They can change shape or break between "
    "updates, and an artifact they write is not covered by the stable contract."
)


def _state() -> LabsState:
    state = labs.describe_state()
    return LabsState(
        enabled = bool(state["enabled"]),
        features = [LabsFeature(**feature) for feature in state["features"]],
        note = _LABS_NOTE,
    )


def _gate(feature: str) -> None:
    try:
        labs.require(feature)
    except labs.LabsDisabled as error:
        raise HTTPException(status_code = 403, detail = str(error)) from error


def _translate(error: Exception) -> HTTPException:
    if isinstance(error, PermissionError):
        return HTTPException(status_code = 403, detail = str(error))
    if isinstance(error, FileNotFoundError):
        return HTTPException(status_code = 404, detail = str(error))
    return HTTPException(status_code = 400, detail = str(error))


# ---------------------------------------------------------------------------
# The switch
# ---------------------------------------------------------------------------


@router.get("", response_model = LabsState)
def read_labs() -> LabsState:
    """The switch and every registered experiment. Readable by any signed-in account."""
    return _state()


@router.put("", response_model = LabsState, dependencies = [Depends(_require_owner)])
def update_labs(payload: LabsUpdate) -> LabsState:
    """Owner-only: enable Labs, or switch one experiment off."""
    if payload.enabled is None and payload.features is None:
        raise HTTPException(status_code = 400, detail = "Nothing to update.")
    try:
        labs.set_state(enabled = payload.enabled, features = payload.features)
    except ValueError as error:
        raise HTTPException(status_code = 400, detail = str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code = 500, detail = str(error)) from error
    return _state()


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


@router.get("/artifacts")
def list_artifacts(limit: int = Query(default = 50, ge = 1, le = 200)) -> dict[str, Any]:
    """Checkpoints this Studio wrote, newest first, with their legal next steps."""
    _gate("artifacts")
    try:
        records = artifacts.scan(limit = limit)
        root = str(artifacts.root())
    except Exception as error:  # noqa: BLE001 - reported as an HTTP failure with its reason
        raise _translate(error) from error
    return {"artifacts": records, "root": root}


@router.get("/artifacts/detail")
def artifact_detail(path: str) -> dict[str, Any]:
    """One artifact. The raw manifest rides along only while that experiment is on."""
    _gate("artifacts")
    try:
        record = artifacts.find(path)
    except Exception as error:  # noqa: BLE001
        raise _translate(error) from error
    if labs.feature_enabled("manifest-viewer") and record.get("manifest_path"):
        try:
            record["raw_manifest"] = artifacts.read_manifest(path)["manifest"]
        except Exception as error:  # noqa: BLE001 - the record is still useful without it
            record["manifest_error"] = str(error)
    return record


@router.get("/artifacts/manifest")
def artifact_manifest(path: str) -> dict[str, Any]:
    """The raw run manifest beside one artifact."""
    _gate("manifest-viewer")
    try:
        return artifacts.read_manifest(path)
    except Exception as error:  # noqa: BLE001
        raise _translate(error) from error


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------


@router.post("/pipelines/plan")
def plan_pipeline(request: PipelinePlanRequest) -> dict[str, Any]:
    """Validate and price a chain from configs only. Starts nothing."""
    _gate("pipelines")
    try:
        return pipelines.preflight(request.spec)
    except Exception as error:  # noqa: BLE001
        raise _translate(error) from error


@router.get("/pipelines")
def list_pipelines() -> dict[str, Any]:
    """Saved pipeline plans, newest first."""
    _gate("pipelines")
    try:
        return {"pipelines": pipelines.list_pipelines(), "root": str(pipelines.root())}
    except Exception as error:  # noqa: BLE001
        raise _translate(error) from error


@router.post("/pipelines")
def save_pipeline(request: PipelineSaveRequest) -> dict[str, Any]:
    """Validate a chain, then save it. A blocked chain still saves, with the block recorded."""
    _gate("pipelines")
    try:
        return {"pipeline": pipelines.save_pipeline(request.spec)}
    except Exception as error:  # noqa: BLE001
        raise _translate(error) from error


@router.delete("/pipelines/{pipeline_id}")
def delete_pipeline(pipeline_id: str) -> dict[str, Any]:
    """Delete one saved pipeline plan. The checkpoints it referenced are untouched."""
    _gate("pipelines")
    try:
        deleted = pipelines.delete_pipeline(pipeline_id)
    except Exception as error:  # noqa: BLE001
        raise _translate(error) from error
    if not deleted:
        raise HTTPException(status_code = 404, detail = f"No pipeline {pipeline_id!r}.")
    return {"deleted": pipeline_id}
