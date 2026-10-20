# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Advanced Studio API: QAT, merging, MMoE, Policy Alignment, curriculum,
dataset blending, auto-eval, and checkpoint splitting.

Every handler imports its engine lazily, so this router can be registered on a
host without torch, safetensors, or lm-eval installed. Long operations (merge
runs, splits, evaluation) go through a small in-process job registry so the UI
can poll while the work streams to disk.
"""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path, PureWindowsPath
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.advanced import (
    blending,
    curriculum,
    eval_harness,
    layer_lr,
    qat,
    rl,
)

router = APIRouter()

_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()
_MAX_JOBS = 100


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class QatRequest(BaseModel):
    enabled: bool = False
    format: str = Field("int4", description = "int4, nf4, or nvfp4")
    group_size: int = 128
    tensor_scale: Optional[float] = None
    skip: Optional[list[str]] = None
    min_numel: int = 4096


class RlValidateRequest(BaseModel):
    method: Literal["dpo", "grpo"] = "dpo"
    config: dict[str, Any] = Field(default_factory = dict)
    dataset_columns: Optional[list[str]] = None
    num_rows: Optional[int] = None


class CurriculumRequest(BaseModel):
    cadence: str = "reasoning:2,code:1"
    total_steps: Optional[int] = None
    datasets: Optional[list[str]] = None


class LayerLrRequest(BaseModel):
    base_learning_rate: float = 2e-4
    num_layers: Optional[int] = None
    zone_multipliers: Optional[dict[str, float]] = None
    layer_multipliers: Optional[dict[str, float]] = None
    depth_curve: Optional[list[float]] = None


class BlendRequest(BaseModel):
    slots: list[dict[str, Any]]
    batch_size: int = 1


class MergePlanRequest(BaseModel):
    # A local directory or a Hugging Face repo id (org/name).
    model_a: str
    model_b: str
    top_k: int = 2
    router_hidden: Optional[int] = None
    temperature: float = 1.0
    layer_start: Optional[int] = None
    layer_end: Optional[int] = None
    hf_token: Optional[str] = Field(None, description = "HuggingFace token for gated/private repos")


class MergeRunRequest(BaseModel):
    model_a: str
    model_b: str
    # Blank or a bare name lands in the Studio outputs folder; see ``_output_dir``.
    output_dir: Optional[str] = None
    method: Literal["linear", "slerp", "ties", "dare"] = "linear"
    weight: float = Field(0.5, ge = 0.0, le = 1.0)
    density: float = Field(0.5, gt = 0.0, le = 1.0)
    seed: int = 3407
    max_shard_gb: float = Field(4.0, gt = 0.0, le = 1024.0)
    hf_token: Optional[str] = None


class MoeRunRequest(BaseModel):
    models: list[str] = Field(..., min_length = 2, max_length = 8)
    output_dir: Optional[str] = None
    top_k: int = 2
    router_hidden: Optional[int] = None
    temperature: float = 1.0
    layer_start: Optional[int] = None
    layer_end: Optional[int] = None
    hf_token: Optional[str] = None


class SplitPlanRequest(BaseModel):
    source: str
    max_shard_gb: Optional[float] = Field(None, gt = 0, le = 1024)
    reserve_fraction: float = Field(0.15, ge = 0.0, lt = 0.9)
    prefer: Literal["ram", "vram", "auto"] = "auto"


class SplitRunRequest(SplitPlanRequest):
    output_dir: Optional[str] = None
    copy_tokenizer: bool = True


class EvalPlanRequest(BaseModel):
    suites: list[str] = Field(..., min_length = 1)
    baseline_model: str
    candidate_model: str
    eval_dataset: Optional[str] = None
    limit: Optional[int] = Field(None, gt = 0)
    batch_size: int = Field(8, gt = 0)


class EvalRunRequest(EvalPlanRequest):
    output_dir: str
    timeout_seconds: int = Field(3600, gt = 0, le = 86400)


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


def _output_dir(value: Optional[str], *, kind: str) -> Path:
    """Resolve a merge/split output folder without pinning the UI to one machine's paths.

    A blank value (or the bare default name the UI prefills) lands under the Studio outputs
    root, so the same request works on Windows and on a Colab runtime, where ``C:\\models``
    does not exist. An absolute path is honoured as given so writing to another drive keeps
    working.
    """
    raw = (value or "").strip() or kind
    expanded = Path(raw).expanduser()
    # A Windows drive path (``C:/models``) is not absolute on POSIX, so test Windows rules too:
    # it must never be joined onto the outputs root. ``C:merged`` stays drive-relative and is
    # therefore not absolute under either rule.
    if expanded.is_absolute() or PureWindowsPath(raw).is_absolute():
        return expanded
    from utils.paths import resolve_output_dir

    try:
        return resolve_output_dir(raw)
    except ValueError as error:
        raise HTTPException(status_code = 400, detail = str(error)) from error


def _start_job(kind: str, work, *, output_dir: Optional[str] = None) -> dict[str, Any]:
    job_id = uuid.uuid4().hex
    record = {
        "job_id": job_id,
        "kind": kind,
        "status": "running",
        "created_at": time.time(),
        "finished_at": None,
        "output_dir": output_dir,
        "result": None,
        "error": None,
    }
    with _JOBS_LOCK:
        _JOBS[job_id] = record
        if len(_JOBS) > _MAX_JOBS:
            finished = sorted(
                (job for job in _JOBS.values() if job["status"] != "running"),
                key = lambda job: job["created_at"],
            )
            for stale in finished[: len(_JOBS) - _MAX_JOBS]:
                _JOBS.pop(stale["job_id"], None)

    def _runner() -> None:
        try:
            result = work()
            record["result"] = result
            record["status"] = "completed"
        except Exception as error:  # noqa: BLE001 - surfaced through the job record
            record["error"] = str(error)
            record["status"] = "failed"
        finally:
            record["finished_at"] = time.time()

    threading.Thread(target = _runner, name = f"advanced-{kind}", daemon = True).start()
    return record


def _job(job_id: str) -> dict[str, Any]:
    with _JOBS_LOCK:
        record = _JOBS.get(job_id)
    if record is None:
        raise HTTPException(status_code = 404, detail = "Unknown advanced job.")
    return record


# ---------------------------------------------------------------------------
# Capabilities / planning
# ---------------------------------------------------------------------------


@router.get("/capabilities")
def capabilities() -> dict[str, Any]:
    from core.advanced import splitter

    memory = splitter.probe_memory()
    return {
        "qat_formats": list(qat.QAT_FORMATS),
        "merge_methods": ["linear", "slerp", "ties", "dare"],
        "rl_methods": list(rl.RL_METHODS),
        "reward_presets": list(rl.REWARD_PRESETS),
        "layer_zones": list(layer_lr.ZONES),
        "eval_suites": eval_harness.list_suites(),
        "eval_harness": eval_harness.harness_available(),
        "memory": memory.to_dict(),
        "torch_available": _module_available("torch"),
        "safetensors_available": _module_available("safetensors"),
        "trl_available": _module_available("trl"),
    }


def _module_available(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@router.post("/qat/validate")
def validate_qat(request: QatRequest) -> dict[str, Any]:
    try:
        cfg = qat.validate_qat_config(request.model_dump())
    except ValueError as error:
        raise HTTPException(status_code = 400, detail = str(error)) from error
    return {"config": cfg.__dict__, "summary": qat.qat_summary(qat.QatSession(config = cfg))}


@router.post("/rl/validate")
def validate_rl(request: RlValidateRequest) -> dict[str, Any]:
    try:
        return rl.validate_rl_config(
            request.method,
            request.config,
            dataset_columns = request.dataset_columns,
            num_rows = request.num_rows,
        )
    except ValueError as error:
        raise HTTPException(status_code = 400, detail = str(error)) from error


@router.post("/curriculum/plan")
def plan_curriculum(request: CurriculumRequest) -> dict[str, Any]:
    try:
        plan = curriculum.build_curriculum_plan(
            request.cadence,
            total_steps = request.total_steps,
            datasets = request.datasets,
        )
    except ValueError as error:
        raise HTTPException(status_code = 400, detail = str(error)) from error
    return plan.to_dict()


@router.post("/layer-lr/plan")
def plan_layer_lr(request: LayerLrRequest) -> dict[str, Any]:
    try:
        cfg = layer_lr.validate_layer_lr_config(
            request.base_learning_rate,
            {
                "zone_multipliers": request.zone_multipliers,
                "layer_multipliers": request.layer_multipliers,
                "depth_curve": request.depth_curve,
            },
        )
    except ValueError as error:
        raise HTTPException(status_code = 400, detail = str(error)) from error
    breakdown = (
        layer_lr.zone_breakdown(request.num_layers) if request.num_layers else []
    )
    return {"config": cfg.to_dict(), "zones": breakdown}


@router.post("/blend/plan")
def plan_blend(request: BlendRequest) -> dict[str, Any]:
    try:
        plan = blending.validate_blend_slots(request.slots, batch_size = request.batch_size)
    except ValueError as error:
        raise HTTPException(status_code = 400, detail = str(error)) from error
    return {**plan.to_dict(), "batch_layout": plan.batch_layout()}


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def _require_checkpoint_source(path: str, label: str, *, allow_hub: bool = True) -> None:
    """Reject a merge/split source that is neither local nor (optionally) a Hub repo."""
    if Path(path).expanduser().exists():
        return
    if allow_hub:
        from core.advanced import merge

        if merge.hf_repo_id(path) is not None:
            return
    expected = (
        "an existing path or a Hugging Face repo id"
        if allow_hub
        else "an existing local checkpoint path"
    )
    raise HTTPException(
        status_code = 400,
        detail = f"{label} must be {expected}: {path}.",
    )


@router.post("/merge/plan")
def plan_merge(request: MergePlanRequest) -> dict[str, Any]:
    from core.advanced import merge

    try:
        spec_a = merge.probe_model_spec(request.model_a, token = request.hf_token)
        spec_b = merge.probe_model_spec(request.model_b, token = request.hf_token)
        alignment = merge.plan_alignment(spec_a, spec_b)
        layer_range = None
        if request.layer_start is not None or request.layer_end is not None:
            layer_range = (
                request.layer_start or 0,
                request.layer_end
                if request.layer_end is not None
                else min(
                    spec_a.num_layers or 0,
                    spec_b.num_layers or 0,
                ),
            )
        moe = merge.plan_moe_merge(
            [spec_a, spec_b],
            top_k = request.top_k,
            router_hidden = request.router_hidden,
            temperature = request.temperature,
            layer_range = layer_range,
        )
    except ValueError as error:
        raise HTTPException(status_code = 400, detail = str(error)) from error
    return {
        "models": [spec_a.to_dict(), spec_b.to_dict()],
        "alignment": alignment.to_dict(),
        "moe": moe.to_dict(),
        "recommended_mode": "mmoe" if alignment.mmoe_required else "standard",
    }


@router.post("/merge/run")
def run_merge(request: MergeRunRequest) -> dict[str, Any]:
    from core.advanced import merge

    _require_checkpoint_source(request.model_a, "Model A")
    _require_checkpoint_source(request.model_b, "Model B")
    output = _output_dir(request.output_dir, kind = "merged")
    for source in (request.model_a, request.model_b):
        try:
            if output.resolve() == Path(source).expanduser().resolve():
                raise HTTPException(
                    status_code = 400,
                    detail = "The output directory must differ from both source checkpoints.",
                )
        except OSError:
            continue

    def work() -> dict[str, Any]:
        return merge.merge_checkpoints(
            request.model_a,
            request.model_b,
            str(output),
            method = request.method,
            weight = request.weight,
            density = request.density,
            seed = request.seed,
            max_shard_bytes = int(request.max_shard_gb * 1024**3),
            token = request.hf_token,
        )

    return _start_job("merge", work, output_dir = str(output))


@router.post("/merge/moe")
def run_moe(request: MoeRunRequest) -> dict[str, Any]:
    from core.advanced import merge, moe

    for index, source in enumerate(request.models):
        _require_checkpoint_source(source, f"Model {index + 1}")
    output = _output_dir(request.output_dir, kind = "moe")

    def work() -> dict[str, Any]:
        snapshots = [
            merge.resolve_checkpoint_source(source, token = request.hf_token)
            for source in request.models
        ]
        specs = [merge.probe_model_spec(source) for source in snapshots]
        layer_range = None
        if request.layer_start is not None or request.layer_end is not None:
            layer_range = (
                request.layer_start or 0,
                request.layer_end
                if request.layer_end is not None
                else min(spec.num_layers or 0 for spec in specs),
            )
        plan = merge.plan_moe_merge(
            specs,
            top_k = request.top_k,
            router_hidden = request.router_hidden,
            temperature = request.temperature,
            layer_range = layer_range,
        )
        return moe.write_moe_bundle(
            str(output),
            plan.to_dict(),
            hidden_size = plan.router_hidden,
            seed = 3407,
        )

    return _start_job("moe", work, output_dir = str(output))


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------


@router.post("/split/plan")
def split_plan(request: SplitPlanRequest) -> dict[str, Any]:
    from core.advanced import splitter

    _require_checkpoint_source(request.source, "Checkpoint", allow_hub = False)
    try:
        return splitter.plan_only(
            request.source,
            max_shard_bytes = (
                int(request.max_shard_gb * 1024**3) if request.max_shard_gb else None
            ),
            reserve_fraction = request.reserve_fraction,
            prefer = request.prefer,
        )
    except ValueError as error:
        raise HTTPException(status_code = 400, detail = str(error)) from error


@router.post("/split/run")
def split_run(request: SplitRunRequest) -> dict[str, Any]:
    from core.advanced import splitter

    _require_checkpoint_source(request.source, "Checkpoint", allow_hub = False)
    output = _output_dir(request.output_dir, kind = "sharded")

    def work() -> dict[str, Any]:
        return splitter.split_checkpoint(
            request.source,
            str(output),
            max_shard_bytes = int((request.max_shard_gb or 4.0) * 1024**3),
            reserve_fraction = request.reserve_fraction,
            prefer = request.prefer,
            copy_tokenizer = request.copy_tokenizer,
        )

    return _start_job("split", work, output_dir = str(output))


# ---------------------------------------------------------------------------
# Auto-eval
# ---------------------------------------------------------------------------


@router.get("/eval/suites")
def eval_suites() -> dict[str, Any]:
    return {
        "suites": eval_harness.list_suites(),
        "harness": eval_harness.harness_available(),
    }


@router.post("/eval/plan")
def eval_plan(request: EvalPlanRequest) -> dict[str, Any]:
    try:
        plan = eval_harness.build_eval_plan(
            request.suites,
            baseline_model = request.baseline_model,
            candidate_model = request.candidate_model,
            eval_dataset = request.eval_dataset,
            limit = request.limit,
            batch_size = request.batch_size,
        )
    except ValueError as error:
        raise HTTPException(status_code = 400, detail = str(error)) from error
    return plan.to_dict()


@router.post("/eval/run")
def eval_run(request: EvalRunRequest) -> dict[str, Any]:
    from core.advanced import eval_runner

    try:
        plan = eval_harness.build_eval_plan(
            request.suites,
            baseline_model = request.baseline_model,
            candidate_model = request.candidate_model,
            eval_dataset = request.eval_dataset,
            limit = request.limit,
            batch_size = request.batch_size,
        )
    except ValueError as error:
        raise HTTPException(status_code = 400, detail = str(error)) from error
    if not eval_harness.harness_available()["available"]:
        raise HTTPException(
            status_code = 409,
            detail = eval_harness.harness_available()["install_hint"],
        )

    def work() -> dict[str, Any]:
        return eval_runner.run_plan(
            plan,
            output_dir = request.output_dir,
            timeout = request.timeout_seconds,
        )

    return _start_job("eval", work)


# ---------------------------------------------------------------------------
# Job status
# ---------------------------------------------------------------------------


@router.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    record = _job(job_id)
    return {
        key: record[key]
        for key in (
            "job_id",
            "kind",
            "status",
            "created_at",
            "finished_at",
            "output_dir",
            "result",
            "error",
        )
    }


@router.get("/jobs")
def list_jobs() -> dict[str, Any]:
    with _JOBS_LOCK:
        records = sorted(_JOBS.values(), key = lambda job: job["created_at"], reverse = True)
    return {
        "jobs": [
            {
                key: record[key]
                for key in ("job_id", "kind", "status", "created_at", "finished_at", "error")
            }
            for record in records[:50]
        ]
    }
