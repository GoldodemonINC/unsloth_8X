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
    merge_eval,
    policy as loading_policy,
    qat,
    resources,
    rl,
    run_manifest,
    tokenizer_check,
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
    # Optional pair to measure: quality at the export quant, parent vs fine-tune.
    parent: Optional[str] = None
    candidate: Optional[str] = None


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
    kind_multipliers: Optional[dict[str, float]] = None
    layer_multipliers: Optional[dict[str, float]] = None
    depth_curve: Optional[list[float]] = None
    # Optional real parameter names, so the preview shows the actual zone map.
    param_names: Optional[list[str]] = None
    # A checkpoint to read the parameter names from when they are not sent.
    model: Optional[str] = None


class BlendRequest(BaseModel):
    slots: list[dict[str, Any]]
    batch_size: int = 1


class MergePlanRequest(BaseModel):
    # A local directory or a Hugging Face repo id (org/name). ``models`` is the
    # current shape (2 to 6 parents); the pair is still accepted so older
    # callers keep working.
    model_a: Optional[str] = None
    model_b: Optional[str] = None
    models: Optional[list[str]] = Field(None, min_length = 2, max_length = 6)
    top_k: int = 2
    router_hidden: Optional[int] = None
    temperature: float = 1.0
    layer_start: Optional[int] = None
    layer_end: Optional[int] = None
    hf_token: Optional[str] = Field(None, description = "HuggingFace token for gated/private repos")


class MergeRunRequest(BaseModel):
    # ``models`` merges 2 to 6 parents; a pair in model_a/model_b still works.
    model_a: Optional[str] = None
    model_b: Optional[str] = None
    models: Optional[list[str]] = Field(None, min_length = 2, max_length = 6)
    # One weight per entry in ``models``, normalized server-side. ``weight`` is
    # the two-parent shorthand (it means [1 - weight, weight]).
    weights: Optional[list[float]] = Field(None, min_length = 2, max_length = 6)
    # Blank or a bare name lands in the Studio outputs folder; see ``_output_dir``.
    output_dir: Optional[str] = None
    method: Literal["linear", "slerp", "ties", "dare"] = "linear"
    weight: float = Field(0.5, ge = 0.0, le = 1.0)
    density: float = Field(0.5, gt = 0.0, le = 1.0)
    seed: int = 3407
    max_shard_gb: float = Field(4.0, gt = 0.0, le = 1024.0)
    hf_token: Optional[str] = None
    # Opt-ins. Both are recorded in merge_report.json and the run manifest.
    allow_remote_code: bool = False
    allow_non_safetensors: bool = False
    allow_tokenizer_conflict: bool = False


class MergeValidateRequest(BaseModel):
    model_a: Optional[str] = None
    model_b: Optional[str] = None
    models: Optional[list[str]] = Field(None, min_length = 2, max_length = 6)
    hf_token: Optional[str] = None
    allow_remote_code: bool = False
    allow_non_safetensors: bool = False


class MergeEvalRequest(BaseModel):
    source: str
    sample_tensors: int = Field(24, gt = 0, le = 512)


class MoeRunRequest(BaseModel):
    # The same 2..6 parent window the weight-space merge accepts; Studio's page
    # cannot offer a seventh slot, and an unbounded list would make the router
    # projection the size of the sum of its experts.
    models: list[str] = Field(..., min_length = 2, max_length = 6)
    output_dir: Optional[str] = None
    top_k: int = 2
    router_hidden: Optional[int] = None
    temperature: float = 1.0
    layer_start: Optional[int] = None
    layer_end: Optional[int] = None
    hf_token: Optional[str] = None
    allow_remote_code: bool = False
    allow_non_safetensors: bool = False


class ResourceEstimateRequest(BaseModel):
    kind: Literal["merge", "moe", "split", "merge_eval", "eval", "qat"]
    sources: Optional[list[str]] = None
    output_dir: Optional[str] = None


class AcquireRequest(BaseModel):
    job_id: str
    kind: Literal["merge", "moe", "split", "merge_eval", "eval", "qat"]
    sources: Optional[list[str]] = None
    wait_seconds: float = Field(0.0, ge = 0.0, le = 3600.0)


class ReleaseRequest(BaseModel):
    job_id: str


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


def _merge_default_name(model_a: str, model_b: str) -> str:
    from utils.paths import default_merge_dir_name

    return default_merge_dir_name(model_a, model_b)


def _merge_sources(
    model_a: Optional[str], model_b: Optional[str], models: Optional[list[str]]
) -> list[str]:
    """The parent list a merge request names, newest shape first.

    ``models`` wins when present; the pair is the fallback so an older client
    keeps working. Blank entries are rejected rather than silently dropped: a
    merge that quietly loses a parent is worse than a 400.
    """
    raw = list(models) if models else [model_a or "", model_b or ""]
    sources = [source.strip() for source in raw]
    if len(sources) < 2:
        raise HTTPException(status_code = 400, detail = "A merge needs at least two models.")
    if any(not source for source in sources):
        raise HTTPException(status_code = 400, detail = "Every merge slot needs a model.")
    if len(set(sources)) != len(sources):
        raise HTTPException(status_code = 400, detail = "Each merge slot must be a different model.")
    return sources


def _merge_labels(count: int) -> list[str]:
    """Display labels: the familiar A/B pair, numbered slots beyond it."""
    if count == 2:
        return ["Model A", "Model B"]
    return [f"Model {index + 1}" for index in range(count)]


def _merge_default_name_for(sources: list[str]) -> str:
    """Default output folder for an N-way merge: first and last parent, plus N."""
    if len(sources) == 2:
        return _merge_default_name(sources[0], sources[1])
    suffix = f"-{len(sources)}way"
    from utils.paths import default_merge_dir_name

    return default_merge_dir_name(sources[0], sources[-1])[: 150 - len(suffix)] + suffix


def _free_output_dir(name: str) -> Path:
    """An unused folder directly under the outputs root, so two runs never overwrite each other."""
    from utils.paths import outputs_root

    root = outputs_root()
    candidate = root / name
    index = 2
    while candidate.exists():
        candidate = root / f"{name}-{index}"
        index += 1
    return candidate


def _library_visible(output: Optional[Path]) -> bool:
    """True when Studio lists *output* as a model (the Library scans the outputs root's children)."""
    if output is None:
        return False
    from utils.paths import outputs_root

    try:
        resolved = output.resolve()
        root = outputs_root().resolve()
    except (OSError, ValueError):
        return False
    return resolved != root and resolved.parent == root


def _output_dir(value: Optional[str], *, kind: str, default_name: Optional[str] = None) -> Path:
    """Resolve a merge/split output folder without pinning the UI to one machine's paths.

    A blank value lands beside the training runs in the Studio outputs root, named after
    the sources, so the same request works on Windows and on a Colab runtime (where
    ``C:\\models`` does not exist) and the result is a model the Library, Export and Chat
    can pick up. An absolute path is honoured as given so writing to another drive keeps
    working.
    """
    raw = (value or "").strip()
    if not raw:
        return _free_output_dir(default_name or kind)
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


def _loading_policy(**flags: Any) -> loading_policy.LoadingPolicy:
    return loading_policy.LoadingPolicy(
        allow_remote_code = bool(flags.get("allow_remote_code", False)),
        allow_non_safetensors = bool(flags.get("allow_non_safetensors", False)),
    )


def _stage_manifest(
    kind: str,
    directory: Optional[str],
    *,
    config: dict[str, Any],
    inputs: dict[str, Any],
    artifacts: dict[str, Any],
    seeds: Optional[dict[str, Any]] = None,
    policy: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    """Write the stage's run manifest beside its artifacts.

    Failing to write it must not lose a finished run's output, so the caller
    records the error instead of raising.
    """
    if not directory:
        return None
    try:
        manifest = run_manifest.build_manifest(
            kind,
            config = config,
            inputs = inputs,
            artifacts = artifacts,
            seeds = seeds or {},
            policy = policy or {},
        )
        target = run_manifest.write_manifest(directory, manifest)
        return str(target)
    except OSError:
        return None


def _start_job(
    kind: str,
    work,
    *,
    output_dir: Optional[str] = None,
    sources: Optional[list[str]] = None,
    with_progress: bool = False,
) -> dict[str, Any]:
    """Reserve resources, then run *work* on a thread.

    The lease is taken here, on the request thread, so a job that cannot run is
    a 409 naming the short resource instead of a background thread that dies
    with a CUDA OOM. The lease is released when the work finishes.
    """
    job_id = uuid.uuid4().hex
    lease = resources.acquire_for_job(
        job_id, kind, sources = sources, output_dir = output_dir
    )
    if lease["status"] != "granted":
        raise HTTPException(status_code = 409, detail = lease["reason"])
    record = {
        "job_id": job_id,
        "kind": kind,
        "status": "running",
        "created_at": time.time(),
        "finished_at": None,
        "output_dir": output_dir,
        "library_visible": _library_visible(Path(output_dir) if output_dir else None),
        "resources": {
            "status": lease["status"],
            "need": (lease.get("estimate") or {}).get("need"),
            "notes": (lease.get("estimate") or {}).get("notes", []),
        },
        "run_manifest": None,
        "result": None,
        "error": None,
        # Long jobs (a merge over many parents) publish a step here so the page
        # can show a bar instead of a spinner. None means "not reporting yet".
        "progress": None,
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

    def _report_progress(payload: dict[str, Any]) -> None:
        record["progress"] = {**payload, "updated_at": time.time()}

    def _runner() -> None:
        try:
            result = work(_report_progress) if with_progress else work()
            record["result"] = result
            record["status"] = "completed"
            if isinstance(result, dict):
                record["run_manifest"] = result.get("run_manifest")
        except Exception as error:  # noqa: BLE001 - surfaced through the job record
            record["error"] = str(error)
            record["status"] = "failed"
            record["progress"] = {
                "stage": "failed",
                "message": str(error),
                "updated_at": time.time(),
            }
        finally:
            record["finished_at"] = time.time()
            resources.BROKER.release(job_id)

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


# The platform scope table, in one place. Each entry answers "does this work on a
# CPU-only host?" honestly, because the product's wedge is that merging does.
_FEATURES = (
    {
        "key": "merge",
        "label": "Merge (weight blending)",
        "cpu": True,
        "gpu": True,
        "note": "Streams one tensor pair at a time, so it fits a CPU host with ~2x the largest tensor free.",
    },
    {
        "key": "split",
        "label": "VRAM Fit (checkpoint sharding)",
        "cpu": True,
        "gpu": True,
        "note": "Splits oversized checkpoints into loadable shards; needs disk, not a GPU.",
    },
    {
        "key": "structural_validation",
        "label": "Structural validation",
        "cpu": True,
        "gpu": True,
        "note": "Load, tokenizer and NaN/scale checks. Says nothing about task quality.",
    },
    {
        "key": "alignment",
        "label": "Policy alignment",
        "cpu": True,
        "gpu": True,
        "note": "Preference data checks and planning; training the adapter needs a GPU.",
    },
    {
        "key": "training",
        "label": "Training",
        "cpu": False,
        "gpu": True,
        "note": "Recipes here need a GPU. The studio's CoRA method is the one recipe with a CPU path.",
    },
    {
        "key": "moe",
        "label": "MMoE expert fusion",
        "cpu": False,
        "gpu": True,
        "note": "The bundle can be written anywhere; training the router needs a GPU.",
    },
    {
        "key": "benchmarks",
        "label": "Benchmarks (eval harness)",
        "cpu": False,
        "gpu": True,
        "note": "lm-eval needs a GPU to be practical; structural validation is the CPU tier.",
    },
    {
        "key": "rl",
        "label": "Reinforcement learning",
        "cpu": False,
        "gpu": True,
        "note": "DPO/GRPO train the model, so they need a GPU.",
    },
    {
        "key": "qat",
        "label": "Quantization-aware fine-tuning",
        "cpu": False,
        "gpu": True,
        "note": "Fake-quant is applied in the forward pass of a training run.",
    },
)

PLATFORM_LINE = (
    "On any CPU (8 GB+ available): merge (streaming), alignment, VRAM Fit sharding, "
    "structural validation. Training, RL, QAT, MMoE, and benchmarks need a GPU; "
    "Colab covers MMoE and small training runs."
)


@router.get("/capabilities")
def capabilities() -> dict[str, Any]:
    from core.advanced import merge, splitter

    memory = splitter.probe_memory()
    has_gpu = bool(memory.gpus)
    # Mirrors merge.max_merge_models(), from the probe this response already has:
    # a three-parent CPU tier, the six-parent GPU tier.
    max_merge_models = merge.MAX_MERGE_MODELS if has_gpu else merge.MAX_MERGE_MODELS_CPU
    features = [
        {**feature, "available_here": feature["gpu"] if has_gpu else feature["cpu"]}
        for feature in _FEATURES
    ]
    return {
        "qat_formats": list(qat.QAT_FORMATS),
        "merge_methods": ["linear", "slerp", "ties", "dare"],
        "rl_methods": list(rl.RL_METHODS),
        "reward_presets": list(rl.REWARD_PRESETS),
        "layer_zones": list(layer_lr.ZONES),
        "layer_param_kinds": list(layer_lr.PARAM_KINDS),
        "eval_suites": eval_harness.list_suites(),
        "eval_harness": eval_harness.harness_available(),
        "memory": memory.to_dict(),
        "features": features,
        "platform_line": PLATFORM_LINE,
        "has_gpu": has_gpu,
        "torch_available": _module_available("torch"),
        "safetensors_available": _module_available("safetensors"),
        "trl_available": _module_available("trl"),
        "merge_streams": True,
        "merge_tier": "gpu" if has_gpu else "cpu",
        "max_merge_models": max_merge_models,
        "merge_cpu_note": (
            "CPU merge is streaming only: parents are never resident at once (equal-shaped "
            "tensors are blended a row chunk at a time), so 8 GB of available RAM is a real "
            "tier. The basic tier blends two or three checkpoints; a GPU host takes up to six."
        ),
    }


@router.get("/resources")
def resource_state() -> dict[str, Any]:
    """Live VRAM/RAM/disk, the headroom limits, and who holds a lease."""
    return resources.BROKER.snapshot()


@router.post("/resources/estimate")
def resource_estimate(request: ResourceEstimateRequest) -> dict[str, Any]:
    try:
        return resources.estimate(
            request.kind, sources = request.sources, output_dir = request.output_dir
        )
    except ValueError as error:
        raise HTTPException(status_code = 400, detail = str(error)) from error


@router.post("/jobs/acquire")
def acquire_resources(request: AcquireRequest) -> dict[str, Any]:
    """Lease resources for a job started outside this router (training, MCP)."""
    result = resources.BROKER.acquire(
        request.job_id,
        request.kind,
        None,
        sources = request.sources,
        wait_seconds = request.wait_seconds,
    )
    if result["status"] == "denied":
        raise HTTPException(status_code = 409, detail = result["reason"])
    return result


@router.post("/jobs/release")
def release_resources(request: ReleaseRequest) -> dict[str, Any]:
    resources.BROKER.release(request.job_id)
    return {"released": request.job_id, "leases": resources.BROKER.leases()}


@router.get("/outputs")
def output_locations() -> dict[str, Any]:
    """Where Studio keeps the models it can list, quantize and serve."""
    from utils.paths import exports_root, outputs_root

    return {
        "outputs_root": str(outputs_root()),
        "exports_root": str(exports_root()),
    }


@router.get("/merge/history")
def merge_history(limit: int = 20) -> dict[str, Any]:
    """Recent merges and MMoE bundles, newest first, read from their run manifests.

    The Merge page reloads a past run's parents and settings from here, so a merge
    can be reproduced without retyping five repo ids from a folder name.
    """
    from core.advanced.merge_history import list_merge_runs
    from utils.paths import outputs_root

    bounded = max(1, min(int(limit), 100))
    return {"entries": list_merge_runs(outputs_root(), limit = bounded)}


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
    payload: dict[str, Any] = {
        "config": cfg.__dict__,
        "summary": qat.qat_summary(qat.QatSession(config = cfg)),
        "claim": (
            f"Quantization-aware fine-tuning at {cfg.format} (group {cfg.group_size}). The "
            "forward pass runs on the fake-quantized weights, so the run learns to "
            "compensate for that grid. Nothing is claimed for other precisions."
        ),
        "export_target": {"format": cfg.format, "group_size": cfg.group_size},
    }
    if request.parent and request.candidate:
        try:
            payload["export_quality"] = qat.validate_checkpoints_at_export(
                request.parent, request.candidate, config = cfg
            )
        except ValueError as error:
            raise HTTPException(status_code = 400, detail = str(error)) from error
    else:
        payload["export_quality"] = {
            "status": "not_measured",
            "verdict": "unmeasured",
            "reason": (
                "Give both parent and candidate checkpoints to measure quality at the "
                "export quant; without them this endpoint only validates the config."
            ),
        }
    return payload


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
    payload = plan.to_dict()
    # Labelled wherever it is exposed: it is a schedule preview until an
    # equal-compute A/B exists, and not a peer of weight blending.
    payload["label"] = "experimental"
    payload["note"] = (
        "Curriculum ordering is experimental. Treat any gain over plain blending "
        "as unproven until an equal-compute A/B says otherwise."
    )
    return payload


@router.post("/layer-lr/plan")
def plan_layer_lr(request: LayerLrRequest) -> dict[str, Any]:
    try:
        cfg = layer_lr.validate_layer_lr_config(
            request.base_learning_rate,
            {
                "zone_multipliers": request.zone_multipliers,
                "kind_multipliers": request.kind_multipliers,
                "layer_multipliers": request.layer_multipliers,
                "depth_curve": request.depth_curve,
            },
        )
    except ValueError as error:
        raise HTTPException(status_code = 400, detail = str(error)) from error
    breakdown = (
        layer_lr.zone_breakdown(request.num_layers) if request.num_layers else []
    )
    param_names = list(request.param_names or [])
    num_layers = request.num_layers
    note = None
    if request.model and not param_names:
        param_names, discovered_layers = _checkpoint_param_names(request.model)
        num_layers = num_layers or discovered_layers
        if not param_names:
            note = (
                "Parameter names could not be read from that checkpoint; the preview "
                "shows the depth strip only."
            )
    preview = layer_lr.zone_preview(
        cfg, num_layers = num_layers, param_names = param_names or None
    )
    payload: dict[str, Any] = {
        "config": cfg.to_dict(),
        "zones": breakdown,
        "preview": preview,
        "num_layers": num_layers,
        "zone_definition_hash": None,
        "notes": [note] if note else [],
    }
    if param_names:
        payload["coverage"] = preview.get("coverage")
    else:
        payload["notes"].append(
            "Send param_names or a local model path to see every parameter's zone before training."
        )
    try:
        payload["zone_definition_hash"] = run_manifest.zone_definition_hash(cfg.to_dict())
    except (TypeError, ValueError):
        pass
    return payload


def _checkpoint_param_names(source: str) -> tuple[list[str], int | None]:
    """Tensor names of a local checkpoint, read from shard headers only.

    A Hub id is not downloaded here: a planning call must stay cheap, and the
    names arrive once the merge has a local snapshot.
    """
    from core.advanced import merge

    if not Path(source).expanduser().exists():
        return [], None
    try:
        entries, _ = merge._open_state(source)
    except (ValueError, OSError):
        return [], None
    names = sorted(entries)
    depths = [layer_lr.layer_index(name) for name in names]
    known = [depth for depth in depths if depth is not None]
    return names, (max(known) + 1 if known else None)


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


def _tokenizer_preview(sources: list[str], token: str | None = None) -> dict[str, Any]:
    """Tokenizer comparison for local checkpoints *and* Hub repos.

    A Hub parent is compared from its config/tokenizer JSONs alone (a few KB), so
    the disagreement that would refuse the merge is visible on Inspect -- before
    the weights are downloaded and before ``Run merge`` is pressed. Without this
    the UI could not offer the override the refusal message tells you to use.
    """
    try:
        return tokenizer_check.compare_tokenizers(
            sources, label = "merge", token = token
        ).to_dict()
    except ValueError as error:
        return {"status": "not_checked", "ok": None, "reason": str(error)}


@router.post("/merge/validate")
def validate_merge(request: MergeValidateRequest) -> dict[str, Any]:
    """Run the pre-merge gates without writing anything.

    This is what the UI calls when both slots are filled: tokenizer agreement,
    pickle weight files, and remote-code requirements, each with the reason it
    would block (or warn about) a run.
    """
    policy = _loading_policy(
        allow_remote_code = request.allow_remote_code,
        allow_non_safetensors = request.allow_non_safetensors,
    )
    sources = _merge_sources(request.model_a, request.model_b, request.models)
    inspections: list[dict[str, Any]] = []
    for label, source in zip(_merge_labels(len(sources)), sources):
        if Path(source).expanduser().exists():
            try:
                inspections.append(
                    {
                        "label": label,
                        **loading_policy.enforce_source_policy(
                            source, policy, label = label
                        ),
                        "blocked": False,
                    }
                )
            except loading_policy.PolicyViolation as error:
                inspections.append(
                    {
                        "label": label,
                        "source": source,
                        "allowed": False,
                        "blocked": True,
                        "blocked_reason": str(error),
                        "warnings": [str(error)],
                    }
                )
        else:
            inspections.append(
                {
                    "label": label,
                    "source": source,
                    "allowed": None,
                    "blocked": False,
                    "notes": ["Not downloaded yet; the policy is applied when it is."],
                }
            )
    tokenizer = _tokenizer_preview(sources, request.hf_token)
    blocked = any(item.get("blocked") for item in inspections)
    result: dict[str, Any] = {
        "ok": (not blocked) and tokenizer.get("status") != "conflict",
        "blocked": blocked or tokenizer.get("status") == "conflict",
        "policy": policy.to_dict(),
        "sources": inspections,
        "tokenizer": tokenizer,
    }
    if tokenizer.get("status") == "conflict":
        result["blocked_reason"] = (
            "The parents' tokenizers disagree on a special token; the merge would refuse "
            "to write. "
            + " ".join(tokenizer.get("conflicts") or [])
            + " Build it anyway with the override (allow_tokenizer_conflict in the API) "
            "if that is the eos/pad you want the result to use, or align the parents' "
            "special tokens first."
        )
    return result


@router.post("/merge/eval")
def eval_merged_model(request: MergeEvalRequest) -> dict[str, Any]:
    """Structural merge-eval: is the merged folder loadable and sane?

    This is the CPU tier. It deliberately does not claim task quality; the
    verdict vocabulary keeps ``pareto-positive``/``within-noise``/``regressed``
    for the benchmark tier so the two cannot be confused.
    """
    return merge_eval.structural_eval(
        request.source, sample_tensors = request.sample_tensors
    )


@router.post("/merge/plan")
def plan_merge(request: MergePlanRequest) -> dict[str, Any]:
    from core.advanced import merge

    sources = _merge_sources(request.model_a, request.model_b, request.models)
    try:
        has_gpu = merge.host_has_gpu()
        limit = merge.max_merge_models(has_gpu, sources = sources)
        merge.check_model_count(len(sources), limit)
        specs = [
            merge.probe_model_spec(source, token = request.hf_token)
            for source in sources
        ]
        # The same group target the merge itself aligns to, so the preview and
        # the run agree even when 3+ parents disagree on a dimension.
        alignments = merge.plan_alignment_many(specs)
        layer_range = None
        if request.layer_start is not None or request.layer_end is not None:
            layer_range = (
                request.layer_start or 0,
                request.layer_end
                if request.layer_end is not None
                else min(spec.num_layers or 0 for spec in specs),
            )
        moe = merge.plan_moe_merge(
            specs,
            top_k = request.top_k,
            router_hidden = request.router_hidden,
            temperature = request.temperature,
            layer_range = layer_range,
        )
    except ValueError as error:
        raise HTTPException(status_code = 400, detail = str(error)) from error
    suggested = _free_output_dir(_merge_default_name_for(sources))
    estimate = resources.estimate(
        "merge", sources = sources, output_dir = suggested
    )
    return {
        "models": [spec.to_dict() for spec in specs],
        # The pair summary stays a pair comparison (its notes are the "2048 vs
        # 1024" lines the cards render); the per-parent list carries the group
        # target every parent is actually cropped to.
        "alignment": (
            merge.plan_alignment(specs[0], specs[1]) if len(specs) == 2 else alignments[1]
        ).to_dict(),
        "alignments": [alignment.to_dict() for alignment in alignments],
        "moe": moe.to_dict(),
        "recommended_mode": (
            "mmoe"
            if any(alignment.mmoe_required for alignment in alignments[1:])
            else "standard"
        ),
        "tier": "gpu" if has_gpu else "cpu",
        "max_models": limit,
        "parent_cap_reason": "gpu" if has_gpu else "ram",
        # Prefilled into the output field so a merge lands in the folder Studio scans.
        "suggested_output_dir": str(suggested),
        # Plan before compute: peak memory, the output folder, and the tokenizer verdict.
        "estimates": {
            "need": estimate["need"],
            "notes": estimate["notes"],
            "streaming": True,
            "model_bytes": estimate["model_bytes"],
        },
        "tokenizer": _tokenizer_preview(sources, request.hf_token),
    }


@router.post("/merge/run")
def run_merge(request: MergeRunRequest) -> dict[str, Any]:
    from core.advanced import merge

    sources = _merge_sources(request.model_a, request.model_b, request.models)
    try:
        limit = merge.max_merge_models(merge.host_has_gpu(), sources = sources)
        merge.check_model_count(len(sources), limit)
    except ValueError as error:
        raise HTTPException(status_code = 400, detail = str(error)) from error
    for label, source in zip(_merge_labels(len(sources)), sources):
        _require_checkpoint_source(source, label)
    weights = request.weights
    if weights is None:
        # The pair shorthand: [1 - w, w]. An N-way merge with no weights is even.
        weights = [1.0 - request.weight, request.weight] if len(sources) == 2 else None
    if weights is not None and len(weights) != len(sources):
        raise HTTPException(
            status_code = 400,
            detail = f"Expected {len(sources)} weights, got {len(weights)}.",
        )
    if weights is not None and sum(weights) <= 0:
        raise HTTPException(
            status_code = 400, detail = "Merge weights must sum to a positive number."
        )
    policy = _loading_policy(
        allow_remote_code = request.allow_remote_code,
        allow_non_safetensors = request.allow_non_safetensors,
    )
    output = _output_dir(
        request.output_dir,
        kind = "merged",
        default_name = _merge_default_name_for(sources),
    )
    for source in sources:
        try:
            if output.resolve() == Path(source).expanduser().resolve():
                raise HTTPException(
                    status_code = 400,
                    detail = "The output directory must differ from every source checkpoint.",
                )
        except OSError:
            continue

    def work(progress) -> dict[str, Any]:
        return merge.merge_checkpoints_many(
            sources,
            str(output),
            method = request.method,
            weights = weights,
            density = request.density,
            seed = request.seed,
            max_shard_bytes = int(request.max_shard_gb * 1024**3),
            token = request.hf_token,
            policy = policy,
            allow_tokenizer_conflict = request.allow_tokenizer_conflict,
            progress = progress,
        )

    return _start_job(
        "merge",
        work,
        output_dir = str(output),
        sources = sources,
        with_progress = True,
    )


@router.post("/merge/moe")
def run_moe(request: MoeRunRequest) -> dict[str, Any]:
    from core.advanced import merge, moe

    for index, source in enumerate(request.models):
        _require_checkpoint_source(source, f"Model {index + 1}")
    policy = _loading_policy(
        allow_remote_code = request.allow_remote_code,
        allow_non_safetensors = request.allow_non_safetensors,
    )
    output = _output_dir(
        request.output_dir,
        kind = "moe",
        default_name = _merge_default_name_for(list(request.models)),
    )

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
        tokenizer = tokenizer_check.check_before_merge(snapshots)
        result = moe.write_moe_bundle(
            str(output),
            plan.to_dict(),
            hidden_size = plan.router_hidden,
            seed = 3407,
        )
        result["tokenizer"] = {
            "status": tokenizer.status,
            "warnings": list(tokenizer.warnings),
        }
        result["run_manifest"] = _stage_manifest(
            "moe",
            str(output),
            config = {"top_k": request.top_k, "plan": plan.to_dict()},
            inputs = {source: run_manifest.checkpoint_digest(source) for source in snapshots},
            artifacts = {"output_dir": str(output), "bundle": "moe_bundle.json"},
            seeds = {"router_seed": 3407},
            policy = {"loading": policy.to_dict()},
        )
        return result

    return _start_job("moe", work, output_dir = str(output), sources = request.models)


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
    output = _output_dir(
        request.output_dir,
        kind = "sharded",
        default_name = f"sharded-{Path(request.source).expanduser().name}",
    )

    def work() -> dict[str, Any]:
        result = splitter.split_checkpoint(
            request.source,
            str(output),
            max_shard_bytes = int((request.max_shard_gb or 4.0) * 1024**3),
            reserve_fraction = request.reserve_fraction,
            prefer = request.prefer,
            copy_tokenizer = request.copy_tokenizer,
        )
        result["run_manifest"] = _stage_manifest(
            "split",
            str(output),
            config = {
                "max_shard_bytes": int((request.max_shard_gb or 4.0) * 1024**3),
                "reserve_fraction": request.reserve_fraction,
                "prefer": request.prefer,
                "copy_tokenizer": request.copy_tokenizer,
            },
            inputs = {"source": run_manifest.checkpoint_digest(request.source)},
            artifacts = {
                "output_dir": str(output),
                "shards": result.get("shards") or result.get("shard_count"),
                "total_bytes": result.get("total_bytes"),
            },
        )
        return result

    return _start_job("split", work, output_dir = str(output), sources = [request.source])


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
        result = eval_runner.run_plan(
            plan,
            output_dir = request.output_dir,
            timeout = request.timeout_seconds,
        )
        result["run_manifest"] = _stage_manifest(
            "eval",
            request.output_dir,
            config = {
                "suites": list(request.suites),
                "batch_size": request.batch_size,
                "limit": request.limit,
                "eval_dataset": request.eval_dataset,
            },
            inputs = {
                "baseline": run_manifest.checkpoint_digest(request.baseline_model),
                "candidate": run_manifest.checkpoint_digest(request.candidate_model),
                "dataset": run_manifest.dataset_identity(request.eval_dataset),
            },
            artifacts = {"output_dir": request.output_dir},
        )
        return result

    return _start_job(
        "eval",
        work,
        output_dir = request.output_dir,
        sources = [request.baseline_model, request.candidate_model],
    )


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
            "library_visible",
            "resources",
            "progress",
            "run_manifest",
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
