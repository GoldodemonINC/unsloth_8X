# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Route-level checks for the advanced API.

These exercise the request models and the source validation without starting a
training run or touching the network.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from core.advanced import tokenizer_check  # noqa: E402
from routes import advanced  # noqa: E402


@pytest.fixture()
def client() -> TestClient:
    app = fastapi.FastAPI()
    app.include_router(advanced.router, prefix = "/api/advanced")
    return TestClient(app)


@pytest.fixture()
def started_jobs(monkeypatch) -> list[dict]:
    """Capture what a run route hands the job registry instead of really running it."""
    jobs: list[dict] = []

    def fake_start_job(
        kind,
        work,
        *,
        output_dir = None,
        sources = None,
        with_progress = False,
        growth_ratio = None,
    ):
        jobs.append(
            {
                "kind": kind,
                "output_dir": output_dir,
                "sources": sources,
                "with_progress": with_progress,
                "growth_ratio": growth_ratio,
                "work": work,
            }
        )
        return {
            "job_id": "job-1",
            "kind": kind,
            "status": "running",
            "created_at": 0.0,
            "finished_at": None,
            "output_dir": output_dir,
            # Same helper the real registry uses, so the route contract stays under test.
            "library_visible": advanced._library_visible(
                Path(output_dir) if output_dir else None
            ),
            "progress": None,
            "result": None,
            "error": None,
        }

    monkeypatch.setattr(advanced, "_start_job", fake_start_job)
    return jobs


def test_capabilities_exposes_merge_and_split_surfaces(client: TestClient):
    response = client.get("/api/advanced/capabilities")
    assert response.status_code == 200
    body = response.json()
    assert set(body["merge_methods"]) == {"linear", "slerp", "ties", "dare"}
    assert "memory" in body and "qat_formats" in body


def test_merge_plan_rejects_unknown_sources(client: TestClient, tmp_path):
    response = client.post(
        "/api/advanced/merge/plan",
        json={
            "model_a": str(tmp_path / "missing-a"),
            "model_b": str(tmp_path / "missing-b"),
            "top_k": 2,
            "temperature": 1.0,
        },
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "Hugging Face repo id" in detail
    assert "missing-a" in detail


def test_merge_run_rejects_unknown_sources_before_starting(client: TestClient, tmp_path):
    response = client.post(
        "/api/advanced/merge/run",
        json={
            "model_a": str(tmp_path / "missing-a"),
            "model_b": str(tmp_path / "missing-b"),
            "output_dir": str(tmp_path / "out"),
            "method": "linear",
            "weight": 0.5,
            "density": 0.5,
            "max_shard_gb": 4,
        },
    )
    assert response.status_code == 400
    assert "Model A" in response.json()["detail"]


def test_splitter_stays_local_only(client: TestClient):
    # The splitter streams existing safetensors; a Hub id is not a valid source.
    response = client.post(
        "/api/advanced/split/plan",
        json={"source": "unsloth/Llama-3.2-1B", "prefer": "auto"},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "existing local checkpoint path" in detail


def test_merge_plan_accepts_repo_ids_structurally(client: TestClient, monkeypatch):
    """A repo id passes validation and reaches the probe (stubbed, no network)."""
    from core.advanced import merge as merge_module

    seen: list[tuple[str, str | None]] = []

    def fake_probe(source: str, *, token: str | None = None):
        seen.append((source, token))
        return merge_module.ModelSpec(
            source = f"hf://{source}",
            architecture = "llama",
            hidden_size = 2048,
            num_layers = 16,
            num_heads = 8,
            intermediate_size = 8192,
            vocab_size = 128256,
        )

    monkeypatch.setattr(merge_module, "probe_model_spec", fake_probe)

    response = client.post(
        "/api/advanced/merge/plan",
        json={
            "model_a": "unsloth/Llama-3.2-1B",
            "model_b": "unsloth/Llama-3.2-1B-Instruct",
            "top_k": 2,
            "temperature": 1.0,
            "hf_token": "hf_test",
        },
    )
    assert response.status_code == 200, response.text
    assert seen == [
        ("unsloth/Llama-3.2-1B", "hf_test"),
        ("unsloth/Llama-3.2-1B-Instruct", "hf_test"),
    ]
    body = response.json()
    assert body["recommended_mode"] == "standard"
    assert body["models"][0]["hidden_size"] == 2048


# ---------------------------------------------------------------------------
# Output folders
#
# The UI prefills a bare name, not a machine-specific path, so the same request has to work
# on Windows and on a Colab runtime where ``C:\\models`` does not exist.
# ---------------------------------------------------------------------------


def _two_checkpoints(tmp_path: Path) -> tuple[str, str]:
    sources = []
    for name in ("model-a", "model-b"):
        path = tmp_path / name
        path.mkdir()
        sources.append(str(path))
    return sources[0], sources[1]


def test_a_bare_merge_output_name_lands_under_the_outputs_root(
    client: TestClient, tmp_path, started_jobs
):
    from utils.paths import resolve_output_dir

    model_a, model_b = _two_checkpoints(tmp_path)
    response = client.post(
        "/api/advanced/merge/run",
        json={
            "model_a": model_a,
            "model_b": model_b,
            "output_dir": "merged-test",
            "method": "linear",
            "weight": 0.5,
            "density": 0.5,
            "max_shard_gb": 4,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    resolved = str(resolve_output_dir("merged-test"))
    assert body["output_dir"] == resolved
    assert Path(body["output_dir"]).is_absolute()
    assert Path(body["output_dir"]).name == "merged-test"
    assert [(job["kind"], job["output_dir"]) for job in started_jobs] == [
        ("merge", resolved)
    ]


def test_a_blank_merge_output_dir_lands_beside_the_training_runs(
    client: TestClient, tmp_path, started_jobs, monkeypatch
):
    """The Library, Export and Chat scan the outputs root's children, so a default must be one."""
    import utils.paths

    outputs = tmp_path / "outputs"
    outputs.mkdir()
    monkeypatch.setattr(utils.paths, "outputs_root", lambda: outputs)

    model_a, model_b = _two_checkpoints(tmp_path)
    response = client.post(
        "/api/advanced/merge/run",
        json={"model_a": model_a, "model_b": model_b, "output_dir": "  "},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert Path(body["output_dir"]).parent == outputs
    assert Path(body["output_dir"]).name == "merge-model-a-model-b"
    assert body["library_visible"] is True
    assert [(job["kind"], job["output_dir"]) for job in started_jobs] == [
        ("merge", body["output_dir"])
    ]


def test_a_second_merge_of_the_same_pair_gets_its_own_folder(
    client: TestClient, tmp_path, started_jobs, monkeypatch
):
    import utils.paths

    outputs = tmp_path / "outputs"
    (outputs / "merge-model-a-model-b").mkdir(parents = True)
    monkeypatch.setattr(utils.paths, "outputs_root", lambda: outputs)

    model_a, model_b = _two_checkpoints(tmp_path)
    response = client.post(
        "/api/advanced/merge/run",
        json={"model_a": model_a, "model_b": model_b},
    )
    assert response.status_code == 200, response.text
    assert response.json()["output_dir"] == str(outputs / "merge-model-a-model-b-2")


def test_merge_plan_suggests_a_folder_studio_can_list(client: TestClient, tmp_path, monkeypatch):
    import utils.paths
    from core.advanced import merge as merge_module

    outputs = tmp_path / "outputs"
    outputs.mkdir()
    monkeypatch.setattr(utils.paths, "outputs_root", lambda: outputs)
    monkeypatch.setattr(
        merge_module,
        "probe_model_spec",
        lambda source, *, token = None: merge_module.ModelSpec(
            source = source,
            architecture = "llama",
            hidden_size = 2048,
            num_layers = 16,
            num_heads = 8,
            intermediate_size = 8192,
            vocab_size = 128256,
        ),
    )

    response = client.post(
        "/api/advanced/merge/plan",
        json={"model_a": "org/Tiny-A", "model_b": "org/Tiny-B"},
    )
    assert response.status_code == 200, response.text
    suggested = Path(response.json()["suggested_output_dir"])
    assert suggested.parent == outputs
    assert suggested.name == "merge-Tiny-A-Tiny-B"


def test_outputs_reports_the_folder_studio_scans(client: TestClient, tmp_path, monkeypatch):
    import utils.paths

    monkeypatch.setattr(utils.paths, "outputs_root", lambda: tmp_path / "outputs")
    monkeypatch.setattr(utils.paths, "exports_root", lambda: tmp_path / "exports")
    body = client.get("/api/advanced/outputs").json()
    assert body == {
        "outputs_root": str(tmp_path / "outputs"),
        "exports_root": str(tmp_path / "exports"),
    }


def test_an_absolute_merge_output_dir_is_used_as_given(client: TestClient, tmp_path, started_jobs):
    model_a, model_b = _two_checkpoints(tmp_path)
    target = tmp_path / "elsewhere" / "merged"
    response = client.post(
        "/api/advanced/merge/run",
        json={"model_a": model_a, "model_b": model_b, "output_dir": str(target)},
    )
    assert response.status_code == 200, response.text
    assert response.json()["output_dir"] == str(target)
    # Outside the scanned folder: perfectly writable, but no Library/Export/Chat entry.
    assert response.json()["library_visible"] is False


def test_a_windows_drive_path_is_used_as_given_on_any_runtime(client: TestClient, tmp_path, started_jobs):
    # ``C:/models`` is not absolute under POSIX rules; it must still never be joined onto the
    # outputs root, or a Windows path would silently land inside the Colab/Linux folder.
    model_a, model_b = _two_checkpoints(tmp_path)
    response = client.post(
        "/api/advanced/merge/run",
        json={"model_a": model_a, "model_b": model_b, "output_dir": "C:/models/merged"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["output_dir"] == str(Path("C:/models/merged").expanduser())


def test_a_drive_relative_name_stays_inside_the_outputs_root(client: TestClient, tmp_path, started_jobs):
    # ``C:merged`` is drive-relative, not absolute: it must be contained, never left to the
    # process working directory.
    from utils.paths import resolve_output_dir

    model_a, model_b = _two_checkpoints(tmp_path)
    response = client.post(
        "/api/advanced/merge/run",
        json={"model_a": model_a, "model_b": model_b, "output_dir": "C:merged"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["output_dir"] == str(resolve_output_dir("C:merged"))


def test_an_output_dir_that_escapes_the_outputs_root_is_rejected(client: TestClient, tmp_path):
    source = tmp_path / "checkpoint"
    source.mkdir()
    response = client.post(
        "/api/advanced/split/run",
        json={"source": str(source), "output_dir": "../../escape", "prefer": "auto"},
    )
    assert response.status_code == 400
    assert ".." in response.json()["detail"]


def test_a_bare_split_output_name_lands_under_the_outputs_root(
    client: TestClient, tmp_path, started_jobs
):
    from utils.paths import resolve_output_dir

    source = tmp_path / "checkpoint"
    source.mkdir()
    response = client.post(
        "/api/advanced/split/run",
        json={"source": str(source), "output_dir": "sharded-test", "prefer": "auto"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["output_dir"] == str(resolve_output_dir("sharded-test"))

def test_the_default_merge_output_is_a_model_the_library_lists(client: TestClient, tmp_path, monkeypatch):
    """End-to-end: no folder given, and Studio's own model scan still finds the result.

    Library, Export and Chat all list the outputs root's direct children, so the default
    has to be one of them and hold a config plus weights.
    """
    pytest.importorskip("safetensors")
    import numpy as np
    from safetensors.numpy import save_file

    import utils.paths
    from utils.models.model_config import scan_trained_models
    from utils.paths import storage_roots

    outputs = tmp_path / "outputs"
    outputs.mkdir()
    monkeypatch.setattr(utils.paths, "outputs_root", lambda: outputs)
    monkeypatch.setattr(storage_roots, "outputs_root", lambda: outputs)

    def checkpoint(name: str) -> Path:
        path = tmp_path / name
        path.mkdir()
        (path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen2",
                    "hidden_size": 4,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 1,
                    "intermediate_size": 8,
                    "vocab_size": 16,
                }
            ),
            encoding = "utf-8",
        )
        save_file(
            {"model.embed_tokens.weight": np.zeros((16, 4), dtype = np.float32)},
            str(path / "model.safetensors"),
        )
        return path

    model_a = checkpoint("tiny-a")
    model_b = checkpoint("tiny-b")
    created = client.post(
        "/api/advanced/merge/run",
        json={"model_a": str(model_a), "model_b": str(model_b)},
    ).json()
    assert created["output_dir"] == str(outputs / "merge-tiny-a-tiny-b")

    deadline = time.time() + 60
    while time.time() < deadline:
        job = advanced._job(created["job_id"])
        if job["status"] != "running":
            break
        time.sleep(0.05)
    assert job["status"] == "completed", job["error"]

    listed = {name: model_type for name, _path, model_type in scan_trained_models(str(outputs))}
    assert listed.get("merge-tiny-a-tiny-b") == "merged"

    # The Export page lists checkpoints from its own scan; its ?run= preselect matches here.
    from utils.models.checkpoints import scan_checkpoints

    exported = {name: checkpoints for name, checkpoints, _meta in scan_checkpoints(str(outputs))}
    assert exported["merge-tiny-a-tiny-b"][0][1] == str(outputs / "merge-tiny-a-tiny-b")


def _tokenized_checkpoint(
    directory: Path, *, eos: str = "</s>", eos_id: int = 2, pad: str | None = None, pad_id: int | None = None
) -> str:
    pytest.importorskip("numpy")
    pytest.importorskip("safetensors")
    import numpy as np
    from safetensors.numpy import save_file

    directory.mkdir(parents = True, exist_ok = True)
    (directory / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "intermediate_size": 8,
                "vocab_size": 32,
                "eos_token_id": eos_id,
            }
        ),
        encoding = "utf-8",
    )
    special: dict[str, object] = {"eos_token": eos}
    config_tokens: dict[str, object] = {"eos_token_id": eos_id}
    if pad is not None:
        special["pad_token"] = pad
        config_tokens["pad_token_id"] = pad_id
    (directory / "special_tokens_map.json").write_text(json.dumps(special), encoding = "utf-8")
    (directory / "tokenizer_config.json").write_text(json.dumps(config_tokens), encoding = "utf-8")
    save_file(
        {"model.embed_tokens.weight": np.zeros((32, 8), dtype = np.float32)},
        str(directory / "model.safetensors"),
    )
    return str(directory)


def test_capabilities_reports_the_platform_scope_honestly(client: TestClient):
    body = client.get("/api/advanced/capabilities").json()
    features = {feature["key"]: feature for feature in body["features"]}
    assert features["merge"]["cpu"] is True
    assert features["split"]["label"] == "VRAM Fit (checkpoint sharding)"
    assert features["training"]["cpu"] is False
    assert "Colab" in body["platform_line"]
    assert body["merge_streams"] is True
    assert "streaming only" in body["merge_cpu_note"]


def test_curriculum_plan_is_labelled_experimental(client: TestClient):
    body = client.post(
        "/api/advanced/curriculum/plan", json={"cadence": "reasoning:1,code:1"}
    ).json()
    assert body["label"] == "experimental"
    assert "unproven" in body["note"]
    assert body["stages"]


def test_merge_validate_blocks_conflicting_tokenizers(client: TestClient, tmp_path: Path):
    a = _tokenized_checkpoint(tmp_path / "a", eos = "</s>", eos_id = 2)
    b = _tokenized_checkpoint(tmp_path / "b", eos = "<|end|>", eos_id = 9)
    body = client.post(
        "/api/advanced/merge/validate", json={"model_a": a, "model_b": b}
    ).json()
    assert body["ok"] is False
    assert body["blocked"] is True
    assert body["tokenizer"]["status"] == "conflict"
    assert "refuse to write" in body["blocked_reason"]


def test_merge_validate_sees_a_hub_conflict_before_the_weights_download(
    client: TestClient, tmp_path: Path, monkeypatch
):
    """A Hub parent is checked from its JSON metadata, so Inspect can show the block.

    Previously a repo id reported ``not_checked`` here, so the UI never rendered the
    override it needs, while ``merge/run`` refused inside the job -- a dead end.
    """
    a = _tokenized_checkpoint(
        tmp_path / "a", eos = "<|endoftext|>", eos_id = 151643, pad = "<|endoftext|>", pad_id = 151643
    )
    b = _tokenized_checkpoint(
        tmp_path / "b", eos = "<|im_end|>", eos_id = 151645, pad = "<|im_end|>", pad_id = 151645
    )
    roots = {"org/a": a, "org/b": b}

    def fake_download(repo_id: str, filename: str, token: str | None) -> str:
        target = Path(roots[repo_id]) / filename
        if not target.is_file():
            raise FileNotFoundError(filename)
        return str(target)

    monkeypatch.setattr(tokenizer_check, "_hub_download", fake_download)

    body = client.post(
        "/api/advanced/merge/validate",
        json={"model_a": "hf://org/a", "model_b": "hf://org/b", "hf_token": "hf_test"},
    ).json()

    assert body["tokenizer"]["status"] == "conflict"
    assert body["tokenizer"]["ok"] is False
    assert body["blocked"] is True
    # The refusal has to name the way forward, since the UI renders it verbatim.
    assert "allow_tokenizer_conflict" in body["blocked_reason"]


def test_merge_validate_passes_matching_tokenizers(client: TestClient, tmp_path: Path):
    a = _tokenized_checkpoint(tmp_path / "a")
    b = _tokenized_checkpoint(tmp_path / "b")
    body = client.post(
        "/api/advanced/merge/validate", json={"model_a": a, "model_b": b}
    ).json()
    assert body["ok"] is True
    assert body["blocked"] is False
    assert body["tokenizer"]["status"] == "ok"


def test_merge_validate_refuses_pickle_weights(client: TestClient, tmp_path: Path):
    source = tmp_path / "pickled"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "qwen2"}), encoding = "utf-8")
    (source / "pytorch_model.bin").write_bytes(b"\0")
    other = _tokenized_checkpoint(tmp_path / "b")
    body = client.post(
        "/api/advanced/merge/validate",
        json={"model_a": str(source), "model_b": other},
    ).json()
    assert body["ok"] is False
    assert body["sources"][0]["blocked"] is True
    assert "allow_non_safetensors" in body["sources"][0]["blocked_reason"]


def test_layer_lr_plan_returns_a_checked_zone_map(client: TestClient):
    body = client.post(
        "/api/advanced/layer-lr/plan",
        json={
            "base_learning_rate": 2e-4,
            "num_layers": 6,
            "param_names": [
                "model.embed_tokens.weight",
                "model.layers.0.mlp.experts.2.w1.weight",
                "model.layers.5.self_attn.q_proj.weight",
                "model.norm.weight",
            ],
        },
    ).json()
    assert body["coverage"]["exact_one_zone"] is True
    assert body["preview"]["zone_map"]
    assert body["zone_definition_hash"]


def test_layer_lr_plan_reports_module_kind_groups(client: TestClient):
    body = client.post(
        "/api/advanced/layer-lr/plan",
        json={
            "base_learning_rate": 2e-4,
            "num_layers": 6,
            "zone_multipliers": {"early": 0.5},
            "kind_multipliers": {"attention": 0.5, "mlp": 2.0},
            "param_names": [
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.mlp.down_proj.weight",
                "model.norm.weight",
            ],
        },
    ).json()
    assert body["preview"]["kind_multipliers"]["attention"] == 0.5
    kinds = {entry["kind"]: entry for entry in body["preview"]["kind_map"]}
    assert kinds["attention"]["params"] == 1
    assert kinds["mlp"]["params"] == 1
    attention = next(
        entry for entry in body["preview"]["entries"] if entry["kind"] == "attention"
    )
    # zone 0.5 * kind 0.5 on a 2e-4 base.
    assert attention["multiplier"] == pytest.approx(0.25)
    assert attention["lr"] == pytest.approx(5e-5)
    assert body["coverage"]["exact_one_zone"] is True


def test_layer_lr_plan_rejects_unknown_kinds(client: TestClient):
    response = client.post(
        "/api/advanced/layer-lr/plan",
        json={"kind_multipliers": {"attn": 0.5}},
    )
    assert response.status_code == 400
    assert "Unknown parameter kinds" in response.json()["detail"]


def test_qat_validate_measures_at_the_export_quant(client: TestClient, tmp_path: Path):
    pytest.importorskip("numpy")
    pytest.importorskip("safetensors")
    import numpy as np
    from safetensors.numpy import save_file

    paths = []
    for name, seed in (("parent", 1), ("candidate", 2)):
        directory = tmp_path / name
        directory.mkdir()
        rng = np.random.default_rng(seed)
        save_file(
            {"model.layers.0.mlp.down_proj.weight": rng.normal(size = (64, 128)).astype(np.float32)},
            str(directory / "model.safetensors"),
        )
        (directory / "config.json").write_text("{\"model_type\": \"qwen2\"}", encoding = "utf-8")
        paths.append(str(directory))

    body = client.post(
        "/api/advanced/qat/validate",
        json={"enabled": True, "format": "int4", "parent": paths[0], "candidate": paths[1]},
    ).json()
    assert body["export_quality"]["status"] == "measured"
    assert body["export_quality"]["format"] == "int4"
    assert body["export_quality"]["tensors_compared"] == 1
    assert "claims" not in body["claim"].lower()

    config_only = client.post("/api/advanced/qat/validate", json={"enabled": True}).json()
    assert config_only["export_quality"]["status"] == "not_measured"
    assert config_only["export_quality"]["verdict"] == "unmeasured"


def test_resources_endpoint_reports_leases_and_estimates(client: TestClient, tmp_path: Path):
    snapshot = client.get("/api/advanced/resources").json()
    assert set(snapshot["limits"]) == {"vram_bytes", "ram_bytes", "disk_bytes"}
    assert "leases" in snapshot and "available" in snapshot

    source = _tokenized_checkpoint(tmp_path / "model")
    estimate = client.post(
        "/api/advanced/resources/estimate", json={"kind": "merge", "sources": [source]}
    ).json()
    assert estimate["kind"] == "merge"
    assert estimate["need"]["vram_bytes"] == 0


def test_jobs_acquire_denies_a_request_that_cannot_fit(client: TestClient, tmp_path: Path, monkeypatch):
    source = _tokenized_checkpoint(tmp_path / "model")
    monkeypatch.setattr(
        advanced.resources.BROKER,
        "_probe",
        lambda: {
            "vram_free_bytes": 0,
            "vram_total_bytes": 0,
            "ram_free_bytes": 1024,
            "ram_total_bytes": 1024,
            "disk_free_bytes": 1024,
            "gpus": [],
        },
    )
    response = client.post(
        "/api/advanced/jobs/acquire",
        json={"job_id": "probe", "kind": "eval", "sources": [source]},
    )
    assert response.status_code == 409
    assert "offers at most" in response.json()["detail"]


def test_start_job_records_the_lease_and_releases_it():
    job = advanced._start_job("merge", lambda: {"status": "completed"})
    deadline = time.time() + 10
    while time.time() < deadline and advanced._job(job["job_id"])["status"] == "running":
        time.sleep(0.01)
    finished = advanced._job(job["job_id"])
    assert finished["status"] == "completed"
    assert finished["resources"]["status"] == "granted"
    assert "run_manifest" in finished
    # The lease is released when the work finishes, or nothing else can start.
    assert all(
        lease["job_id"] != job["job_id"] for lease in advanced.resources.BROKER.leases()
    )


def test_a_finished_merge_carries_its_structural_report(
    client: TestClient, tmp_path: Path, started_jobs, monkeypatch
):
    """Eval → keep or discard: the job's own result says whether to keep the folder."""
    from core.advanced import merge as merge_module

    model_a = _tokenized_checkpoint(tmp_path / "a")
    model_b = _tokenized_checkpoint(tmp_path / "b")
    output = tmp_path / "out"
    output.mkdir()
    monkeypatch.setattr(
        merge_module,
        "merge_checkpoints_many",
        lambda *args, **kwargs: {"metrics": {"blended_tensors": 2}},
    )
    client.post(
        "/api/advanced/merge/run",
        json={"model_a": model_a, "model_b": model_b, "output_dir": str(output)},
    )

    result = started_jobs[-1]["work"](lambda payload: None)

    # The merge's own report survives beside the check.
    assert result["metrics"] == {"blended_tensors": 2}
    assert result["structural_eval"]["verdict"] == "unloadable"
    # An empty output folder is not a model, so the gate has to say discard.
    assert result["structural_eval"]["gate"]["decision"] == "discard"


def test_the_harness_install_route_reports_what_it_did(client: TestClient, monkeypatch):
    monkeypatch.setattr(
        advanced.eval_harness,
        "install",
        lambda **kwargs: {
            "installed": True,
            "available": True,
            "harness": {},
            "command": ["python", "-m", "pip", "install", "lm-eval"],
            "output": "Successfully installed lm-eval",
            "note": "lm_eval imports in this environment now.",
        },
    )

    body = client.post("/api/advanced/eval/install").json()

    assert body["installed"] is True
    assert body["available"] is True


def test_the_harness_install_route_maps_a_failed_install_to_500(
    client: TestClient, monkeypatch
):
    """A pip failure is the user's answer, not a silent no-op."""

    def boom(**kwargs):
        raise advanced.eval_harness.HarnessInstallError(
            "`python -m pip install lm-eval` exited 1. no matching distribution"
        )

    monkeypatch.setattr(advanced.eval_harness, "install", boom)

    response = client.post("/api/advanced/eval/install")

    assert response.status_code == 500
    assert "exited 1" in response.json()["detail"]


def test_a_finished_reshape_carries_its_structural_report(
    client: TestClient, tmp_path: Path, started_jobs, monkeypatch
):
    """Loadability after reshaping: the scaler's result carries the same gate."""
    from core.advanced import scaling as scaling_module

    source = _tokenized_checkpoint(tmp_path / "base")
    planned = client.post("/api/advanced/scale/plan", json={"source": source}).json()
    output = tmp_path / "scaled"
    output.mkdir()
    monkeypatch.setattr(
        scaling_module,
        "run",
        lambda *args, **kwargs: {
            "final_params": planned["default_target_params"],
            "verified": True,
            "step0_max_logit_diff": 0.0,
        },
    )
    client.post(
        "/api/advanced/scale/run",
        json={
            "source": source,
            "target_params": planned["default_target_params"],
            "output_dir": str(output),
        },
    )

    result = started_jobs[-1]["work"](lambda payload: None)

    # The reshape verified its logits, but the folder it wrote holds no weights,
    # so the loadability gate has to say discard on its own.
    assert result["structural_eval"]["verdict"] == "unloadable"
    assert result["structural_eval"]["gate"]["decision"] == "discard"


def test_a_scale_run_hands_the_planned_growth_to_the_lease(
    client: TestClient, tmp_path: Path, started_jobs, monkeypatch
):
    from core.advanced import scaling as scaling_module

    source = _tokenized_checkpoint(tmp_path / "base")
    planned = client.post("/api/advanced/scale/plan", json = {"source": source}).json()
    output = tmp_path / "scaled"
    output.mkdir()
    monkeypatch.setattr(
        scaling_module,
        "run",
        lambda *args, **kwargs: {
            "final_params": planned["default_target_params"],
            "verified": True,
            "step0_max_logit_diff": 0.0,
        },
    )
    client.post(
        "/api/advanced/scale/run",
        json = {
            "source": source,
            "target_params": planned["default_target_params"],
            "output_dir": str(output),
        },
    )

    assert started_jobs[-1]["kind"] == "scale"
    assert started_jobs[-1]["growth_ratio"] == pytest.approx(
        planned["default_target_params"] / planned["base_params"]
    )


def test_the_completion_check_warns_on_a_merge_that_only_loads(tmp_path: Path):
    """All-zero sampled weights load, so it is review rather than a clean keep."""
    report = advanced._completion_check(Path(_tokenized_checkpoint(tmp_path / "merged")))

    assert report["verdict"] == "structurally-valid"
    assert report["gate"]["decision"] == "review"


def test_merge_run_hands_its_sources_to_the_broker(client: TestClient, tmp_path: Path, started_jobs):
    model_a, model_b = _two_checkpoints(tmp_path)
    client.post(
        "/api/advanced/merge/run", json={"model_a": model_a, "model_b": model_b}
    )
    assert started_jobs[-1]["sources"] == [model_a, model_b]


# ---------------------------------------------------------------------------
# N-way merges (2 to 6 parents)
# ---------------------------------------------------------------------------


def _n_checkpoints(tmp_path: Path, count: int) -> list[str]:
    return [
        _tokenized_checkpoint(tmp_path / f"parent{index}") for index in range(count)
    ]


def test_merge_plan_reports_every_parent(client: TestClient, tmp_path: Path):
    models = _n_checkpoints(tmp_path, 3)
    body = client.post(
        "/api/advanced/merge/plan",
        json={"models": models, "top_k": 2, "temperature": 1.0},
    ).json()

    assert len(body["models"]) == 3
    assert len(body["alignments"]) == 3
    assert len(body["moe"]["experts"]) == 3
    # The pair summary is still there for the cards that read it.
    assert body["alignment"]["mode"] == "identical"


def test_merge_run_accepts_six_parents(
    client: TestClient, tmp_path: Path, started_jobs, monkeypatch
):
    """Six parents stay the GPU tier's ceiling."""
    from core.advanced import merge as merge_module

    monkeypatch.setattr(merge_module, "host_has_gpu", lambda: True)
    models = _n_checkpoints(tmp_path, 6)
    response = client.post(
        "/api/advanced/merge/run",
        json={
            "models": models,
            "weights": [1, 1, 1, 1, 1, 1],
            "method": "linear",
            "density": 0.5,
            "max_shard_gb": 4,
        },
    )

    assert response.status_code == 200
    assert started_jobs[0]["sources"] == models
    # The page needs a progress stream for a run this long.
    assert started_jobs[0]["with_progress"] is True


def test_merge_run_allows_four_small_parents_on_a_cpu_host(
    client: TestClient, tmp_path: Path, monkeypatch, started_jobs
):
    """A CPU host is capped by RAM, not by the missing GPU."""
    from core.advanced import merge as merge_module

    monkeypatch.setattr(merge_module, "host_has_gpu", lambda: False)
    monkeypatch.setattr(merge_module, "free_ram_bytes", lambda: 16 * 1024**3)
    models = _n_checkpoints(tmp_path, 4)
    response = client.post(
        "/api/advanced/merge/run",
        json={"models": models, "method": "linear", "density": 0.5, "max_shard_gb": 4},
    )

    assert response.status_code == 200
    assert started_jobs[-1]["sources"] == models


def test_merge_run_stops_at_two_when_ram_is_short(
    client: TestClient, tmp_path: Path, monkeypatch, started_jobs
):
    """A host without room for a third row chunk says so before starting."""
    from core.advanced import merge as merge_module

    monkeypatch.setattr(merge_module, "host_has_gpu", lambda: False)
    monkeypatch.setattr(merge_module, "free_ram_bytes", lambda: 40 * 1024**2)
    models = _n_checkpoints(tmp_path, 3)
    response = client.post(
        "/api/advanced/merge/run",
        json={"models": models, "method": "linear", "density": 0.5, "max_shard_gb": 4},
    )

    assert response.status_code == 400
    assert "can merge at most 2" in response.json()["detail"]
    assert started_jobs == []


def test_merge_run_rejects_seven_parents(client: TestClient, tmp_path: Path):
    models = _n_checkpoints(tmp_path, 7)
    response = client.post(
        "/api/advanced/merge/run",
        json={"models": models, "method": "linear", "density": 0.5, "max_shard_gb": 4},
    )
    assert response.status_code == 422


def test_merge_run_rejects_duplicate_parents(client: TestClient, tmp_path: Path, started_jobs):
    models = _n_checkpoints(tmp_path, 2)
    response = client.post(
        "/api/advanced/merge/run",
        json={
            "models": [models[0], models[0]],
            "method": "linear",
            "density": 0.5,
            "max_shard_gb": 4,
        },
    )
    assert response.status_code == 400
    assert "different model" in response.json()["detail"]
    assert started_jobs == []


def test_merge_run_rejects_a_weight_per_wrong_parent_count(
    client: TestClient, tmp_path: Path, started_jobs
):
    models = _n_checkpoints(tmp_path, 3)
    response = client.post(
        "/api/advanced/merge/run",
        json={
            "models": models,
            "weights": [1, 1],
            "method": "linear",
            "density": 0.5,
            "max_shard_gb": 4,
        },
    )
    assert response.status_code == 400
    assert "Expected 3 weights" in response.json()["detail"]
    assert started_jobs == []


def test_merge_validate_checks_every_parent(client: TestClient, tmp_path: Path):
    models = _n_checkpoints(tmp_path, 3)
    body = client.post(
        "/api/advanced/merge/validate", json={"models": models}
    ).json()
    assert len(body["sources"]) == 3
    assert body["tokenizer"]["status"] == "ok"


def test_start_job_publishes_merge_progress(client: TestClient):
    """The merge page's bar reads /jobs/{id}.progress, so the runner must fill it."""
    import time

    def work(progress):
        progress({"stage": "merging", "done": 1, "total": 4, "message": "1/4 tensors"})
        return {"ok": True}

    record = advanced._start_job("merge", work, with_progress = True)
    job = record
    for _ in range(200):
        job = advanced._job(record["job_id"])
        if job["status"] != "running":
            break
        time.sleep(0.01)

    assert job["status"] == "completed"
    assert job["progress"]["stage"] == "merging"
    assert job["progress"]["done"] == 1
    assert job["progress"]["total"] == 4
    assert "updated_at" in job["progress"]
