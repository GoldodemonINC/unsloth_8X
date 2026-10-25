# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from pathlib import Path

import pytest

from core.labs import housekeeping


def _write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(text, encoding = "utf-8")


def test_scan_sizes_outputs_and_marks_the_labs_folders_private(tmp_path, monkeypatch):
    outputs = tmp_path / "outputs"
    exports = tmp_path / "exports"
    hub = tmp_path / "hub"
    datasets = tmp_path / "datasets"
    outputs.mkdir()
    exports.mkdir()
    (outputs / "run-a").mkdir()
    _write(outputs / "run-a" / "model.safetensors", "a" * 32)
    _write(outputs / "run-a" / "run_manifest.json", '{"stage": "train"}')
    (outputs / "labs-pipelines").mkdir()
    _write(outputs / "labs-pipelines" / "plan.json", "{}")
    (exports / "bundle").mkdir()
    _write(exports / "bundle" / "model.gguf", "g" * 8)
    (hub / "models--org--name").mkdir(parents = True)
    _write(hub / "models--org--name" / "blob", "h" * 16)
    (datasets / "dataset--one").mkdir(parents = True)
    _write(datasets / "dataset--one" / "blob", "d" * 4)

    monkeypatch.setattr(
        housekeeping,
        "_paths",
        lambda: {
            "outputs": outputs,
            "exports": exports,
            "hf-hub": hub,
            "hf-datasets": datasets,
        },
    )
    monkeypatch.setattr(housekeeping, "_running_outputs", lambda: [])

    report = housekeeping.scan()
    names = {entry["name"]: entry for entry in report["entries"]}
    assert "labs-pipelines" not in names
    run_bytes = 32 + len('{"stage": "train"}')
    assert names["run-a"]["bytes"] == run_bytes
    assert names["run-a"]["kind"] == "run"
    assert names["run-a"]["deletable"] is True
    assert names["run-a"]["manifest"]["stage"] == "train"
    assert names["models--org--name"]["kind"] == "hf-model"
    assert names["bundle"]["kind"] == "export"
    assert report["total_bytes"] == run_bytes + 8 + 16 + 4
    assert [root["id"] for root in report["roots"]] == [
        "outputs",
        "exports",
        "hf-hub",
        "hf-datasets",
    ]


def test_delete_refuses_missing_confirmation_and_paths_outside_the_roots(tmp_path, monkeypatch):
    outputs = tmp_path / "outputs"
    outside = tmp_path / "outside"
    outputs.mkdir()
    outside.mkdir()
    target = outputs / "run-a"
    target.mkdir()
    _write(target / "config.json", "{}")

    monkeypatch.setattr(housekeeping, "_paths", lambda: {"outputs": outputs})
    monkeypatch.setattr(housekeeping, "_running_outputs", lambda: [])

    with pytest.raises(ValueError):
        housekeeping.delete([str(target)], confirm = False)

    result = housekeeping.delete([str(outside)], confirm = True)
    assert result["deleted"] == []
    assert result["refused"][0]["path"] == str(outside)

    # The outside path is refused; the folder the scan offered is removed.
    final = housekeeping.delete([str(target)], confirm = True)
    assert final["freed_bytes"] > 0
    assert not target.exists()


def test_delete_refuses_a_running_jobs_output(tmp_path, monkeypatch):
    outputs = tmp_path / "outputs"
    running = outputs / "run-a"
    running.mkdir(parents = True)
    _write(running / "config.json", "{}")
    monkeypatch.setattr(housekeeping, "_paths", lambda: {"outputs": outputs})
    monkeypatch.setattr(housekeeping, "_running_outputs", lambda: [str(running)])

    result = housekeeping.delete([str(running)], confirm = True)
    assert result["deleted"] == []
    assert "right now" in result["refused"][0]["reason"]
    assert running.is_dir()


def test_protected_reason_never_offers_a_storage_root(tmp_path):
    root = tmp_path / "outputs"
    root.mkdir()
    assert housekeeping._protected_reason(root, "outputs", root)
    assert housekeeping._protected_reason(root / "labs-pipelines", "outputs", root)
    nested = root / "run" / "nested"
    nested.mkdir(parents = True)
    assert housekeeping._protected_reason(nested, "outputs", root)
