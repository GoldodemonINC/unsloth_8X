# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from pathlib import Path

from core.advanced import nrc_license, run_manifest


def test_a_written_manifest_carries_the_license_and_a_notice(tmp_path: Path) -> None:
    manifest = run_manifest.build_manifest("merge")
    run_manifest.write_manifest(tmp_path, manifest)

    document = run_manifest.read_manifest(tmp_path)
    assert document is not None
    assert document["license"]["name"] == "NRC-3"
    assert document["license"]["upstream_unchanged"] is True

    notice = (tmp_path / nrc_license.LICENSE_FILE).read_text(encoding = "utf-8")
    assert "NRC-3" in notice
    assert "rebranded" in notice.lower()
    assert "at scale" in notice
    assert "keep that source's license" in notice


def test_a_stage_that_brings_its_own_license_keeps_it(tmp_path: Path) -> None:
    manifest = run_manifest.build_manifest("merge")
    manifest["license"] = {"name": "custom", "version": "9"}

    run_manifest.write_manifest(tmp_path, manifest)

    document = run_manifest.read_manifest(tmp_path)
    assert document is not None
    assert document["license"] == {"name": "custom", "version": "9"}


def test_rewriting_the_manifest_leaves_one_identical_notice(tmp_path: Path) -> None:
    run_manifest.write_manifest(tmp_path, run_manifest.build_manifest("merge"))
    first = (tmp_path / nrc_license.LICENSE_FILE).read_text(encoding = "utf-8")

    run_manifest.update_manifest(tmp_path, events = [])

    assert (tmp_path / nrc_license.LICENSE_FILE).read_text(encoding = "utf-8") == first
    assert len(list(tmp_path.glob(f"*{nrc_license.LICENSE_FILE}"))) == 1


def test_an_unwritable_folder_does_not_fail_the_stage() -> None:
    assert nrc_license.stamp(Path("/nonexistent-x8-dir")) is None


def test_a_saved_custom_pipeline_carries_the_license(
    tmp_path: Path, monkeypatch
) -> None:
    from core.labs import pipelines

    monkeypatch.setattr(pipelines, "root", lambda: tmp_path)
    monkeypatch.setattr(
        pipelines,
        "preflight",
        lambda spec: {
            "final_params": 1,
            "total_output_bytes": 2,
            "blocked": False,
            "blocked_reason": None,
            "stages": [{"kind": "merge"}],
        },
    )

    record = pipelines.save_pipeline(
        {"name": "two merges", "source": "model", "stages": []}
    )

    assert record["license"]["name"] == "NRC-3"
    assert (tmp_path / nrc_license.LICENSE_FILE).is_file()
