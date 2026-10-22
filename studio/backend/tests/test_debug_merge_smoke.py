# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("safetensors")

from core.training import debug_fixture


def test_the_smoke_merges_every_method_through_the_real_engine(tmp_path: Path):
    report = debug_fixture.run_merge_smoke(tmp_path / "smoke")

    assert report["ok"] is True
    assert [check["method"] for check in report["checks"]] == [
        "linear",
        "slerp",
        "ties",
        "dare",
    ]
    for check in report["checks"]:
        assert check["status"] == "completed"
        # N-way SLERP folds a running result, so only a pair of parents streams.
        if check["method"] != "slerp":
            assert check["streamed_tensors"] >= 1
        assert check["mean"] == pytest.approx(check["expected"], abs = 0.02)
        assert Path(report["workdir"]).is_dir()


def test_a_broken_merge_method_fails_the_smoke(tmp_path: Path, monkeypatch):
    from core.advanced import merge

    original = merge._stream_linear

    def _broken(*args, **kwargs):
        raise RuntimeError("blend broke")

    monkeypatch.setattr(merge, "_stream_linear", _broken)

    with pytest.raises(RuntimeError, match = "blend broke"):
        debug_fixture.run_merge_smoke(tmp_path / "broken")

    monkeypatch.setattr(merge, "_stream_linear", original)
