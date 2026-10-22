# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for export capability gating.

Export is supported on ``get_device() in {CUDA, XPU, MLX}`` and on a CPU-only host that has a
working torch, where the CPU export path does the merge; a torch-aware reason is reported when no
runtime exists (pytorch_not_installed / mlx_unavailable / detection_failed). The backend must
import without PyTorch. The matrix mocks the hardware probes; wiring is checked with ast so it
runs on CPU.
"""

import ast
import builtins
from pathlib import Path

import pytest

import utils.hardware.hardware as hw

_BACKEND = Path(__file__).resolve().parent.parent


def _src(rel):
    return (_BACKEND / rel).read_text(encoding = "utf-8")


def _func_src(rel, name):
    src = _src(rel)
    node = next(
        n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    return ast.get_source_segment(src, node)


# -- capability matrix --------------------------------------------------------------------------


def _patch(monkeypatch, *, torch: bool, device, apple: bool):
    monkeypatch.setattr(hw, "_has_torch", lambda: torch)
    monkeypatch.setattr(hw, "get_device", lambda: device)
    monkeypatch.setattr(hw, "is_apple_silicon", lambda: apple)


def test_cpu_with_torch_supports_export_through_the_cpu_path(monkeypatch):
    # PyTorch present but no accelerator: the merge runs through transformers + PEFT on the CPU,
    # so export is offered and the response says so.
    _patch(monkeypatch, torch = True, device = hw.DeviceType.CPU, apple = False)
    monkeypatch.setattr(hw, "current_chat_only_verdict", lambda: ("no_gpu", None))
    cap = hw.export_capability()
    assert cap["export_supported"] is True
    assert cap["export_unsupported_reason"] is None
    assert cap["export_unsupported_message"] is None
    assert cap["export_cpu_only"] is True
    assert "CPU" in cap["export_cpu_note"]
    # The note names what still needs an accelerator, so a missing FP8 pill reads as intended.
    assert "FP8" in cap["export_cpu_note"] and "4-bit" in cap["export_cpu_note"]
    # torchao quantizes through accelerator kernels, so the CPU path reports it unavailable even
    # though the package itself may load.
    assert cap["torchao_export_supported"] is False


def test_gpu_present_but_unusable_still_exports_on_cpu(monkeypatch):
    """A CPU torch build on a GPU host exports on the CPU and keeps the repair advice."""
    _patch(monkeypatch, torch = True, device = hw.DeviceType.CPU, apple = False)
    monkeypatch.setattr(hw, "current_chat_only_verdict", lambda: ("torch_cpu_build", "2.11.0+cpu"))
    monkeypatch.setattr(hw, "CHAT_ONLY_MISMATCH_VENDORS", ("nvidia",))
    cap = hw.export_capability()
    assert cap["export_supported"] is True
    assert cap["export_cpu_only"] is True
    assert "CPU" in cap["export_cpu_note"]


def test_cuda_with_torch_supports_export(monkeypatch):
    _patch(monkeypatch, torch = True, device = hw.DeviceType.CUDA, apple = False)
    cap = hw.export_capability()
    assert cap["export_supported"] is True
    assert cap["export_unsupported_reason"] is None
    assert cap["export_unsupported_message"] is None


def test_xpu_with_torch_supports_export(monkeypatch):
    _patch(monkeypatch, torch = True, device = hw.DeviceType.XPU, apple = False)
    assert hw.export_capability()["export_supported"] is True


def test_mlx_without_torch_supports_export(monkeypatch):
    # Apple Silicon MLX exports without PyTorch.
    _patch(monkeypatch, torch = False, device = hw.DeviceType.MLX, apple = True)
    assert hw.export_capability()["export_supported"] is True


def test_no_torch_non_apple_reports_pytorch_missing(monkeypatch):
    _patch(monkeypatch, torch = False, device = hw.DeviceType.CPU, apple = False)
    # Both branches below sit AFTER the gpu-present-but-unusable one, so they are only
    # reachable on a host with no accelerator at all. The verdict is module state that
    # detection writes and that current_chat_only_verdict() re-derives from a 60 second
    # inventory cache, so without pinning it here this reads whatever an earlier test in
    # the same worker left behind: a run that has already faked an NVIDIA host answers
    # "torch_cpu_build" and never reaches the branch being asserted. The rest of this file
    # already pins the verdict wherever it matters.
    monkeypatch.setattr(hw, "current_chat_only_verdict", lambda: ("no_gpu", None))
    cap = hw.export_capability()
    assert cap["export_supported"] is False
    assert cap["export_unsupported_reason"] == "pytorch_not_installed"
    assert "PyTorch is not installed" in cap["export_unsupported_message"]


def test_apple_without_mlx_reports_mlx_unavailable(monkeypatch):
    # Apple + CPU means the MLX stack is missing; reason is mlx_unavailable regardless of torch.
    for has_torch in (False, True):
        _patch(monkeypatch, torch = has_torch, device = hw.DeviceType.CPU, apple = True)
        cap = hw.export_capability()
        assert cap["export_supported"] is False
        assert cap["export_unsupported_reason"] == "mlx_unavailable"
        assert "MLX" in cap["export_unsupported_message"]


def test_apple_no_torch_install_reports_no_torch(monkeypatch):
    # GGUF-only by request: the message must not send the user to `unsloth studio update`.
    _patch(monkeypatch, torch = False, device = hw.DeviceType.CPU, apple = True)
    monkeypatch.setattr(hw, "current_chat_only_verdict", lambda: ("no_torch", None))
    cap = hw.export_capability()
    assert cap["export_supported"] is False
    assert cap["export_unsupported_reason"] == "no_torch"
    assert "--no-torch" in cap["export_unsupported_message"]
    assert "unsloth studio update" not in cap["export_unsupported_message"]


# -- import safety without PyTorch --------------------------------------------------------------


def test_export_backend_imports_without_torch(monkeypatch):
    """core/export/export.py must import on a --no-torch host (unsloth/torch blocked) and return a
    clean 'PyTorch is not installed' message from an export attempt, not crash at import."""
    import importlib
    import sys

    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        top = name.split(".")[0]
        if top in {"torch", "unsloth"}:
            raise ImportError(f"simulated: {top} not installed")
        return real_import(name, *args, **kwargs)

    # Drop any preloaded copies so the guarded import paths re-run under the block.
    for m in [k for k in sys.modules if k.split(".")[0] in {"torch", "unsloth"}]:
        monkeypatch.delitem(sys.modules, m, raising = False)
    monkeypatch.delitem(sys.modules, "core.export.export", raising = False)
    monkeypatch.setattr(builtins, "__import__", blocking_import)

    mod = importlib.import_module("core.export.export")
    assert mod._IS_MLX is False
    assert mod.torch is None
    assert mod._export_runtime_available() is False

    be = mod.ExportBackend.__new__(mod.ExportBackend)
    be.current_model = None
    be.current_tokenizer = None
    be.is_peft = False
    be._audio_type = None
    ok, message, out = be.export_merged_model("/tmp/does-not-matter")
    assert ok is False
    assert "PyTorch is not installed" in message


# -- endpoint / backend wiring (ast) ------------------------------------------------------------


def test_main_endpoints_expose_export_capability():
    m = _src("main.py")
    # Both system endpoints spread export_capability() into their response.
    assert m.count("**export_capability()") >= 2
    assert '"/api/system/hardware"' in m and '"/api/system"' in m


def test_routes_guard_mutating_endpoints():
    r = _src("routes/export.py")
    assert "async def _ensure_export_supported()" in r
    # load + all four export endpoints call the guard.
    assert r.count("await _ensure_export_supported()") >= 5


def test_export_methods_check_runtime():
    e = _src("core/export/export.py")
    assert "def _export_runtime_available()" in e
    # Each export method returns the clear message when the runtime is missing.
    assert e.count("_export_runtime_available()") >= 5
    assert "_PYTORCH_MISSING_MESSAGE" in e


def test_export_capability_reads_no_torch_helper():
    cap = _func_src("utils/hardware/hardware.py", "export_capability")
    assert "_has_torch()" in cap and "DeviceType.MLX" in cap and "is_apple_silicon()" in cap


def test_a_failed_detection_is_reported_as_such(monkeypatch):
    """Do not send the user to fix something that is not wrong. ensure_hardware_detected()
    records CPU + "detection_failed" when the probe raises, so the host looks CPU-only to
    export_capability; reporting no_accelerator (or pytorch_not_installed) there points the
    remediation at hardware or an install that may both be fine."""
    from utils.hardware import hardware as hw

    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CPU)
    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", "detection_failed")

    cap = hw.export_capability()

    assert cap["export_supported"] is False
    assert cap["export_unsupported_reason"] == "detection_failed"
    assert "detection failed" in cap["export_unsupported_message"].lower()


def test_a_cpu_host_without_a_usable_torch_keeps_the_reason(monkeypatch):
    """The CPU branch needs a working torch: without one the old refusals still apply."""
    from utils.hardware import hardware as hw

    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CPU)
    monkeypatch.setattr(hw, "CHAT_ONLY_REASON", "no_gpu")
    monkeypatch.setattr(hw, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(hw, "_has_torch", lambda: False)
    monkeypatch.setattr(hw, "current_chat_only_verdict", lambda: ("no_gpu", None))

    cap = hw.export_capability()

    assert cap["export_supported"] is False
    assert cap["export_unsupported_reason"] == "pytorch_not_installed"
    assert cap.get("export_cpu_only") is not True
