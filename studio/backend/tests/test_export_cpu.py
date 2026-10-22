# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the CPU export backend.

Import-light on purpose: the module must not import torch at module scope, because the hosts it
exists for are exactly the ones where unsloth/torch cannot provide an accelerator, and the pure
format helpers are the part that decides what those hosts are offered.
"""

import json

from core.export import cpu_export


def test_cpu_host_probe_uses_hardware_device(monkeypatch):
    import utils.hardware as hw

    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CPU)
    assert cpu_export.cpu_export_host() is True
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    assert cpu_export.cpu_export_host() is False


def test_cpu_host_probe_fails_closed(monkeypatch):
    import utils.hardware as hw

    def boom():
        raise RuntimeError("detection exploded")

    monkeypatch.setattr(hw, "get_device", boom)
    assert cpu_export.cpu_export_host() is False


def test_sixteen_bit_merged_is_allowed_on_cpu():
    assert cpu_export.cpu_merged_format_error("16-bit (FP16)", None) is None
    assert cpu_export.cpu_merged_format_error("16-bit (BF16)", None) is None


def test_gpu_only_merged_formats_name_the_cpu_alternatives():
    fp4 = cpu_export.cpu_merged_format_error("4-bit (FP4)", None)
    assert fp4 and "GGUF" in fp4
    compressed = cpu_export.cpu_merged_format_error("16-bit (FP16)", "w4a16")
    assert compressed and "accelerator" in compressed


def test_gguf_quant_allow_list_matches_what_llama_quantize_can_emit():
    assert cpu_export.cpu_gguf_quant_error("Q4_K_M") is None
    assert cpu_export.cpu_gguf_quant_error(["q8_0", "f16"]) is None
    assert cpu_export.cpu_gguf_quant_error("not_quantized") is None
    error = cpu_export.cpu_gguf_quant_error("iq2_xxs")
    assert error and "iq2_xxs" in error


def test_bf16_request_keeps_bf16(monkeypatch):
    assert cpu_export.cpu_merged_dtype_label("16-bit (BF16)") == "bfloat16"


def test_adapter_export_copies_the_checkpoint_files(tmp_path, monkeypatch):
    checkpoint = tmp_path / "run" / "checkpoint-10"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_text("weights", encoding="utf-8")
    (checkpoint / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    # Trainer state is not part of the adapter and must not be copied into the export.
    (checkpoint / "optimizer.pt").write_text("big", encoding="utf-8")
    (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")

    target = tmp_path / "export"
    monkeypatch.setattr(
        "utils.paths.resolve_export_write_dir", lambda value: target
    )
    backend = cpu_export.CpuExportBackend()
    backend.current_checkpoint = str(checkpoint)

    success, message, output = backend.export_lora_adapter(str(target))

    assert success is True
    assert output == str(target.resolve())
    assert (target / "adapter_model.safetensors").read_text(encoding="utf-8") == "weights"
    assert (target / "adapter_config.json").is_file()
    assert (target / "tokenizer_config.json").is_file()
    assert not (target / "optimizer.pt").exists()
    assert not (target / "trainer_state.json").exists()
    assert (target / "export_metadata.json").is_file()
    assert (target / "README.md").is_file()
    assert "adapter" in message.lower()


def test_adapter_export_refuses_a_merged_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint-10"
    checkpoint.mkdir()
    target = tmp_path / "export"
    monkeypatch.setattr(
        "utils.paths.resolve_export_write_dir", lambda value: target
    )
    backend = cpu_export.CpuExportBackend()
    backend.current_checkpoint = str(checkpoint)

    success, message, output = backend.export_lora_adapter(str(target))

    assert success is False
    assert "not a PEFT model" in message
    assert output is None


def test_gguf_lora_adapter_is_refused_with_the_working_alternative(tmp_path):
    checkpoint = tmp_path / "checkpoint-10"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    backend = cpu_export.CpuExportBackend()
    backend.current_checkpoint = str(checkpoint)

    success, message, _ = backend.export_lora_adapter(
        str(tmp_path / "export"), gguf=True
    )

    assert success is False
    assert "merged model as GGUF" in message


def test_merged_export_without_a_model_is_refused():
    backend = cpu_export.CpuExportBackend()
    success, message, _ = backend.export_merged_model("/tmp/nowhere")
    assert success is False
    assert "No model loaded" in message


class _FakeModel:
    def __init__(self, calls):
        self.calls = calls

    def merge_and_unload(self):
        self.calls.append("merge")
        return self

    def to(self, dtype):
        self.calls.append(("to", str(dtype)))
        return self

    def save_pretrained(self, path, safe_serialization = True):
        self.calls.append(("model", path, safe_serialization))


class _FakeTokenizer:
    def __init__(self, calls):
        self.calls = calls

    def save_pretrained(self, path):
        self.calls.append(("tokenizer", path))


def test_merged_export_folds_the_adapter_and_writes_the_folder(tmp_path, monkeypatch):
    calls = []
    checkpoint = tmp_path / "checkpoint-10"
    checkpoint.mkdir()
    output = tmp_path / "merged"
    monkeypatch.setattr(
        "utils.paths.resolve_export_write_dir", lambda value: output
    )

    backend = cpu_export.CpuExportBackend()
    backend.current_model = _FakeModel(calls)
    backend.current_tokenizer = _FakeTokenizer(calls)
    backend.is_peft = True
    backend.base_model = "org/base"
    backend.current_checkpoint = str(checkpoint)

    success, message, path = backend.export_merged_model(str(output))

    assert success is True, message
    assert path == str(output.resolve())
    assert calls[0] == "merge"
    assert ("model", str(output), True) in calls
    assert ("tokenizer", str(output)) in calls
    assert (output / "export_metadata.json").is_file()
    metadata = json.loads((output / "export_metadata.json").read_text(encoding = "utf-8"))
    assert metadata["base_model"] == "org/base"
    assert metadata["export_device"] == "cpu"
    assert (output / "README.md").is_file()


def test_merged_export_refuses_quantized_formats(tmp_path, monkeypatch):
    calls = []
    output = tmp_path / "merged"
    monkeypatch.setattr(
        "utils.paths.resolve_export_write_dir", lambda value: output
    )
    backend = cpu_export.CpuExportBackend()
    backend.current_model = _FakeModel(calls)
    backend.current_tokenizer = _FakeTokenizer(calls)

    success, message, _ = backend.export_merged_model(
        str(output), compressed_method = "w8a8"
    )

    assert success is False
    assert "w8a8" in message
    assert calls == []


def test_merged_export_rejects_gpu_only_formats_before_touching_a_model():
    backend = cpu_export.CpuExportBackend()
    success, message, _ = backend.export_merged_model("/tmp/nowhere", format_type="4-bit (FP4)")
    assert success is False
    assert "accelerator" in message
