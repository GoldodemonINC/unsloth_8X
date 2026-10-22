# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU-only export path.

Unsloth's loaders need an accelerator, so every export used to be refused on a CPU-only host.
Merging a LoRA into its base, saving the adapter and converting an already-merged model to GGUF
are all plain weight operations, so they run here through transformers / PEFT instead. Nothing in
this module imports unsloth: on a CPU install that import raises, which is exactly the runtime the
path exists for.

Supported on CPU: 16-bit merged, PEFT adapter, and GGUF (via llama.cpp, GPU disabled).
Not supported on CPU: 4-bit (bnb) merges and compressed-tensors / torchao quantized merges, which
need CUDA or NVIDIA-only kernels. Those are refused with a message naming the working formats.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from loggers import get_logger
from utils.paths import ensure_dir

logger = get_logger(__name__)

# Formats the merged CPU path can write. Everything else needs an accelerator.
CPU_MERGED_FORMATS = ("16-bit (FP16)", "16-bit (BF16)")
# Quantizations llama-quantize accepts, mirrored from unsloth_zoo's allow-list.
CPU_GGUF_QUANTS = (
    "f32",
    "f16",
    "bf16",
    "q8_0",
    "q6_k",
    "q5_k_m",
    "q5_k_s",
    "q4_k_m",
    "q4_k_s",
    "q4_0",
    "q4_1",
    "q3_k_m",
    "q2_k",
)
_ADAPTER_CONFIG = "adapter_config.json"
_ADAPTER_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")
_TOKENIZER_FILES = ("tokenizer_config.json", "tokenizer.json")
# What an adapter export carries. Optimizer/scheduler/trainer state in a checkpoint is not part of
# the adapter and can be many times its size, so the copy is an allow-list, not a directory clone.
_ADAPTER_EXPORT_NAMES = frozenset(
    (
        _ADAPTER_CONFIG,
        *_ADAPTER_WEIGHTS,
        "README.md",
        "config.json",
        "generation_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
        "chat_template.json",
        "merges.txt",
        "vocab.json",
        "vocab.txt",
        "tokenizer.model",
    )
)
_ADAPTER_EXPORT_PREFIXES = ("tokenizer", "sentencepiece", "modules_to_save")


def cpu_export_host() -> bool:
    """True when this host has no usable accelerator, so export must run on the CPU.

    MLX is excluded by the caller: Apple Silicon exports through its own backend.
    """
    try:
        from utils.hardware import DeviceType, get_device

        return get_device() == DeviceType.CPU
    except Exception as exc:  # detection unavailable reads as "not CPU", keeping the old refusal
        logger.debug("CPU export probe failed: %s", exc)
        return False


def cpu_merged_dtype_label(requested: Optional[str] = None) -> str:
    """The torch dtype a merged CPU export is stored in, as a torch attribute name.

    A CPU merges in fp32 (most x86 CPUs emulate bfloat16 in software), and the finished weights are
    cast to what the format says: fp16, or bf16 when that was chosen. Casting is cheap; computing in
    fp16/bf16 on a CPU is not.
    """
    if requested and "BF16" in requested:
        return "bfloat16"
    if requested and "FP16" in requested:
        return "float16"
    return "float32"


def cpu_merged_format_error(format_type: str, compressed_method: Optional[str]) -> Optional[str]:
    """Why this merged format cannot be written on CPU, or None when it can."""
    if compressed_method:
        return (
            f"'{compressed_method}' export needs an accelerator. On CPU, export the 16-bit merged "
            "model or GGUF instead."
        )
    if format_type not in CPU_MERGED_FORMATS:
        return (
            f"'{format_type}' export needs an accelerator (bitsandbytes 4-bit or an NVIDIA-only "
            "quantizer). On CPU, choose 16-bit merged or GGUF."
        )
    return None


def cpu_gguf_quant_error(quantization_method) -> Optional[str]:
    """Why these GGUF quantizations cannot be produced on CPU, or None when they can."""
    methods = (
        list(quantization_method)
        if isinstance(quantization_method, (list, tuple))
        else [quantization_method]
    )
    for method in methods:
        normalized = str(method or "q8_0").lower()
        if normalized in ("not_quantized", "fast_quantized", "quantized"):
            continue
        if normalized not in CPU_GGUF_QUANTS:
            return (
                f"GGUF quantization '{method}' is not supported on CPU. Choose one of: "
                f"{', '.join(CPU_GGUF_QUANTS)}."
            )
    return None


def _cast_for_storage(model, dtype_name: str):
    """Cast finished weights to the export's storage dtype; fp32 is stored as merged."""
    if dtype_name == "float32":
        return model
    try:
        import torch
    except ImportError:  # this module imports without torch; a real export never runs without it
        return model
    return model.to(getattr(torch, dtype_name))


def _adapter_dir(checkpoint: Path) -> bool:
    return (checkpoint / _ADAPTER_CONFIG).is_file()


def _is_adapter_export_file(relative: Path) -> bool:
    """Whether a file inside a checkpoint belongs to the adapter artifact.

    Trainer state (optimizer/scheduler/rng) and logs live beside the adapter weights and are not
    part of what inference needs, so the export copies only weights, config and tokenizer files.
    """
    name = relative.name
    if name in _ADAPTER_EXPORT_NAMES:
        return True
    return name.startswith(_ADAPTER_EXPORT_PREFIXES)


def _checkpoint_has_tokenizer(checkpoint: Path) -> bool:
    return any((checkpoint / name).is_file() for name in _TOKENIZER_FILES)


def _write_model_card(directory: Path, base_model: Optional[str], source: Optional[str], formats: str):
    """A minimal card, so the exported folder is self-describing without the Unsloth formatter."""
    lines = [
        "# Exported model",
        "",
        f"Exported on a CPU-only host with Unsloth Studio ({formats}).",
        "",
    ]
    if base_model:
        lines.append(f"- Base model: `{base_model}`")
    if source:
        lines.append(f"- Source checkpoint: `{source}`")
    lines.append("")
    try:
        (directory / "README.md").write_text("\n".join(lines), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not write export README: %s", exc)


def _write_export_metadata(directory: Path, base_model: Optional[str], source: Optional[str]):
    metadata = {
        "base_model": base_model,
        "source_checkpoint": str(Path(source).resolve()) if source and Path(source).exists() else None,
        "export_device": "cpu",
    }
    try:
        with open(directory / "export_metadata.json", "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
    except OSError as exc:
        logger.warning("Could not write export metadata: %s", exc)


def _push_folder(directory: Path, repo_id: str, hf_token, private: bool) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    repo_url = api.create_repo(repo_id, private=private, exist_ok=True)
    resolved = getattr(repo_url, "repo_id", repo_id)
    api.upload_folder(folder_path=str(directory), repo_id=resolved, repo_type="model")
    return resolved


def _convert_merged_to_gguf(
    model_dir: Path,
    out_dir: Path,
    quantization_methods: List[str],
    model_name: str,
    model_dtype: str = "f16",
    imatrix: Optional[str] = None,
) -> List[str]:
    """Convert an already-merged HF folder to GGUF with llama.cpp on the CPU.

    Mirrors unsloth.save.save_to_gguf's two-pass route: convert once to f16, then quantize each
    requested method, so every quant comes from identical weights. llama.cpp is installed without
    GPU support, which is the configuration this host can actually build.
    """
    from unsloth_zoo.llama_cpp import (
        _download_convert_hf_to_gguf,
        check_llama_cpp,
        convert_to_gguf,
        install_llama_cpp,
        quantize_gguf,
        use_local_gguf,
    )

    try:
        quantizer_location, _converter = check_llama_cpp()
    except Exception:
        logger.info("llama.cpp not found; installing a CPU build for GGUF conversion")
        quantizer_location, _converter = install_llama_cpp(gpu_support=False, print_output=False)

    ensure_dir(out_dir)
    produced: List[str] = []
    with use_local_gguf():
        converter_path, supported_text_archs, supported_vision_archs = (
            _download_convert_hf_to_gguf()
        )
        initial_files, _is_vlm = convert_to_gguf(
            model_name=model_name,
            input_folder=str(model_dir),
            model_dtype=model_dtype,
            quantization_type=model_dtype,
            converter_location=converter_path,
            supported_text_archs=supported_text_archs,
            supported_vision_archs=supported_vision_archs,
            max_shard_size="50GB",
            print_output=False,
        )

    moved: List[str] = []
    for path in initial_files:
        if not os.path.exists(path):
            raise RuntimeError(
                f"GGUF conversion produced no output at {path}. Check free disk space and retry."
            )
        target = Path(out_dir) / Path(path).name
        shutil.move(path, target)
        moved.append(str(target))
    produced.extend(moved)

    base_gguf = next((p for p in moved if "-mmproj" not in Path(p).name.lower()), None)
    for method in quantization_methods:
        normalized = str(method or "q8_0").lower()
        if normalized in ("not_quantized", "fast_quantized", "quantized"):
            normalized = {"not_quantized": "f16", "fast_quantized": "q8_0", "quantized": "q4_k_m"}[
                normalized
            ]
        if normalized == model_dtype:
            continue  # the initial conversion already is this format
        if base_gguf is None:
            raise RuntimeError("GGUF conversion produced no model file to quantize.")
        output = Path(out_dir) / f"{model_name}.{normalized.upper()}.gguf"
        kwargs = dict(
            input_gguf=base_gguf,
            output_gguf=str(output),
            quant_type=normalized,
            quantizer_location=quantizer_location,
            print_output=False,
        )
        if imatrix:
            kwargs["imatrix"] = imatrix
        quantize_gguf(**kwargs)
        produced.append(str(output))
    return produced


class CpuExportBackend:
    """Export a checkpoint on a CPU-only host, without Unsloth.

    The public surface matches the calls ExportBackend makes into this class, so the orchestrator
    and worker need no protocol change.
    """

    def __init__(self):
        self.current_model = None
        self.current_tokenizer = None
        self.current_checkpoint: Optional[str] = None
        self.base_model: Optional[str] = None
        self.is_peft = False
        self.is_vision = False
        self.decision = None
        self.diffusion_lm_profile = None
        self._model_dtype: Optional[str] = None

    # -- loading ---------------------------------------------------------

    def load_checkpoint(
        self,
        checkpoint_path: str,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        trust_remote_code: bool = False,
        hf_token=None,
        base_model: Optional[str] = None,
        **_ignored,
    ) -> Tuple[bool, str]:
        """Load a merged checkpoint or a LoRA adapter onto the CPU in 16-bit.

        ``load_in_4bit`` is accepted for signature parity and ignored: bitsandbytes 4-bit needs a
        GPU, and a CPU merge should start from full-precision weights anyway.
        """
        del max_seq_length, load_in_4bit
        from hub.utils.hf_tokens import normalize_token

        token = normalize_token(hf_token)
        # The anonymous sentinel means "read without a credential"; transformers spells that None.
        token = None if token is False else token
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.release()
            checkpoint = Path(checkpoint_path).expanduser()
            if not checkpoint.exists():
                return False, f"Checkpoint not found: {checkpoint}"

            is_adapter = _adapter_dir(checkpoint)
            if is_adapter:
                from utils.models import get_base_model_from_lora

                base = base_model or get_base_model_from_lora(str(checkpoint))
                if not base:
                    return False, "Could not determine base model for adapter"
            else:
                base = None
            model_id = base or str(checkpoint)

            dtype_name = cpu_merged_dtype_label()
            dtype = getattr(torch, dtype_name)
            logger.info(
                "Loading %s on CPU as %s (no accelerator on this host)",
                "adapter" if is_adapter else "checkpoint",
                dtype_name,
            )

            tokenizer_source = (
                str(checkpoint) if _checkpoint_has_tokenizer(checkpoint) else model_id
            )
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source,
                token=token,
                trust_remote_code=trust_remote_code,
            )
            load_kwargs = dict(
                token=token,
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=True,
            )
            try:
                # transformers 5 renamed torch_dtype to dtype; keep the older spelling working.
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, dtype=dtype, **load_kwargs
                )
            except TypeError as exc:
                if "dtype" not in str(exc):
                    raise
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, torch_dtype=dtype, **load_kwargs
                )
            if is_adapter:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, str(checkpoint))

            model.eval()
            self.current_model = model
            self.current_tokenizer = tokenizer
            self.current_checkpoint = str(checkpoint)
            self.base_model = base
            self.is_peft = is_adapter
            self._model_dtype = dtype_name
            kind = "adapter" if is_adapter else "merged model"
            return True, f"Loaded {kind} on CPU ({dtype_name}) successfully"
        except Exception as exc:
            logger.error("CPU checkpoint load failed: %s", exc)
            import traceback

            logger.error(traceback.format_exc())
            return False, f"Failed to load checkpoint on CPU: {exc}"

    def release(self):
        self.current_model = None
        self.current_tokenizer = None
        self.current_checkpoint = None
        self.base_model = None
        self.is_peft = False
        self._model_dtype = None
        try:
            import gc

            gc.collect()
        except Exception:
            pass

    # -- exporting -------------------------------------------------------

    def _merged_model(self):
        """The model with the adapter folded in, or the plain model for a full checkpoint.

        The merged model replaces the wrapper, so a second export neither merges twice nor holds a
        second copy of the weights.
        """
        if not self.is_peft:
            return self.current_model
        merged = self.current_model.merge_and_unload()
        self.current_model = merged
        self.is_peft = False
        return merged

    def _save_merged(
        self, directory: Path, format_type: str = "16-bit (FP16)"
    ) -> Path:
        model = self._merged_model()
        dtype_name = cpu_merged_dtype_label(format_type)
        # Cast only the finished weights: the merge math above runs in fp32, which a CPU does
        # natively. In place, so the loaded weights are not held twice.
        model = _cast_for_storage(model, dtype_name)
        ensure_dir(directory)
        model.save_pretrained(str(directory), safe_serialization=True)
        self.current_tokenizer.save_pretrained(str(directory))
        _write_export_metadata(directory, self.base_model, self.current_checkpoint)
        _write_model_card(
            directory,
            self.base_model,
            self.current_checkpoint,
            f"16-bit merged ({dtype_name})",
        )
        return directory

    def export_merged_model(
        self,
        save_directory: str,
        format_type: str = "16-bit (FP16)",
        push_to_hub: bool = False,
        repo_id: Optional[str] = None,
        hf_token=None,
        private: bool = False,
        compressed_method: Optional[str] = None,
        **_ignored,
    ) -> Tuple[bool, str, Optional[str]]:
        error = cpu_merged_format_error(format_type, compressed_method)
        if error:
            return False, error, None
        if self.current_model is None or self.current_tokenizer is None:
            return False, "No model loaded. Please select a checkpoint first.", None
        try:
            if save_directory:
                from utils.paths import resolve_export_write_dir

                target = Path(resolve_export_write_dir(save_directory))
                logger.info(
                    "Saving 16-bit merged model on CPU to %s as %s",
                    target,
                    cpu_merged_dtype_label(format_type),
                )
                self._save_merged(target, format_type)
                output_path: Optional[str] = str(target.resolve())
            else:
                output_path = None

            if push_to_hub:
                if not repo_id or not hf_token:
                    return False, "Repository ID and Hugging Face token required for Hub upload", None
                if output_path is None:
                    with tempfile.TemporaryDirectory(prefix="unsloth-cpu-export-") as staging:
                        self._save_merged(Path(staging), format_type)
                        repo_id = _push_folder(Path(staging), repo_id, hf_token, private)
                else:
                    repo_id = _push_folder(Path(output_path), repo_id, hf_token, private)
                logger.info("Merged CPU export pushed to %s", repo_id)
                return True, f"Model exported successfully and pushed to {repo_id}", output_path

            return True, "Model exported successfully (CPU)", output_path
        except Exception as exc:
            logger.error("CPU merged export failed: %s", exc)
            import traceback

            logger.error(traceback.format_exc())
            return False, f"Export failed: {exc}", None

    def export_base_model(
        self,
        save_directory: str,
        push_to_hub: bool = False,
        repo_id: Optional[str] = None,
        hf_token=None,
        private: bool = False,
        base_model_id: Optional[str] = None,
        **_ignored,
    ) -> Tuple[bool, str, Optional[str]]:
        if base_model_id:
            self.base_model = base_model_id
        return self.export_merged_model(
            save_directory,
            push_to_hub=push_to_hub,
            repo_id=repo_id,
            hf_token=hf_token,
            private=private,
        )

    def export_lora_adapter(
        self,
        save_directory: str,
        push_to_hub: bool = False,
        repo_id: Optional[str] = None,
        hf_token=None,
        private: bool = False,
        gguf: bool = False,
        gguf_outtype: str = "q8_0",
        adapter_format: Optional[str] = None,
        **_ignored,
    ) -> Tuple[bool, str, Optional[str]]:
        """Copy the trained PEFT adapter out. The files are the artifact, so no merge is needed."""
        del gguf_outtype
        if adapter_format and adapter_format != "peft":
            return (
                False,
                "Only the PEFT adapter format is available on CPU hosts; MLX adapters are built "
                "on Apple Silicon.",
                None,
            )
        if gguf:
            return (
                False,
                "GGUF LoRA adapter files are not produced on CPU. Export the merged model as GGUF "
                "instead, or save the PEFT adapter.",
                None,
            )
        source = Path(self.current_checkpoint) if self.current_checkpoint else None
        if source is None or not _adapter_dir(source):
            return False, "This is not a PEFT model. No adapter to export.", None
        try:
            from utils.paths import resolve_export_write_dir

            target = Path(resolve_export_write_dir(save_directory))
            ensure_dir(target)
            copied = 0
            for entry in source.rglob("*"):
                if not entry.is_file():
                    continue
                relative = entry.relative_to(source)
                if not _is_adapter_export_file(relative):
                    continue
                destination = target / relative
                ensure_dir(destination.parent)
                shutil.copy2(entry, destination)
                copied += 1
            _write_export_metadata(target, self.base_model, str(source))
            _write_model_card(target, self.base_model, str(source), "PEFT adapter")
            logger.info("Copied %d adapter files on CPU to %s", copied, target)

            output_path = str(target.resolve())
            if push_to_hub:
                if not repo_id or not hf_token:
                    return False, "Repository ID and Hugging Face token required for Hub upload", None
                repo_id = _push_folder(target, repo_id, hf_token, private)
                return True, f"Adapter exported and pushed to {repo_id}", output_path
            return True, "Adapter exported successfully (CPU)", output_path
        except Exception as exc:
            logger.error("CPU adapter export failed: %s", exc)
            return False, f"Adapter export failed: {exc}", None

    def export_gguf(
        self,
        save_directory: str,
        quantization_method="Q4_K_M",
        push_to_hub: bool = False,
        repo_id: Optional[str] = None,
        hf_token=None,
        imatrix_file=None,
        private: bool = False,
        npu_q4nx: bool = False,
        **_ignored,
    ) -> Tuple[bool, str, Optional[str]]:
        del npu_q4nx
        if self.current_model is None or self.current_tokenizer is None:
            return False, "No model loaded. Please select a checkpoint first.", None
        error = cpu_gguf_quant_error(quantization_method)
        if error:
            return False, error, None
        methods = (
            list(quantization_method)
            if isinstance(quantization_method, (list, tuple))
            else [quantization_method]
        )
        model_name = Path(self.current_checkpoint or "model").name or "model"
        imatrix = str(imatrix_file) if imatrix_file and imatrix_file is not True else None
        try:
            from utils.paths import resolve_export_write_dir

            target = Path(resolve_export_write_dir(save_directory)) if save_directory else None
            parent = target.parent if target else Path(tempfile.mkdtemp(prefix="unsloth-gguf-"))
            ensure_dir(parent)
            with tempfile.TemporaryDirectory(prefix="unsloth-cpu-merge-", dir=str(parent)) as staging:
                staging_dir = Path(staging)
                logger.info("Merging on CPU to feed the GGUF converter: %s", staging_dir)
                self._save_merged(staging_dir)
                out_dir = target if target else parent
                produced = _convert_merged_to_gguf(
                    staging_dir,
                    out_dir,
                    methods,
                    model_name,
                    model_dtype="f16",
                    imatrix=imatrix,
                )
            logger.info("CPU GGUF export produced: %s", produced)
            output_path = str(out_dir.resolve())
            if push_to_hub:
                if not repo_id or not hf_token:
                    return False, "Repository ID and Hugging Face token required for Hub upload", None
                repo_id = _push_folder(Path(output_path), repo_id, hf_token, private)
                return True, f"GGUF exported and pushed to {repo_id}", output_path
            return True, "GGUF exported successfully (CPU)", output_path
        except Exception as exc:
            logger.error("CPU GGUF export failed: %s", exc)
            import traceback

            logger.error(traceback.format_exc())
            return False, f"GGUF export failed: {exc}", None
