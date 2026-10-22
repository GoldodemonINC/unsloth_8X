# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

_METHODS = ("linear", "slerp", "ties", "dare")
_VALUES = (1.0, 2.0, 6.0)

def build_checkpoint(
    directory: Path,
    value: float,
    *,
    rows: int = 16,
    hidden: int = 8,
    dtype: str = "float32",
) -> Path:
    import numpy as np
    from safetensors.numpy import save_file

    directory.mkdir(parents = True, exist_ok = True)
    (directory / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "hidden_size": hidden,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "intermediate_size": hidden,
                "vocab_size": rows,
            }
        ),
        encoding = "utf-8",
    )
    weights = {
        "model.embed_tokens.weight": np.full((rows, hidden), float(value)),
        "model.norm.weight": np.full((hidden,), float(value)),
    }
    save_file(
        {key: array.astype(dtype) for key, array in weights.items()},
        str(directory / "model.safetensors"),
    )
    return directory

def _merged_mean(output: Path, key: str = "model.embed_tokens.weight") -> float:
    import numpy as np
    from safetensors import safe_open

    index = json.loads(
        (output / "model.safetensors.index.json").read_text(encoding = "utf-8")
    )
    shard = output / index["weight_map"][key]
    with safe_open(str(shard), framework = "np") as handle:
        return float(np.asarray(handle.get_tensor(key), dtype = np.float32).mean())

def run_merge_smoke(workdir: Path | str | None = None) -> dict[str, Any]:
    import numpy as np

    from core.advanced import merge

    root = Path(workdir) if workdir is not None else Path(
        tempfile.mkdtemp(prefix = "unsloth-debug-merge-")
    )
    root.mkdir(parents = True, exist_ok = True)
    parents = [
        build_checkpoint(root / f"parent-{index}", value)
        for index, value in enumerate(_VALUES)
    ]
    checks: list[dict[str, Any]] = []
    for method in _METHODS:
        output = root / f"merged-{method}"
        report = merge.merge_checkpoints_many(
            [str(parent) for parent in parents], str(output), method = method
        )
        reference = merge.blend_tensors(
            [np.full((16, 8), value, dtype = np.float32) for value in _VALUES],
            method = method,
            density = 0.5,
            seed = 3407,
        )
        expected = float(np.asarray(reference, dtype = np.float32).mean())
        mean = _merged_mean(output)
        streamed = int(report["metrics"]["streamed_tensors"])
        checks.append(
            {
                "method": method,
                "status": report["status"],
                "streamed_tensors": streamed,
                "shards": report["metrics"]["shards"],
                "mean": round(mean, 4),
                "expected": round(expected, 4),
                "ok": report["status"] == "completed"
                and abs(mean - expected) <= 0.02
                and (streamed >= 1 or method == "slerp"),
            }
        )
    failed = [check for check in checks if not check["ok"]]
    return {
        "ok": not failed,
        "workdir": str(root),
        "parents": [str(parent) for parent in parents],
        "checks": checks,
        "failed": [check["method"] for check in failed],
    }
