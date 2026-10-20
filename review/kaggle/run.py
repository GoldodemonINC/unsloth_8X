"""Verify the X8 training stack on a free Kaggle GPU.

Kaggle's free tier gives ~30 GPU hours a week (T4 x2 or P100, 12-hour sessions,
internet allowed), which is the practical replacement when a Colab quota runs out.
This kernel is the whole test: clone `x8-main`, install it the same way the Studio
installer does, run the backend tests for the features that are new here, then run
a real QLoRA fine-tune on a 1B model and prove the adapter loads back.

Everything is also appended to /kaggle/working/result.txt so the outcome survives
in the kernel artifact even when the log view is awkward.

Run it as a Kaggle script kernel (`kaggle kernels push`) or paste it into a
notebook. Either way: Accelerator = GPU T4 x2, Internet = On.
"""

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

REPO_URL = "https://github.com/GoldodemonINC/unsloth_8X.git"
BRANCH = "x8-main"
REPO = "/kaggle/working/unsloth_8X"
BACKEND = os.path.join(REPO, "studio/backend")
RESULT = "/kaggle/working/result.txt"
WORK = "/kaggle/working"

_transcript: list[str] = []
_started = time.time()


def say(message):
    print(message, flush=True)
    _transcript.append(message)


def flush_result():
    try:
        with open(RESULT, "w", encoding="utf-8") as handle:
            handle.write("\n".join(_transcript))
    except Exception:
        pass


def finish(code):
    """Exit for a script kernel, but stay alive when this file is pasted into a notebook."""
    flush_result()
    if "ipykernel" in sys.modules:
        say(f"exit code would be {code}")
        return
    sys.exit(code)


def sh(cmd, cwd=None, check=True, env=None, tail=6000):
    say(f"$ {cmd}")
    run_env = dict(os.environ)
    run_env.setdefault("PYTHONUNBUFFERED", "1")
    run_env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    run_env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if env:
        run_env.update(env)
    result = subprocess.run(
        cmd, shell=True, cwd=cwd, text=True, capture_output=True, env=run_env
    )
    output = (result.stdout or "") + (result.stderr or "")
    say(output[-tail:])
    flush_result()
    if result.returncode != 0 and check:
        raise SystemExit(f"FAILED: {cmd}")
    return result


def step(title):
    say(f"\n===== {title} =====")


# ── 1. Hardware ─────────────────────────────────────────────────────────────
step("environment")
sh("python -V")
sh("df -h /kaggle | tail -1", check=False)
gpu = sh(
    "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader",
    check=False,
)
if gpu.returncode != 0:
    say("No GPU visible. Kaggle -> Session options -> Accelerator -> GPU T4 x2, then rerun.")
    finish(2)
    raise SystemExit(2)
sh(
    "python -c \"import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))\""
)

# ── 2. Clone this fork ──────────────────────────────────────────────────────
step("clone")
sh(f"rm -rf {REPO}")
sh(f"git clone --quiet --depth 1 --branch {BRANCH} {REPO_URL} {REPO}")
sha = subprocess.run(
    ["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True
).stdout.strip()
say(f"checked out {BRANCH} @ {sha[:12]}")
say("compare with `git rev-parse HEAD` in the checkout you edited")

# ── 3. Install ──────────────────────────────────────────────────────────────
# setup.sh --local is the installer the Colab notebook and `unsloth studio update`
# run: it overlays this checkout editable and installs the pinned ML stack. The
# frontend build, llama.cpp prebuilt, whisper and audio.cpp are skipped - training
# does not need any of them, and each costs minutes.
step("install (studio setup.sh --local, no frontend / llama.cpp)")
install_env = {
    "SKIP_STUDIO_FRONTEND": "1",
    "SKIP_PREBUILT_INSTALL": "1",
    "UNSLOTH_SKIP_NODE_INSTALL": "1",
    "UNSLOTH_SKIP_WHISPER_INSTALL": "1",
    "UNSLOTH_SKIP_AUDIO_CPP_INSTALL": "1",
}
install = sh(
    f"chmod +x {REPO}/studio/setup.sh && bash {REPO}/studio/setup.sh --local",
    cwd=os.path.join(REPO, "studio"),
    check=False,
    env=install_env,
    tail=4000,
)

VENV_PYTHON = os.path.expanduser("~/.unsloth/studio/unsloth_studio/bin/python")


def resolve_python():
    """The interpreter the installer filled, or the kernel's own if it used that."""
    for candidate in (os.environ.get("UNSLOTH_STUDIO_PYTHON"), VENV_PYTHON, sys.executable):
        if not candidate or not os.path.isfile(candidate):
            continue
        probe = subprocess.run(
            [candidate, "-c", "import unsloth, os; print(os.path.realpath(unsloth.__file__))"],
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0:
            say(f"using {candidate} -> {probe.stdout.strip()}")
            return candidate, probe.stdout.strip()
    return sys.executable, ""


PYTHON, UNSLOTH_PATH = resolve_python()
# The installer already exits non-zero on a real failure; this is the only signal
# that matters downstream, so the fallback only runs when the import is missing.
if not UNSLOTH_PATH:
    step("install fallback: pip overlay")
    say(f"setup.sh exit {install.returncode}; importing unsloth from {PYTHON} failed")
    sh(f"{PYTHON} -m pip install -q -e {REPO} unsloth-zoo")
    sh(f"{PYTHON} -m pip install -q -U bitsandbytes accelerate")
    PYTHON, UNSLOTH_PATH = resolve_python()
    if not UNSLOTH_PATH:
        finish(3)
        raise SystemExit("could not install unsloth from the checkout")

say(f"unsloth resolves to {UNSLOTH_PATH}")
sh(f"{PYTHON} -m pip show unsloth | head -8", check=False)
if not os.path.realpath(UNSLOTH_PATH).startswith(os.path.realpath(REPO)):
    say("WARN: unsloth is not coming from the checkout - the overlay did not win")

# ── 4. Backend tests for the features that are new here ─────────────────────
step("backend tests")
sh(f"{PYTHON} -m pip install -q pytest structlog httpx fastapi pydantic pyyaml")


def run_tests(test_files):
    """pytest, installing any backend dep the run reports missing."""
    command = (
        f"{PYTHON} -m pytest "
        + " ".join(test_files)
        + " -q --no-header -p no:cacheprovider"
    )
    result = None
    for attempt in range(6):
        result = sh(command, cwd=BACKEND, check=False, tail=8000)
        output = (result.stdout or "") + (result.stderr or "")
        missing = set(re.findall(r"No module named '([A-Za-z0-9_.]+)'", output))
        packages = sorted(
            {name.split(".")[0].replace("_", "-") for name in missing} if missing else set()
        )
        if result.returncode == 0 or not packages:
            return result
        say(f"installing missing test deps: {packages} (attempt {attempt + 1})")
        sh(f"{PYTHON} -m pip install -q {' '.join(packages)}", check=False)
    return result


test_files = [
    "tests/test_moe_expert_scope.py",
    "tests/test_eval_dataset_selector.py",
    "tests/test_advanced_layer_lr_groups.py",
    "tests/test_advanced_zone_map.py",
    "tests/test_advanced_train_manifest.py",
]
present = [name for name in test_files if os.path.isfile(os.path.join(BACKEND, name))]
say(f"running {len(present)}/{len(test_files)}: {present}")
tests_run = run_tests(present) if present else None
tests_ok = tests_run is not None and tests_run.returncode == 0

# ── 5. Real training on the GPU ─────────────────────────────────────────────
# Not a mock: a 1B model, 4-bit, LoRA, real batches, a separate eval dataset, then
# a reload of the saved adapter. Also exercises the streaming dataset path whose
# `_iterable_batch_size` NameError is what killed the last Colab run.
step("GPU training smoke")
smoke = r'''
import math
import os
import sys
import time

import torch

from unsloth import FastLanguageModel

START = time.time()
print("torch", torch.__version__, "cuda", torch.cuda.is_available())

# The streaming repair: a partially built unsloth-zoo carries the call sites but
# not the helper, and every mapped streaming batch then dies with a NameError.
import unsloth  # noqa: F401  (installs the repair when the zoo needs it)
import unsloth_zoo.dataset_utils as dataset_utils

assert hasattr(dataset_utils, "_iterable_batch_size"), (
    "unsloth_zoo.dataset_utils has no _iterable_batch_size, and the import repair did not add it"
)
print("streaming helper present:", dataset_utils._iterable_batch_size(None))

from datasets import load_dataset

stream = load_dataset("unsloth/alpaca-cleaned", split="train", streaming=True)
mapped = stream.map(lambda row: {**row, "text": (row["instruction"] or "") + "\n" + (row["output"] or "")})
first = next(iter(mapped))
assert isinstance(first["text"], str) and first["text"].strip()
print("streaming worked, first text chars:", len(first["text"]))

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B",
    max_seq_length = 1024,
    load_in_4bit = True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)
trainable = [p for p in model.parameters() if p.requires_grad]
before = [p.detach().float().clone() for p in trainable]
print("trainable tensors:", len(trainable), "params:", sum(p.numel() for p in trainable))

rows = load_dataset("unsloth/alpaca-cleaned", split="train[:400]")

def to_text(batch):
    return {
        "text": [
            (instruction or "") + "\n" + (output or "")
            for instruction, output in zip(batch["instruction"], batch["output"])
        ]
    }

rows = rows.map(to_text, batched = True, remove_columns = rows.column_names)
rows = rows.filter(lambda row: bool(row["text"].strip()))
split = rows.train_test_split(test_size = 32, seed = 3407)
train_ds, eval_ds = split["train"], split["test"]
print("train rows:", len(train_ds), "eval rows:", len(eval_ds))

from trl import SFTConfig, SFTTrainer

bf16 = torch.cuda.is_bf16_supported()
trainer = SFTTrainer(
    model = model,
    processing_class = tokenizer,
    train_dataset = train_ds,
    eval_dataset = eval_ds,
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 2,
        warmup_steps = 2,
        max_steps = 20,
        learning_rate = 2e-4,
        logging_steps = 1,
        eval_strategy = "steps",
        eval_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none",
        save_strategy = "no",
        fp16 = not bf16,
        bf16 = bf16,
    ),
)
result = trainer.train()
losses = [row["loss"] for row in trainer.state.log_history if "loss" in row]
assert losses, "training produced no loss values"
assert all(math.isfinite(value) for value in losses), losses
print("loss first/last:", losses[0], losses[-1])
assert losses[-1] < losses[0] * 1.5, f"loss did not move sensibly: {losses[0]} -> {losses[-1]}"

evals = [row["eval_loss"] for row in trainer.state.log_history if "eval_loss" in row]
assert evals, "eval_strategy=steps ran but no eval_loss was logged"
assert all(math.isfinite(value) for value in evals), evals
print("eval losses:", evals)

changed = sum(
    1
    for old, param in zip(before, trainable)
    if not torch.equal(old, param.detach().float())
)
assert changed > 0, "no LoRA tensor changed during training"
print(f"changed tensors: {changed}/{len(before)}")

adapter = "/kaggle/working/lora_adapter"
model.save_pretrained(adapter)
tokenizer.save_pretrained(adapter)
weights = [name for name in os.listdir(adapter) if name.endswith((".safetensors", ".bin"))]
assert weights, f"adapter dir has no weights: {os.listdir(adapter)}"
print("adapter files:", sorted(os.listdir(adapter))[:6])

from peft import PeftModel

reloaded = PeftModel.from_pretrained(model, adapter)
assert reloaded is not None
print("adapter reloaded into the live model")

print("seconds:", round(time.time() - START, 1))
print("SMOKE_OK")
'''

smoke_path = os.path.join(WORK, "train_smoke.py")
with open(smoke_path, "w", encoding="utf-8") as handle:
    handle.write(smoke)
smoke_run = sh(
    f"{PYTHON} {smoke_path}",
    cwd=BACKEND,
    check=False,
    env={"PYTHONPATH": BACKEND},
    tail=12000,
)
smoke_ok = smoke_run.returncode == 0

# ── 6. Summary ──────────────────────────────────────────────────────────────
step("summary")
say(f"commit           : {sha[:12]} ({BRANCH})")
say(f"unsloth from     : {UNSLOTH_PATH}")
say(f"backend tests    : {'PASS' if tests_ok else 'FAIL'} ({len(present)} files)")
say(f"gpu training     : {'PASS' if smoke_ok else 'FAIL'}")
say(f"elapsed          : {round(time.time() - _started, 1)}s")
finish(0 if (tests_ok and smoke_ok) else 1)
