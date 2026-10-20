# Testing X8 training on a free GPU

Colab's free GPU quota is not a published number and Google does not top it up
when it runs out, so when a Colab run dies with the quota banner the options are
paid Colab units or somewhere else. This folder is somewhere else.

## Kaggle (no credit card, 30 GPU hours a week)

Kaggle Notebooks give roughly **30 GPU hours per week**, T4 x2 or P100, 12-hour
sessions, with internet enabled. Phone verification is required before a GPU can
be used. Quotas are a weekly allowance rather than a guarantee, and the reset
time is not published.

[`run.py`](run.py) is the whole test in one script:

1. clones `x8-main` from `GoldodemonINC/unsloth_8X` (shallow) and prints the SHA,
2. installs it with the repo's own installer, `studio/setup.sh --local`, minus the
   frontend build, llama.cpp prebuilt, whisper and audio.cpp, so it is the same
   Python stack Studio uses,
3. runs the backend tests for the features that are new here (MoE expert scope,
   eval dataset, layer-LR groups, zone map, training manifest),
4. runs a real QLoRA fine-tune: Llama-3.2-1B in 4-bit, 20 steps on alpaca rows,
   a separate eval split with `eval_strategy="steps"`, then saves the adapter and
   reloads it with PEFT,
5. walks a streaming dataset through `.map()`, which is the `_iterable_batch_size`
   NameError path that killed the last Colab run.

Expect ~15-25 minutes for the install and ~5 minutes for the training. Everything
is also written to `/kaggle/working/result.txt`, so the verdict survives even if
the log view is awkward. Exit code is 0 only when both the tests and the training
pass.

### Option A - paste it into a notebook (no CLI, no API token)

1. Create a Kaggle account and verify your phone (required for GPU).
2. **Create -> Notebook**.
3. **File -> Import Notebook** and upload `run.py`, or paste its contents into one
   cell and run it.
4. In the right-hand panel: **Accelerator -> GPU T4 x2**, **Internet -> On**.
5. Run all. The last lines are `backend tests: PASS`, `gpu training: PASS`.

### Option B - push it as a kernel

With a Kaggle API token at `~/.kaggle/kaggle.json`:

```bash
pip install kaggle
cd review/kaggle
kaggle kernels push                 # uploads run.py using kernel-metadata.json
kaggle kernels status goldodemon/unsloth-x8-training-smoke
kaggle kernels output goldodemon/unsloth-x8-training-smoke -p out
```

The metadata file already sets `enable_gpu` and `enable_internet`. Change `id`
to your own Kaggle username before pushing.

## If Kaggle is not an option

- **Lightning AI Studio** - free credits to start, phone verification, T4/L4/L40S.
  The exact monthly GPU-hour allowance is not published; its own docs say signup
  credits plus more when a card is added.
- **Paid Colab units** - the documented fix for an exhausted free quota; roughly
  $10 buys enough units for a few T4 hours.
- **Modal** - $30/month of free compute, but a payment method is required on file.
- **Rented GPU** - RunPod / Vast.ai / Lambda prices a T4 near $0.2-0.4/hour, so a
  full verification run costs well under a dollar.
- **SageMaker Studio Lab** - free T4 for 4 hours a day, but AWS has closed it to
  new customers.
