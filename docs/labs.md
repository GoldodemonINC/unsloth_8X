# Labs (Settings → Labs)

One installation-wide switch that opens the unfinished features, and one
registry that decides what "unfinished" means. Labs is off by default: every
route behind it answers 403 until the switch is on, so a feature is closed even
when its page is reached by URL.

## The switch

`core/labs/__init__.py` owns the state:

| Thing | Where |
| --- | --- |
| Stored value | `app_settings` key `labs_state`, `{"enabled": bool, "features": {id: bool}}` |
| Read | `labs.read_state()` — a failed read means **off**, never on |
| Write | `labs.set_state(...)`, reached through `PUT /api/labs` (owner only) |
| Gate | `labs.require(feature_id)` raises `LabsDisabled`, which routes turn into a 403 |

`GET /api/labs` is readable by any signed-in account because the UI needs the
switch to decide what to draw. The write is behind
`policy.require_owner`, like the other installation-wide settings.

Two rules keep the state honest:

* Unknown feature ids are refused, not stored. A stale value cannot enable a
  feature that no longer exists.
* `feature_enabled()` is `master AND feature`. Turning one experiment off
  closes only that experiment's routes; turning Labs off closes all of them.

## The registry

`FEATURES` in `core/labs/__init__.py` is the single list. The Settings tab
renders what the server sends, so adding an entry is the whole UI change; do
not copy the list into the frontend.

| Id | Kind | What it opens |
| --- | --- | --- |
| `artifacts` | studio | The Artifacts studio: produced checkpoints, provenance, legal next steps |
| `pipelines` | studio | The Pipelines planner: plan a merge → scale → train → quantize → benchmark chain |
| `benchmarks` | studio | The Benchmark studio: score an artifact, keep the report, compare later |
| `cmt` | studio | CMT: per-dataset splits, capacity-matched expansion, new-blocks-only training |
| `housekeeping` | studio | The Cleanup studio: output and HF-cache sizes, guarded delete |
| `auto-scale` | feature | The wizard card that grows a checkpoint by the size its own datasets justify |
| `manifest-viewer` | feature | The raw `run_manifest.json` beside an artifact |
| `scale-from-artifact` | feature | Hand one artifact's path to the Scaler |
| `training-continuation` | feature | Make an artifact the Training model in one click |

Each row in the Settings tab toggles one of these. A feature that is off says so
on its own page (`LabsOffCard`) rather than disappearing.

## Benchmarks keep evidence, not the checkpoint

A benchmark record lives under `outputs/labs-benchmarks/records/<id>.json` and
never edits the artifact it describes. The eval job still writes
`auto_eval_report.json` and `.md` under its own run directory; the record is the
small, listable summary the page reads after the job is gone.

The studio reuses the existing `eval_harness`/`eval_runner` pair. It does not
add a second benchmark implementation, so the suites and the `lm-eval` install
check stay in one place. A delta under 1.5 points is labelled within noise, the
same convention the post-training Regression Guard uses.

## CMT is one job and one experiment

CMT (`core/labs/cmt.py`) loads the base on CPU, splits every local dataset into
its own train and held-out rows, expands the checkpoint with
`force_small_growth=True`, freezes every inherited parameter, and trains only
the inserted blocks. Each held-out split becomes one held-out perplexity task in
the same `cpt_train` call, so the per-dataset delta comes from the training run
itself rather than a second eval pass.

The sizing rule is `clamp(((1.7 * megatokens + 12) / 10) * 1M, 1M, 700M)`, with
the row fallback at 1M per 10k rows. It is a heuristic, and the page says so.
Two limits are honest, not bugs:

* The engine inserts whole blocks. If the requested growth is smaller than one
  block, the plan reports the block's size as the real growth.
* A growth below the Scaler's 2x floor is deliberately forced here. The manifest
  records `floor_bypassed: true`; no other stage gets that flag.

CMT v1 reads local `.jsonl`, `.json`, `.parquet`, `.csv` and `.txt` files. It
does not download Hub datasets during planning.

### CMT is also a method in the trainer

While the `cmt` experiment is on, the wizard's method picker offers **CMT** next
to LoRA/QLoRA/Full/CPT. Picking it swaps the dataset, preset, parameter and
advanced cards for the CMT card, because CMT owns its datasets, per-dataset
splits, eval and steps. The trainer's Start button is replaced by a note: CMT
never posts `/api/training/start`, and `startFreshTrainingRun()` refuses the
method before a payload is built. Everything the run needs travels to
`/api/labs/cmt/run` instead.

Two consequences are deliberate:

* CMT is offered only on a CUDA-class host. It loads in fp16, expands the
  checkpoint and trains dense new blocks; CPU and Apple Silicon hosts cannot.
* Turning the experiment off resets a persisted `cmt` selection to `lora`, so a
  stale browser cannot sit on an engine whose routes now answer 403.

`toBackendTrainingType("cmt")` throws. That is the boundary: the trainer's
`training_type` literal has no CMT member, and the tests pin the refusal.

## Scale before training

`core/labs/autoscale.py` shares CMT's sizing rule and then stops: it expands the
checkpoint, saves it, and leaves training to the method the user picks. It is
the data-sized alternative to the Scaler's slider.

The wizard shows it as **Scale from the dataset** whenever `auto-scale` is on
and the method is not CMT (CMT does the same sizing internally). It accepts one
local dataset path per line, plans without loading a weight, then runs one job
that writes a `scaled/` checkpoint and its manifest. *Use as the training model*
points the wizard at that folder; the run itself stays a separate, normal start.

Keeping growth and training separate is on purpose. Re-scaling from inside every
Start would grow the model again on each retry and make an accidental rerun
permanent. The card makes the growth a step the user can see and keep.

## Why Pipelines does not run

The planner prices a chain from `config.json` arithmetic only: parameter
projections, fp16 bytes, the bytes each GGUF level writes, and the refusals the
engines already enforce (growth floor, cap, tokenizer mismatch, quantize
ordering). It never loads weights.

Executing the chain needs durability the job layer does not have yet — jobs live
in an in-memory dict, leases are taken on the request thread, and a restart
loses the run. Shipping a run button on top of that would lose an overnight
pipeline. The plan is the part that can be right today, and it is what tells a
user a chain is impossible before they spend the GPU hours.

## Adding an experiment

1. Add one entry to `FEATURES` with an `id`, a label, a description, a status
   (`alpha` or `beta`) and `studio`.
2. Call `labs.require("<id>")` at the top of every route the experiment adds.
   Hiding the page is not gating it.
3. Gate the page with `<LabsFeatureGate feature="<id>">`; a deep link then shows
   the switch and where to flip it.
4. Add the route to `CHAT_ONLY_ALLOWED` only if it works with no accelerator.
5. Write the test: the route is 403 with the experiment off and 200 with it on.

## What Labs must never do

* Rewrite or migrate an existing artifact in place. An experiment may write a
  new folder; older builds have to be able to read what they already had.
* Stand in for a safety review. Destructive behavior gets built properly, not
  hidden behind the switch.
* Delete outputs when Labs is turned off. The switch hides entry points; the
  files stay where they are.

## Tests

Backend (`studio/backend/tests/`):

```bash
pytest tests/test_labs.py tests/test_labs_artifacts.py \
  tests/test_labs_benchmarks.py tests/test_labs_cmt.py \
  tests/test_labs_autoscale.py tests/test_labs_housekeeping.py -q
```

Frontend (`studio/frontend/`):

```bash
node --experimental-strip-types --test tests/labs-feature-gate.test.ts
node --experimental-strip-types --test tests/training-method-cmt.test.ts
node --experimental-strip-types --test tests/sidebar-nav-backend-parity.test.ts
```

The sidebar parity test compares `SIDEBAR_NAV_ITEM_DEFAULTS` (backend) with
`DEFAULT_CUSTOMIZATION.sidebarNav` (frontend), including order: the `labs` row
was added to both lists and to `SIDEBAR_NAV_ITEM_IDS`, and the backend's
`SidebarNavItemId` literal, in the same position.
