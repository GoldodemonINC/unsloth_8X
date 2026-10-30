# X8 — Xperimental features

This branch carries the experimental features and nothing else. No desktop shell,
no HTTP server, no CLI, no training or inference stack. Each feature is a folder
you can read on its own, with the test that came with it.

Everything here comes from `x8-main`; the app still carries the same code and is
where these features actually run.

## Layout

```
features/
  artifacts/     created files, mirrors and previews for a plan's outputs
  autoscale/     "scale before training": sizes a growth from the dataset
  benchmarks/    eval-plan execution and stored benchmark records
  cmt/           curriculum-mixed training from token budgets
  housekeeping/  what the outputs, exports and caches cost, and what may go
  pipelines/     multi-stage plans with preflight and per-stage manifests
shared/
  merge, scaling, tensor_io, ...   the machinery the features call
  paths.py                          where outputs and exports live
```

`features/<name>/<name>.py` is the feature; `test_<name>.py` beside it is its test.
`shared/` holds the nine modules the features import — including the weight-blending
engine and the checkpoint scaler — so the folders above stay readable without
dragging in the rest of the app.

## What is deliberately not here

- **The app**: `studio/frontend`, `studio/src-tauri`, `studio/backend/routes`,
  the settings gate, and the server that hosts these features.
- **The rest of the backend**: inference, training, export, RAG, research.
- **`tests/test_labs.py`**: it only exercised the Labs settings gate through the
  HTTP route, which is app code. Its behaviour is unchanged on `x8-main`.

Two references to app modules survive on purpose, both already written to fail
softly rather than raise:

- `features/housekeeping/housekeeping.py` reads the running-job registry from
  `routes.advanced` inside a `try`. The comment there says an import failure means
  there is nothing to protect, so cleanup simply stops skipping live jobs.
- `shared/run_manifest.py` stamps `studio_version()` from `utils.studio_version`
  inside a `try`, falling back to `"unknown"`.

## The one behaviour that changed

`shared/paths.py` replaces Studio's `utils.paths.storage_roots` for the four call
sites that needed a directory. Studio's version resolves through its account
workspace, which enforces per-account isolation and refuses paths that escape it —
machinery that belongs to the app. The replacement keeps the part that picks the
directory: `UNSLOTH_STUDIO_HOME`, then `STUDIO_HOME`, then `UNSLOTH_HOME/studio`,
then `~/.unsloth/studio`, with `outputs` and `exports` beneath it, and
`HF_HOME`/`HF_HUB_CACHE` for the caches. Run beside Studio, it reads the same
folders Studio lists; run alone, it does not consult Studio's stored settings.

## Tests

```bash
python -m pytest features shared
```

`conftest.py` puts the repository root on `sys.path` so `features.*` and `shared.*`
import regardless of where pytest is invoked from. The modules import each other by
package name, so nothing works from inside a single feature folder.

Current state: **46 passed, 1 failed**. The failure is
`features/artifacts/test_artifacts.py::test_growth_below_the_floor_is_blocked_with_the_number`,
which expects a 1.5x growth to be refused at a 2x floor; `scaling.GROWTH_FLOOR` is
`1.0` today. It fails the same way on `x8-main` and was not introduced here.
