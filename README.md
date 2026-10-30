# Unsloth X8

A fork of [Unsloth](https://github.com/unslothai/unsloth) that keeps the training stack and pushes on the Studio: the Scaler can grow **or** shrink a checkpoint, and Labs is a set of working engines rather than a preview.

The upstream project is the fastest way to fine-tune open models on one GPU. This fork keeps that and adds a layer for the work that comes *around* a run — sizing it, scaling the model first, chaining stages, scoring the result, and knowing what a produced folder is allowed to do next.

> **Branch:** `x8-main` is the branch this fork is developed on. `main` mirrors the imported upstream tree.

---

## What this fork adds

| Area | What it does |
| --- | --- |
| **Scaler, growth** | Any target above the base is allowed — 1.11x and 1.26x included, not just 2x. Ceiling stays base + 3B. Below 2x the plan says the gain is unmeasured instead of refusing. |
| **Scaler, shrinking** | Depth drops whole transformer blocks to land near a smaller target (a 28-block 1.7B loses ~50M per block, so five or six blocks is ~1.4B). Width narrows each MLP instead, ranking channels by their gate/up/down weights. Edges protected, at least half the depth and a quarter of the width kept. |
| **Fit check** | Sizes a run from the card before anything starts, so an out-of-memory job is caught at plan time. |
| **Labs engines** | CMT, Scale-before-training, Benchmarks, Artifacts, Pipelines, the shrink repair preset and Cleanup — each backed by tests, behind a single switch. |
| **Run records** | Every merge, scale, CMT and pipeline output writes `run_manifest.json` and a copy of the license notice beside the checkpoint. |
| **Licensing** | Additions original to this fork are under [NRC-3](LICENSE-NRC-3.md); inherited code keeps its upstream license. |

### The two Scaler directions

**Grow** inserts blocks, widens the feed-forward, or both. Depth growth is bit-exact at step 0 — the new block contributes exactly zero until training unlocks it — so the checkpoint computes the base function while it waits for continued pretraining. Width growth is within ~1e-5 of the base logits, not bit-exact, because float32 re-associates the widened sum.

**Shrink** removes blocks or narrows each block's MLP. Neither can be function-preserving: a removed block is a removed residual contribution, and a removed channel is a smaller projection. The run reports the drift it measured against the base and hands you a checkpoint that needs a recovery pass; the Labs **shrink repair** preset is that pass — it measures the gap, trains, measures again, and passes or fails the result against a tolerance you set. Shrinking is the right tool when you have a fine-tune whose behaviour you want at a smaller shape; if you only need training to fit in less VRAM, QLoRA or a smaller base checkpoint is less work.

Both directions plan from `config.json` alone, so the slider answers before any weights load.

---

## Run it

### Colab (free T4)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GoldodemonINC/unsloth_8X/blob/x8-main/studio/Unsloth_Studio_X8_Colab.ipynb)

The notebook clones `x8-main`, installs this checkout editable over any PyPI wheel, verifies the import came from the checkout, and serves the UI through Colab's port proxy.

### Local

```bash
git clone -b x8-main https://github.com/GoldodemonINC/unsloth_8X.git
cd unsloth_8X

bash studio/setup.sh --local     # installs this checkout editable, plus prebuilt engines
unsloth studio -p 8888           # http://127.0.0.1:8888
```

`setup.sh --local` is what makes the fork's own code win over a published `unsloth` wheel; run it from the checkout, not from a copied folder. Set `UNSLOTH_SKIP_WHISPER_INSTALL=1` and `UNSLOTH_SKIP_AUDIO_CPP_INSTALL=1` to skip the speech/audio engines when you only need training and merging — that is what the Colab notebook does.

### Development loop

Two terminals, backend first:

```bash
python studio/backend/run.py --port 8888
```

```bash
cd studio/frontend
npm install
npm run dev        # http://127.0.0.1:5173, proxies /api to 127.0.0.1:8888
```

The desktop shell (`studio/src-tauri/`) wraps the same UI; the Vite dev server and the desktop app talk to the same backend.

---

## What is in here

| Path | What it is |
| --- | --- |
| `unsloth/` | Unsloth Core: PEFT/LoRA kernels, model loaders, patchers, `models/scaling.py` (grow + shrink) |
| `studio/backend/` | FastAPI server: training worker, merges, Scaler, splitter, export, Labs routes, hardware probe |
| `studio/frontend/` | React + Vite + Tailwind UI, TypeScript, node's own test runner |
| `studio/src-tauri/` | Desktop shell hosting the Studio UI |
| `studio/backend/core/labs/` | The Labs engines: CMT, auto-scale, benchmarks, artifacts, pipelines |
| `studio/backend/core/advanced/` | Merge, MoE, splitter, Scaler, eval harness, run manifests |
| `install.sh`, `studio/setup.sh` | Installers: pin the Python stack and prebuilt inference engines |
| `docs/` | Repository docs (see below) |

### Studio surfaces

The sidebar is the map of the app; rows are gated on the measured hardware verdict.

- **Chat, Projects, Library, Model hub** — run and organize models.
- **Images, Video, Audio** — media workspaces.
- **Train** — the wizard: method picker, datasets, fit check, advanced/Scaler section, run preview.
- **Merge** — CPU streaming merge or GPU full-load merge, two to six parents.
- **Scaler** — grow or shrink a checkpoint before training it.
- **Recipes** — Data Recipes for building datasets.
- **Export** — merged, adapter, GGUF and quantized export.
- **Labs** — hidden until Settings → Labs is on. Artifacts, Pipelines, Benchmarks, CMT, Scale-before-training, Run manifest, Cleanup.
- **API** — the OpenAI-compatible server and request monitor.

Merging, scaling and export are engines under the run layer (jobs, resource arbitration, `run_manifest.json`), not separate applications.

---

## Checks

From `studio/frontend/`:

```bash
npm run typecheck    # tsc over src and tests
npm test             # node --test over tests/**/*.test.ts
npm run i18n:check   # locale overlay parity
npm run biome:check  # lint + format
```

Backend tests run from `studio/backend/`:

```bash
python -m pytest tests -q
```

The Scaler's own suites are `tests/test_advanced_scaling.py`,
`tests/test_advanced_scaling_routes.py` and `tests/test_engine_shrink.py`; the
last one builds a tiny Llama and performs the block surgery for real.

From the repository root: `pre-commit run --all-files` (Ruff and the repo's lint scripts).

---

## Docs

| Doc | Covers |
| --- | --- |
| [docs/labs.md](docs/labs.md) | The Labs switch, the experiment registry, Artifacts and Pipelines |
| [docs/simulated-training.md](docs/simulated-training.md) | The Debugging simulator that runs a job with no hardware behind it |
| [docs/theming.md](docs/theming.md) | Palettes, accent tokens, changing the UI color |
| [docs/README.md](docs/README.md) | Repository map and the local CRLF trap |

---

## License

This repository carries three sets of terms, and they do not overlap:

- **Unsloth Core** (`unsloth/`) is **Apache-2.0**, as inherited from upstream.
- **The Studio application** (`studio/`) is **AGPL-3.0** — see [studio/LICENSE.AGPL-3.0](studio/LICENSE.AGPL-3.0).
- **Original additions in this fork** — the Labs engines, the trainer surface and the run records — are under **[NRC-3](LICENSE-NRC-3.md)**. Files that carry the `NRC-3` header are the ones covered; the Scaler lives in upstream-licensed files and keeps those terms.

NRC-3 permits private, internal and commercial use, modification, and sharing with attribution. It reserves five things, and each needs written permission: relaunching the work as a rebranded product (a different name, logo or interface, a near-miss name, or an "official" claim), running it as a hosted service for other people, public distribution of packaged builds at scale — by a person or a company, not only a company — folding it into a product or service you sell, and stripping its notices or relicensing it. Publishing your changed source on a code host stays allowed. Code inherited from upstream, and every model weight, tokenizer or dataset, keeps whatever license came with it, and this license adds nothing to those parts.

Produced checkpoints carry a `LICENSE-NRC-3.txt` beside their `run_manifest.json`, so a folder keeps those terms wherever it is copied.

---

## Contributing

Issues and pull requests are welcome. Small fixes can go straight to a PR; for anything larger, open an issue with the shape you have in mind first so the approach can be agreed before you spend the time.

The most self-contained places to start are `studio/backend/core/labs/` (each engine is a module with its own tests), `studio/frontend/src/features/labs/`, and `studio/backend/core/advanced/` — custom eval tasks, the repair preset and adapter merging each live in one module there with its own tests.

If a suite looks mysteriously red on Windows, check the line endings before changing the test: `.gitattributes` asks for LF under `studio/frontend/`, but `core.autocrlf=true` checks files out as CRLF and several tests pin exact source text. [docs/README.md](docs/README.md) has the details.
