# Unsloth repository docs

These docs describe this checkout, not the hosted product site
(<https://unsloth.ai/docs>). They are deliberately small: each file covers one
thing a contributor has to know before changing it.

| Doc | Covers |
| --- | --- |
| [simulated-training.md](simulated-training.md) | The Debugging simulator that runs a training job with no hardware behind it |
| [labs.md](labs.md) | The Labs switch, its experiment registry, and the Artifacts and Pipelines studios |
| [theming.md](theming.md) | Palettes, accent tokens, and how to change the UI color |

## What is in this repository

| Path | What it is |
| --- | --- |
| `unsloth/` | Unsloth Core: the Python training library (PEFT/LoRA kernels, model loaders, patchers) |
| `studio/backend/` | The Studio server: FastAPI routes, training worker, merges, export, hardware probe |
| `studio/frontend/` | The Studio UI: React + Vite + Tailwind, TypeScript, node's own test runner |
| `studio/src-tauri/` | Desktop shell (Tauri) that hosts the Studio UI |
| `install.sh`, `install.ps1` | One-line installers; both pin the Python stack and prebuilt engines |
| `scripts/` | Maintenance and benchmark scripts used by CI |
| `docs/` | This directory |

## The Studio surfaces

The sidebar is the map of the app. Rows are defined in
`studio/frontend/src/components/app-sidebar.tsx` (`navRows`) and every row is
gated on the same measured hardware verdict (`usePlatformStore`):

- **Chat**, **Projects**, **Library**, **Model hub** — run and organize models.
- **Images**, **Video**, **Audio** — the media workspaces.
- **Train** — the training wizard (`features/studio/wizard/`), with the method
  picker, advanced/scaler section, and run preview.
- **Merge** — model merging (`features/advanced/merge-page.tsx`), CPU streaming
  merge or GPU full-load merge.
- **Scaler** — parameter-count scaling (`features/advanced/scale-page.tsx`).
- **Recipes** — Data Recipes for building datasets.
- **Export** — merged, adapter, GGUF and quantized export.
- **Labs** — the experimental row, hidden until Settings → Labs is on. It hosts
  the **Artifacts** studio (what each produced checkpoint allows next), the
  **Pipelines** planner (chain the stages and price them before any compute),
  **Benchmarks** (score an artifact and keep the report), **CMT** (capacity
  matched to the data, then train only the new blocks) and **Cleanup** (what
  outputs and caches are using, with a guarded delete).
- **API** — the OpenAI-compatible server and request monitor.

Merging, scaling and export are engines under the run layer (jobs, resource
arbitration, `run_manifest.json`) rather than separate applications.

## Checks to run before a change lands

From `studio/frontend/`:

```bash
npm run typecheck    # tsc over src and tests
npm test             # node --test over tests/**/*.test.ts
npm run i18n:check   # locale overlay parity
npm run biome:check  # lint + format
```

From the repository root: `pre-commit run --all-files` (Ruff and the repo's own
lint scripts; Python only).

### A local trap worth knowing

`git config core.autocrlf=true` on Windows checks these files out with CRLF even
though `.gitattributes` asks for `eol=lf` under `studio/frontend/`. Many tests
pin exact source text with `\n` in the regex, so they fail locally and pass on
Linux CI. If a suite looks mysteriously red, check the line endings of the file
under test before changing the test:

```bash
file studio/frontend/src/i18n/locales/en.ts   # "with CRLF line terminators" is the tell
```

A clean LF checkout is the only way to get a representative local run.
