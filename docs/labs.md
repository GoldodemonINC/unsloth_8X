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
| `manifest-viewer` | feature | The raw `run_manifest.json` beside an artifact |
| `scale-from-artifact` | feature | Hand one artifact's path to the Scaler |
| `training-continuation` | feature | Make an artifact the Training model in one click |

Each row in the Settings tab toggles one of these. A feature that is off says so
on its own page (`LabsOffCard`) rather than disappearing.

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
pytest tests/test_labs.py tests/test_labs_artifacts.py -q
```

Frontend (`studio/frontend/`):

```bash
node --experimental-strip-types --test tests/labs-feature-gate.test.ts
node --experimental-strip-types --test tests/sidebar-nav-backend-parity.test.ts
```

The sidebar parity test compares `SIDEBAR_NAV_ITEM_DEFAULTS` (backend) with
`DEFAULT_CUSTOMIZATION.sidebarNav` (frontend), including order: the `labs` row
was added to both lists and to `SIDEBAR_NAV_ITEM_IDS`, and the backend's
`SidebarNavItemId` literal, in the same position.
