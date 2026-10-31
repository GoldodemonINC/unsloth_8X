# Context

Working notes for picking the repository up cold. Scope: the webhook delivery and
training-progress work, plus the Labs growth-policy and CPU-capability pass that
followed it.

Files this work owns:

- `studio/backend/core/training/webhooks.py` — the events side: the per-run progress
  state, the terminal-claim policy, the enabled/url config gate, and the event entry
  points.
- `studio/backend/core/training/webhook_payload.py` — the wire contract: the four
  header names and `build_payload`, imported by both sides below.
- `studio/backend/core/training/loss_anomaly.py` — the detector alone: the window
  constants, the retention bound and its floor, the series view, the tracker.
- `studio/backend/core/training/webhook_delivery.py` — sending alone: signing, the
  pinned transport, proxy refusal, the retry loop, delivery records and threads.
- `studio/backend/utils/webhook_settings.py` — webhook settings I/O and the address policy.
- `studio/backend/routes/settings.py` — settings and test-event routes, delivery response models.
- `studio/backend/tests/test_webhooks.py` — 104 tests covering all of the above.
- `studio/backend/tests/test_end_to_end_webhook.py` — the end-to-end path: eviction
  past the bound, detection at the absolute step, signed delivery to a loopback
  sink, and the disabled gate.

## What changed, and why

**Per-run memory bound.** `observe_progress` used to keep every loss point of a run
forever (~104 bytes/point, so ~1 GB for a ten-million-step run). A run now retains
only its newest 8,192 points; absolute `total`/`base` counters keep step attribution
correct after eviction. Detector state derived from older points — baseline prefix
heap, recent-window medians and sums, plateau deques, worst spike, NaN step — is kept
separately and is fixed-size, so detection is unaffected: spike, divergence, plateau
and NaN all still fire and attribute correctly deep into a long stream (proven at
200,000 steps, and measured flat at 1.85 MB of retained state from 10k to 1M steps).

**The bound is now tied to the windows, not to a lucky literal.**
`_floor_terms()` holds the four window-to-bound relations as named `_FloorTerm`
rows and `_worst_term()` is the single argmax over them: `retained_points_floor()`
reads the worst term's requirement and `_retention_refusal()` builds the message
from that same term, so which relation dominates is decided in one place. Today the
floor is 7,211 against the shipped 8,192. Three
relations make that floor necessary: the recent-window eviction loop reads one
position behind the window start, the divergence baseline must finish filling before
`base` overtakes it, and the spike baseline needs its full look-back below the first
scanned index. Lowering the bound below the floor — or raising any window constant
past the bound — fails at import with an `AssertionError` that states the broken
relation with both sides named and neither blamed, for example `MAX_RETAINED_POINTS
= 8192 is below the floor 12011 required by BASELINE_WINDOW = 6000`.
The check is an `if`/`raise`, not an `assert`, so it holds under `python -O`
exactly as under a normal interpreter. `test_the_retention_bound_stays_above_every_detector_window`
fails on a broken bound and, independently, fails if any single term is dropped from
the floor while the runtime check would still pass, and
`test_the_import_refusal_names_the_offending_window_and_its_value` walks each window
in turn and checks that the message names it, then lowers the bound and checks that
the message leads with the bound. The raise itself is pinned by
`test_the_shipped_module_imports_clean_through_the_import_system` and
`test_a_broken_relation_refuses_on_a_real_import`: they copy the real file into a
temp directory and import it through a genuine subprocess (one row under `python
-O`), so deleting the `if`/`raise` or downgrading it to an `assert` fails the suite.

**Failure classification by kind, not by message matching.** Every delivery outcome
carries a `failure_kind` drawn from a fixed set (`invalid`, `unresolvable`, `blocked`,
`proxy`, `certificate`, `http`, `transport`). The retry loop reads only the kind:
pre-flight refusals record exactly one attempt with no backoff; certificate errors,
3xx and 4xx outside {408, 425, 429} stop immediately; transport errors, 5xx and
408/425/429 retry up to three attempts with 0.5 s / 1.5 s delays. A blocked,
unresolvable or proxied target therefore never sleeps through a backoff it cannot
succeed from.

**Source-aware proxy advice.** Delivery is direct-only: a proxy resolves the target
name itself, which would defeat address pinning, so a configured proxy is refused as
`proxy` rather than silently honoured. The message names the source that is actually
in play — an environment proxy (`HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY`) is told to add
the host to `NO_PROXY`, while a system proxy (Windows registry) is told that
`NO_PROXY` is not read for it and to use the operating system's proxy bypass list
instead. Loopback targets are exempt.

**One owner for the address policy.** `classify_webhook_host` in
`studio/backend/utils/webhook_settings.py` is the single definition of what is
blocked, returning `loopback`, `allowed`, `link-local`, `multicast`, `unspecified` or
`reserved`. Literal-URL validation and the resolved-address check both call it, so
changing the rule means changing one function. Ordering that must not drift: loopback
is checked before the blocked categories (`::1` classifies as `reserved`), and
IPv4-mapped IPv6 is unmapped first.

**Classification reaches API consumers.** `WebhookDeliveryInfo` in
`studio/backend/routes/settings.py` projects `failure_kind`, so
`GET /api/settings/webhooks` distinguishes a certificate failure from a timeout in
JSON, and the delivery history consumes it: `webhook-settings-section.tsx`
renders the wording for the row's kind (`settings.general.webhooks.failureInvalid`
… `failureTransport`, present in every locale), an unknown or absent kind falls
back to the raw `error`, and a failure with neither falls back to `failureOther`.

**The suite states each behavior once.** `test_webhooks.py` is written as a contract:
repeated cases became parametrized tables with one row per case — the four delivery
outcomes (unreachable, 404, 500-then-200, 500), the two terminal-claim shapes, the
http/https blocked resolution, and both legacy-secret read paths — and three tests
that only restated coverage a broader case already held were deleted (the
pinned-address certificate case, the 20k spike, the retry-record duplicate).
Assertions name bounds (`prefix_count`, `entries()`) instead of reaching into heap
internals. The flat-retention, floor-invariant and 200k-attribution tests are
unchanged.

Expected values that come from the module's shipped constants are
hand-computed literals with one owner: `_BOUND`, `_FLOOR`, `_BELOW_FLOOR`,
`_BASELINE_WINDOW`, and a single `_WINDOW_REQUIREMENTS` table shared by both
retention tests. A module constant used as an expectation
(`MAX_RETAINED_POINTS`, the window bounds, `FAILURE_PROXY`/`FAILURE_CERTIFICATE`,
`MAX_TRACKED_RUNS`) is specification rather than computed output and is read from
the module deliberately; exactly one assertion pins the module's bound to the
file's literal.

**The unused delivery parameters are gone.** `webhook_delivery.deliver` had a
`record: bool` flag that no caller ever passed and a `timeout` override that no
caller ever set, so the entry is now always appended, the timeout is always
`REQUEST_TIMEOUT_S`, and the single-use `_record` helper was inlined into the one
place that writes to the ring buffer. `send_test_event` no longer forwards a
timeout either; `max_attempts` stays because the tests pass it. The malformed-URL
table and the address-policy table no longer share rows: every address refusal
(`link-local`, `multicast`, `unspecified`, `reserved`) is asserted by the
address-policy test, which also checks `classify_webhook_host` and the resolver
for the same host, so the URL-shape table keeps only scheme, credential, port and
whitespace cases.

## How the module operates now

Four modules, one direction of flow: `webhooks` imports `webhook_delivery`,
`webhook_payload` and `loss_anomaly`; `webhook_delivery` imports `webhook_payload`;
nothing imports back, so there is no cycle and each piece of state has exactly one
owner.

- `loss_anomaly.py` holds `MAX_RETAINED_POINTS = 8192`, the windows
  (`BASELINE_WINDOW = 3600`, `RECENT_WINDOW = 3600`, `PLATEAU_WINDOW = 2600`,
  `SPIKE_BASELINE = 20`, `CHECK_EVERY = 10`), `_floor_terms()`, `_worst_term()`,
  `retained_points_floor()`, `_retention_refusal()` and the import-time refusal. It
  keeps no global state — a `LossAnomalyTracker` and a `RetainedSeries` are made per
  run by its caller.
- `webhook_payload.py` holds the four wire-header names beside `build_payload`, so
  the events side and the transport side both depend on the contract instead of on
  each other.
- `webhook_delivery.py` holds the retry constants (`MAX_ATTEMPTS = 3`,
  `RETRY_DELAYS = (0.5, 1.5)`, `REQUEST_TIMEOUT_S`, `MAX_DELIVERIES = 25`) and the
  failure kinds, plus the delivery state it actually reads — records ring buffer and
  live threads — under `_delivery_lock`. It imports the header names from
  `webhook_payload`, never defines its own.
- `webhooks.py` holds `MAX_TRACKED_RUNS = 8` and `MAX_TERMINAL_KEYS = 100`, the
  `_progress` map and its lock, the terminal-claim set with its own
  `_terminal_lock`, the `_enabled_config` gate both event paths share, and the
  three event entry points. It re-exports `build_payload`, `get_deliveries` and
  `wait_for_deliveries` for callers that already import it.

A name is patched on the module that reads it: retry constants on `webhook_delivery`,
window constants and the floor on `loss_anomaly`. Patching them through `webhooks`
would silently miss.

Delivery path: resolve the endpoint once and refuse any blocked address, connect
through a pinned connection to the verified socket address, wrap TLS with the
original host name so certificates are checked against the domain rather than the
pinned address, and send the original authority in `Host`. Blocked, unresolvable and
proxied targets fail once with their kind recorded; everything else retries by kind
as described above. Delivery records (ring buffer of 25) carry `failure_kind`.

Progress path: `observe_progress` keeps per-run state for at most 8 runs, trims each
run's curve to the retention bound, and checks for anomalies every 10 points once a
run has at least 8 points. The detector reports `nan`, `divergence`, `spike` and
`plateau`, each alerted at most once per run.

## Growth policy: one owner (this pass)

The first fix pass repaired two bugs but reproduced the duplication that had caused
the first one: `core/labs/pipelines.py::_plan_scale` re-derived the growth rules that
`core/advanced/scaling.py` already owned. That is exactly how the old 2x floor went
stale in the first place — `scaling` changed, `pipelines` silently did not.

`scaling.growth_policy(base_params, *, target_params, added_params)` is now the single
owner of the growth arithmetic. It returns the requested target, the clamped target,
the default (base + 10%, clamped at the ceiling), the ceiling itself, the no-growth
refusal sentence, and the measured-range note. Both callers ask it:

- `scaling.plan()` assembles its report from that one result: `floor_params`,
  `max_params`, `default_target_params`, `target_params`, `target_below_floor` and the
  `no_growth` refusal all come from it, so the Scaler no longer computes them twice.
- `pipelines._plan_scale` passes the chain's *running* parameter count as the base. It
  cannot delegate to `scaling.plan()`, because `plan()` derives its base from the
  stage's own `config.json`, while a scale stage in the middle of a chain grows from
  whatever the previous stage produced.

Two product decisions deliberately stay in `pipelines`, because the two surfaces
really do differ: the Scaler *clamps* a target past `CAP_ADDED_PARAMS` down to the
ceiling (the slider cannot ask for more), while a pipeline stage *refuses* it and asks
you to split the run. `growth_policy` reports `over_cap` and `requested_params`;
`pipelines` renders its own refusal from them, so the ceiling arithmetic still has one
owner and only the verdict differs.

**CPU capability is explicit on every cost-model row.**
`core/advanced/resources.py::_COST_MODEL` had no `cpu` key on any row, so
`cpu_capable = bool(model.get("cpu"))` read "absent" as "cannot". Every row now states
it: `True` for merge, moe, split, scale, merge_eval and autoscale; `False` for eval,
qat and cmt.

That does move two outcomes on a GPU-less host, and those two are the point of the
change. Sized against a real source checkpoint, the base file asked `scale` for 4.0
times the checkpoint in video memory and `autoscale` for 3.0 times, against a machine
reporting none, so both were refused with 409; their `vram` is now zeroed when there is
no GPU and the note says the stage runs on CPU, so both lease. `eval`, `qat` and `cmt`
are genuinely GPU-bound: denied before this branch, denied after, same GPU message.
With no source to size, every kind already leased both before and after, because the
video-memory ask is a multiple of zero.

**Review response: the lease sizes RAM from the checkpoint the job will write.**
The Major on this branch pointed out that on a GPU-less host `vram` is zeroed, so RAM
is the only admission gate for `scale`, and the row reserved 1.0 times the *source*
checkpoint while `scaling.run` really peaks at two *expanded* copies — it saves, then
reloads to verify. A grown job could therefore be granted a lease the host cannot
hold and die of memory exhaustion.

`estimate` now takes `growth_ratio`, the planned checkpoint size as a multiple of the
source, and multiplies the `ram` term by it. `scale`'s own multiplier went from 1.0 to
2.0 so it matches the peak its own comment documents. `/scale/run` derives the ratio
from `plan()`'s `base_params` and `target_params`, and `/labs/auto-scale/run` from
`autoscale.plan()`'s `expansion`; both hand it to `_start_job`, which passes it to
`acquire_for_job`, which passes it to `estimate`. A ratio below 1 clamps to 1, so a
shrink cannot shrink the reservation. `vram` and `disk` multipliers are untouched:
`vram` governs the GPU path, where moving it would change grant outcomes this branch
is meant to hold still.

## Verification

This pass: `126 passed` across `test_labs_artifacts.py`, `test_advanced_scaling.py`,
`test_advanced_resources.py`, `test_advanced_routes.py`,
`test_advanced_scaling_routes.py`, `test_labs.py`, `test_labs_autoscale.py` and
`test_labs_cmt.py`; `67 passed` across the whole `tests/test_labs_*` set. Three tests
pin the new structure: `test_the_scale_refusal_is_the_scaler_own_policy` and
`test_the_default_growth_target_is_the_scaler_default` fail if `pipelines` stops
delegating, and `test_every_cost_model_row_states_its_cpu_capability` fails if a row
loses its `cpu` key. A live `_start_job` probe against this host's real memory probe
(gpus = [], vram_free = 0) reproduced the grant/deny table exactly.

`133 passed`: 104 in `studio/backend/tests/test_webhooks.py`, 2 in
`studio/backend/tests/test_end_to_end_webhook.py`, plus 27 adjacent
(`test_loss_anomaly_parity.py`, `test_parent_guard.py`). The webhook suites include a
real TLS sink with an in-test CA, a wrong-name certificate rejection, a DNS-rebinding
case, proxied-send refusal, and route-level assertions for surfaced errors.

A bounded backend run over the webhook surface and every `routes/settings` importer —
53 files, 3,858 tests — passes 3,783 with 70 skipped and the same 35 failures as the
pre-change baseline (identical failure set: symlink privilege, the Unix-only
`resource` module, HOME/XDG path assertions), so none of this thread's changes moved
that surface.

Review round on PR #20: `134 passed` across `test_advanced_resources.py`,
`test_advanced_routes.py`, `test_advanced_scaling_routes.py`, `test_labs_autoscale.py`,
`test_labs.py`, `test_labs_artifacts.py` and `test_advanced_scaling.py`, then
`17 passed` in `test_labs_cmt.py` and `test_labs_repair.py` — the suites that reach
`_start_job`, whose signature grew. No comments were added and line endings stayed
uniform; `git add` reports 126 insertions and 9 deletions across the six files, so
nothing was rewritten wholesale.

## Performance: `observe_progress` at 10k, 50k, 100k steps

One point per training step over a decaying-loss curve, min of three runs each,
Python 3.12.13 on this host:

| steps | total | steady µs/point | checks disabled | anomalies |
| --- | --- | --- | --- | --- |
| 10000 | 0.25 s | 31.6 | 11.9 | 1 |
| 50000 | 2.67 s | 55.7 | 23.8 | 1 |
| 100000 | 6.17 s | 51.2 | 24.0 | 1 |

Steady cost is the last 5,000 points of each run. The wall-clock ratios are not
linear because the first 8,192 points run in the cheap pre-retention regime at
about 13 µs each: a 10k run is mostly pre-retention, a 100k run is not, so the
mix differs even though neither regime grows.

Three findings:

- **Cost does not grow with run length.** With checks disabled the last 5,000
  points cost 23.79 µs each at 50k and 24.01 µs each at 100k. Sampling
  5,000-point windows across one 100k run gives 48–137 µs with no upward trend;
  that spread is machine noise from the concurrent sessions on this host.
- **Detection stays bounded.** Retention holds exactly `MAX_RETAINED_POINTS`
  (8192) points at every depth, a check runs every 10 points, and each run
  surfaced exactly one anomaly — a plateau on this curve.
- **The front trim is the base cost.** `del points[:1]` on the 8,192-element
  retained list measured 18.4 µs in isolation against 19.2 µs for the whole
  checks-disabled path: one memmove of 8,191 pointers per point, roughly half
  of the steady cost.

The trim was deliberately left alone. Batching it would break
`test_end_to_end_webhook.py`, which asserts `len(state["points"]) == bound`
exactly after eviction, and a ring-buffer or deque replacement would rewrite the
`RetainedSeries` indexing the parity suite depends on. The saving is about 1.8 s
on a run that takes hours, which does not pay for either risk.

## Not part of this work

The NRC-2 → NRC-3 licence rename is not part of the webhook work described here,
but it is no longer sitting uncommitted: this session committed it as `ee59feed3`
on `x8-main` (the Studio tree — `LICENSE-NRC-3.md`, `core/advanced/nrc_license.py`,
`README.md`, `scripts/release_x8_local.py`, and the header-only edits across the
labs and advanced modules, `core/training/fit.py` and the frontend labs files,
51 files) and as `e835b40cb` on `XE-Main` (the features tree, 13 files), both
pushed. Still uncommitted: `docs/releases/2026.5.0.md` and the other untracked
files in the working tree.

## Next task

Delivered. The six paths that belong to this thread — `Context.md`,
`studio/backend/core/advanced/scaling.py`, `studio/backend/core/labs/pipelines.py`,
`studio/backend/core/advanced/resources.py`,
`studio/backend/tests/test_labs_artifacts.py` and
`studio/backend/tests/test_advanced_resources.py` — are committed as `56ffc47cf`
on branch `x8-growth-policy`, pushed to the `x8` remote, and open as PR #20
against `x8-main` with no auto-merge:
https://github.com/GoldodemonINC/unsloth_8X/pull/20

Two further commits ride the same branch: `eaa37fbeb` records the `observe_progress`
benchmark below, and `297bedb42` answers the review.

The index is clean. The other 43 entries in `git status` belong to concurrent
sessions — 24 modified files under `studio/frontend/` (that session owns the
frontend gate matrix), plus untracked `.playtest-library/`, `AGENTS.md`,
`Project skills/`, `docs/releases/2026.5.0.md` and
`studio/src-tauri/x8-updater.key.pub`. Stage by name, never `git add -A`.

Backend gate for this thread has run: 268 passed, exit 0, across the four webhook
suites and the eight labs and advanced suites. No backend lint gate is wired —
ruff is configured in the root `pyproject.toml` but is not installed in either
virtualenv.

PR #20 has had one review: three findings, all three answered — two corrections to
this file (one of them already fixed in `eaa37fbeb` before the review ran) and the
RAM sizing Major in `297bedb42`. It is waiting on re-review and a merge.

Two engineering items are open and deliberately deferred. The `disk` multiplier on the
`scale` row has the same under-reservation the `ram` fix just closed: the review asked
for RAM, and disk would size from `growth_ratio` the same way, but that is a separate
decision. The front-trim decision recorded under Performance needs
`test_end_to_end_webhook.py` relaxed or `RetainedSeries` moved to a deque, and the
measured saving does not justify either.
