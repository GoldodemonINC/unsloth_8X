# Context

Working notes for picking the repository up cold. Scope: the webhook delivery and
training-progress work done in this session.

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
- `studio/backend/tests/test_webhooks.py` — 98 tests covering all of the above.

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
`retained_points_floor()` derives the smallest safe bound from the window constants;
today it computes 7,211 against the shipped 8,192. Three relations make that floor
necessary: the recent-window eviction loop reads one position behind the window
start, the divergence baseline must finish filling before `base` overtakes it, and
the spike baseline needs its full look-back below the first scanned index. Lowering
the bound below the floor — or raising any window constant past the bound — fails at
import with `AssertionError: MAX_RETAINED_POINTS is smaller than the detector windows
need`. The check is an `if`/`raise`, not an `assert`, so it holds under `python -O`
exactly as under a normal interpreter. `test_the_retention_bound_stays_above_every_detector_window`
fails on a broken bound and, independently, fails if any single term is dropped from
the floor while the runtime check would still pass.

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
JSON. No UI consumes it yet and no text is special-cased in the frontend.

**The suite states each behavior once.** `test_webhooks.py` is written as a contract:
repeated cases became parametrized tables with one row per case — the four delivery
outcomes (unreachable, 404, 500-then-200, 500), the two terminal-claim shapes, the
http/https blocked resolution, and both legacy-secret read paths — and three tests
that only restated coverage a broader case already held were deleted (the
pinned-address certificate case, the 20k spike, the retry-record duplicate).
Assertions name bounds (`prefix_count`, `entries()`) instead of reaching into heap
internals. The flat-retention, floor-invariant and 200k-attribution tests are
unchanged.

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
  `SPIKE_BASELINE = 20`, `CHECK_EVERY = 10`), `retained_points_floor()` and the
  import-time refusal. It keeps no global state — a `LossAnomalyTracker` and a
  `RetainedSeries` are made per run by its caller.
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

## Verification

`125 passed`: 98 in `studio/backend/tests/test_webhooks.py` plus 27 adjacent
(`test_loss_anomaly_parity.py`, `test_parent_guard.py`). The webhook suites include a
real TLS sink with an in-test CA, a wrong-name certificate rejection, a DNS-rebinding
case, proxied-send refusal, and route-level assertions for surfaced errors.

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

1. Make the import refusal name the offending window and its value (today's message
   always blames `MAX_RETAINED_POINTS`, even when a window constant is what moved).
2. Drive a post-eviction anomaly through the full delivery path to a local sink as a
   permanent test — exercised by hand after the split (8,414 points, spike at 8,400,
   signed delivery to a loopback sink), but the suites still do not cover it.
