# Context

Working notes for picking the repository up cold. Scope: the webhook delivery and
training-progress work done in this session.

Files this work owns:

- `studio/backend/core/training/webhooks.py` — delivery transport, failure
  classification, retry policy, progress tracking, anomaly detection, retention bound.
- `studio/backend/utils/webhook_settings.py` — webhook settings I/O and the address policy.
- `studio/backend/routes/settings.py` — settings and test-event routes, delivery response models.
- `studio/backend/tests/test_webhooks.py` — 107 tests covering all of the above.

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

## How the module operates now

Constants at the top of `webhooks.py`: `MAX_RETAINED_POINTS = 8192`, windows
`BASELINE_WINDOW = 3600`, `RECENT_WINDOW = 3600`, `PLATEAU_WINDOW = 2600`,
`SPIKE_BASELINE = 20`, `CHECK_EVERY = 10`; `MAX_TRACKED_RUNS = 8`,
`MAX_DELIVERIES = 25`, `MAX_ATTEMPTS = 3`, `RETRY_DELAYS = (0.5, 1.5)`; then
`retained_points_floor()` and the import-time refusal.

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

`134 passed`: 107 in `studio/backend/tests/test_webhooks.py` plus 27 adjacent
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
2. Split `webhooks.py` (929 lines carrying signing, pinned transport, proxy policy,
   retry, delivery records, the progress entry and the detector) so transport and
   detection stop sharing one file.
3. Drive a post-eviction anomaly through the full delivery path to a local sink, the
   one end-to-end case the suites still do not cover.
