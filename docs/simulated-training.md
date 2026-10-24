# Simulated training (Settings → Debugging)

A debug switch that starts a training job against a fake trainer, so the whole
Studio flow — wizard, readiness checks, Start, progress, completion — can be
exercised on a machine with no GPU and no dataset.

## Turning it on

Settings → **Debugging** → *Simulated training*. The toggle writes to the
backend (`/api/settings/debug/training-simulation`, stored as
`debug_simulation`), and the value is mirrored into
`features/settings/stores/debug-training-simulation-store.ts` so the wizard and
the Start button follow it without a reload.

## What it changes

With the switch on:

- The method list stops applying host limits, so Full, CPT, LoRA and RL are all
  selectable — see `isTrainingMethodSupportedOnDevice(..., simulated)` in
  `features/training/lib/training-methods.ts`.
- LoRA variants stop applying the bitsandbytes limit, so LoftQ is selectable —
  `isTrainingLoraVariantSupportedOnDevice(..., simulated)`.
- The wizard stops rewriting the method to CoRA on a CPU host
  (`features/studio/wizard/training-wizard.tsx`).
- Readiness stops marking the config unsupported
  (`features/training/hooks/use-training-readiness.ts`).
- Starting skips model, dataset, token and consent preparation and posts a
  fixed `debug/simulated-model` payload
  (`features/training/lib/start-fresh-training-run.ts`).
- The sidebar keeps Train reachable on a chat-only host, and the route guard
  lets `/studio` through.

## What it does not change

Three host checks are deliberately still in force, which is why a simulated run
can be *selected* and still show Start as disabled:

| Check | Where |
| --- | --- |
| Model type supported on this device | `isTrainingModelTypeSupportedOnDevice`, called without a `simulated` argument in `validation.ts` |
| Embedding / chat-audio rows on MLX | `validation.ts`, the `deviceType === "mac"` branch |
| Dataset-modality mismatch | `hasIncompatibleTrainingModalities`, used by `deriveTrainingReadiness` |

If you need the simulator to cover those too, thread the flag through them — and
add the audio-on-a-Mac case to `tests/training-validation.test.ts`, which is the
combination that currently selects fine and then refuses to start.

The start path also skips validation entirely, so the decision-model owner check
that readiness enforces is not applied on that route. Worth deciding which side
is authoritative before changing either.

## Two gates, one decision

The client reads a settings flag; the backend decides with
`core.training.debug_simulation.is_enabled()` (`studio/backend/routes/training.py`,
`_start_simulated_training`). If they disagree, the UI promises a fake run and
the server performs a real one. Keep the toggle as the single source of truth and
let the server flag follow it.

## Tests

- `tests/training-method-cora.test.ts` — every method and LoRA variant is offered
  when simulated.
- `tests/training-validation.test.ts` — a simulated run ignores the host's method
  limits.
- `tests/training-decision-model.test.ts` — readiness is live for a config the
  host would refuse.
- `tests/chat-only-route-guard.test.ts` — `/studio` opens on a chat-only host.
