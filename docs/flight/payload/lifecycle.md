# flight.payload.lifecycle

**Source:** `packages/flight/src/flight/payload/lifecycle.py`
**Kind:** module

## Purpose

The lifecycle module executes the INIT graph's effect intents (`SELFTEST`,
`MODEL_LOAD`, `HOME`, `VERIFY_INIT`). The graph stays pure: it emits intents
and consumes terminal results. The executor owns a bounded pending-job capacity
of one and a bounded completion mailbox. Flight starts one lazy daemon worker.
SIL and GSE pump the same executor synchronously on the control thread and do
not start the worker. The executor never touches the bus or the gimbal HAL and
never mutates the inference holder. It reads the holder's factory (or the
installed session as the scripted candidate fallback) and produces typed
completions the control owner polls.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `LifecycleToken` | dataclass | Exact activation key, control revision, containment generation |
| `LifecycleObservation` | dataclass | Frozen tick snapshot, home target, and issue time for one check |
| `SelfTestService` | protocol | `check(observation, cancel) -> Result[str | None, FaultCode]` |
| `HomeArrivalService` | protocol | `check(observation, cancel) -> Result[str | None, FaultCode]` |
| `InitializationVerifier` | protocol | `verify(observation, cancel) -> Result[InitVerificationResult, FaultCode]` |
| `ObservedSelfTest` | class | Default self-test over fresh encoder and inhibit-confirmed feedback |
| `ExactHomeArrival` | class | Default HOME check: exact target on a fresh later sample |
| `PendingInitializationVerifier` | class | Default verifier; always reports current-key PENDING |
| `LifecyclePoll` | dataclass | Drained results plus a verified candidate session to install |
| `LifecycleExecutor` | class | Bounded executor: submit, poll, cancel, shutdown; flight daemon or synchronous pump |

## Inputs and outputs

`submit(effects, token, now)` queues pending effect intents deduplicated by
exact `(kind, effect_id)` under the token; each worker invocation is bounded by
a finite monotonic deadline (constructor `effect_deadline_s`, default 30 s).
`poll(token, observation, now)` returns terminal `EffectResult`s for the
current token plus at most one verified candidate `RuntimeSession` for the
control owner to install. One poll claims at most one job. The completion is
mailed for the next poll. `cancel()` drops pending work and flags a blocked
call; `shutdown()` flags the worker and joins it with a bounded timeout when
a daemon exists. Shutdown never waits on the SDK. A blocked daemon may remain
until its call returns. Synchronous mode has no thread to join.

## Behavior

1. Flight lazily starts one daemon on the first claimed intent and reuses it
   across INIT reentries. SIL and GSE set `synchronous` and run each claimed
   job on the caller inside `poll`. One poll runs at most one job. A `PENDING`
   result does not run again in that same poll. The completion waits in the
   mailbox until the next `poll`. Pending capacity is one job and completions
   are bounded, so repeated reentry cannot grow the worker count. Synchronous
   mode never starts a thread.
2. Pending self-test and home jobs are claimed again on a later poll with
   refreshed observations under the original intent deadline. `VERIFY_INIT`
   re-arms its invocation deadline each time the verifier returns a
   current-key `PENDING`, so a responsive verifier may hold INIT indefinitely;
   a blocked verifier call still fails at its bound, and a wrong-key or
   blank-evidence response never re-arms.
3. `MODEL_LOAD` runs `factory.load` then `warm_up` and returns the candidate
   session in the completion; the worker never installs it.
4. A `VERIFY_INIT` completion carrying a current-key `VERIFIED` result with
   nonempty evidence, all three prerequisites succeeded, and a warm candidate
   is exposed to the control owner as `LifecyclePoll.install`. Blank evidence
   or a wrong activation key cannot promote. An `Err` verifier response becomes
   a failed `VERIFY_INIT` result; an explicit `FAILED` status is left for the
   graph.
5. Completions for a different token, a different kind or id, a cancelled
   generation, or a submitted-but-exited intent are dropped by `poll`.
6. Service or factory exceptions become typed failed effects at the worker
   boundary; nothing raises into the control path.

## Errors and faults

Service and factory `Err` codes pass through as the effect result's fault.
Unexpected exceptions are typed at the public boundary: `MODEL_LOAD` load or
warm-up exceptions become `MODEL_CORRUPT`; self-test, home, and verifier
exceptions become `GIMBAL_FAULT`; cooperative cancellation surfaces as
`INFERENCE_TIMEOUT`. Deadline expiry maps `MODEL_LOAD` to `INFERENCE_TIMEOUT`
and every other kind to `GIMBAL_FAULT`. Nothing raises into the control
thread.

## Messages

None. The worker publishes nothing; the control owner emits lifecycle
telemetry and graph outcomes.

## Configuration

The executor takes a finite positive `effect_deadline_s` (default 30 s) that
bounds each invocation and each control-side wait for a completion, measured
from issue (inclusive `>=` expiry). It never bounds the SDK call itself. On
the flight daemon path a blocked call may outlive the deadline inside the
worker. On the synchronous path the same call runs during `poll`, and the
mailed completion is visible on the next poll. The home target comes from
`gimbal.home_el_deg` through the app; feedback freshness follows the encoder
freshness window.

`synchronous` defaults to false. `flight.core.main` keeps the daemon.
`sim.sil.validation.build_validation_system` passes true. `SilHarness` and the
GSE in-process backend share that pumped executor.

## Constraints

The executor holds no locks across service or SDK calls. Motion/pose HAL stays
in the control reference path; the lifecycle worker never calls gimbal
methods. `ObservedSelfTest` attests only to a fresh valid encoder sample and
inhibit-confirmed gimbal observation without containment - it is not a camera
or hardware self-test. `ExactHomeArrival` requires the encoder sample to match
the configured target within numerical equality (1e-12), be fresh per the
feedback window, be strictly later than the HOME intent issue time, and carry
valid health without containment; HOME arrival is not stability evidence.
`PendingInitializationVerifier` never grants success; production verification
has no pass-by-default option.

## Related documents

- [`flight.payload.app`](app.md)
- [`flight.payload.inference.runtime`](inference/runtime.md)
- [`flight.payload.graphs.init`](graphs/init.md)
