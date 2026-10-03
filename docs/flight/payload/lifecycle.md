# flight.payload.lifecycle

**Source:** `packages/flight/src/flight/payload/lifecycle.py`
**Kind:** module

## Purpose

The lifecycle module executes the INIT graph's asynchronous effect intents
(`SELFTEST`, `MODEL_LOAD`, `HOME`, `VERIFY_INIT`) off the control thread. The
graph stays pure: it emits intents and consumes terminal results; the executor
owns one lazy worker thread, a bounded pending-job capacity of one, and a
bounded completion mailbox. The worker never touches the bus or the gimbal HAL
and never mutates the inference holder - it reads the holder's factory (or the
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
| `LifecycleExecutor` | class | One-worker bounded executor: submit, poll, cancel, shutdown |

## Inputs and outputs

`submit(effects, token, now)` queues pending effect intents deduplicated by
exact `(kind, effect_id)` under the token; each worker invocation is bounded by
a finite monotonic deadline (constructor `effect_deadline_s`, default 30 s).
`poll(token, observation, now)` returns terminal `EffectResult`s for the
current token plus at most one verified candidate `RuntimeSession` for the
control owner to install. `cancel()` drops pending work and flags a blocked
worker; `shutdown()` flags the worker and joins it with a bounded timeout, so
shutdown itself never waits on the SDK - a blocked daemon worker may still
remain until its call returns.

## Behavior

1. The worker lazily starts on the first submitted intent and is reused across
   INIT reentries; pending capacity is one job and completions are bounded, so
   repeated reentry cannot grow the worker count.
2. Pending self-test and home jobs are re-polled through the same worker with
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
bounds each worker invocation and each control-side wait for a completion,
measured from issue (inclusive `>=` expiry); it never bounds the SDK call
itself, which may outlive the deadline inside the daemon worker. The home
target comes from `gimbal.home_el_deg` through the app; feedback freshness
follows the encoder freshness window.

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
