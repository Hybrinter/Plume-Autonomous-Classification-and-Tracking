# ADR-FLIGHT-0009: Integrate the real system-mode authority with SAFE-to-INIT recovery

**Status:** Accepted
**Date:** 2026-10-03
**Topic:** interface
**Supersedes:** ADR-FLIGHT-0008
**Superseded-by:** none
**Related:** ADR-FLIGHT-0007

## Context

ADR-FLIGHT-0008 froze the request/transition/activation message separation but
left two points open: the runtime authority itself was a pending external
dependency, and the frozen contract routed recovery as `EXIT_SAFE` to `IDLE`.
The delivered authority exists as scoped modules under `flight.system_modes`
(a pure transition table plus an app shell) and must be adapted to the frozen
epoch/sequence activation contract and the payload-graph architecture: the
payload owns a verified INIT lifecycle that completes into `IDLE`, and the
graph set has exactly five system modes. With a real INIT graph on the bus,
an authorized recovery that lands directly in `IDLE` would skip the lifecycle
gates INIT exists to enforce.

## Decision

- Adopt the delivered authority under `flight.system_modes` as the single
  mode authority: a pure `transitions.decide` over the explicit `MODE_EDGES`
  table, plus the `SystemModesApp` shell that drains requests, routed
  commands, fault safety evidence, and sync requests and publishes the
  transition record and any activation.
- Activation identity stays `(epoch, sequence)` from the frozen contract: the
  composition root supplies the epoch and the authority allocates strictly
  increasing sequences; a matching-epoch sync replays the current activation
  unchanged and a foreign-epoch sync is ignored.
- Recovery is `SAFE -> INIT`, not `SAFE -> IDLE`. In `SAFE`, only a ground
  `EXIT_SAFE` to `INIT` is accepted while no SAFE-triggering fault is active;
  that activation carries `recovery_authorized`. Both the fault-owned latch
  and the payload containment release only on that authorized `SAFE -> INIT`
  record with matching request identity, epoch, and sequence.
- `INIT -> IDLE` is keyed: the authority accepts it only for the payload's
  own verified completion request (`graph_intent:init_complete`) carrying the
  exact `ActivationKey` of the current INIT activation. A stale key from a
  previous INIT entry cannot complete a later one.
- The shell gates every non-`SAFE` transition on fresh fault-owned safety
  evidence (matching epoch, strictly increasing nonnegative evidence
  sequence, finite nonfuture observation time, within the watchdog
  interval). Missing or stale evidence fails closed; `SAFE` is always
  decidable. A SAFE request decided or merely observed in a tick wins over
  that tick's `EXIT_SAFE`.
- Ground commands keep their targets: `EXIT_SAFE` (ARM/EXECUTE at the
  router, routed `EXECUTE` at the authority), `SET_MODE` (`mode` param), and
  `GIMBAL_STOW` (mapped to a `SET_MODE` `STOW` request). Local payload pose
  commands (`GIMBAL_HOME`, `GIMBAL_GOTO`, `GIMBAL_HOLD`, `GIMBAL_RESUME`)
  never reach the authority.
- The composition root owns the authority as a peer subsystem app: wired by
  `build_apps`, scheduled with the other apps, and heartbeat-monitored. SIL
  `step_once` ticks it before the payload's activation drain, after the
  command router, and after the FDIR tick, so boot SAFE, ground commands,
  and fault requests are decided in their deterministic cycles. The
  `publish_activation` injection remains an explicit fixture seam that seeds
  the authority coherently; it is not acceptance proof.
- The separation contract stands: requests and transition records never
  select behavior, ordinary activations never clear fault-owned or hardware
  latches, and containment does not wait for the authority.

## Consequences

- Every behavior change requires an accepted activation allocated by the
  authority; denied requests produce audit records only.
- Recovery from any SAFE latched state passes through the verified INIT
  lifecycle before `IDLE` is reachable; there is no shortcut from `SAFE` to
  `OPERATE` or direct `IDLE` resume.
- Contained hardware cannot be recovered by an ordinary activation: only
  the authorized `SAFE -> INIT` record with a fresh unconsumed request ID
  releases the latches.
- Late subscribers synchronize by replayed activation under the matching
  epoch; the bus retains nothing for them otherwise.

## Alternatives considered

- Keep `SAFE -> IDLE` recovery -- skips the verified INIT lifecycle gates
  and lets an authorized recovery select `IDLE` without the payload's
  initialization evidence.
- Co-locate authority inside the fault app -- re-couples evidence production
  to arbitration and removes the independent fail-closed check.
