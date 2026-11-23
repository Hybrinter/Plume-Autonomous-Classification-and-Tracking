# flight.system_modes.app

**Source:** `packages/flight/src/flight/system_modes/app.py`
**Kind:** app shell

## Purpose

`SystemModesApp` is the system-mode authority app shell. Each tick it drains safety evidence,
mode requests, routed system-mode commands, and sync requests. It decides each request with
`decide` and publishes the outcome.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SUBSYSTEM` | constant | `"system_modes"`: routing target and heartbeat name |
| `AuthorityState` | class | Mutable active activation, sequence, transition count, and evidence |
| `SystemModesApp` | class | Frozen authority app with config, bus, clock, epoch, and subscriptions |
| `SystemModesApp.from_config` | function | Builds the app and subscribes to its inputs |
| `SystemModesApp.tick` | method | Runs one drain, decide, and publish cycle |
| `SystemModesApp.run` | method | Periodic loop with heartbeats until `stop_event` is set |

## Inputs and outputs

- `from_config(cfg, bus, clock, epoch)` returns a `SystemModesApp` with no active mode. Its
  first tick activates `SAFE`.
- `tick()` takes no arguments and returns `None`.
- `run(stop_event)` runs until the event is set.

## Behavior

1. Drain `SafetyStateMsg` and keep the newest accepted evidence. Acceptance requires the
   injected epoch to match, a nonnegative strictly increasing `evidence_sequence`, and a
   finite nonfuture `observed_s`; stale, replayed, wrong-epoch, or nonfinite evidence never
   replaces the accepted state.
2. With no active mode, decide a `SAFE` boot request (request ID `<epoch>-boot`). The system
   always boots into `SAFE` and waits for an operator command.
3. Drain `SystemModeRequestMsg` and decide each one as a `SUBSYSTEM` request.
4. Drain `RoutedCommandMsg` for the `system_modes` target. `EXIT_SAFE` requires the routed
   `EXECUTE` phase and becomes an `EXIT_SAFE` request for `INIT`. `SET_MODE` becomes a
   `SET_MODE` request for its `mode` parameter. `GIMBAL_STOW` becomes a `SET_MODE` request
   for `STOW`.
5. For each decision, publish one `SystemModeTransitionMsg`. For an accepted decision,
   allocate the next sequence and publish one `SystemModeActivatedMsg` keyed by
   `(epoch, sequence)`.
6. For each command, publish one execution `CommandAckMsg` correlated to the original
   source/sequence/opcode: `ACCEPTED` on an accepted decision, else `REJECTED` with
   `COMMAND_INVALID`.
7. Drain `SystemModeSyncRequestMsg`. A request under the matching epoch republishes the
   current activation unchanged; a foreign epoch is ignored.

Before any request reaches the table the shell applies two gates. Every non-`SAFE`
transition requires fresh accepted evidence (within `watchdog_interval_s`); missing or stale
evidence fails closed. `INIT -> IDLE` additionally requires the payload's own verified
completion request (`requested_by="payload"`, reason `graph_intent:init_complete`) carrying
the exact `ActivationKey(epoch, sequence)` of the current INIT activation; a wrong-key,
wrong-epoch, missing-key, or non-payload completion is denied with a record.

Subsystem requests are decided before ground commands in one tick, and a SAFE request
accepted or merely observed in a tick denies that tick's `EXIT_SAFE` outright. Duplicate
requests are not coalesced: each gets its own transition record.

## Errors and faults

A `SET_MODE` with an unknown `mode`, an `EXIT_SAFE` without routed `EXECUTE`, or an
unsupported opcode gets a `REJECTED` ACK with `COMMAND_INVALID` and no transition record.
The app does not raise at runtime.

## Messages

| Direction | Type |
| --- | --- |
| Subscribe | `SystemModeRequestMsg`, `RoutedCommandMsg`, `SafetyStateMsg`, `SystemModeSyncRequestMsg` |
| Publish | `SystemModeTransitionMsg`, `SystemModeActivatedMsg`, `CommandAckMsg`, `HeartbeatMsg` |

## Configuration

Reads `FaultConfig` via `cfg.fault`:

| Field | Use |
| --- | --- |
| `watchdog_interval_s` | Loop wait, heartbeat interval, and evidence freshness bound |

## Constraints

- `SystemModesApp` is frozen; mutable state lives in `AuthorityState`.
- The epoch comes from the composition root. The app does not create it.
- A sync replay does not allocate a sequence or publish a transition record.
- `SAFE` is always decidable without fresh evidence; every other transition fails closed
  when evidence is missing or stale.
- The freshness check lives in the shell so `decide` stays pure on the latest accepted
  evidence.

## Related documents

- [`flight.system_modes`](../system_modes.md)
- [`flight.system_modes.transitions`](transitions.md)
- [`flight.fault.app`](../fault/app.md)
- [`flight.core.composition`](../core/composition.md)
