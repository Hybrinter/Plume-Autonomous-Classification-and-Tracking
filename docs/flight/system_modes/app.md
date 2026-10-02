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

- `from_config(cfg, bus, clock, epoch)` returns a `SystemModesApp` with no active mode. Its first tick activates `SAFE`.
- `tick()` takes no arguments and returns `None`.
- `run(stop_event)` runs until the event is set.

## Behavior

1. Drain `SafetyStateMsg` and keep the newest latch flag and active faults as evidence.
2. With no active mode, decide a `SAFE` boot request (request ID `<epoch>-boot`). The system
   always boots into `SAFE` and waits for an operator command.
3. Drain `SystemModeRequestMsg` and decide each one as a `SUBSYSTEM` request.
4. Drain `RoutedCommandMsg` for the `system_modes` target. `EXIT_SAFE` becomes an `EXIT_SAFE`
   request for `INIT`. `SET_MODE` becomes a `SET_MODE` request for its `mode` parameter.
5. For each decision, publish one `SystemModeTransitionMsg`. For an accepted decision, allocate
   the next sequence and publish one `SystemModeActivatedMsg` keyed by `(epoch, sequence)`.
6. For each command, publish one execution `CommandAckMsg`: `ACCEPTED` on an accepted decision,
   else `REJECTED` with `COMMAND_INVALID`.
7. Drain `SystemModeSyncRequestMsg`. Republish the current activation with its existing key.

Requests are decided before commands in one tick. Duplicate requests are not coalesced.

## Errors and faults

A `SET_MODE` with an unknown `mode` or an unsupported opcode gets a `REJECTED` ACK with
`COMMAND_INVALID` and no transition record. The app does not raise at runtime.

## Messages

| Direction | Type |
| --- | --- |
| Subscribe | `SystemModeRequestMsg`, `RoutedCommandMsg`, `SafetyStateMsg`, `SystemModeSyncRequestMsg` |
| Publish | `SystemModeTransitionMsg`, `SystemModeActivatedMsg`, `CommandAckMsg`, `HeartbeatMsg` |

## Configuration

Reads `FaultConfig` via `cfg.fault`:

| Field | Use |
| --- | --- |
| `watchdog_interval_s` | Loop wait and heartbeat interval |

## Constraints

- `SystemModesApp` is frozen; mutable state lives in `AuthorityState`.
- The epoch comes from the composition root. The app does not create it.
- A sync replay does not allocate a sequence or publish a transition record.

## Related documents

- [`flight.system_modes`](../system_modes.md)
- [`flight.system_modes.transitions`](transitions.md)
- [`flight.fault.app`](../fault/app.md)
- [`flight.core.composition`](../core/composition.md)
