# flight.fault.app

**Source:** `packages/flight/src/flight/fault/app.py`
**Kind:** app shell

## Purpose

`FaultApp` is the FDIR app shell. Each tick it drains heartbeats, steps the
pure mode manager, runs the heartbeat watchdog, handles mode commands, and
publishes `SafetyStateMsg`. Software boots latched `SAFE`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ModeShell` | class | Mutable holder for `SystemModeState` and active faults |
| `FaultApp` | class | Frozen FDIR app with config, bus, clock, and subscriptions |
| `FaultApp.from_config` | function | Builds the app and subscribes to heartbeats, faults, requests, and commands |
| `FaultApp.initial_entries` | method | Seeds watchdog entries for all monitored subsystems |
| `FaultApp.tick` | method | Runs one watchdog, mode-manager, and safety-state cycle |
| `FaultApp.run` | method | Periodic loop until `stop_event` is set |

## Inputs and outputs

- `from_config(cfg, bus, clock, monitored)` returns a `FaultApp` with fresh subscriptions.
- `initial_entries()` returns a `dict[str, WatchdogEntry]` keyed by monitored subsystem name.
- `tick(entries, now)` takes the current watchdog dict and monotonic seconds. It returns the
  updated entries dict.
- `run(stop_event)` runs until the event is set.

## Behavior

1. Call `begin_tick` and clear the per-tick active-fault set.
2. Drain all pending `HeartbeatMsg` values and reset miss counts for matching subsystems.
3. Drain all pending `FaultEventMsg` values and step the mode manager. SAFE-triggering codes
   latch `SAFE`.
4. Call `check_heartbeats` and step the mode manager for any `WATCHDOG_EXPIRE` faults.
5. Drain `ModeRequestMsg` values (homing, stow, model suspend or resume) and step the
   manager.
6. Drain routed mode commands (`ENTER_INIT`, `ENTER_OPERATE`, `ENTER_IDLE`, `ENTER_STOW`).
   Ack `ACCEPTED` when the edge fires. Ack `REJECTED` when the state does not change.
7. Publish `SafetyStateMsg` with the current `SystemMode`, latch, reason, and active faults
   from this tick.

## Errors and faults

The app publishes `ModeChangeMsg` when the mode manager accepts an edge. It publishes
`CommandAckMsg` with `REJECTED` when a mode command is refused. It does not raise at runtime.

## Messages

| Direction | Type |
| --- | --- |
| Subscribe | `HeartbeatMsg`, `FaultEventMsg`, `RoutedCommandMsg`, `ModeRequestMsg` |
| Publish | `ModeChangeMsg`, `SafetyStateMsg`, `CommandAckMsg` |

## Configuration

Reads `FaultConfig` via `cfg.fault`:

| Field | Use |
| --- | --- |
| `watchdog_interval_s` | Tick and loop wait interval |
| `watchdog_max_miss_count` | Consecutive misses before `WATCHDOG_EXPIRE` |

## Constraints

- `FaultApp` is frozen. Mutable state lives in `ModeShell` and the threaded entries dict.
- Heartbeats from subsystems not in `monitored` are ignored.
- `tick` takes `now` explicitly for deterministic tests.
- The loop uses `stop_event.wait(timeout=...)` for immediate shutdown.
- Boot mode is latched `SAFE` with no fault reason required.

## Related documents

- [`flight.fault`](../fault.md)
- [`flight.fault.mode`](mode.md)
- [`flight.fault.watchdog`](watchdog.md)
- [`flight.fault.policy`](policy.md)
- [`flight.core.composition`](../core/composition.md)
