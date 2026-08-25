# flight.fault.app

**Source:** `packages/flight/src/flight/fault/app.py`
**Kind:** app shell

## Purpose

`FaultApp` is the FDIR app shell. Each tick it drains heartbeats, routes fault events through the
policy, runs the heartbeat watchdog, handles `EXIT_SAFE` commands, and publishes `SafetyStateMsg`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SafetyLatch` | class | Mutable SAFE latch state owned by the app shell |
| `FaultApp` | class | Frozen FDIR app with config, bus, clock, and subscriptions |
| `FaultApp.from_config` | function | Builds the app and subscribes to heartbeats, faults, and routed commands |
| `FaultApp.initial_entries` | method | Seeds watchdog entries for all monitored subsystems |
| `FaultApp.tick` | method | Runs one watchdog, fault-routing, and safety-state cycle |
| `FaultApp.run` | method | Periodic loop until `stop_event` is set |

## Inputs and outputs

- `from_config(cfg, bus, clock, monitored)` returns a `FaultApp` with fresh subscriptions.
- `initial_entries()` returns a `dict[str, WatchdogEntry]` keyed by monitored subsystem name.
- `tick(entries, now)` takes the current watchdog dict and monotonic seconds; it returns the
  updated entries dict.
- `run(stop_event)` runs until the event is set.

## Behavior

1. Drain all pending `HeartbeatMsg` values and reset miss counts for matching subsystems.
2. Drain all pending `FaultEventMsg` values and publish `ModeChangeMsg(SAFE)` for SAFE-triggering
   codes; latch SAFE on each trigger.
3. Call `check_heartbeats` and route any `WATCHDOG_EXPIRE` faults through the same policy.
4. Drain routed `EXIT_SAFE` commands; un-latch SAFE when the latch is set and no SAFE-triggering
   fault fired this tick.
5. Publish `SafetyStateMsg` with the current latch state and active fault codes from this tick.

## Errors and faults

The app publishes `ModeChangeMsg(SAFE)` for faults in `SAFE_TRIGGERING_FAULTS`. It publishes
`CommandAckMsg` with `REJECTED` when `EXIT_SAFE` is refused. It does not raise at runtime.

## Messages

| Direction | Type |
| --- | --- |
| Subscribe | `HeartbeatMsg`, `FaultEventMsg`, `RoutedCommandMsg` |
| Publish | `ModeChangeMsg`, `SafetyStateMsg`, `CommandAckMsg` |

## Configuration

Reads `FaultConfig` via `cfg.fault`:

| Field | Use |
| --- | --- |
| `watchdog_interval_s` | Tick and loop wait interval |
| `watchdog_max_miss_count` | Consecutive misses before `WATCHDOG_EXPIRE` |

## Constraints

- `FaultApp` is frozen; mutable state lives in `SafetyLatch` and the threaded entries dict.
- Heartbeats from subsystems not in `monitored` are ignored.
- `tick` takes `now` explicitly for deterministic tests.
- The loop uses `stop_event.wait(timeout=...)` for immediate shutdown.

## Related documents

- [`flight.fault`](../fault.md)
- [`flight.fault.watchdog`](watchdog.md)
- [`flight.fault.policy`](policy.md)
- [`flight.core.composition`](../core/composition.md)
