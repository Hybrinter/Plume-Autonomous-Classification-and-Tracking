# flight.fault

**Source:** `packages/flight/src/flight/fault`
**Kind:** subsystem app

## Purpose

The fault package runs FDIR for the flight software. It watches subsystem heartbeats, routes
`FaultEventMsg` values to mode changes, and publishes the SAFE latch state. Producing subsystems
raise their own faults; this package routes them and emits `WATCHDOG_EXPIRE` when heartbeats stop.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`app`](fault/app.md) | module | FDIR app shell: bus I/O, watchdog cycle, SAFE exit handling |
| [`watchdog`](fault/watchdog.md) | pure module | Heartbeat miss counting and `WATCHDOG_EXPIRE` emission |
| [`policy`](fault/policy.md) | pure module | SAFE-triggering fault set and mode-change message construction |

## Package interface

Re-exports `FaultApp`, `WatchdogEntry`, `build_entries`, `check_heartbeats`, `SAFE_TRIGGERING_FAULTS`,
`decide_mode_change`, `enter_safe_mode`, and `exit_safe_mode`.

## Interactions

The fault app subscribes to `HeartbeatMsg`, `FaultEventMsg`, and `RoutedCommandMsg`. It publishes
`ModeChangeMsg`, `SafetyStateMsg`, and `CommandAckMsg`. The composition root passes the monitored
subsystem name tuple from `MONITORED_SUBSYSTEMS`. The fault app does not use HAL drivers.

## Constraints

- `watchdog` and `policy` are pure modules with no bus, clock, or I/O access.
- Watchdog timing uses monotonic seconds; message timestamps use wall-clock ISO strings.
- Thermal and power threshold checks live in their producing subsystems, not here.
- The fault app does not monitor its own heartbeat.

## Related documents

- [`flight.core.composition`](core/composition.md)
- [`flight.payload`](payload.md)
- [`flight.iss_iface`](iss_iface.md)
- [`flight.thermal`](thermal.md)
- [`flight.electrical`](electrical.md)
