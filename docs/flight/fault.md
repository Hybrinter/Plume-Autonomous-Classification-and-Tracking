# flight.fault

**Source:** `packages/flight/src/flight/fault`
**Kind:** subsystem app

## Purpose

The fault package runs FDIR for the flight software. It watches subsystem heartbeats, routes
`FaultEventMsg` values through the system mode manager, and publishes the current
`SystemMode`. Producing subsystems raise their own faults. This package routes them and emits
`WATCHDOG_EXPIRE` when heartbeats stop.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`app`](fault/app.md) | module | FDIR app shell: bus I/O, watchdog cycle, mode manager |
| [`mode`](fault/mode.md) | pure module | Legal `SystemMode` edges and `ModeChangeMsg` construction |
| [`watchdog`](fault/watchdog.md) | pure module | Heartbeat miss counting and `WATCHDOG_EXPIRE` emission |
| [`policy`](fault/policy.md) | pure module | SAFE-triggering fault set and SAFE-entry message construction |

## Package interface

Re-exports `FaultApp`, `WatchdogEntry`, `build_entries`, `check_heartbeats`,
`SAFE_TRIGGERING_FAULTS`, `decide_mode_change`, `enter_safe_mode`, and `enter_init_mode`.

## Interactions

The fault app subscribes to `HeartbeatMsg`, `FaultEventMsg`, `ModeRequestMsg`, and
`RoutedCommandMsg`. It publishes `ModeChangeMsg`, `SafetyStateMsg`, and `CommandAckMsg`.
The composition root passes the monitored subsystem name tuple from `MONITORED_SUBSYSTEMS`.
The fault app does not use HAL drivers.

## Constraints

- `watchdog`, `policy`, and `mode` are pure modules with no bus, clock, or I/O access.
- Watchdog timing uses monotonic seconds. Message timestamps use wall-clock ISO strings.
- Thermal and power threshold checks live in their producing subsystems, not here.
- The fault app does not monitor its own heartbeat.
- Only this package publishes `ModeChangeMsg`.

## Related documents

- [`flight.core.composition`](core/composition.md)
- [`flight.payload`](payload.md)
- [`flight.iss_iface`](iss_iface.md)
- [`flight.thermal`](thermal.md)
- [`flight.electrical`](electrical.md)
