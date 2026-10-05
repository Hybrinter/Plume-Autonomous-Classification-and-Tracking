# flight.fault

**Source:** `packages/flight/src/flight/fault`
**Kind:** subsystem app

## Purpose

The fault package runs FDIR for the flight software. It watches subsystem heartbeats, routes
`FaultEventMsg` values to SAFE requests for the external mode authority, and publishes the
fault-owned safety evidence. Producing subsystems raise their own faults; this package routes
them and emits `WATCHDOG_EXPIRE` when heartbeats stop.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`app`](fault/app.md) | module | FDIR app shell: bus I/O, watchdog cycle, recovery gating |
| [`watchdog`](fault/watchdog.md) | pure module | Heartbeat miss counting and `WATCHDOG_EXPIRE` emission |
| [`policy`](fault/policy.md) | pure module | SAFE-triggering fault set and mode-request construction |

## Package interface

Re-exports `FaultApp`, `WatchdogEntry`, `build_entries`, `check_heartbeats`, `SAFE_TRIGGERING_FAULTS`,
`decide_mode_request`, `enter_safe_request`, and `recovery_authorized`.

## Interactions

The fault app subscribes to `HeartbeatMsg`, `FaultEventMsg`, and `SystemModeActivatedMsg`. It
publishes `SystemModeRequestMsg` and `SafetyStateMsg`. The composition root passes the
monitored subsystem name tuple from `MONITORED_SUBSYSTEMS` and the activation epoch. The
fault app does not use HAL drivers.

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
