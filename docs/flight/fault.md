# flight.fault

**Source:** `packages/flight/src/flight/fault`
**Kind:** subsystem app

## Purpose

The fault package runs FDIR for the flight software. It watches subsystem heartbeats,
routes fault events to SAFE mode, and publishes the inhibit authority for command exit.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`app`](fault/app.md) | app shell | Bus loop for watchdog ticks and SAFE latching |
| [`policy`](fault/policy.md) | pure module | Maps fault codes to SAFE mode changes |
| [`watchdog`](fault/watchdog.md) | pure module | Counts missed heartbeats per subsystem |

## Package interface

The package re-exports `FaultApp`, `WatchdogEntry`, `build_entries`, `check_heartbeats`,
`SAFE_TRIGGERING_FAULTS`, `decide_mode_change`, `enter_safe_mode`, and `exit_safe_mode`.

## Interactions

The fault app subscribes to `HeartbeatMsg`, `FaultEventMsg`, and `RoutedCommandMsg`.
It publishes `ModeChangeMsg`, `SafetyStateMsg`, and `CommandAckMsg` for EXIT_SAFE handling.
Other subsystems raise `FaultEventMsg`; this package routes them. The watchdog emits
`WATCHDOG_EXPIRE` when a monitored subsystem misses too many heartbeats.

## Constraints

The `watchdog` and `policy` modules are pure. They take time as arguments and thread state
in and out. `FaultApp` owns the bus, the clock, and mutable latch state. The fault app does
not monitor its own heartbeat. Thermal and power limit checks live in their producing apps.

## Related documents

- [`flight.core.composition`](core/composition.md)
