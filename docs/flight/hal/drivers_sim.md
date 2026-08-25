# flight.hal.drivers_sim

**Source:** `packages/flight/src/flight/hal/drivers_sim`
**Kind:** driver set

## Purpose

This package holds in-process stand-in drivers for SIL, tests, and sim-selecting
composition roots. Each driver satisfies the matching HAL Protocol structurally.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`sensor`](drivers_sim/sensor.md) | driver | `SimSensor` replays scripted mosaic frames |
| [`gimbal`](drivers_sim/gimbal.md) | driver | `SimGimbal` first-order gimbal dynamics |
| [`station`](drivers_sim/station.md) | driver | `SimStationLink` replays inbound CCSDS packets |
| [`scalar`](drivers_sim/scalar.md) | driver | `SimScalarSensor` replays scalar readings |
| [`launch_lock`](drivers_sim/launch_lock.md) | driver | `SimLaunchLock` in-memory launch lock |

## Package interface

Re-exports: `SimGimbal`, `SimLaunchLock`, `SimScalarSensor`, `SimSensor`, `SimStationLink`.

`SimLaunchLock` is the only launch-lock implementation. No real driver exists yet.

## Interactions

The SIL composition root and `select_drivers` construct these drivers from scripted inputs
(`SimDriverInputs`). Apps receive the same Protocol types as in flight.

Sim drivers script data for deterministic tests. End-of-script behavior differs by device
type (stall, hold-last, or empty queue).

## Constraints

- Real and sim driver packages do not import each other.
- Sim drivers return `Result` and do not raise on runtime faults.
- `SimGimbal` integrates pose lazily on every public call. The injected clock must advance
  between SIL steps or the pose does not move.
- `SimStationLink.sent` is a test inspection hook with no real-driver counterpart.

## Related documents

- [`flight.hal`](hal.md)
- [`flight.hal.interfaces`](interfaces.md)
- [`flight.core.select_drivers`](core/select_drivers.md)
