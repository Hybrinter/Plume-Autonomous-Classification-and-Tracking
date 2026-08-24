# flight.hal.interfaces

**Source:** `packages/flight/src/flight/hal/interfaces`
**Kind:** package

## Purpose

This package defines runtime-checkable Protocols for every injected hardware and storage
abstraction. Each Protocol names the typed surface that real and sim drivers satisfy
structurally.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`sensor`](interfaces/sensor.md) | module | `ImagingSensor` acquire-only camera Protocol |
| [`gimbal`](interfaces/gimbal.md) | module | `GimbalActuator` and `GimbalPosition` |
| [`station`](interfaces/station.md) | module | `StationLink` byte-level CCSDS transport |
| [`scalar`](interfaces/scalar.md) | module | `ScalarSensor` single-value housekeeping read |
| [`launch_lock`](interfaces/launch_lock.md) | module | `LaunchLock` motorized pin Protocol |
| [`storage`](interfaces/storage.md) | module | `StorageWriter` and `StorageReader` |

## Package interface

Re-exports: `GimbalActuator`, `GimbalPosition`, `ImagingSensor`, `LaunchLock`,
`ScalarSensor`, `StationLink`, `StorageReader`, `StorageWriter`.

## Interactions

Subsystem apps hold Protocol-typed constructor arguments. The composition root selects
concrete drivers and passes them in. No app imports a driver module.

## Constraints

- Drivers satisfy these Protocols structurally. They do not subclass the Protocol classes.
- Every public method returns `Result[..., FaultCode]`.
- `@runtime_checkable` enables `isinstance` checks at the composition root and in tests.
- `LaunchLock` has a sim driver only today. No real driver exists yet.

## Related documents

- [`flight.hal`](hal.md)
- [`flight.hal.drivers_real`](drivers_real.md)
- [`flight.hal.drivers_sim`](drivers_sim.md)
