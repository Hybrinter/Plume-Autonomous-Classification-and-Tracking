# flight.hal.interfaces

**Source:** `packages/flight/src/flight/hal/interfaces/`
**Kind:** package

## Purpose

Defines runtime-checkable Protocol types for each device class. Apps import only this
package. The composition root injects a concrete driver that satisfies each Protocol.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`gimbal`](interfaces/gimbal.md) | module | Gimbal actuator Protocol and position type |
| [`sensor`](interfaces/sensor.md) | module | Imaging sensor Protocol |
| [`scalar`](interfaces/scalar.md) | module | Housekeeping scalar sensor Protocol |
| [`station`](interfaces/station.md) | module | Station byte-link Protocol |
| [`storage`](interfaces/storage.md) | module | Storage writer and reader Protocols |
| [`launch_lock`](interfaces/launch_lock.md) | module | Launch-lock pin Protocol |

## Package interface

Re-exports `GimbalActuator`, `GimbalPosition`, `ImagingSensor`, `LaunchLock`,
`ScalarSensor`, `StationLink`, `StorageReader`, and `StorageWriter`.

## Interactions

Apps receive injected Protocol implementations and call their methods directly. No bus
messages are defined in this package. `MosaicFrame` values pass by direct call from the
sensor driver to the payload app.

## Constraints

- Every public method returns `Result[T, FaultCode]`.
- Each Protocol carries `@runtime_checkable`.
- No real `LaunchLock` driver exists. Only `SimLaunchLock` implements the Protocol
  today.

## Related documents

- [`flight.hal`](../hal.md)
- [`flight.hal.drivers_real`](drivers_real.md)
- [`flight.hal.drivers_sim`](drivers_sim.md)
