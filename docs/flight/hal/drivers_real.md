# flight.hal.drivers_real

**Source:** `packages/flight/src/flight/hal/drivers_real/`
**Kind:** driver set

## Purpose

Provides concrete drivers for real hardware. Only composition roots import this package.
Each driver satisfies its HAL Protocol structurally.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`gimbal`](drivers_real/gimbal.md) | driver | Serial PTU gimbal driver |
| [`sensor`](drivers_real/sensor.md) | driver | PySpin FLIR Blackfly S camera driver |
| [`scalar`](drivers_real/scalar.md) | stub | Housekeeping scalar sensor stub |
| [`station`](drivers_real/station.md) | driver | TCP-in / UDP-out CCSDS station link |

## Package interface

Re-exports `RealGimbal`, `RealScalarSensor`, `RealSensor`, and `RealStationLink`.

## Interactions

`flight.core.select_drivers` constructs these drivers when the profile selects a
`real` axis. Each instance is injected into an app as its Protocol type. There is no
`RealLaunchLock`.

## Constraints

- Importing this package does not require any hardware SDK.
- Constructing `RealSensor` or `RealGimbal` may raise `ImportError` when the SDK is
  absent.
- `RealScalarSensor` is a stub. It always returns `Ok(0.0)`.
- This package must not import `drivers_sim`.

## Related documents

- [`flight.hal`](../hal.md)
- [`flight.hal.interfaces`](interfaces.md)
- [`flight.hal.drivers_sim`](drivers_sim.md)
