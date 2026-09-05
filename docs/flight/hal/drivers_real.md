# flight.hal.drivers_real

**Source:** `packages/flight/src/flight/hal/drivers_real`
**Kind:** driver set

## Purpose

This package holds concrete drivers for real flight hardware. The flight composition root
constructs these drivers when the environment config selects a real axis.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`sensor`](drivers_real/sensor.md) | driver | `RealSensor` FLIR Blackfly S over PySpin |
| [`gimbal`](drivers_real/gimbal.md) | stub | `RealGimbal` torque-command stub |
| [`ephemeris`](drivers_real/ephemeris.md) | stub | `RealIssEphemeris` always `Err(EPHEMERIS_FAULT)` |
| [`station`](drivers_real/station.md) | driver | `RealStationLink` TCP-in / UDP-out CCSDS link |
| [`scalar`](drivers_real/scalar.md) | stub | `RealScalarSensor` placeholder (returns 0.0) |

## Package interface

Re-exports: `RealGimbal`, `RealIssEphemeris`, `RealScalarSensor`, `RealSensor`,
`RealStationLink`.

There is no `RealLaunchLock`. Launch-lock hardware is not integrated yet.

## Interactions

Only `flight.core.main` and `flight.core.select_drivers` import this package. Apps receive
the resulting Protocol implementations through `build_apps`.

`RealSensor` passes `MosaicFrame` values by direct call to the payload app. `RealStationLink`
carries raw CCSDS bytes to and from iss_iface. `RealGimbal` accepts torque and pose
commands from the payload gimbal path. `RealIssEphemeris` is a stub.

## Constraints

- Importing this module does not require vendor SDKs. Construction of `RealSensor`
  may raise `ImportError` when PySpin is absent.
- Real and sim driver packages do not import each other.
- `RealScalarSensor` is a stub. It always returns `Ok(0.0)`.
- `RealGimbal` is a stub. Commands return `Ok` and do not move hardware.
- `RealIssEphemeris` is a stub. `read_state` returns `Err(EPHEMERIS_FAULT)`.
- Drivers return `Result` on runtime faults. Only startup misconfiguration raises.

## Related documents

- [`flight.hal`](hal.md)
- [`flight.hal.interfaces`](interfaces.md)
- [`flight.core.select_drivers`](core/select_drivers.md)
