# flight.hal.drivers_real

**Source:** `packages/flight/src/flight/hal/drivers_real`
**Kind:** driver set

## Purpose

This package holds concrete drivers for real flight hardware. The flight composition root
constructs these drivers when the driver config selects a real axis.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`sensor`](drivers_real/sensor.md) | driver | `RealSensor` FLIR Blackfly S over PySpin |
| [`gimbal`](drivers_real/gimbal.md) | driver | `RealGimbal` fail-closed Xeryon rate adapter |
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
carries raw CCSDS bytes to and from iss_iface. `RealGimbal` accepts signed rate and pose
latches from the payload gimbal path and rejects torque. `RealIssEphemeris` is a stub.

## Constraints

- Importing this module does not require PySpin. `RealGimbal` imports the vendored
  Xeryon module at driver import; serial I/O still waits until connect.
- Real and sim driver packages do not import each other.
- `RealScalarSensor` is a stub. It always returns `Ok(0.0)`.
- `RealGimbal` remains motion-disabled until explicit Xeryon audit and watchdog/stow
  validation flags pass. It never calls the vendor shutdown helper that may home.
- `RealIssEphemeris` is a stub. `read_state` returns `Err(EPHEMERIS_FAULT)`.
- Drivers return `Result` on runtime faults. Only startup misconfiguration raises.

## Related documents

- [`flight.hal`](hal.md)
- [`flight.hal.interfaces`](interfaces.md)
- [`flight.core.select_drivers`](core/select_drivers.md)
