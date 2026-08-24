# flight.hal.drivers_sim

**Source:** `packages/flight/src/flight/hal/drivers_sim/`
**Kind:** driver set

## Purpose

Provides sim and SIL stand-in drivers. Only composition roots import this package. Each
driver satisfies its HAL Protocol structurally.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`gimbal`](drivers_sim/gimbal.md) | driver | First-order gimbal dynamics model |
| [`sensor`](drivers_sim/sensor.md) | driver | Scripted mosaic frame replay |
| [`scalar`](drivers_sim/scalar.md) | driver | Scripted scalar reading replay |
| [`station`](drivers_sim/station.md) | driver | Scripted CCSDS packet replay and recording |
| [`launch_lock`](drivers_sim/launch_lock.md) | driver | In-memory launch-lock model |

## Package interface

Re-exports `SimGimbal`, `SimLaunchLock`, `SimScalarSensor`, `SimSensor`, and
`SimStationLink`.

## Interactions

`flight.core.select_drivers` and the SIL composition root construct these drivers when
the profile selects a `sim` axis. `SimLaunchLock` is wired for every profile, including
all-real flight, until a real driver exists.

## Constraints

- End-of-script behavior differs per driver. See each module page.
- This package must not import `drivers_real`.
- `SimStationLink.sent` is a test and SIL inspection hook with no real-driver
  counterpart.

## Related documents

- [`flight.hal`](../hal.md)
- [`flight.hal.interfaces`](interfaces.md)
- [`flight.hal.drivers_real`](drivers_real.md)
