# flight.hal

**Source:** `packages/flight/src/flight/hal`
**Kind:** package

## Purpose

The hardware abstraction layer holds device Protocols and concrete drivers for flight
hardware. Apps depend on the Protocols in `flight.hal.interfaces`. The composition root
injects a real or sim driver bundle at startup.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`interfaces`](hal/interfaces.md) | package | Runtime-checkable device Protocols |
| [`drivers_real`](hal/drivers_real.md) | driver set | Real hardware drivers |
| [`drivers_sim`](hal/drivers_sim.md) | driver set | Simulation and SIL drivers |

## Package interface

None. The top-level `hal` package has no `__init__.py` re-exports. Importers reach into
`hal.interfaces`, `hal.drivers_real`, or `hal.drivers_sim` directly.

## Interactions

Apps receive injected Protocol implementations from `flight.core.composition.build_apps`.
The payload app uses `ImagingSensor` and `GimbalActuator`. The iss_iface app uses
`StationLink`, `StorageReader`, and `StorageWriter`. The thermal and electrical apps
use `ScalarSensor`. The mechanical app uses `LaunchLock`.

Only composition roots (`flight.core.main`, `sim.sil`) import concrete driver packages.
Import-linter contracts enforce this boundary.

## Constraints

- Apps import `flight.hal.interfaces` only. They never import `drivers_real` or
  `drivers_sim`.
- Real and sim driver packages do not import each other.
- Every driver method returns `Result[T, FaultCode]`. Drivers do not raise on runtime
  faults.
- Vendor SDK imports happen inside driver constructors, not at module import time.
- Large artifacts (frames, stored bytes) pass by direct call, not on the message bus.

## Related documents

- [`flight.core.composition`](core/composition.md)
- [`flight.core.select_drivers`](core/select_drivers.md)
