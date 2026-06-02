# `hal` Subsystem Context

Non-obvious, cross-cutting context for the hardware-abstraction layer -- not derivable
from the individual files or their docstrings.

## Three-layer split is enforced, not conventional

- `interfaces/` (Protocols), `drivers_real/`, `drivers_sim/` are isolated by
  `.importlinter` contracts, not just by directory hygiene:
  - `drivers-from-composition-roots-only`: every app subsystem (`payload`, `thermal`,
    `electrical`, `mechanical`, `iss_iface`, `fault`) and `libs` is *forbidden* from
    importing either concrete driver package. Apps depend only on `flight.hal.interfaces`
    and receive a driver injected by the composition root (`flight.core` / `sim`). If app
    code imports a driver, CI fails -- this is the testability guarantee, not a style rule.
  - `drivers-independent` (both directions): real and sim driver packages must not import
    each other. Keep them parallel and standalone; never share helpers across the two.
- `hal/__init__.py` is intentionally empty (no re-exports). Importers must reach into
  `hal.interfaces` / `hal.drivers_*` explicitly, which keeps the layer boundary legible
  and lets the import-linter contracts target sub-packages precisely.

## Lazy-SDK pattern (PySpin)

- Only `drivers_real/sensor.py` touches a vendor SDK. `PySpin` is imported *inside*
  `RealSensor.__init__`, never at module top -- so importing `flight.hal.drivers_real`
  (e.g. from a composition root that selects sim) never requires the FLIR Spinnaker SDK.
  The SDK is the optional `camera` extra; construction is the only place it can raise
  `ImportError`. No other hal driver has an SDK dependency.

## Structural (duck) satisfaction -- on purpose

- Drivers satisfy the Protocols structurally; they do NOT subclass or import the Protocol
  classes. (`SimGimbal` imports `GimbalPosition`, but that is a data type, not the
  `GimbalActuator` Protocol.) The `@runtime_checkable` decorator exists so the composition
  root / tests can `isinstance`-check an injected driver.

## Sim-exhaustion semantics differ deliberately

- Each sim driver scripts data, but end-of-script behavior is chosen to mimic real
  hardware, so they are NOT uniform:
  - `SimSensor`: returns `Err(CAMERA_STALL)` once frames run out (a camera goes silent).
  - `SimScalarSensor`: holds the last reading forever (a housekeeping sensor always reads).
  - `SimStationLink`: returns `Ok(None)` once inbound is drained (empty inbound queue).
- `SimStationLink.downlinked` is a test/SIL inspection hook with no real-driver counterpart.

## Real drivers are safe-default stubs

- Every real driver is a stub pending hardware: methods return `Ok(...)` / safe defaults
  (`RealGimbal.read_position` -> origin; `RealScalarSensor.read` -> 0.0; `RealStationLink`
  inert) except `RealSensor.acquire_frame`, which returns `Err(CAMERA_STALL)` so a
  mis-wired flight build fails loudly rather than feeding fake frames downstream.
