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

## Acquire-only contract for ImagingSensor (ADR 0007)

- `ImagingSensor.acquire_frame()` returns a `MosaicFrame` (raw `(H, W)` uint16 2x2-CFA
  mosaic plane plus `timestamp_utc`, `frame_id`, `exposure_us`, `gain_db`). Drivers acquire
  only -- no demosaic, calibration, or normalization inside any driver. Those stages are pure
  functions in `flight.payload.preprocess`.
- `MosaicFrame` is NOT a bus message: it is passed by direct call from the injected sensor
  driver to the payload app. Frames never cross the bus (co-location invariant; large arrays
  never go on the bus).
- `RawFrameMsg` and `MessageType.RAW_FRAME` were removed in 2026-06-09 as part of the mosaic
  contract switchover (ADR 0007). There is no bus message for raw or separated band stacks.

## Fake-PySpin test pattern

- `RealSensor` is tested in CI without the FLIR Spinnaker SDK by injecting a fake `PySpin`
  module via `monkeypatch.setitem(sys.modules, "PySpin", fake)` before constructing the
  sensor. This exercises the lazy-import contract (the SDK is only imported inside
  `__init__`, not at module top) and all driver behavior (acquire, incomplete image, timeout,
  exposure/gain writes). See `packages/flight/tests/test_real_sensor_pyspin.py`.

## Real drivers are safe-default stubs

- Every real driver is a stub pending hardware: methods return `Ok(...)` / safe defaults
  (`RealGimbal.read_position` -> origin; `RealScalarSensor.read` -> 0.0; `RealStationLink`
  inert). `RealSensor` is the one fully-implemented real driver (PySpin acquisition + control
  plane, with the fake-SDK CI tests described above). Construction of `RealSensor` raises
  `ImportError` if PySpin is absent; all other methods are safe to call in tests via the fake.
