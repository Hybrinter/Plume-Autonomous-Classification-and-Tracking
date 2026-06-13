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

## Lazy-SDK pattern (PySpin, pyserial)

- Two real drivers touch a vendor SDK, each imported *inside* `__init__`, never at module top:
  `RealSensor` imports `PySpin` (FLIR Spinnaker, the `camera` extra) and `RealGimbal` imports
  `serial` (pyserial). Importing `flight.hal.drivers_real` from a sim-selecting composition root
  requires neither SDK; construction is the only place either can raise `ImportError`.

## Closed-loop GimbalActuator surface (ADR 0008)

- `GimbalActuator` is `goto_angle` / `set_rate` / `home` / `stow` / `read_position` (returns a
  *timestamped* `GimbalPosition`) / `read_stow_switch`. The old `send_command(GimbalCommandMsg)`
  delta path is deleted. Drivers clamp the *hardware* envelope (travel +-90/+-45, max hardware
  slew); the arbiter clamps the *mission* envelope -- defense in depth, two independent limits.
- **`SimGimbal` has real first-order dynamics.** Position integrates *lazily*: every public call
  first advances the pose by the clock time elapsed since the previous call, so the one driver is
  honest under both the threaded flight loop (`RealClock`) and the stepped SIL (`ManualClock`).
  RATE integrates the clamped commanded rate; ABSOLUTE/STOW/HOME approach the target with a
  first-order exponential clamped to the slew envelope. The SIL **must advance the clock** between
  steps or the pose never moves. Encoder reads add seeded Gaussian noise; the stow switch closes
  once stow was commanded and the pose is within 0.5 deg of the stow pose.

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

## Fake-SDK test pattern (PySpin and pyserial)

- Real drivers with an SDK are tested in CI by injecting a fake module via
  `monkeypatch.setitem(sys.modules, "<sdk>", fake)` before construction. `RealSensor` uses a fake
  `PySpin` (`test_real_sensor_pyspin.py`); `RealGimbal` uses a scriptable fake `serial` whose
  port records writes and replays queued response lines (`test_real_gimbal_serial.py`). This
  exercises the lazy-import contract and the full driver logic (count conversion, envelope clamps,
  `*`/`!` response handling) without the physical SDK or hardware. The verb set (PP/TP/PS/TS) is a
  documented reference assumption, not a validated wire protocol -- HIL bring-up confirms it.

## Real driver implementation status

- `RealSensor` (PySpin) and `RealGimbal` (serial PTU, ADR 0008) are fully implemented, with the
  fake-SDK CI tests above; constructing either raises `ImportError` if its SDK is absent, and
  `RealGimbal` raises `ValueError` on an empty `serial_port`. `RealScalarSensor` and
  `RealStationLink` remain safe-default stubs pending their hardware integration.
