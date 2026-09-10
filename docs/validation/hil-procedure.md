# HIL (Hardware-in-the-Loop) Validation Procedure

> **STATUS: DEFINED, NOT RUN.** This procedure is specified ahead of hardware. It is **not**
> executed in CI and requires the full flight hardware bench (camera, gimbal, radio/socket link).
> Do not mark any requirement `verified` from HIL until this procedure has actually been run.

## What HIL exercises

HIL runs every axis **real** (`profiles/hil.toml`: all five axes `"real"`,
`host="jetson_aarch64"`): the PySpin camera (`RealSensor`), the serial gimbal (`RealGimbal`,
requires `config.gimbal.serial_port` nonempty), the real ONNX detector, the socket station link
(`RealStationLink`), and `RealClock`. It is the highest-fidelity venue short of flight.

## Prerequisites

- Full bench: camera connected (PySpin SDK present), XD-C gimbal on its serial port, radio or
  socket bridge to the ground station emulator. Provision a hardware-specific
  `settings_default.txt`; do not commit it to this repository.
- The HIL socket harness backend (`gse.harness.SocketBackend`) -- **deferred** (raises
  `NotImplementedError("PIL/HIL socket backend deferred")`). Bench runners are the next,
  human-gated effort.

## Procedure (when hardware exists)

1. Provision the bench and verify each SDK loads (PySpin, pyserial, onnxruntime) -- these imports
   are lazy and only resolve when the real drivers are constructed.
2. Load config: `load_config("config/default.toml", "profiles/hil.toml")`.
3. Construct drivers with `select_drivers(config, RealClock())` (no `sim_inputs` needed -- every
   axis selects a real branch), then call `RealGimbal.initialize()` from the composition root
   before starting the scheduler. Confirm that it starts Xeryon v1.88, loads the external settings,
   applies `INFO=4`, `POLI=2 ms`, degree units, −45°/+45° hardware soft limits, and completes
   index search with motor/closed-loop/encoder status valid. The real sensor branch also applies
   `set_exposure_us(config.sensor.default_exposure_us)` and `set_gain_db(config.sensor.default_gain_db)`,
   exiting on `Err`.
4. Start the real `Scheduler`; drive scenarios from the ground station, including realtime-only
   assertions.
5. Record evidence against HIL-venue requirements; update `vcrm.toml` only after a clean run.

## XD-C gimbal checklist

Record timestamps, configuration values, controller readback, and operator initials for each item:

1. Connection: verify the configured port, 115200 baud, `Stage.XRTU_40_109`, axis `X`, and the
   supplied settings file; verify no communication starts before `initialize()`.
2. Index/sign: run `find_index()`, confirm encoder-at-index/valid/closed-loop status, then issue
   a small positive command and confirm the measured elevation sign. Record `direction_sign` and
   `index_offset_deg`.
3. Limits: prove position commands clamp to 0°..+45° for imaging and that only `stow()` may
   command −45°. Prove rate clamp, 0.01°/s quantization, 0.005°/s stop deadband, and safe scan
   reversal ordering.
4. Feedback: compare successive EPOS/TIME samples to an independent reference; verify derived
   velocity and timestamp-wrap handling. Hold TIME constant for more than 250 ms and confirm a
   stale-feedback gimbal fault and stop.
5. Lease/faults: issue RATE without renewal, wait three seconds, and confirm `stopScan()` and
   `rate_lease_expired`. Inject/induce each status fault and verify it maps to SAFE-triggering
   `GIMBAL_FAULT`; verify `reset_faults()` is required to clear a latched duty fault.
6. Duty/stow: confirm ordinary motion stops at 105 s continuous motor-on time, then verify SAFE
   stow may use the final 15 s of the documented 120 s allowance. Confirm the manufacturer's
   cooldown rule before operational release; this software does not claim cooldown compliance.
7. Teardown: stop the scheduler and confirm ordered shutdown stops motion and closes Xeryon.

## Notes

- The `lock` (LaunchLock) axis remains a permanent VCRM gap: no device, no config field, no HIL
  coverage. It is documented, never tested.
