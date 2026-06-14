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

- Full bench: camera connected (PySpin SDK present), gimbal on its serial port, radio or socket
  bridge to the ground station emulator.
- The HIL socket harness backend (`gse.harness.SocketBackend`) -- **deferred** (raises
  `NotImplementedError("PIL/HIL socket backend deferred")`). Bench runners are the next,
  human-gated effort.

## Procedure (when hardware exists)

1. Provision the bench and verify each SDK loads (PySpin, pyserial, onnxruntime) -- these imports
   are lazy and only resolve when the real drivers are constructed.
2. Load config: `load_config("config/default.toml", "profiles/hil.toml")`.
3. Construct drivers with `select_drivers(config, RealClock())` (no `sim_inputs` needed -- every
   axis selects a real branch). The real sensor branch also applies
   `set_exposure_us(config.sensor.default_exposure_us)` and `set_gain_db(config.sensor.default_gain_db)`,
   exiting on `Err`.
4. Start the real `Scheduler`; drive scenarios from the ground station, including realtime-only
   assertions.
5. Record evidence against HIL-venue requirements; update `vcrm.toml` only after a clean run.

## Notes

- The `lock` (LaunchLock) axis remains a permanent VCRM gap: no device, no config field, no HIL
  coverage. It is documented, never tested.
