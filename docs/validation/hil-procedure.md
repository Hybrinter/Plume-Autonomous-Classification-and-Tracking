# HIL (Hardware-in-the-Loop) Validation Procedure

> **STATUS: DEFINED, NOT RUN.** This procedure is specified ahead of hardware. It is **not**
> executed in CI and requires the full flight hardware bench (camera, gimbal, radio/socket link).
> Do not mark any requirement `verified` from HIL until this procedure has actually been run.

## What HIL exercises

HIL runs every axis **real** (`profiles/hil.toml`: all axes `"real"`,
`host="jetson_aarch64"`): the PySpin camera (`RealSensor`), the torque-stub gimbal
(`RealGimbal`), the real ONNX detector, the socket station link
(`RealStationLink`), `RealIssEphemeris` (stub), and `RealClock`. It is the
highest-fidelity venue short of flight.

## Prerequisites

- Full bench: camera connected (PySpin SDK present), gimbal torque interface, radio or socket
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
   `set_exposure_us(config.sensor.initial_exposure_us)` and `set_gain_db(config.sensor.initial_gain_db)`,
   exiting on `Err`.
4. Start the real `Scheduler`; drive scenarios from the ground station, including realtime-only
   assertions.
5. Record evidence against HIL-venue requirements; update `vcrm.toml` only after a clean run.

## Notes

- The `lock` (LaunchLock) axis remains a permanent VCRM gap: no device, no config field, no HIL
  coverage. It is documented, never tested.
- Orin Nano Super HIL compute uses MAXN SUPER (`nvpmodel`) and `jetson_clocks`. Module TDP is
  25 W. Payload-bus `power_limit_w` stays 55 W. The camera is USB3 Blackfly, not CSI.
