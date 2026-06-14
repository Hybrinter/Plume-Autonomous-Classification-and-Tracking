# PIL (Processor-in-the-Loop) Validation Procedure

> **STATUS: DEFINED, NOT RUN.** This procedure is specified ahead of hardware so the seam and
> profile exist and are import-clean. It is **not** executed in CI and requires a Jetson target
> plus the real compute/link/clock stack. Do not mark any requirement `verified` from PIL until
> this procedure has actually been run on hardware.

## What PIL exercises

PIL runs the flight apps with the compute, link, and clock axes set to **real** while the sensor
and gimbal remain **sim** (`profiles/pil.toml`: `sensor="sim"`, `gimbal="sim"`,
`compute="real"`, `link="real"`, `clock="real"`, `host="jetson_aarch64"`). This proves the real
ONNX detector, the real socket station link, and wall-clock timing on the target board, while
still feeding deterministic scene frames and a sim gimbal.

## Prerequisites

- Jetson aarch64 target with the lean flight image (onnxruntime + the exported model at
  `config.inference.model_path`).
- Networking between the target and a ground station emulator (`packages/gse`
  `StationEmulator`) reachable at `config.link.command_tcp_host:command_tcp_port` /
  `telemetry_udp_host:telemetry_udp_port`.
- The PIL socket harness backend (`gse.harness.SocketBackend`) -- **deferred**; it currently
  raises `NotImplementedError("PIL/HIL socket backend deferred")`. Implementing it is the
  next, human-gated effort (see the CHECKPOINT in the plan).

## Procedure (when hardware exists)

1. Flash the lean flight image to the Jetson and copy `config/default.toml` + `profiles/pil.toml`.
2. On the target, load config as an override:
   `load_config("config/default.toml", "profiles/pil.toml")`.
3. Construct drivers with `flight.core.select_drivers.select_drivers(config, RealClock(), sim_inputs)`
   where `sim_inputs` supplies frames + the scripted detector for the still-sim sensor axis
   (compute/link/clock select real branches automatically).
4. Start the real `Scheduler` (thread-per-app) -- PIL uses the real-time scheduler, not the
   deterministic stepper.
5. From the ground (`StationEmulator`), drive the realtime scenarios, including the
   `ack_within_seconds` realtime-only assertions that the in-process backend skips.
6. Record evidence against the PIL-venue requirements and update `vcrm.toml`
   (`status = "verified"`) only after a clean run.

## Notes

- The `lock` (LaunchLock) axis has no device and no config field; it remains a permanent VCRM gap
  and is not exercised by PIL.
