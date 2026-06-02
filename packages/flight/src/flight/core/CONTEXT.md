# `core` Subsystem Context

Non-obvious, cross-cutting context for the compute / C&DH host. Documents the WHY and the
invariants that are not derivable from the individual files or their docstrings.

---

## build_apps is the driver-agnostic seam

- `composition.build_apps` is the single composition root for the whole app topology, and
  it imports **only** HAL `Protocol`s (`flight.hal.interfaces`) and the apps -- **never** a
  concrete driver. This is the load-bearing design decision: the same wiring serves both
  the flight entry and the SIL. The SIL constructs a `Drivers` bundle of sim drivers and
  calls the identical `build_apps`; nothing about the wiring is duplicated.
- Only `main.py` imports `flight.hal.drivers_real` and `OnnxDetector`. Keep concrete-driver
  imports out of `composition.py` or the SIL reuse breaks.

## What is actually unit-tested

- `main.py` / `build_flight_system` are **runtime-only**: `RealSensor` lazily imports PySpin
  and `OnnxDetector` lazily imports onnxruntime, both absent in CI and on dev machines. So
  `main` is never exercised by the suite -- the tested surface is `build_apps` with sim
  drivers + `ManualClock`. Logic that must be covered belongs in `build_apps`, not `main`.

## MONITORED_SUBSYSTEMS

- `("payload", "iss_iface", "thermal", "electrical")` -- exactly the four apps that run
  persistent heartbeat-emitting loops. The `fault` app is the watchdog and deliberately
  does **not** monitor itself. There are **five** `SystemApps` but only four are monitored.

## Scheduler / bus coupling

- The `MessageBus` is in-process (`queue.Queue` transport), which is *why* apps can run as
  shared-memory daemon threads rather than processes -- the scheduler hands each app the
  single shared bus via `build_apps`, not a per-process copy. One `stop` Event fans out to
  all apps; `stop()` retains the dead threads so `is_running()` stays honest after shutdown.
- Daemon threads mean an unjoined app dies with the process; `main` blocks on a bare
  `threading.Event().wait()` and only `KeyboardInterrupt` triggers a graceful `stop()`.

## config_loader is partially stubbed

- `_validate()` is a placeholder (`return None`) and `_build_pact_config` uses
  `.get(key, Default)` defaults, so a TOML key that does not match a dataclass field is
  **silently ignored** rather than erroring. Until the TODOs land, a typo'd config key reads
  as the Python default with no warning -- do not assume load failure on a bad key.
