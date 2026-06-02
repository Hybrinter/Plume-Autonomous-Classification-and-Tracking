# PACT Flight Software Architecture

> **Status (2026-06-01):** This document describes the current `packages/flight` subsystem-app
> architecture on branch `fsw-restructure`. The legacy `src/pact/` tree (multiprocessing +
> `ops/main.py`) is retained for reference and is being retired; do not build new work against it.
> The design rationale lives in
> `docs/superpowers/specs/2026-05-30-pact-iss-payload-fsw-structure-design.md`.

## System Overview

PACT (Plume Autonomous Capture/Classification Technology) is an **ISS-attached external payload**.
It autonomously detects industrial plumes in multispectral VNIR imagery from orbit, drives a
pointing gimbal to track them, and hands telemetry + downlink products to the station. The ISS
provides the bus services PACT does not implement itself -- primary power, thermal sink, attitude,
and the downlink path to the ground -- so PACT is **not** a free-flyer and implements no RF stack.

Because the payload is continuously commandable and recoverable from the ground through the
station, the software reliability posture is **fail-safe / ground-recoverable / graceful
degradation** rather than fully autonomous fault recovery. Hazardous functions still honor NASA
payload safety inhibits.

The system is a **Python-only** monorepo (the earlier Rust-migration plan is dropped). Heavy
training/tooling code is kept out of the flight image entirely.

---

## Workspace Layout

A `uv` workspace with three packages keeps the flight image lean (no torch/onnxruntime in the
flight dependency set):

```
packages/
  flight/   # the flight software: pact-flight (deps: numpy, scipy, structlog)
  sim/      # SIL harness, scene generation, digital twin: pact-sim (deps: numpy, pact-flight)
  tools/    # training, evaluation, model export (heavy deps live here): pact-tools
```

`import-linter` (`.importlinter`) enforces package isolation and layering; `mypy --strict`,
`ruff`, and `pytest` round out the gates. CI (`.github/workflows/ci.yml`) runs all gates scoped to
`packages/` on Linux + Python 3.14.

---

## The Subsystem-App Model

Each subsystem under `packages/flight/src/flight/` is an isolated **app**: a thin imperative
shell wrapping a pure decision core, communicating with other apps **only** over a typed pub/sub
message bus. No app holds a reference to another app. The composition root (`flight.core`) owns
the bus, the clock, the drivers, and the scheduler.

| Layer | Package | Role |
|-------|---------|------|
| **core** | `flight.core` | Composition root: config load, `build_apps`, the thread `Scheduler`, the flight `main()` entry. |
| **payload** | `flight.payload` | The science pipeline: acquire -> preprocess -> infer -> track -> gimbal-arbitrate. |
| **fault** | `flight.fault` | FDIR: heartbeat watchdog + fault-to-mode policy -> SAFE. |
| **iss_iface** | `flight.iss_iface` | Station bridge: inbound `CommandMsg`, outbound downlink items. |
| **thermal** | `flight.thermal` | Housekeeping: temperature telemetry + `THERMAL_OVER_LIMIT` self-report. |
| **electrical** | `flight.electrical` | Housekeeping: power telemetry + `POWER_OVER_LIMIT` self-report. |
| **mechanical** | `flight.mechanical` | Scaffold only (no concrete device yet). |
| **libs** | `flight.libs` | Shared foundations: `types`, `messages`, `config`, `bus`, `time`, `telemetry`. |
| **hal** | `flight.hal` | Hardware abstraction: `interfaces` (Protocols) + `drivers_real` + `drivers_sim`. |

---

## Dependency Layering

Enforced by the `flight-layers` import-linter contract (higher imports lower; never the reverse):

```
flight.core                                                      (composition root)
  |
  v
flight.{payload, thermal, electrical, mechanical, iss_iface, fault}   (peer apps; no cross-imports)
  |
  v
flight.hal.interfaces                                            (HAL Protocols)
  |
  v
flight.libs                                                      (types < messages; config/bus/time/telemetry)
```

Additional contracts: `flight` must not import `sim`/`tools`; `sim` must not import `tools`;
concrete drivers (`drivers_real`/`drivers_sim`) are reachable **only** from composition roots
(`flight.core` and `sim.sil`) -- apps depend solely on the HAL Protocols; the real and sim driver
sets must not import each other.

---

## The Message Bus

`flight.libs.bus.MessageBus` is a typed pub/sub bus routed by **exact message type**:
`subscribe(MsgType)` returns a `Subscription[MsgType]`; `publish(msg)` delivers a copy to every
subscription registered for `type(msg)`. Transport is in-process `queue.Queue` (what the SIL and
unit tests use); a multiprocessing-backed transport can replace the queue factory later without
touching app code.

**Queue-ownership invariant:** only the composition root constructs the bus and injects
subscriptions into apps. Apps never construct queues; pure cores never touch the bus.

Three standard envelopes give the "everything is commandable, everything is telemetered" property:

- `CommandMsg{target, command_id, params, source, seq}` -- station/ground -> `iss_iface` -> bus -> target app.
- `TelemetryEventMsg{subsystem, event_name, payload}` -- app -> bus -> downlink/storage.
- `FaultEventMsg` / `HeartbeatMsg` / `ModeChangeMsg` -- the FDIR event envelopes.

**Large artifacts never go on the bus.** `(C, H, W)` tensors and masks stay in-process (see the
preprocessing co-location invariant); the bus carries compact records only.

---

## The HAL

Each device class is a `@runtime_checkable` `Protocol` in `flight.hal.interfaces`, returning
`Result[T, FaultCode]`:

| Protocol | Real driver | Sim driver |
|----------|-------------|------------|
| `ImagingSensor` | `RealSensor` (lazy PySpin) | `SimSensor` (replays frames) |
| `GimbalActuator` | `RealGimbal` (stub) | `SimGimbal` (integrates deltas) |
| `StationLink` | `RealStationLink` (stub) | `SimStationLink` (scripted in/records out) |
| `ScalarSensor` | `RealScalarSensor` (0.0 stub) | `SimScalarSensor` (replays readings) |

The detector backend is a parallel swap behind `flight.payload.model.DetectorBackend`:
`OnnxDetector` (lazy `onnxruntime`, frozen `.onnx` artifact) for flight, `ScriptedDetector`
(fixed probability mask) for SIL/tests. SDK imports are **lazy** -- importing a driver module
never requires its SDK; only constructing the real driver does. This keeps CI and the lean flight
image SDK-free.

The composition root selects the implementation; **apps never know whether they got a real or sim
driver.**

---

## Inside the Payload App

The payload's stages are internal stages of one app, so a float32 `(C, H, W)` tensor is never
pickled across a process boundary (the **preprocessing co-location invariant**):

```
flight/payload/
  app.py         # PayloadApp: the loop shell (acquire -> publish), holds injected drivers + bus
  preprocess/    # pure fns: select_bands -> radiometric -> quality flags -> ProcessedFrameMsg
  model/         # DetectorBackend: OnnxDetector | ScriptedDetector; shared extract_blobs
  tracking/      # pure: EMA filter, constant-velocity Kalman, IoU blob matcher
  gimbal/        # pure: GimbalArbiter FSM (the resolver), LQR control law, safety gates
  control.py     # PayloadController: the pure composition of tracking + gimbal into one step
```

Per-frame flow inside `PayloadApp.process_frame(raw, state, now)`:

```
raw frame
  -> apply_calibration (identity) -> select_bands (B2,B3,B4,B8) -> compute_quality_flags
  -> ProcessedFrameMsg            [co-located, no queue]
  -> detector.detect(processed)   -> InferenceResultMsg  (published)
  -> PayloadController.step(state, result, now):
        confidence/area gates -> match_blobs (IoU) -> EMA -> Kalman predict/update
        -> arbiter.step (FSM) -> LQR refinement
     -> (new ControlState, GimbalCommandMsg | None, telemetry events)
  -> gimbal.send_command(cmd) + bus.publish(cmd) + publish telemetry
```

`PayloadController` and every tracking/gimbal function are **pure** -- they take state + inputs
and return new state + outputs, with no I/O, no clock access (time is passed in as `now`), and no
bus access. This makes them deterministic, replayable from logs, and trivially unit-testable.

### The gimbal arbiter

The `GimbalArbiter` is the pure FSM resolver over five states (IDLE / ACQUIRING / TRACKING / SCAN
/ SAFE). The tracking controller is a *command source* (it requests pointing); the arbiter is the
*resolver* (it decides whether and how much to command, subject to persistence, rate limit, and
safety gates). Both stay pure; `PayloadController` composes them.

---

## FDIR

`flight.fault` is the failure-detection/isolation/recovery app:

- **Watchdog** (`watchdog.py`, pure): one `WatchdogEntry` per monitored subsystem;
  `check_heartbeats` increments misses for overdue subsystems and emits `WATCHDOG_EXPIRE` at the
  threshold (`watchdog_max_miss_count`, default 3 misses of `watchdog_interval_s`, default 5 s).
- **Policy** (`policy.py`, pure): a `frozenset` `SAFE_TRIGGERING_FAULTS` + `decide_mode_change` map
  a `FaultEventMsg` to a `ModeChangeMsg(SAFE)` or `None`. This replaces the legacy per-`FaultCode`
  callable dispatch table (dynamic dispatch is disallowed); the SAFE-triggering set is preserved
  exactly.
- **App** (`app.py`): subscribes to `HeartbeatMsg` + `FaultEventMsg`, runs the watchdog each tick,
  and publishes `ModeChangeMsg`.

Each producing subsystem **self-reports** its faults (the payload emits `INFERENCE_NAN`; thermal/
electrical emit their over-limit codes), so the FDIR app only watches heartbeats and routes
already-raised faults -- no central sensor polling.

---

## Time

`flight.libs.time.Clock` separates **monotonic** time (control intervals, timeouts, rate limits)
from **wall-clock** time (ISO 8601 message timestamps). Pure cores and app shells receive a
`Clock`; the composition root owns the concrete instance (`RealClock` in flight, `ManualClock` in
SIL/tests). Pure step functions take `now: float` explicitly so they remain deterministic.

---

## Composition Root + Scheduler

`flight.core.composition.build_apps(config, bus, clock, drivers, monitored)` is the single,
**driver-agnostic** wiring point: it constructs all five apps from a `Drivers` bundle (HAL
Protocols + detector) over one shared bus + clock. It imports only Protocols and apps -- never a
concrete driver -- so the identical wiring serves both the flight entry and the SIL.

`flight.core.scheduler.Scheduler` runs each app's `run(stop_event)` in a daemon thread (the bus is
in-process, so threads share it); `start()` launches, `stop()` signals + joins.

`flight.core.main` is the flight composition root: it constructs the real drivers + the ONNX
detector, calls `build_apps`, and runs the scheduler. It executes only on flight hardware (real
drivers + `onnxruntime` are absent in CI), so the driver-agnostic `build_apps` is what is
unit-tested.

---

## Software-in-the-Loop (SIL)

`packages/sim` stands up the **real flight apps** against sim drivers via the same `build_apps`:

- `sim.scene.plume` -- synthetic zeroed `(4, 256, 256)` frames + a `ScriptedDetector` whose fixed
  mask yields one stable central plume blob.
- `sim.sil.build_sil_system` -- constructs the sim drivers, bundles `Drivers`, and calls
  `build_apps` (the exact flight wiring).
- `sim.sil.SilHarness` -- a deterministic single-threaded stepper (no scheduler threads): each
  step acquires + processes one frame, samples housekeeping, pumps the ISS bridge, publishes
  per-subsystem liveness heartbeats, then runs the FDIR tick -- all over the shared bus with `now`
  advanced explicitly.

Two integration tests prove the closed loop and run in the default CI gate (not `e2e`):
1. **Nominal data path:** a plume scene drives payload detection -> a gimbal command (the SimGimbal
   moves off origin) + telemetry, with no spurious SAFE.
2. **FDIR path:** a 95 C reading exceeds the 80 C limit -> `THERMAL_OVER_LIMIT` -> the FDIR app
   publishes `ModeChangeMsg(SAFE)`.

The digital twin (`sim.twin`) is a deferred scaffold; `SimGimbal`'s delta integration is
sufficient pointing dynamics for the current SIL.

---

## Conventions (summary)

Full rules live in `.claude/rules/`. The load-bearing ones:

- **Result, not exceptions.** Library code returns `Result[T, E]`; process entry points may raise
  only for unrecoverable startup failures. Never read `.value` without an `Ok`/`Err` check first.
- **Frozen dataclasses** (`slots=True` for pure data structs); construct modified copies with
  `dataclasses.replace()`. No dynamic dispatch; statically-typed `Protocol` interfaces only.
- **Strong typing everywhere**, mypy `--strict`. `mypy_path` resolves cross-package imports to the
  workspace src trees (without it, `flight.*` collapses to `Any` and strict checking degrades).
- **Enum values mirror member names** (`IDLE = "IDLE"`); numpy arrays carry a shape/dtype comment;
  `structlog` everywhere with `subsystem` + `event` fields; line length 100; PEP 758 parenless
  `except A, B:` is the ruff-format-normalized idiom on Python 3.14.

---

## Status & Remaining Work

Built and CI-green end-to-end: tooling, `libs`, `hal`, `core` foundations, the full `payload`
pipeline, `fault`, `iss_iface`, `thermal`, `electrical`, the composition root + scheduler, and the
SIL closed-loop integration. Open items:

- **Retire `src/pact/`** (legacy tree) and then widen the CI gates from `packages/` to the whole
  repo. This is a deliberate, user-gated deletion step.
- **`mechanical`** subsystem -- build when a concrete mechanism device is identified.
- **`tools/` migration** -- move the legacy torch training/inference (`InferenceEngine`, dataset,
  train, quantize, model export-to-ONNX) into `pact-tools`.
- **Real driver integration** -- `RealSensor` (PySpin), `RealGimbal`, `RealStationLink`,
  `RealScalarSensor` are stubs pending the flight hardware + ISS avionics interface.
- **Model-upload transport** -- chunked model upload (legacy `comms/uplink.py`) as a future
  consumer of the `iss_iface` command/downlink transport.
- **CI housekeeping** -- bump `actions/checkout`/`setup-uv` off the deprecated Node 20 runtime.
