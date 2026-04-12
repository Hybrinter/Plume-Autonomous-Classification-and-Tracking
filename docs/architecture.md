# PACT Software Architecture

## System Overview

PACT (Plume Autonomous Capture Technology) autonomously detects industrial smoke-stack plumes
in multispectral VNIR imagery, drives an active gimbal to track detected plumes, stores
imagery and metadata with integrity checksums, and downlinks data for ground-based ML
retraining -- all with no real-time ground-in-the-loop control.

The software is organized into ten subsystems, each running as an independent process or
thread. All inter-subsystem data passes through typed, frozen-dataclass messages on named
queues. The codebase is Python written in a Rust-idiomatic style (`frozen=True` dataclasses,
`enum.Enum` discriminants, `Result[T,E]` error handling) to enable mechanical translation
to Rust once behavior is verified.

**Phase I scope:** pretrained model inference, gimbal control, on-board storage, CCSDS
downlink, safe model uplink with rollback, and fault detection. Phase II items are called
out explicitly throughout.

---

## Subsystems

| Subsystem        | Role                                                                       |
|------------------|----------------------------------------------------------------------------|
| **types**        | Dependency root: all enums, frozen message dataclasses, config dataclasses |
| **model**        | U-Net/ResNet-34 segmentation -- forward pass, blob extraction, training    |
| **preprocessing**| Band selection, radiometric calibration, quality flagging, ROI crop        |
| **controller**   | Gimbal arbiter state machine, blob tracker, EMA filter, Kalman + LQR      |
| **imaging**      | FLIR Blackfly S GigE Vision interface, frame acquisition, stall detection  |
| **comms**        | CCSDS encoding, priority downlink queue, chunked uplink, staged model deploy|
| **storage**      | Frame persistence with SHA-256 checksums and append-only manifests         |
| **telemetry**    | Health event aggregation and CCSDS telemetry packet formatting              |
| **fault**        | Heartbeat watchdog, fault detection, safe-mode entry                       |
| **ops**          | Process orchestrator: queue creation, config loading, mode FSM             |

---

## Process Topology

All queues are created in `ops/main.py` and passed as arguments to each subsystem.
No subsystem creates its own queues.

```
imaging_process        --[raw_frame_queue: RawFrameMsg]-->        inference_process
                                                                   (preprocessing runs here)
inference_process      --[inference_queue: InferenceResultMsg]--> controller_process
controller_process     --[gimbal_queue: GimbalCommandMsg]-->       (hardware gimbal stub)
controller_process     --[telemetry_queue: TelemetryEventMsg]-->   telemetry_process
inference_process      --[storage_queue: StorageWriteMsg]-->       storage_process
storage_process        --[downlink_queue: DownlinkItemMsg]-->       comms_process
telemetry_process      --[downlink_queue: DownlinkItemMsg]-->       comms_process
comms_process          --[uplink_queue: UploadChunkMsg]-->          ops/main.py
any subsystem          --[fault_queue: FaultEventMsg]-->            fault_process
any subsystem          --[heartbeat_queue: HeartbeatMsg]-->         fault_process
fault_process          --[mode_queue: ModeChangeMsg]-->             ops/main.py
```

**Preprocessing co-location:** preprocessing runs as a plain function call inside the
inference process (`_run_inference_process()` in `ops/main.py`), not as a separate process.
This avoids serializing large numpy arrays over a `multiprocessing.Queue` on the hot path.
The `RawFrameMsg -> ProcessedFrameMsg` transformation is a function call, not a queue hop.

---

## Concurrency Model

| Subsystem   | Primitive                          | Rationale                                               |
|-------------|------------------------------------|---------------------------------------------------------|
| imaging     | `threading.Thread` + `queue.Queue` | I/O-bound GigE Vision DMA; GIL not a bottleneck         |
| inference   | `multiprocessing.Process`          | GPU-bound; requires true process isolation (REQ-AIML-COMP-002) |
| controller  | `multiprocessing.Process`          | CPU-bound; isolated from GIL for deterministic scheduling |
| storage     | `threading.Thread` + `queue.Queue` | I/O-bound disk writes                                   |
| comms       | `asyncio`                          | Many concurrent I/O waiters; no CPU-heavy loops         |
| telemetry   | `threading.Thread` + `queue.Queue` | I/O-bound serialization and queue puts                  |
| fault       | `multiprocessing.Process`          | GIL immunity: watchdog must fire during GPU kernels     |
| ops         | Main process                       | Spawns all others; runs mode FSM loop                   |

---

## Dependency Layer Order

The import graph is strictly layered. Lower layers must never import from higher layers;
a circular import is a build-breaking bug.

```
types                           <- dependency root; imports only Python stdlib
  |
  v
model / preprocessing / imaging <- import from types only
  |
  v
controller                      <- imports from types + preprocessing
  |
  v
storage / telemetry / comms     <- import from types + model
  |
  v
fault                           <- imports from types + all subsystem message types
  |
  v
ops                             <- imports everything; orchestration layer
```

---

## State Machines

### Gimbal Arbiter (controller/)

Five states. Transitions are evaluated once per `InferenceResultMsg`. The arbiter is a
pure function -- all state is explicit in `ArbiterState`.

```
                  blob detected, confidence gate passed,
                  persistence_count < acquire_persistence_frames
IDLE ------------------------------------------------------------> ACQUIRING
  ^                                                                    |
  |   consecutive_miss >= release_persistence_frames                  | persistence_count >=
  <----------------------- TRACKING <---------------------------------+  acquire_persistence_frames
  |                             |
  |   consecutive_miss >=       | blob detected
  |   release_persistence       | (resets miss counter)
  |   frames                    |
  <-----------------------------+

IDLE ---- idle_seconds > scan_entry_idle_seconds --------------> SCAN
SCAN ---- blob detected ----------------------------------------> ACQUIRING

any state ---- FaultEventMsg received -------------------------> SAFE
SAFE ---- ground command (Phase II) ---------------------------> IDLE
```

**Guards applied before arbiter is called (in `process.py`):**
- `apply_confidence_gate`: mean blob confidence >= `confidence_gate` (default 0.55)
- `apply_min_area_gate`: blob area >= `min_blob_area_px` (default 15 px)
- `check_deadband`: displacement in `[min_deadband_px, max_deadband_px]`; > max -> `GIMBAL_RUNAWAY`
- `check_rate_limit`: enforce <= `retarget_rate_limit_hz` (default 0.5 Hz) between commands

### Ops Mode FSM (ops/)

```
IDLE ---- first blob tracked ----------------------------------> ACTIVE
ACTIVE ---- idle_seconds > scan_entry_idle_seconds ------------> SCAN
SCAN ---- blob detected ----------------------------------------> ACTIVE
ACTIVE/SCAN ---- FaultEventMsg ---------------------------------> SAFE
SAFE ---- ground command (Phase II) ---------------------------> IDLE
```

Mode transitions are applied in `ops/main.py` by draining `mode_queue` each iteration.

---

## Critical-Path Data Flow

The path from raw frame to gimbal command and stored artifact:

```
1. imaging_process
   +-- FlirBlackflyCamera.acquire_frame()    ->  RawFrameMsg
       +-- raw_frame_queue.put(msg)

2. _run_inference_process()  [ops/main.py -- same process as inference]
   +-- raw_frame_queue.get()                 ->  RawFrameMsg
   +-- apply_calibration(bands)              ->  calibrated bands (function call)
   +-- select_bands(calibrated)              ->  4-band tensor (function call)
   +-- compute_quality_flags(...)            ->  frozenset[FrameUsabilityTag]
   +-- ProcessedFrameMsg(...)                ->  in-memory object (no queue)
   +-- InferenceEngine.run(processed)        ->  Ok(InferenceResultMsg)
   +-- inference_queue.put(inference_result) ->  to controller_process
   +-- storage_queue.put(StorageWriteMsg)    ->  to storage_process

3. controller_process
   +-- inference_queue.get()                 ->  InferenceResultMsg
   +-- apply_safety_gates(blobs)             ->  filtered blobs
   +-- GimbalArbiter.step(state, result, now) -> (new_state, GimbalCommandMsg, events)
   +-- gimbal_queue.put(command)             ->  to hardware stub
   +-- telemetry_queue.put(events)           ->  to telemetry_process

4. storage_process  [concurrent with step 3]
   +-- storage_queue.get()                   ->  StorageWriteMsg
   +-- write raw_bands.npy + sha256 verify
   +-- write processed_tensor.pt + sha256 verify
   +-- manifest.append(JSON line)            ->  atomic commit point
   +-- downlink_queue.put(DownlinkItemMsg)   ->  to comms_process
```

---

## Design Decisions

### Preprocessing inside inference process

Preprocessing runs as a plain function call inside `_run_inference_process()` in `ops/main.py`.
Moving it to a separate process would require serializing `(C, H, W)` float32 numpy arrays
through `multiprocessing.Queue` (pickle round-trip) on every frame -- unacceptable on the
hot path. The function-call approach has zero serialization cost.

### Pure-function gimbal arbiter

`GimbalArbiter.step()` is a pure function with no side effects. This makes the arbiter
deterministic, trivially unit-testable (no mocks needed), and replay-able from logs. All
mutable state lives in `ArbiterState`, which is passed in and returned. The caller
(`process.py`) owns the queue, the clock, and the state variable.

### Greedy IoU blob matching

Blob matching uses greedy intersection-over-union (descending sort, first match wins) rather
than optimal Hungarian assignment. For the expected case of < 10 blobs per frame, greedy
matching is O(n^2) with negligible cost and is fully deterministic. Hungarian assignment
would add a scipy dependency and complexity for no practical benefit at this blob count.

### EMA asymmetric initialization

The EMA centroid filter returns the raw observation unmodified on the first frame
(`initialized = False` -> return new_centroid). This avoids a "phantom history" effect where
the first smoothed output is biased toward an arbitrary initial state. Behavior differs
between frame 1 and all subsequent frames.

### Comm window as weekday gate

The TDRSS comm window is enforced as a UTC weekday check (`MON-FRI`). This is not an
approximation -- ISS data dumps are constrained to weekdays by the ISS-ground interface
protocol, and the dump schedule is fixed. No orbital contact prediction is needed or correct.

### Shared downlink queue

Both storage and telemetry write to the same `downlink_queue` (a `queue.PriorityQueue`).
The queue is thread-safe; no synchronization is needed. A single queue keeps priority
ordering global -- telemetry health packets at priority 0 always drain before imagery at
priority 2 or 3.

### Config loaded once at startup

`ops/config_loader.py` loads and merges `config/default.toml` + `config/flight.toml` once
at startup, producing frozen `PactConfig` dataclass instances distributed to each subsystem
as constructor arguments. No subsystem reads TOML at runtime. No dynamic reconfiguration --
changes require restart. This guarantees all processes see consistent configuration.

---

## Configuration Reference

Key non-obvious parameters. Full defaults in `config/default.toml`.

| Section         | Key                           | Default       | Effect |
|-----------------|-------------------------------|---------------|--------|
| `[controller]`  | `confidence_gate`             | 0.55          | Min mean blob confidence (post-sigmoid). Blobs below this are filtered before the arbiter. |
| `[controller]`  | `ema_alpha`                   | 0.4           | EMA smoothing: 0.4 x new + 0.6 x prev. Higher = more responsive, less filtering. |
| `[controller]`  | `min_deadband_px`             | 20            | Displacements < 20 px produce no gimbal command (jitter suppression). |
| `[controller]`  | `max_deadband_px`             | 250           | Displacements > 250 px trigger `GIMBAL_RUNAWAY` fault. |
| `[controller]`  | `retarget_rate_limit_hz`      | 0.5           | Max gimbal command rate (1 command / 2 s). Prevents motor overload. |
| `[controller]`  | `acquire_persistence_frames`  | 3             | Consecutive frames with blob before ACQUIRING -> TRACKING. |
| `[controller]`  | `release_persistence_frames`  | 5             | Consecutive misses before TRACKING -> IDLE. |
| `[inference]`   | `latency_budget_ms`           | 500.0         | **Placeholder** -- pending Jetson Xavier benchmark. `InferenceEngine.run()` returns `Err(INFERENCE_TIMEOUT)` if exceeded. |
| `[fault]`       | `watchdog_interval_s`         | 5.0           | Heartbeat send interval per subsystem. |
| `[fault]`       | `watchdog_max_miss_count`     | 3             | Missed heartbeats before `PROCESS_DIED` fault (15 s total). |
| `[comms]`       | `max_daily_downlink_bytes`    | 1073741824    | 1 GB/day cap enforced by `DownlinkQueue.dequeue()`. |

---

## Test Strategy

Three tiers, all in `tests/`:

**Unit tests** -- isolated, pure-function tests with no real processes. MockCamera replaces
`FlirBlackflyCamera`. All threshold-sensitive functions are parameterized with below/at/above
boundary values. Result types are unwrapped with `assert isinstance(result, Ok)` before
accessing `.value`. Fixture catalogue in `tests/conftest.py`.

**Integration tests** -- multi-subsystem pipeline tests:
- `test_inference_pipeline`: full preprocessing + inference chain with a synthetic frame
- `test_controller_pipeline`: `InferenceResultMsg` -> safety gates -> arbiter -> command
- `test_comms_pipeline`: downlink queue priority ordering + CCSDS packet round-trip

**E2E smoke test** (`tests/e2e/test_full_pipeline_smoke.py`) -- all processes spawned
simultaneously. Synthetic `InferenceResultMsg` is injected onto the inference queue for the
first 3 frames to force ACQUIRING/TRACKING transitions (the untrained model produces random
output). Marked `@pytest.mark.e2e`, 60-second timeout.

---

## Known Gaps / Phase II

| Subsystem    | What's Stubbed                                   | Impact |
|--------------|--------------------------------------------------|--------|
| imaging      | `FlirBlackflyCamera` PySpin integration          | All CI runs use `MockCamera`; no hardware path exercised |
| model        | TensorRT INT8 quantization (`quantize.py`)       | Inference runs in FP32; Jetson latency uncharacterized |
| model        | `latency_budget_ms` (500 ms placeholder)         | Timeout fault threshold is unvalidated |
| preprocessing| `MOTION_SMEAR` detection                         | Always absent from quality flags |
| comms        | TDRSS modem hardware interface                   | Downlink writes to file/socket only |
| comms        | CCSDS secondary headers + CRC-16/CCITT           | Primary header only; payload uses `pickle.dumps` |
| comms        | Uplink chunk reassembly timeout                  | Incomplete uplinks accumulate indefinitely |
| storage      | Compression for raw `.npy` files                 | High disk consumption per frame |
| storage      | LRU eviction on storage full                     | `STORAGE_FULL` fault halts all writes permanently |
| telemetry    | Thermal/power sensor HAL                         | Both fields are hardcoded `0.0` |
| fault        | Safe-mode exit via ground command                | System cannot exit safe mode autonomously |
| ops          | Process restart on crash                         | Crashed subsystem transitions system to safe mode; no recovery |

---

## Planned Subsystems (Phase II)

### Power / Thermal Management

The Jetson Xavier reports real-time power draw via the INA3221 sensor on the carrier board.
Power consumption is a reliable proxy for thermal load. A future `power/` subsystem would:
- Poll Xavier power draw periodically via the system power sensor interface
- Feed real watt readings into `fault/detector.py` `check_power()` and `check_thermal()`
  (currently called with mocked `0.0` values)
- Replace the placeholder thermal limits in `fault/` with real thresholds derived from
  Xavier thermal envelope specifications

This removes the last two mocked fault detectors and completes the fault detection loop.
