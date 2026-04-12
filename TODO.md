# PACT Next Steps

---

## Code Correctness (Phase I Bugs)

- [ ] `src/pact/controller/lqr.py` -- `LqrController.from_config()` silently falls back to
      proportional gain when the DARE solver fails; this violates the `Result[T,E]` contract.
      Change it to return `Err(FaultCode.CONTROLLER_FAULT)` and let the caller decide.

- [ ] `src/pact/types/messages.py` -- Add `schema_version: int = 1` to `RawFrameMsg` and
      `InferenceResultMsg` before the first flight integration test. Without this, any schema
      change during a multi-day mission will silently corrupt queued messages.

- [ ] `src/pact/comms/uplink.py` -- Uplink chunk reassembly has no timeout. An incomplete
      uplink accumulates indefinitely. Add a configurable `uplink_reassembly_timeout_s` and
      emit `FaultCode.MODEL_CORRUPT` when it expires.

---

## CI / Config Integrity

- [ ] Add a CI check that asserts default field values in `src/pact/types/config.py` match
      `config/default.toml` exactly. Currently divergence is silent and breaks test
      reproducibility. A simple pytest fixture that loads both and diffs them is sufficient.

---

## Test Coverage Gaps

- [ ] Parameterize all threshold-sensitive unit tests (confidence gate, deadband, rate limit,
      quality flags) with below/at/above boundary values -- verify boundary behavior is tested,
      not just the happy path.

- [ ] Add a unit test for EMA asymmetric initialization: assert frame-1 output equals the raw
      centroid, frame-2 output differs for the same input. Documents the intentional behavior.

- [ ] Add a test for `DownlinkPriority` queue ordering: enqueue items at all four priority
      levels out of order; assert dequeue order is always HEALTH_TELEMETRY first.

- [ ] Add a test for the storage date-rollover edge case: inject a `StorageWriteMsg` with a
      timestamp straddling midnight and assert two separate date subdirectories are created.

---

## Phase I Completions (Hardware / Integration)

- [ ] `src/pact/imaging/camera.py` -- Implement real PySpin GigE Vision acquisition in
      `FlirBlackflyCamera`. All CI currently runs through `MockCamera`; the hardware path is
      untested.

- [ ] `src/pact/controller/process.py` -- Replace `send_gimbal_command()` stub with real
      serial/CAN driver once hardware interface is specified.

- [ ] `src/pact/preprocessing/quality.py` -- Implement `MOTION_SMEAR` detection using a
      gimbal-slew-rate heuristic (angular velocity from consecutive `GimbalCommandMsg`
      timestamps vs. configured exposure). Currently always absent from quality flags.

- [ ] Benchmark `InferenceEngine.run()` on the Jetson Xavier NX with the trained model and
      replace the `latency_budget_ms = 500.0` placeholder in `config/default.toml` with a
      real value. The current timeout fault threshold is completely unvalidated.

---

## Phase II

- [ ] **Power / Thermal subsystem** (`src/pact/power/`) -- Poll the Jetson Xavier INA3221
      sensor for real-time power draw. Feed readings into `fault/detector.py`
      `check_power()` and `check_thermal()`, which currently receive mocked `0.0`. This
      completes the fault detection loop.

- [ ] **CCSDS full packet framing** -- Add secondary headers, authentication fields, and
      CRC-16/CCITT to `src/pact/comms/ccsds.py`. Current payload uses `pickle.dumps` (interim).

- [ ] **Safe-mode exit** -- Implement ground command handling in `ops/main.py` to call
      `fault_process.exit_safe_mode()`. System currently cannot exit safe mode autonomously.

- [ ] **Process restart on crash** -- Add restart logic in `ops/main.py` for non-fault crashes.
      Currently a crashed subsystem transitions to safe mode with no recovery path.

- [ ] **Storage LRU eviction** -- Implement an eviction policy in `src/pact/storage/writer.py`
      so `STORAGE_FULL` does not permanently halt writes. Evict oldest frames by manifest
      timestamp.

- [ ] **TensorRT INT8 quantization** -- Implement `src/pact/model/quantize.py` (currently a
      stub). Required for Jetson inference latency targets.

- [ ] **Telemetry sensor HAL** -- Wire real thermal and power readings into
      `SystemHealthSnapshot`. Currently both fields are hardcoded `0.0`.
