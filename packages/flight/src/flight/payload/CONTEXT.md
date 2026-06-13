# `payload` Subsystem Context

Non-obvious, cross-cutting context for the payload app. Documents design decisions and
invariants you cannot derive by reading the individual files or their docstrings.

---

## Science pipeline is one app, not a process chain

`app.py` collapses the legacy imaging + inference + controller *processes* into internal
stages of a single in-process loop. Preprocessing -> detection -> control run as plain
function calls inside `process_frame()`.

- **Co-location invariant:** `ProcessedFrameMsg` is constructed locally and handed directly
  to `detector.detect()`. It is **never published on the bus** and never crosses a queue --
  this avoids per-frame pickling of the `(C, H, W)` tensor. Only `InferenceResultMsg`,
  arbiter telemetry/commands, faults, and heartbeats reach the bus. Do not put
  `ProcessedFrameMsg` on a queue.

## Ingest pipeline order (as of 2026-06-09 mosaic contract)

`process_frame(raw: MosaicFrame, state, now, slew_rate_deg_per_s, gimbal_pos, safe_commanded,
safe_cleared)` runs these pure functions in order:

1. `calibrate_mosaic(mosaic, calib)` -- bad-pixel repair then `(raw - dark) / flat` on the
   raw mosaic plane. Returns `Err(FRAME_MALFORMED)` on shape mismatch or
   `Err(INFERENCE_NAN)` if non-finite output would propagate.
2. `separate_bands(calibrated)` -- 2x2 CFA -> `(4, H/2, W/2)` float32 band planes in
   `SensorConfig.mosaic_layout` cell order. Returns `Err(FRAME_MALFORMED)` on odd dims.
3. `normalize_dn(planes, bit_depth)` -- `clip(dn / (2**bit_depth - 1), 0, 1)` float32.
4. `select_bands(normalized, layout, band_names)` -- reorder planes into
   `InferenceConfig.input_bands` order. Returns `Err(FRAME_MALFORMED)` on name mismatch.
5. `compute_quality_flags(selected, exposure_us, slew_rate_deg_per_s, ifov_deg_per_px, ...)`.

**Mode-dependent ROI (as of 2026-06-11 pointing phase, ADR 0008).** After quality flags run on
the *full* band plane, `process_frame` selects the inference tensor by arbiter state:
- **Search mode** (not TRACKING, or estimator not yet initialized): the full plane is decimated
  to the inference input size -- `tensor = selected[:, ::factor, ::factor]`,
  `crop_origin_px = (0, 0)`, `scale_factor = 1/factor`.
- **TRACKING mode** (estimator initialized): a full-resolution `crop_to_roi` window is cropped
  around the Kalman-estimated boresight-error target -- `scale_factor = 1.0`, `crop_origin_px`
  the clamped top-left. `from_config` requires the plane to be an equal integer multiple of the
  inference input on both axes (uniform decimation); it raises `ValueError` otherwise.

**Calibration injection:** `PayloadApp` holds `calib: MosaicCalibration` (constructed by the
composition root from `calibration_io.load_calibration` on flight, or
`calibration_io.build_identity_calibration` for SIL). Identity calibration is SIL-only. A
bad `calibration_dir` raises `SystemExit` in `main()` before the scheduler starts.

**Slew-rate smear input:** `run()` derives the slew rate from consecutive
`gimbal.read_position()` diffs over the elapsed time since the previous frame. The first
frame and any failed encoder read use 0.0 (smear gate degrades gracefully). `ifov_deg_per_px`
comes from `SensorConfig` (0.02 deg/px at the 1024 geometry; see ADR 0008).

## `PayloadController.step` is the pure composition root (error-space, as of ADR 0008)

`control.py:step(state, result, now, gimbal_pos, safe_commanded, safe_cleared)` runs the whole
control loop as one pure function (no I/O, no clock read) and returns
`(ControlState, GimbalRequest | None, list[TelemetryEventMsg], FaultCode | None)`:
confidence gate -> min-area gate -> `match_blobs` -> `boresight_error_deg` /
`target_displacement_px` -> EMA -> Kalman (predict **always**, update **only if EMA
initialized**) -> deadband/strike gate -> `arbiter.step` -> LQR refine -> runaway check.
State is threaded through frozen `ControlState`, replaced never mutated. Non-obvious points:

- **Estimators live in boresight-error degree space.** The EMA and Kalman estimate the target's
  angular error from boresight (not absolute pixels), so the LQR setpoint is the zero vector
  (`u = -K x`). The **published slew rate is `-u = K x`** -- the LQR's `u` acts on the error
  velocity in the plant model, so the gimbal must slew the opposite way to shrink the error.
  Getting this sign wrong slews *away* from the target (caught by the Task 9 command-direction
  SIL test).
- Kalman **update** runs only when `ema.initialized`; a lost target resets EMA to uninitialized
  (the filter coasts on predict-only). LQR refinement applies only to a RATE request when
  `ema.initialized`; STOW/ABSOLUTE are never refined.
- **The safety gates are wired now.** `check_deadband` suppresses RATE below `min_deadband_px`
  and escalates to `GIMBAL_RUNAWAY` above `max_deadband_px` after `max_deadband_strike_count`
  strikes; `check_runaway` (encoder-vs-commanded rate divergence, RATE mode only) is the second
  fault source. STOW/ABSOLUTE requests are never deadband-suppressed (safing and scan must always
  actuate). The commanded RATE is threaded out in `ControlState` as the runaway monitor's input
  for the *next* frame.

## Arbiter is the FSM resolver; tracking is a command source

`GimbalArbiter.step(state, result, error_deg, now, safe_commanded, safe_cleared)` is the pure FSM
(IDLE/ACQUIRING/TRACKING/SCAN/SAFE) that *decides whether and in what mode* to command, emitting a
typed `GimbalRequest` (RATE in TRACKING, ABSOLUTE for the reversing SCAN raster, STOW on SAFE
entry) -- never a bus message. The tracking stack (EMA/Kalman/LQR) only refines the *magnitude* of
an already-decided TRACKING command. Keep the arbiter free of I/O and estimator math.

- **SAFE latches in the arbiter.** A drained `safe_commanded` (or any non-zero `result.mode_flags`)
  transitions to SAFE and returns a STOW request; while SAFE, no further requests are produced and
  blobs are ignored until `safe_cleared` (a ground `ModeChangeMsg(non-SAFE)`) returns it to IDLE.
- **TRACKING release hysteresis:** a blobless TRACKING frame increments `miss_count`; release to
  IDLE only at `release_persistence_frames`; any blob resets it. The SCAN raster is ABSOLUTE and
  reverses at +-30 deg.

## Gotchas

- **Monotonic `now`, deltas only.** `app.py` passes `clock.monotonic_s()` as `now`, while
  several docstrings (control.py, arbiter.py) say "Unix timestamp." This is safe *only*
  because `now` is consumed exclusively as rate-limit/interval differences. Never compare
  `now` against a wall-clock value or persist it as an absolute time.
- **Pointing error is boresight-relative (ADR 0008).** `pointing.boresight_error_deg` inverts the
  preprocess crop/decimation transform, measures the offset *from the plane center*, and scales by
  `SensorConfig.ifov_deg_per_px`. Sign convention: image +x -> +az, image +y (down) -> -el. The
  old `PIXEL_TO_DEG` constant and absolute-centroid math are deleted -- a live reference to
  `PIXEL_TO_DEG`, `az_delta_deg`, or `send_command` is a bug.
- **ROI crop is live now.** `crop_to_roi`/`backproject_pixel` are used in `process_frame` (see the
  mode-dependent ROI note above): decimated full plane in search, full-res Kalman-centered crop in
  TRACKING. They are no longer dead exports.
- **Detector backend is swappable** (`ScriptedDetector` for SIL/tests, `OnnxDetector` for
  flight). `onnxruntime` is imported lazily in `OnnxDetector.__init__`, so importing the
  payload package never requires it. The composition root picks the backend.
- **Band planes are half-resolution.** A 1024x1024 mosaic yields (4, 512, 512) band planes --
  inherent to the 2x2 CFA geometry. The plane (`sensor.{height,width}_px // 2`) must be at least
  the inference input and an equal integer multiple of it on both axes (for uniform search-mode
  decimation); `from_config` validates this at startup and raises `ValueError` on mismatch.
