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

`process_frame(raw: MosaicFrame, state, now, slew_rate_deg_per_s)` runs these pure functions
in order:

1. `calibrate_mosaic(mosaic, calib)` -- bad-pixel repair then `(raw - dark) / flat` on the
   raw mosaic plane. Returns `Err(FRAME_MALFORMED)` on shape mismatch or
   `Err(INFERENCE_NAN)` if non-finite output would propagate.
2. `separate_bands(calibrated)` -- 2x2 CFA -> `(4, H/2, W/2)` float32 band planes in
   `SensorConfig.mosaic_layout` cell order. Returns `Err(FRAME_MALFORMED)` on odd dims.
3. `normalize_dn(planes, bit_depth)` -- `clip(dn / (2**bit_depth - 1), 0, 1)` float32.
4. `select_bands(normalized, layout, band_names)` -- reorder planes into
   `InferenceConfig.input_bands` order. Returns `Err(FRAME_MALFORMED)` on name mismatch.
5. `compute_quality_flags(selected, exposure_us, slew_rate_deg_per_s, ifov_deg_per_px, ...)`.

**Calibration injection:** `PayloadApp` holds `calib: MosaicCalibration` (constructed by the
composition root from `calibration_io.load_calibration` on flight, or
`calibration_io.build_identity_calibration` for SIL). Identity calibration is SIL-only. A
bad `calibration_dir` raises `SystemExit` in `main()` before the scheduler starts.

**Slew-rate smear input:** `run()` derives the slew rate from consecutive
`gimbal.read_position()` diffs over the elapsed time since the previous frame. The first
frame and any failed encoder read use 0.0 (smear gate degrades gracefully). `ifov_deg_per_px`
comes from `SensorConfig` (0.04 deg/px, formerly the hardcoded `PIXEL_TO_DEG`).

## `PayloadController.step` is the pure composition root

`control.py:step()` reproduces the *entire* legacy controller loop as one pure function
(no I/O, no clock read): confidence gate -> min-area gate -> `match_blobs` ->
EMA -> Kalman (predict **always**, update **only if EMA initialized**) ->
`arbiter.step` -> LQR refine. State is threaded through frozen `ControlState`, replaced
never mutated. Faithful-reproduction quirks that look like bugs but are intentional:

- Kalman **update** runs only when `ema.initialized`; a lost target resets EMA to
  uninitialized (so the filter coasts on predict-only).
- LQR refinement **overrides** the arbiter's az/el deltas, but **only** when a command
  already exists *and* `ema.initialized` -- otherwise the arbiter's raw pixel-derived
  deltas stand.

## Arbiter is the FSM resolver; tracking is a command source

`GimbalArbiter.step` is the pure FSM (IDLE/ACQUIRING/TRACKING/SCAN/SAFE) that *decides
whether and in what mode* to command. The tracking stack (EMA/Kalman/LQR) only refines the
*magnitude* of an already-decided TRACKING command. Keep the arbiter free of I/O and
estimator math.

## Gotchas

- **Monotonic `now`, deltas only.** `app.py` passes `clock.monotonic_s()` as `now`, while
  several docstrings (control.py, arbiter.py) say "Unix timestamp." This is safe *only*
  because `now` is consumed exclusively as rate-limit/interval differences. Never compare
  `now` against a wall-clock value or persist it as an absolute time.
- **No crop in the loop.** Frames are passed at full resolution (`crop_origin_px=(0,0)`,
  `scale_factor=1.0`). `crop_to_roi`/`backproject_pixel` are exported from `preprocess` but
  deliberately unused here pending the pointing phase.
- **Detector backend is swappable** (`ScriptedDetector` for SIL/tests, `OnnxDetector` for
  flight). `onnxruntime` is imported lazily in `OnnxDetector.__init__`, so importing the
  payload package never requires it. The composition root picks the backend.
- **Band planes are half-resolution.** A 512x512 mosaic yields (4, 256, 256) band planes --
  inherent to the 2x2 CFA geometry. `InferenceConfig.input_height_px` and
  `input_width_px` must equal `sensor_cfg.height_px // 2` and `sensor_cfg.width_px // 2`;
  `from_config` validates this at startup and raises `ValueError` on mismatch.
