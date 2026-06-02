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
  deliberately unused here; raw frames must already match the identity-calibration shape.
- **Detector backend is swappable** (`ScriptedDetector` for SIL/tests, `OnnxDetector` for
  flight). `onnxruntime` is imported lazily in `OnnxDetector.__init__`, so importing the
  payload package never requires it. The composition root picks the backend.
- **`PIXEL_TO_DEG = 0.04` is duplicated** in `control.py` and `gimbal/arbiter.py`; keep them
  in sync if you retune the FOV constant.
- **Identity calibration is a placeholder** (zero dark / unit flat from `InferenceConfig`
  shape), so `apply_calibration` is currently a no-op pending real sensor characterization.
