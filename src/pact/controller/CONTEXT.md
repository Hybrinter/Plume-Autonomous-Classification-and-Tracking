# controller/ -- Agent Context

## Purpose

Gimbal arbiter state machine (IDLE/ACQUIRING/TRACKING/SCAN/SAFE), blob persistence
tracking, EMA centroid smoothing, deadband + rate-limit safety gates, and Kalman/LQR
targeting. The arbiter is the safety-critical core -- all other modules in this package
support it.

## Defining Design Decision

`GimbalArbiter.step(state, result, now)` is a pure function. No queue access, no I/O,
no side effects. The caller supplies the Unix timestamp (`now`) so behavior is fully
deterministic and time-independent in tests. All mutable state is in `ArbiterState`
(passed in, returned out). This design is from ADR-001.

## Invariants

- All four safety gates run in `process.py` *before* the arbiter is called. The arbiter
  sees only pre-filtered blobs and never needs to re-check gates.
- `PIXEL_TO_DEG = 0.04` (at 420 km altitude) is defined in `arbiter.py`. Import it --
  never redefine it.
- `match_blobs(prev_blobs, new_blobs, iou_threshold)` takes exactly 3 positional args
  and returns `tuple[BlobMeta, ...]`. It assigns new IDs internally (max existing + 1).

## Gotchas

- **EMA asymmetric initialization:** on the first frame, the EMA filter returns the raw
  centroid without smoothing (`initialized = False`). Frame 1 output != frame 2+ output
  for equivalent inputs.
- **Deadband is three-valued:** `< min_deadband_px` -> no command; `[min, max]` -> command
  issued; `> max_deadband_px` -> `Err(GIMBAL_RUNAWAY)`. It is not a simple pass/fail gate.
- `GimbalArbiter.__init__` requires `cfg: ControllerConfig` -- the no-argument constructor
  no longer exists.

## Phase II Gaps

- `LqrController.from_config()` silently falls back to proportional gain if the DARE
  solver fails. This violates the `Result[T,E]` contract -- should return `Err()`.
- `send_gimbal_command()` in `process.py` is a hardware stub -- real serial/CAN driver
  not yet integrated.
