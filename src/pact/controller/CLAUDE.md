# Controller Subsystem

## Purpose
Gimbal safety arbiter state machine, blob tracker, EMA filter, and safety gates.

## Satisfies
- REQ-AIML-GIMB-001, REQ-AIML-GIMB-002, REQ-AIML-GIMB-003, REQ-AIML-GIMB-004,
  REQ-AIML-GIMB-005, REQ-AIML-GIMB-006, REQ-AIML-GIMB-007, REQ-AIML-GIMB-008
- REQ-AIML-DATA-006, REQ-AIML-DATA-007, REQ-AIML-DATA-008, REQ-AIML-DATA-009
- REQ-GIMB-HIGH-001, REQ-GIMB-HIGH-002, REQ-GIMB-HIGH-003, REQ-GIMB-HIGH-004

## Owns (produces)
- `GimbalCommandMsg` — issued to the gimbal hardware interface after all safety gates pass
- `TelemetryEventMsg` — emitted on every arbiter state transition
- `HeartbeatMsg` — sent to the fault watchdog on each watchdog interval

## Consumes
- `InferenceResultMsg` — received from the inference process queue

## Key Invariants
- `GimbalArbiter` is a **pure function** — `step()` has no side effects, no I/O, no queue
  access. It maps `(ArbiterState, InferenceResultMsg, float) → (ArbiterState,
  Optional[GimbalCommandMsg], list[TelemetryEventMsg])`.
- `GimbalArbiter` holds **no queue references**. All queue interaction lives in `process.py`.
- All safety gates (`apply_confidence_gate`, `apply_min_area_gate`, `check_deadband`,
  `check_rate_limit`) run **before** the arbiter is called. The arbiter only sees
  pre-filtered blobs.
- The EMA filter state is threaded through `ArbiterState`; it is never mutated in place.
- See `controller/adr/ADR-001` for the rationale behind the pure-function arbiter design.
- `PIXEL_TO_DEG: float = 0.04` is defined in `arbiter.py` and imported by `process.py`. Do not
  redefine it elsewhere.
- `match_blobs(prev_blobs, new_blobs, iou_threshold)` takes exactly 3 positional arguments and
  returns `tuple[BlobMeta, ...]`. It assigns fresh IDs internally (max existing ID + 1). No
  external counter is needed or accepted.
- `ArbiterState` now has `scan_pan_deg: float = 0.0` as a field (added to track raster scan
  position).

## New Modules (added this session)

### `kalman.py` — 2D constant-velocity Kalman filter
- `KalmanState` (frozen dataclass): `x: object` (np.ndarray float64, shape (4,)), `P: object` (4×4 covariance)
- `KalmanFilter` (frozen dataclass): `F`, `H`, `Q`, `R` matrices; `from_config(cfg)` static factory
- `predict(kf, state) -> KalmanState`: x = F·x, P = F·P·Fᵀ + Q
- `update(kf, state, observation) -> Ok[KalmanState] | Err[FaultCode.GIMBAL_RUNAWAY]`
- State vector: [pan_deg, tilt_deg, pan_rate, tilt_rate]; observation: [pan_deg, tilt_deg]

### `lqr.py` — Discrete-time LQR controller
- `LqrController` (frozen dataclass): `K: object` (2×4 gain matrix), `max_slew_deg_s: float`
- `from_config(cfg) -> LqrController`: solves DARE via `scipy.linalg.solve_discrete_are`
- `compute_control(controller, state_error) -> np.ndarray`: u = −K·e, clamped to ±max_slew_deg_s

## Concurrency
`multiprocessing.Process` + `multiprocessing.Queue` — the controller is downstream of the
inference process (CPU-heavy) and must be isolated from the GIL to guarantee deterministic
scheduling of safety-critical gimbal commands.

## Known Gaps / TODOs
- `send_gimbal_command()` inside `process.py` is a **stub**. Physical gimbal hardware driver
  not yet integrated. Replace with the vendor serial/CAN API before flight integration.
- `GimbalArbiter.__init__` requires `cfg: ControllerConfig` — the no-argument constructor no
  longer exists. Always construct as `GimbalArbiter(controller_cfg)`.
- LQR `from_config()` silently falls back to proportional gain if the DARE solver fails. This
  violates the `Result[T,E]` contract — should be changed to return `Err()` in a future session.
- `KalmanFilter.update()` allocates `np.eye(4)` on every call (hot path). Pre-compute into
  `KalmanFilter` as a frozen field in a future session.
