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

## Concurrency
`multiprocessing.Process` + `multiprocessing.Queue` — the controller is downstream of the
inference process (CPU-heavy) and must be isolated from the GIL to guarantee deterministic
scheduling of safety-critical gimbal commands.

## Known Gaps / TODOs
- `send_gimbal_command()` inside `process.py` is a **stub**. Physical gimbal hardware driver
  not yet integrated. Replace with the vendor serial/CAN API before flight integration.
- `GimbalArbiter.step()` body is `...` (stub). Full state machine logic to be implemented
  once hardware-in-the-loop testing begins.
- The SCAN state slew pattern (raster vs. circular) is not yet specified — placeholder only.
