# Phase 5d -- Payload Controller (pure control composition) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `flight.payload.control` -- a single pure `PayloadController.step()` that reproduces the legacy controller orchestration (confidence/area gates -> blob match -> EMA -> Kalman predict/update -> gimbal arbiter FSM -> LQR refinement) over a bundled `ControlState`, with thorough tests, `mypy --strict`/`ruff` clean, gates green.

**Architecture:** `flight/payload/control.py` composes the already-migrated `flight.payload.tracking` (filter/kalman/tracker) and `flight.payload.gimbal` (arbiter/lqr/safety) into one pure control core. `ControlState` is a frozen dataclass bundling the three threaded states (`ArbiterState`, `EmaFilterState`, `KalmanState`). `PayloadController.from_config(cfg)` builds the immutable arbiter/Kalman/LQR once; `step(state, result, now)` is pure (no I/O, no clock access -- `now` is passed in) and returns `(new_state, command_or_none, telemetry_events)`. This is the payload's pure control core; the I/O app shell that drives it (acquire -> preprocess -> infer -> control -> command/bus) is Phase 5e.

**Tech Stack:** Python 3.14, numpy, frozen dataclasses, pytest, mypy --strict, ruff, import-linter.

---

## Context for the implementer

- This reproduces the orchestration in `src/pact/controller/process.py` (the loop body, ~lines 147-214). The sequence per inference result, threading `ControlState`:
  1. `apply_confidence_gate(result.blobs, cfg.confidence_gate)` then `apply_min_area_gate(., cfg.min_blob_area_px)`.
  2. `match_blobs(state.arbiter.tracked_blobs, tuple(gated), cfg.blob_iou_match_threshold)`.
  3. EMA: if matched, `ema = ema_update(state.ema, matched[0].centroid_raw, cfg.ema_alpha)`; else reset `EmaFilterState(centroid=(0.0, 0.0), initialized=False)`.
  4. Kalman: `kalman = predict(kf, state.kalman)` ALWAYS; if `ema.initialized`, `obs = [ema.centroid * PIXEL_TO_DEG]` and `upd = update(kf, kalman, obs)`; on `Ok`, adopt `upd.value`.
  5. `filtered = replace(result, blobs=matched)`; `new_arbiter, command, telemetry = arbiter.step(state.arbiter, filtered, now)`.
  6. LQR refine: if `command is not None and ema.initialized`, `u = compute_control(lqr, kalman.x)`, `cmd_interval_s = 1.0 / max(cfg.retarget_rate_limit_hz, 1e-6)`, `command = replace(command, az_delta_deg=u[0]*cmd_interval_s, el_delta_deg=u[1]*cmd_interval_s)`.
  `PIXEL_TO_DEG = 0.04` (same value as in arbiter.py).
- Source signatures already migrated and available: `ArbiterState`, `GimbalArbiter(cfg).step(state, result, now) -> (ArbiterState, GimbalCommandMsg | None, list[TelemetryEventMsg])`; `KalmanFilter.from_config(cfg)`, `KalmanFilter.initial_state(pan_deg, tilt_deg)`, `predict(kf, state)`, `update(kf, state, obs) -> Ok[KalmanState] | Err[FaultCode]`; `EmaFilterState`, `ema_update(state, centroid, alpha)`; `match_blobs(prev, new, iou_threshold)`; `LqrController.from_config(cfg)`, `compute_control(controller, state_error) -> np.ndarray`; `apply_confidence_gate`, `apply_min_area_gate`. Confirm each against the source before finalizing.
- MUST pass `uv run mypy packages` (strict) and `uv run ruff check packages`. Do NOT modify `src/pact/`. Stage only named files. Commit locally; no push. ASCII only. Tests annotated `-> None`. No `.importlinter` change (payload internal + payload->libs allowed).

## File structure (created in this phase)

```
packages/flight/src/flight/payload/control.py        # PayloadController + ControlState
packages/flight/tests/test_payload_controller.py     # NEW
```

---

## Task 1: `flight.payload.control`

**Files:** `packages/flight/src/flight/payload/control.py`

- [ ] **Step 1: Create `control.py`**

```python
"""Payload control: composes tracking estimators and the gimbal FSM/law into one pure step.

Reproduces the per-inference-result orchestration the legacy controller process performed:
confidence/area gates -> blob match -> EMA smoothing -> Kalman predict/update -> gimbal
arbiter FSM -> LQR refinement. Pure: no I/O, no clock access (now is passed in). All state is
threaded through ControlState (frozen; replaced, never mutated).
"""

from dataclasses import dataclass, replace

import numpy as np

from flight.libs.config import ControllerConfig
from flight.libs.messages import GimbalCommandMsg, InferenceResultMsg, TelemetryEventMsg
from flight.libs.types import GimbalState, Ok
from flight.payload.gimbal import (
    ArbiterState,
    GimbalArbiter,
    LqrController,
    apply_confidence_gate,
    apply_min_area_gate,
    compute_control,
)
from flight.payload.tracking import (
    EmaFilterState,
    KalmanFilter,
    KalmanState,
    ema_update,
    match_blobs,
    predict,
    update,
)

PIXEL_TO_DEG = 0.04


@dataclass(frozen=True, slots=True)
class ControlState:
    """Bundled control state threaded across frames (replaced each step, never mutated)."""

    arbiter: ArbiterState
    ema: EmaFilterState
    kalman: KalmanState


@dataclass(frozen=True)
class PayloadController:
    """Pure payload control core composing the tracking estimators and gimbal FSM/law."""

    cfg: ControllerConfig
    arbiter: GimbalArbiter
    kf: KalmanFilter
    lqr: LqrController

    @staticmethod
    def from_config(cfg: ControllerConfig) -> "PayloadController":
        """Build the immutable arbiter, Kalman filter, and LQR controller from config."""
        return PayloadController(
            cfg=cfg,
            arbiter=GimbalArbiter(cfg),
            kf=KalmanFilter.from_config(cfg),
            lqr=LqrController.from_config(cfg),
        )

    def initial_state(self) -> ControlState:
        """Return the starting control state: IDLE arbiter, uninitialized EMA, zeroed Kalman."""
        return ControlState(
            arbiter=ArbiterState(
                gimbal_state=GimbalState.IDLE,
                tracked_blobs=(),
                idle_duration_s=0.0,
                last_command_time=0.0,
                current_target_id=None,
            ),
            ema=EmaFilterState(centroid=(0.0, 0.0), initialized=False),
            kalman=KalmanFilter.initial_state(0.0, 0.0),
        )

    def step(
        self,
        state: ControlState,
        result: InferenceResultMsg,
        now: float,
    ) -> tuple[ControlState, GimbalCommandMsg | None, list[TelemetryEventMsg]]:
        """Run one pure control step, returning the new state, an optional command, and events.

        Args:
            state: The control state from the previous frame.
            result: The detection result for this frame.
            now: Wall-clock seconds (Unix), supplied by the caller (never read here).

        Returns:
            (new_state, gimbal_command_or_none, telemetry_events).
        """
        cfg = self.cfg
        gated = apply_confidence_gate(result.blobs, cfg.confidence_gate)
        gated = apply_min_area_gate(gated, cfg.min_blob_area_px)
        matched = match_blobs(state.arbiter.tracked_blobs, tuple(gated), cfg.blob_iou_match_threshold)

        if matched:
            ema = ema_update(state.ema, matched[0].centroid_raw, cfg.ema_alpha)
        else:
            ema = EmaFilterState(centroid=(0.0, 0.0), initialized=False)

        kalman = predict(self.kf, state.kalman)
        if ema.initialized:
            obs = np.array(
                [ema.centroid[0] * PIXEL_TO_DEG, ema.centroid[1] * PIXEL_TO_DEG],
                dtype=np.float64,
            )
            updated = update(self.kf, kalman, obs)
            if isinstance(updated, Ok):
                kalman = updated.value

        filtered = replace(result, blobs=matched)
        new_arbiter, command, telemetry = self.arbiter.step(state.arbiter, filtered, now)

        if command is not None and ema.initialized:
            state_error = np.asarray(kalman.x, dtype=np.float64)
            u = compute_control(self.lqr, state_error)
            cmd_interval_s = 1.0 / max(cfg.retarget_rate_limit_hz, 1e-6)
            command = replace(
                command,
                az_delta_deg=float(u[0]) * cmd_interval_s,
                el_delta_deg=float(u[1]) * cmd_interval_s,
            )

        new_state = ControlState(arbiter=new_arbiter, ema=ema, kalman=kalman)
        return new_state, command, telemetry
```

Note: confirm `arbiter.step` returns exactly `(ArbiterState, GimbalCommandMsg | None, list[TelemetryEventMsg])` and that `match_blobs` returns `tuple[BlobMeta, ...]` (so `matched[0].centroid_raw` is valid). Adjust the local typing only if strict mypy requires it; do not change behavior.

- [ ] **Step 2: Verify and commit**

Run: `uv run mypy packages` -> Success. `uv run ruff check packages` -> passed.
```bash
git add packages/flight/src/flight/payload/control.py
git commit -m "feat(payload): add PayloadController pure control composition"
```

---

## Task 2: `test_payload_controller.py`

**Files:** `packages/flight/tests/test_payload_controller.py`

- [ ] **Step 1: Write the test**

```python
"""Tests for the PayloadController pure control composition."""

import numpy as np

from flight.libs.config import ControllerConfig
from flight.libs.messages import BlobMeta, InferenceResultMsg
from flight.libs.types import GimbalState, MessageType
from flight.payload.control import ControlState, PayloadController


def _result_with_blob(frame_id: int, *, with_blob: bool) -> InferenceResultMsg:
    """Build an InferenceResultMsg, optionally carrying one strong, stable blob."""
    mask = np.zeros((16, 16), dtype=np.float32)
    blobs: tuple[BlobMeta, ...] = ()
    if with_blob:
        blobs = (
            BlobMeta(
                blob_id=1,
                bbox=(100, 100, 150, 150),
                centroid_raw=(125.0, 125.0),
                pixel_area=200,
                mean_confidence=0.85,
                persistence_count=1,
            ),
        )
    return InferenceResultMsg(
        msg_type=MessageType.INFERENCE_RESULT,
        timestamp_utc="2026-06-01T00:00:00.000Z",
        frame_id=frame_id,
        mask=mask,
        blobs=blobs,
        model_version="test",
        inference_ms=0.0,
        mode_flags=0,
    )


def test_initial_state_is_idle() -> None:
    """The controller starts IDLE with no tracked blobs."""
    controller = PayloadController.from_config(ControllerConfig())
    state = controller.initial_state()
    assert state.arbiter.gimbal_state is GimbalState.IDLE
    assert state.arbiter.tracked_blobs == ()


def test_no_detection_stays_idle_no_command() -> None:
    """With no blobs, the controller stays IDLE and issues no command."""
    controller = PayloadController.from_config(ControllerConfig())
    state = controller.initial_state()
    state, command, _events = controller.step(state, _result_with_blob(1, with_blob=False), now=1.0)
    assert state.arbiter.gimbal_state is GimbalState.IDLE
    assert command is None


def test_persistent_blob_progresses_to_tracking_and_commands() -> None:
    """A stable, strong blob over several frames drives the FSM to TRACKING and issues a command."""
    controller = PayloadController.from_config(ControllerConfig())
    state = controller.initial_state()
    now = 0.0
    saw_command = False
    for frame_id in range(1, 9):
        now += 1.0
        state, command, _events = controller.step(state, _result_with_blob(frame_id, with_blob=True), now)
        if command is not None:
            saw_command = True
    assert state.arbiter.gimbal_state is GimbalState.TRACKING
    assert saw_command
```

Note: confirm `BlobMeta` and `InferenceResultMsg` fields against `flight.libs.messages`. If the FSM's exact end state differs (e.g. requires more frames or a different persistence threshold), read the migrated arbiter to set the right frame count / assertion -- the goal is to demonstrate IDLE -> ... -> TRACKING with a command, faithfully to the migrated FSM.

- [ ] **Step 2: Verify and commit**

Run: `uv run pytest packages/flight/tests/test_payload_controller.py -v` -> PASS. `uv run mypy packages` -> Success. `uv run ruff check packages` -> passed.
```bash
git add packages/flight/tests/test_payload_controller.py
git commit -m "test(payload): add PayloadController tests"
```

---

## Task 3: Full gate sweep

**Files:** none (verification)

- [ ] **Step 1: Run every gate exactly as CI does**

```bash
uv run ruff check packages
uv run ruff format --check packages
uv run mypy packages
uv run lint-imports
uv run pytest packages -m "not e2e"
```
Expected: all pass; `lint-imports` 7 contracts kept; pytest includes the new controller tests.

- [ ] **Step 2: If `ruff format --check packages` flags new files**, run `uv run ruff format packages`, re-check, commit:
```bash
git add packages
git commit -m "style: ruff-format payload control"
```
(Skip if nothing needed reformatting.)

---

## Risks & notes

- **Faithful orchestration:** the step sequence must match `src/pact/controller/process.py` exactly (gate -> match -> EMA -> Kalman predict/[update] -> arbiter -> LQR-refine). The Kalman update is conditional on `ema.initialized`; LQR refine is conditional on `command is not None and ema.initialized`.
- **mypy strict + numpy:** `kalman.x` is typed `object`; wrap with `np.asarray(..., dtype=np.float64)` before `compute_control` (as shown). Keep the minimal typing.
- **Test robustness:** the persistence-to-TRACKING test depends on `acquire_persistence_frames` (default 3) and the rate limit (0.5 Hz; `now` advances 1.0s/frame). If the migrated FSM needs a different frame count to reach TRACKING, adjust the loop bound, not the FSM.
- **Deferred:** the I/O app shell (acquire/preprocess/infer/command/bus loop) is Phase 5e; storage of big artifacts waits on the storage service.

## Self-review (against the spec)

- **Spec coverage (Section 7 tracking controller):** the FSM + estimator (Kalman/EMA) + control law (LQR) are composed into one pure `step`, exactly as the spec's tracking controller intends; it remains a pure core (the gimbal arbiter is the resolver inside it).
- **Placeholder scan:** no TBD/TODO; full code given for `control.py` and the tests, with verify-against-source notes.
- **Type/name consistency:** `PayloadController`/`ControlState`/`step`/`from_config`/`initial_state` are used identically in the module and tests; all composed symbols match the names exported by `flight.payload.tracking` and `flight.payload.gimbal`.
