"""Payload control: composes tracking estimators and the gimbal FSM/law into one pure step.

Reproduces the per-inference-result orchestration the legacy controller process performed:
confidence/area gates -> blob match -> EMA smoothing -> Kalman predict/update -> gimbal
arbiter FSM -> LQR refinement. Pure: no I/O, no clock access (now is passed in). All state is
threaded through ControlState (frozen; replaced, never mutated).
"""

from __future__ import annotations

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
    def from_config(cfg: ControllerConfig) -> PayloadController:
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

        Notes:
            Faithfully reproduces the legacy controller loop (process.py): the Kalman update
            is applied only when the EMA is initialized; the LQR refinement overrides the
            arbiter's deltas only when a command exists and the EMA is initialized.
        """
        cfg = self.cfg
        gated = apply_confidence_gate(result.blobs, cfg.confidence_gate)
        gated = apply_min_area_gate(gated, cfg.min_blob_area_px)
        matched = match_blobs(
            state.arbiter.tracked_blobs, tuple(gated), cfg.blob_iou_match_threshold
        )

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
