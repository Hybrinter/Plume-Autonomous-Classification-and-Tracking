"""Payload control: composes tracking estimators and the gimbal FSM/law into one pure step.

Reproduces the per-inference-result orchestration the legacy controller process performed,
now in boresight-error degree space: confidence/area gates -> blob match -> boresight error
(via IFOV with crop back-projection) -> EMA smoothing -> Kalman predict/update -> deadband +
encoder-runaway safety gates (finally wired into the live path) -> gimbal arbiter FSM -> LQR
refinement. Pure: no I/O, no clock access (now is passed in). All state is threaded through
ControlState (frozen; replaced, never mutated).

The EMA and Kalman now estimate the target's boresight error in degrees, so the LQR setpoint
("target at boresight") is the zero vector and u = -K x needs no explicit subtraction. The
deadband suppresses RATE commands below the minimum displacement and escalates above the
maximum after max_deadband_strike_count strikes; STOW/ABSOLUTE requests are never suppressed
(safing and scan must always actuate). The encoder-runaway monitor compares the commanded rate
threaded from the previous frame against the measured encoder motion.

Satisfies: REQ-AIML-GIMB-002, REQ-AIML-GIMB-006, REQ-AIML-GIMB-007, REQ-GIMB-HIGH-001.
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass, replace

# third-party
import numpy as np

# internal
from flight.hal.interfaces import GimbalPosition
from flight.libs.config import ControllerConfig, SensorConfig
from flight.libs.messages import InferenceResultMsg, TelemetryEventMsg
from flight.libs.types import Err, FaultCode, GimbalCommandMode, GimbalState, Ok
from flight.payload.gimbal import (
    INITIAL_RUNAWAY_STATE,
    ArbiterState,
    GimbalArbiter,
    GimbalRequest,
    LqrController,
    RunawayState,
    apply_confidence_gate,
    apply_min_area_gate,
    boresight_error_deg,
    check_deadband,
    check_runaway,
    compute_control,
    target_displacement_px,
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


@dataclass(frozen=True, slots=True)
class ControlState:
    """Bundled control state threaded across frames (replaced each step, never mutated).

    Attributes:
        arbiter: The gimbal FSM state.
        ema: The EMA centroid filter state (in boresight-error degrees).
        kalman: The Kalman estimator state (in boresight-error degrees).
        runaway: The encoder-divergence runaway monitor state.
        deadband_strikes: Consecutive max-deadband violations not yet escalated.
        commanded_az_rate_deg_per_s: Azimuth rate commanded last frame (runaway input).
        commanded_el_rate_deg_per_s: Elevation rate commanded last frame (runaway input).
    """

    arbiter: ArbiterState
    ema: EmaFilterState
    kalman: KalmanState
    runaway: RunawayState
    deadband_strikes: int
    commanded_az_rate_deg_per_s: float
    commanded_el_rate_deg_per_s: float


@dataclass(frozen=True)
class PayloadController:
    """Pure payload control core composing the tracking estimators and gimbal FSM/law.

    Attributes:
        cfg: ControllerConfig (gates, persistence, slew/runaway tuning).
        arbiter: The pure GimbalArbiter FSM.
        kf: The constant-velocity Kalman filter.
        lqr: The discrete-LQR control law.
        plane_width_px: Band-plane width in pixels (sensor width / 2); boresight x = w/2.
        plane_height_px: Band-plane height in pixels (sensor height / 2); boresight y = h/2.
        ifov_deg_per_px: Instantaneous field of view per band-plane pixel (degrees).
    """

    cfg: ControllerConfig
    arbiter: GimbalArbiter
    kf: KalmanFilter
    lqr: LqrController
    plane_width_px: int
    plane_height_px: int
    ifov_deg_per_px: float

    @staticmethod
    def from_config(cfg: ControllerConfig, sensor: SensorConfig) -> PayloadController:
        """Build the immutable arbiter, Kalman filter, LQR, and pointing geometry from config.

        Inputs:
            cfg (ControllerConfig): Controller tuning.
            sensor (SensorConfig): Sensor geometry; the band plane is the mosaic halved,
                so plane_{width,height}_px = sensor.{width,height}_px // 2.

        Outputs:
            PayloadController: A fully constructed pure control core.
        """
        return PayloadController(
            cfg=cfg,
            arbiter=GimbalArbiter(cfg),
            kf=KalmanFilter.from_config(cfg),
            lqr=LqrController.from_config(cfg),
            plane_width_px=sensor.width_px // 2,
            plane_height_px=sensor.height_px // 2,
            ifov_deg_per_px=sensor.ifov_deg_per_px,
        )

    def initial_state(self) -> ControlState:
        """Return the starting control state: IDLE arbiter, uninitialized EMA, zeroed Kalman.

        Outputs:
            ControlState: IDLE arbiter, uninitialized EMA, zeroed Kalman, fresh runaway
            monitor, no deadband strikes, and zero commanded rates.
        """
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
            runaway=INITIAL_RUNAWAY_STATE,
            deadband_strikes=0,
            commanded_az_rate_deg_per_s=0.0,
            commanded_el_rate_deg_per_s=0.0,
        )

    def step(
        self,
        state: ControlState,
        result: InferenceResultMsg,
        now: float,
        gimbal_pos: GimbalPosition | None,
        safe_commanded: bool,
        safe_cleared: bool,
    ) -> tuple[ControlState, GimbalRequest | None, list[TelemetryEventMsg], FaultCode | None]:
        """Run one pure control step in boresight-error degree space.

        Inputs:
            state (ControlState): The control state from the previous frame.
            result (InferenceResultMsg): The detection result (carries crop_origin_px and
                scale_factor for boresight back-projection).
            now (float): Monotonic seconds, supplied by the caller (never read here).
            gimbal_pos (GimbalPosition | None): Latest encoder read for the runaway monitor,
                or None when unavailable.
            safe_commanded (bool): True to latch SAFE and stow this frame.
            safe_cleared (bool): True to exit SAFE to IDLE this frame.

        Outputs:
            tuple[ControlState, GimbalRequest | None, list[TelemetryEventMsg], FaultCode | None]:
            (new_state, request, telemetry_events, fault). fault is GIMBAL_RUNAWAY when the
            deadband strike limit is reached or the encoder diverges from the commanded rate.

        Notes:
            The deadband gate suppresses RATE requests below min_deadband_px and escalates
            above max_deadband_px after max_deadband_strike_count strikes; STOW/ABSOLUTE are
            never suppressed. The LQR refines the RATE request only when the EMA is
            initialized (u = -K x with the boresight-zero setpoint).
        """
        cfg = self.cfg
        gated = apply_confidence_gate(result.blobs, cfg.confidence_gate)
        gated = apply_min_area_gate(gated, cfg.min_blob_area_px)
        matched = match_blobs(
            state.arbiter.tracked_blobs, tuple(gated), cfg.blob_iou_match_threshold
        )

        error_deg: tuple[float, float] | None = None
        displacement: float | None = None
        if matched:
            error_deg = boresight_error_deg(
                matched[0].centroid_raw,
                result.crop_origin_px,
                result.scale_factor,
                self.plane_width_px,
                self.plane_height_px,
                self.ifov_deg_per_px,
            )
            displacement = target_displacement_px(
                matched[0].centroid_raw,
                result.crop_origin_px,
                result.scale_factor,
                self.plane_width_px,
                self.plane_height_px,
            )

        if error_deg is not None:
            ema = ema_update(state.ema, error_deg, cfg.ema_alpha)
        else:
            ema = EmaFilterState(centroid=(0.0, 0.0), initialized=False)

        kalman = predict(self.kf, state.kalman)
        if ema.initialized:
            obs = np.array([ema.centroid[0], ema.centroid[1]], dtype=np.float64)
            updated = update(self.kf, kalman, obs)
            if isinstance(updated, Ok):
                kalman = updated.value

        # Deadband + max-displacement strike gate (finally wired; REQ-AIML-GIMB-006/007).
        fault: FaultCode | None = None
        deadband_strikes = 0
        suppress_rate_command = False
        if displacement is not None:
            db = check_deadband(displacement, cfg.min_deadband_px, cfg.max_deadband_px)
            if isinstance(db, Err):
                deadband_strikes = state.deadband_strikes + 1
                suppress_rate_command = True
                if deadband_strikes >= cfg.max_deadband_strike_count:
                    fault = FaultCode.GIMBAL_RUNAWAY
            elif not db.value:
                suppress_rate_command = True

        filtered = replace(result, blobs=matched)
        new_arbiter, request, telemetry = self.arbiter.step(
            state.arbiter, filtered, error_deg, now, safe_commanded, safe_cleared
        )

        if request is not None and request.mode is GimbalCommandMode.RATE and ema.initialized:
            u = compute_control(self.lqr, np.asarray(kalman.x, dtype=np.float64))
            limit = cfg.max_slew_rate_deg_per_s
            request = replace(
                request,
                az_deg=float(min(max(u[0], -limit), limit)),
                el_deg=float(min(max(u[1], -limit), limit)),
            )
        if request is not None and request.mode is GimbalCommandMode.RATE and suppress_rate_command:
            request = None

        cmd_az, cmd_el = state.commanded_az_rate_deg_per_s, state.commanded_el_rate_deg_per_s
        rate_mode_active = cmd_az != 0.0 or cmd_el != 0.0
        new_runaway, runaway_fault = check_runaway(
            state.runaway,
            gimbal_pos,
            cmd_az,
            cmd_el,
            rate_mode_active,
            cfg.runaway_rate_tolerance_deg_per_s,
            cfg.runaway_strike_count,
        )
        if fault is None:
            fault = runaway_fault

        is_rate_request = request is not None and request.mode is GimbalCommandMode.RATE
        next_cmd_az = request.az_deg if is_rate_request and request is not None else 0.0
        next_cmd_el = request.el_deg if is_rate_request and request is not None else 0.0
        new_state = ControlState(
            arbiter=new_arbiter,
            ema=ema,
            kalman=kalman,
            runaway=new_runaway,
            deadband_strikes=deadband_strikes,
            commanded_az_rate_deg_per_s=next_cmd_az,
            commanded_el_rate_deg_per_s=next_cmd_el,
        )
        return new_state, request, telemetry, fault
