"""Payload control: ingest vision, outer LQG, and inner rate PI (pure).

ingest_vision gates blobs, matches them by IoU, forms an area-weighted CoM, converts
that CoM to boresight-error degrees, and writes an EMA vision mailbox.
step_outer predicts the dual-axis Kalman filter, updates from the encoder, consumes
the mailbox through the rewind ring, runs the arbiter, and holds a rate reference r.
step_inner converts r to SI and delegates to the rate PI. step() is ingest plus one
outer tick so the frame-synchronous app path stays valid until the control thread
owns inner ticks.

Satisfies: REQ-AIML-GIMB-002, REQ-AIML-GIMB-006, REQ-AIML-GIMB-007, REQ-GIMB-HIGH-001.
"""

from __future__ import annotations

# stdlib
import math
from dataclasses import dataclass, replace

# third-party
import numpy as np

# internal
from flight.hal.interfaces import GimbalPosition
from flight.libs.config import ControllerConfig, GimbalConfig, SensorConfig
from flight.libs.messages import BlobMeta, InferenceResultMsg, TelemetryEventMsg
from flight.libs.types import FaultCode, GimbalCommandMode, GimbalState, MessageType, Ok
from flight.payload.gimbal import (
    INITIAL_RATE_SERVO_STATE,
    ArbiterState,
    GimbalArbiter,
    GimbalRequest,
    LqrController,
    RateServoState,
    apply_confidence_gate,
    apply_min_area_gate,
    area_weighted_com_px,
    boresight_error_deg,
    compute_control,
    rate_servo_step,
    reset_servo,
)
from flight.payload.tracking import (
    DualKalmanState,
    EmaFilterState,
    EstimatorRing,
    EstimatorSnapshot,
    KalmanFilter,
    apply_vision,
    ema_update,
    empty_ring,
    match_blobs,
    predict,
    push_snapshot,
    update_enc,
)


def _clip(value: float, limit: float) -> float:
    """Symmetric clip to [-limit, limit]."""
    return min(max(value, -limit), limit)


@dataclass(frozen=True, slots=True)
class ControlState:
    """Bundled control state threaded across ingest and inner/outer ticks.

    Attributes:
        arbiter: The gimbal FSM state.
        ema: The EMA centroid filter state (in boresight-error degrees).
        kalman: The dual-axis 4-state Kalman estimator.
        servo: Inner rate-PI state (SI units).
        ring: Outer-tick rewind snapshots.
        r_az_deg_s: Held azimuth rate reference (deg/s).
        r_el_deg_s: Held elevation rate reference (deg/s).
        seen_vision: True after the first accepted vision update.
        vis_z_az: Mailbox azimuth vision measurement (deg), or None.
        vis_z_el: Mailbox elevation vision measurement (deg), or None.
        vis_t_shutter: Mailbox capture time, or None.
        last_outer_time_s: Monotonic time of the last outer tick (0 before the first).
        vision_blobs: Matched blobs from the last ingest (for the arbiter).
        vision_mode_flags: Inference mode_flags from the last ingest.
        pending_vision_frame: True when ingest has not yet been consumed by step_outer.
    """

    arbiter: ArbiterState
    ema: EmaFilterState
    kalman: DualKalmanState
    servo: RateServoState
    ring: EstimatorRing
    r_az_deg_s: float
    r_el_deg_s: float
    seen_vision: bool
    vis_z_az: float | None
    vis_z_el: float | None
    vis_t_shutter: float | None
    last_outer_time_s: float
    vision_blobs: tuple[BlobMeta, ...]
    vision_mode_flags: int
    pending_vision_frame: bool


@dataclass(frozen=True)
class PayloadController:
    """Pure payload control core: vision ingest, outer LQG, inner rate PI.

    Attributes:
        cfg: ControllerConfig (gates, persistence, LQG, servo, Δt bounds).
        gimbal_cfg: GimbalConfig (inertia, damping, travel envelope).
        arbiter: The pure GimbalArbiter FSM.
        kf: The per-axis 4-state Kalman filter.
        lqr: The continuous-LQR control law.
        plane_width_px: Band-plane width in pixels (sensor width / 2).
        plane_height_px: Band-plane height in pixels (sensor height / 2).
        ifov_deg_per_px: Instantaneous field of view per band-plane pixel (degrees).
        j: Inertia estimate, shape (2, 2), kg m^2.
        b: Damping estimate, shape (2, 2), N m s / rad.
    """

    cfg: ControllerConfig
    gimbal_cfg: GimbalConfig
    arbiter: GimbalArbiter
    kf: KalmanFilter
    lqr: LqrController
    plane_width_px: int
    plane_height_px: int
    ifov_deg_per_px: float
    j: np.ndarray  # np.ndarray[float64, (2, 2)]
    b: np.ndarray  # np.ndarray[float64, (2, 2)]

    @staticmethod
    def from_config(
        cfg: ControllerConfig,
        sensor: SensorConfig,
        gimbal: GimbalConfig | None = None,
    ) -> PayloadController:
        """Build the immutable arbiter, Kalman filter, LQR, servo plant, and geometry.

        Inputs:
            cfg (ControllerConfig): Controller tuning.
            sensor (SensorConfig): Sensor geometry; the band plane is the mosaic halved,
                so plane_{width,height}_px = sensor.{width,height}_px // 2.
            gimbal (GimbalConfig | None): Inertia, damping, and travel. Defaults to
                GimbalConfig() when omitted.

        Outputs:
            PayloadController: A fully constructed pure control core.
        """
        gimbal_cfg = gimbal if gimbal is not None else GimbalConfig()
        return PayloadController(
            cfg=cfg,
            gimbal_cfg=gimbal_cfg,
            arbiter=GimbalArbiter(cfg),
            kf=KalmanFilter.from_config(cfg),
            lqr=LqrController.from_config(cfg),
            plane_width_px=sensor.width_px // 2,
            plane_height_px=sensor.height_px // 2,
            ifov_deg_per_px=sensor.ifov_deg_per_px,
            j=np.array(gimbal_cfg.J_kg_m2, dtype=np.float64).reshape(2, 2),
            b=np.array(gimbal_cfg.B_nms_per_rad, dtype=np.float64).reshape(2, 2),
        )

    def initial_state(self) -> ControlState:
        """Return the starting control state: IDLE arbiter, zeroed estimators, r = 0.

        Outputs:
            ControlState: IDLE arbiter, uninitialized EMA, zeroed Kalman and servo,
            empty rewind ring, r = 0, and no vision yet.
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
            servo=INITIAL_RATE_SERVO_STATE,
            ring=empty_ring(self.cfg.estimator_ring_len),
            r_az_deg_s=0.0,
            r_el_deg_s=0.0,
            seen_vision=False,
            vis_z_az=None,
            vis_z_el=None,
            vis_t_shutter=None,
            last_outer_time_s=0.0,
            vision_blobs=(),
            vision_mode_flags=0,
            pending_vision_frame=False,
        )

    def ingest_vision(
        self,
        state: ControlState,
        result: InferenceResultMsg,
        capture_monotonic_s: float,
    ) -> ControlState:
        """Gate blobs, match by IoU, form CoM, EMA-smooth, and write the vision mailbox.

        Inputs:
            state (ControlState): Prior control state.
            crop origin and scale from the inference result (identity after full-plane ingest).
            capture_monotonic_s (float): Shutter time of this frame (monotonic seconds).

        Outputs:
            ControlState: Updated EMA, mailbox, matched blobs, and pending-frame flag.
            Kalman, servo, ring, and r are unchanged.
        """
        cfg = self.cfg
        gated = apply_confidence_gate(result.blobs, cfg.confidence_gate)
        gated = apply_min_area_gate(gated, cfg.min_blob_area_px)
        matched = match_blobs(state.vision_blobs, tuple(gated), cfg.blob_iou_match_threshold)
        com = area_weighted_com_px(matched)
        if com is None:
            return replace(
                state,
                ema=EmaFilterState(centroid=(0.0, 0.0), initialized=False),
                vis_z_az=None,
                vis_z_el=None,
                vis_t_shutter=None,
                vision_blobs=matched,
                vision_mode_flags=result.mode_flags,
                pending_vision_frame=True,
                arbiter=replace(state.arbiter, tracked_blobs=matched),
            )
        error_deg = boresight_error_deg(
            com,
            result.crop_origin_px,
            result.scale_factor,
            self.plane_width_px,
            self.plane_height_px,
            self.ifov_deg_per_px,
        )
        ema = ema_update(state.ema, error_deg, cfg.ema_alpha)
        return replace(
            state,
            ema=ema,
            vis_z_az=ema.centroid[0],
            vis_z_el=ema.centroid[1],
            vis_t_shutter=capture_monotonic_s,
            vision_blobs=matched,
            vision_mode_flags=result.mode_flags,
            pending_vision_frame=True,
            arbiter=replace(state.arbiter, tracked_blobs=matched),
        )

    def step_outer(
        self,
        state: ControlState,
        gimbal_pos: GimbalPosition | None,
        now: float,
        dt: float,
        safe_commanded: bool,
        safe_cleared: bool,
    ) -> tuple[ControlState, GimbalRequest | None, list[TelemetryEventMsg], FaultCode | None]:
        """Run one outer LQG tick: predict, encoder update, rewind, LQR, arbiter.

        Inputs:
            state (ControlState): Prior control state.
            gimbal_pos (GimbalPosition | None): Encoder read, or None when unavailable.
            now (float): Monotonic seconds at this outer tick.
            dt (float): Time since the previous outer tick (seconds). Chunked at
                dt_outer_max_s. Values <= 0 skip the predict.
            safe_commanded (bool): True to latch SAFE and stow this tick.
            safe_cleared (bool): True to exit SAFE to IDLE this tick.

        Outputs:
            tuple[ControlState, GimbalRequest | None, list[TelemetryEventMsg], FaultCode | None]:
            (new_state, request, telemetry_events, fault). fault is always None on the
            live path (deadband and encoder-runaway gates are not applied).
        """
        cfg = self.cfg
        kalman = self._predict_chunked(state.kalman, state.r_az_deg_s, state.r_el_deg_s, dt)
        theta_az = gimbal_pos.az_deg if gimbal_pos is not None else None
        theta_el = gimbal_pos.el_deg if gimbal_pos is not None else None
        if theta_az is not None:
            enc_az = update_enc(self.kf, kalman.az, theta_az)
            if isinstance(enc_az, Ok):
                kalman = DualKalmanState(az=enc_az.value, el=kalman.el)
        if theta_el is not None:
            enc_el = update_enc(self.kf, kalman.el, theta_el)
            if isinstance(enc_el, Ok):
                kalman = DualKalmanState(az=kalman.az, el=enc_el.value)

        ring = push_snapshot(
            state.ring,
            EstimatorSnapshot(
                t=now,
                az=kalman.az,
                el=kalman.el,
                u_az=state.r_az_deg_s,
                u_el=state.r_el_deg_s,
                theta_enc_az=theta_az,
                theta_enc_el=theta_el,
            ),
        )
        seen_vision = state.seen_vision
        vis_z_az = state.vis_z_az
        vis_z_el = state.vis_z_el
        vis_t_shutter = state.vis_t_shutter
        if vis_z_az is not None and vis_z_el is not None and vis_t_shutter is not None:
            rewound = apply_vision(
                self.kf,
                ring,
                vis_z_az,
                vis_z_el,
                vis_t_shutter,
                now,
                state.r_az_deg_s,
                state.r_el_deg_s,
                kalman,
            )
            if rewound is not None:
                kalman = rewound
                seen_vision = True
            vis_z_az = None
            vis_z_el = None
            vis_t_shutter = None

        error_deg: tuple[float, float] | None = (
            state.ema.centroid if state.ema.initialized else None
        )
        filtered = InferenceResultMsg(
            msg_type=MessageType.INFERENCE_RESULT,
            timestamp_utc="1970-01-01T00:00:00.000Z",
            frame_id=0,
            mask=np.zeros((1, 1), dtype=np.float32),
            blobs=state.vision_blobs,
            model_version="",
            inference_ms=0.0,
            mode_flags=state.vision_mode_flags,
            crop_origin_px=(0, 0),
            scale_factor=1.0,
        )
        new_arbiter, request, telemetry = self.arbiter.step(
            state.arbiter,
            filtered,
            error_deg,
            now,
            safe_commanded,
            safe_cleared,
            dt=dt,
            has_new_frame=state.pending_vision_frame,
        )

        servo = state.servo
        r_az = 0.0
        r_el = 0.0
        if new_arbiter.gimbal_state is GimbalState.SAFE:
            servo = reset_servo(state.servo)
        elif new_arbiter.gimbal_state is GimbalState.SCAN:
            scan_enc_az = theta_az if theta_az is not None else 0.0
            scan_enc_el = theta_el if theta_el is not None else 0.0
            cmd_az = request.az_deg if request is not None else new_arbiter.scan_pan_deg
            cmd_el = request.el_deg if request is not None else 0.0
            r_az = _clip(cfg.scan_kp * (cmd_az - scan_enc_az), cfg.max_slew_rate_deg_per_s)
            r_el = _clip(cfg.scan_kp * (cmd_el - scan_enc_el), cfg.max_slew_rate_deg_per_s)
        elif new_arbiter.gimbal_state is GimbalState.TRACKING and seen_vision:
            u = compute_control(
                self.lqr,
                np.asarray(kalman.az.x, dtype=np.float64),
                np.asarray(kalman.el.x, dtype=np.float64),
            )
            r_az = float(u[0])
            r_el = float(u[1])
            if request is not None and request.mode is GimbalCommandMode.RATE:
                request = replace(request, az_deg=r_az, el_deg=r_el)
            else:
                request = GimbalRequest(
                    mode=GimbalCommandMode.RATE,
                    az_deg=r_az,
                    el_deg=r_el,
                    reason="tracking_target",
                )
        elif request is not None and request.mode is GimbalCommandMode.RATE:
            request = replace(request, az_deg=0.0, el_deg=0.0)

        new_state = replace(
            state,
            arbiter=new_arbiter,
            kalman=kalman,
            servo=servo,
            ring=ring,
            r_az_deg_s=r_az,
            r_el_deg_s=r_el,
            seen_vision=seen_vision,
            vis_z_az=vis_z_az,
            vis_z_el=vis_z_el,
            vis_t_shutter=vis_t_shutter,
            last_outer_time_s=now,
            pending_vision_frame=False,
        )
        return new_state, request, telemetry, None

    def step_inner(
        self,
        state: ControlState,
        gimbal_pos: GimbalPosition | None,
        dt: float,
    ) -> tuple[ControlState, tuple[float, float]]:
        """Advance the inner rate PI and return (new_state, (tau_az, tau_el) N·m).

        Inputs:
            state (ControlState): Prior control state (holds r in deg/s).
            gimbal_pos (GimbalPosition | None): Encoder read. None yields zero torque.
            dt (float): Inner-tick duration in seconds. Chunked at dt_inner_max_s.

        Outputs:
            (ControlState, (tau_az_nm, tau_el_nm)). Degrees convert to radians at this
            shell; the servo stays in SI.
        """
        if gimbal_pos is None or dt <= 0.0:
            return state, (0.0, 0.0)
        travel = self._travel_saturated(gimbal_pos)
        r_rad_s = (math.radians(state.r_az_deg_s), math.radians(state.r_el_deg_s))
        theta_rad = (math.radians(gimbal_pos.az_deg), math.radians(gimbal_pos.el_deg))
        servo = state.servo
        tau: tuple[float, float] = (0.0, 0.0)
        remaining = dt
        max_dt = self.cfg.dt_inner_max_s
        while remaining > 0.0:
            chunk = remaining if remaining <= max_dt else max_dt
            servo, tau = rate_servo_step(
                servo,
                r_rad_s,
                theta_rad,
                chunk,
                self.j,
                self.b,
                self.cfg.kp,
                self.cfg.ki,
                self.cfg.tau_max_nm,
                self.cfg.ym_lpf_s,
                travel,
            )
            remaining -= chunk
        return replace(state, servo=servo), tau

    def step(
        self,
        state: ControlState,
        result: InferenceResultMsg,
        now: float,
        gimbal_pos: GimbalPosition | None,
        safe_commanded: bool,
        safe_cleared: bool,
    ) -> tuple[ControlState, GimbalRequest | None, list[TelemetryEventMsg], FaultCode | None]:
        """Run ingest_vision then one step_outer (frame-synchronous adapter).

        Inputs:
            state (ControlState): The control state from the previous frame.
            result (InferenceResultMsg): The detection result.
            now (float): Monotonic seconds, supplied by the caller (never read here).
            gimbal_pos (GimbalPosition | None): Latest encoder read, or None.
            safe_commanded (bool): True to latch SAFE and stow this frame.
            safe_cleared (bool): True to exit SAFE to IDLE this frame.

        Outputs:
            tuple[ControlState, GimbalRequest | None, list[TelemetryEventMsg], FaultCode | None]:
            Same contract as step_outer.
        """
        ingested = self.ingest_vision(state, result, now)
        if ingested.last_outer_time_s == 0.0:
            dt = self.cfg.kalman_dt_s
        else:
            dt = max(0.0, now - ingested.last_outer_time_s)
        return self.step_outer(ingested, gimbal_pos, now, dt, safe_commanded, safe_cleared)

    def _predict_chunked(
        self,
        kalman: DualKalmanState,
        u_az: float,
        u_el: float,
        dt: float,
    ) -> DualKalmanState:
        """Predict both axes, splitting dt at dt_outer_max_s."""
        if dt <= 0.0:
            return kalman
        remaining = dt
        max_dt = self.cfg.dt_outer_max_s
        az = kalman.az
        el = kalman.el
        while remaining > 0.0:
            chunk = remaining if remaining <= max_dt else max_dt
            az = predict(self.kf, az, u_az, chunk)
            el = predict(self.kf, el, u_el, chunk)
            remaining -= chunk
        return DualKalmanState(az=az, el=el)

    def _travel_saturated(self, pos: GimbalPosition) -> tuple[bool, bool]:
        """Return True per axis when the encoder is at a travel stop."""
        g = self.gimbal_cfg
        eps = 1e-6
        az_sat = pos.az_deg <= g.az_min_deg + eps or pos.az_deg >= g.az_max_deg - eps
        el_sat = pos.el_deg <= g.el_min_deg + eps or pos.el_deg >= g.el_max_deg - eps
        return az_sat, el_sat
