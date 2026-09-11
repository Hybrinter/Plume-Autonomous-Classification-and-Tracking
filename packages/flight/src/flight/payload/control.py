"""Payload control: cascaded elevation inner/outer loops (pure cores).

Inner: encoder ring -> polynomial y_m -> PI + computed torque.
Outer: CoG intersect -> co-rotating predictor -> residual KF -> r, plus the
TRACKING/REWIND/SAFE arbiter. STOW/HOME/GOTO write r through the position loop
into the same inner PI.

Pure: no I/O, no bus, no clock reads. Time, encoder angle, ISS state, and vision
samples are arguments.

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001, REQ-GIMB-HIGH-003.
"""

from __future__ import annotations

# stdlib
import math
from dataclasses import dataclass, replace

# internal
from flight.libs.config import (
    ControllerConfig,
    EphemerisConfig,
    GimbalConfig,
    PreprocessingConfig,
    SensorConfig,
)
from flight.libs.messages import BlobMeta, InferenceResultMsg, TelemetryEventMsg
from flight.libs.types import FaultCode, GimbalCommandMode, GimbalState, MessageType
from flight.payload.gimbal import (
    ArbiterState,
    GimbalArbiter,
    GimbalRequest,
    apply_confidence_gate,
    apply_min_area_gate,
    fit_rate_timed,
    inner_step,
    intersect_cog,
    outer_rate,
    pinhole_error_rad,
    position_rate,
    predict_los,
)
from flight.payload.tracking import (
    EncoderSample,
    NominalRateSample,
    PredictorReferenceChange,
    ResidualFilter,
    ResidualHistory,
    ResidualState,
    VisionObservation,
    estimate_at,
    match_blobs,
    submit_event,
)


@dataclass(frozen=True, slots=True)
class VisionSample:
    """One vision packet for the outer loop (shell queue payload).

    Attributes:
        t_s: Monotonic shutter time.
        frame_id: Stable frame identifier used to deduplicate delayed observations.
        z_v: Elevation boresight error in radians, or None when no blob.
        p_cog: Band-plane centroid, or None when no blob.
        exposure_us: Live frame exposure.
        blobs: Gated, matched blobs (empty on a miss).
        mode_flags: Inference mode_flags for SAFE latching.
        iss: ISS state at shutter, or None when ephemeris is dead.
        theta_g_rad: Encoder angle interpolated at shutter time, or None when
            the shared encoder stream does not bracket the shutter.
    """

    t_s: float
    frame_id: str
    z_v: float | None
    p_cog: tuple[float, float] | None
    exposure_us: float
    blobs: tuple[BlobMeta, ...]
    mode_flags: int
    iss: IssSample | None
    theta_g_rad: float | None = None


@dataclass(frozen=True, slots=True)
class IssSample:
    """ISS ECI state passed into the outer step (from the ephemeris HAL).

    Attributes:
        r_m: Position meters ECI.
        v_m_s: Inertial velocity m/s ECI.
        utc_s: UTC seconds for Earth rotation.
    """

    r_m: tuple[float, float, float]
    v_m_s: tuple[float, float, float]
    utc_s: float


@dataclass(frozen=True, slots=True)
class ControlState:
    """Bundled control state threaded across inner and outer ticks.

    Attributes:
        arbiter: Gimbal FSM state.
        residual: Latest two-state residual Kalman estimate.
        residual_history: Timestamped residual events and stable posterior anchor.
        encoder_ring: Encoder elevations in radians, oldest to newest.
        integrator: Inner PI integrator.
        r_cog_ecef_m: Last good CoG Earth point, ECEF meters.
        r_rad_s: Last rate reference.
        y_m: Last encoder-rate estimate, rad/s.
        last_inner_s: Monotonic time of the last inner step, or None if not started.
        last_outer_s: Monotonic time of the last outer step, or None if not started.
        last_theta_enc_rad: Last encoder sample, radians, or None.
        integrity_freeze_strikes: Consecutive encoder-freeze inner ticks.
        integrity_lock_strikes: Consecutive lock-fight inner ticks.
        lock_theta_ref_rad: Encoder elevation latched at lock engage, or None.
        lock_ref_s: Monotonic seconds of that latch, or None.
        last_exposure_us: Last live exposure (REWIND smear cap).
        pose_mode: STOW/HOME/ABSOLUTE while the position loop is active.
        pose_el_deg: Position-loop target elevation, degrees.
        last_tau_nm: Last inner torque, N·m.
        last_theta_los: Last predictor elevation, rad.
        last_omega_t_nom: Last co-rotating elevation rate, rad/s.
        last_omega_az_nom: Last unactuated optical-azimuth rate, rad/s.
        last_omega_scene_el: Last elevation scene rate used for smear, rad/s.
        last_e_az: Unactuated optical azimuth error, rad (telemetry only).
    """

    arbiter: ArbiterState
    residual: ResidualState
    residual_history: ResidualHistory
    encoder_ring: tuple[float, ...]
    encoder_timestamp_ring: tuple[float, ...]
    integrator: float
    r_cog_ecef_m: tuple[float, float, float] | None
    r_rad_s: float
    y_m: float
    last_inner_s: float | None
    last_outer_s: float | None
    last_theta_enc_rad: float | None
    integrity_freeze_strikes: int
    integrity_lock_strikes: int
    lock_theta_ref_rad: float | None
    lock_ref_s: float | None
    last_exposure_us: float
    pose_mode: GimbalCommandMode | None
    pose_el_deg: float
    last_tau_nm: float
    last_theta_los: float
    last_omega_t_nom: float
    last_omega_az_nom: float
    last_omega_scene_el: float
    last_e_az: float


@dataclass(frozen=True, slots=True)
class InnerTick:
    """Outputs of one inner_step on the controller.

    Attributes:
        state: Updated ControlState.
        tau_nm: Torque command, N·m.
    """

    state: ControlState
    tau_nm: float


@dataclass(frozen=True, slots=True)
class OuterTick:
    """Outputs of one outer_step on the controller.

    Attributes:
        state: Updated ControlState.
        request: Pose request (STOW on SAFE entry), or None.
        telemetry: Compact pointing plus arbiter transition events.
        fault: Integrity trip from the inner path is published by the shell.
    """

    state: ControlState
    request: GimbalRequest | None
    telemetry: list[TelemetryEventMsg]
    fault: FaultCode | None


@dataclass(frozen=True)
class PayloadController:
    """Pure cascaded elevation controller.

    Attributes:
        cfg: ControllerConfig.
        gimbal: GimbalConfig.
        eph: EphemerisConfig (WGS-84, Earth rate, epoch).
        preprocessing: PreprocessingConfig (smear pixel budget).
        arbiter: GimbalArbiter.
        residual_filt: ResidualFilter.
        plane_width_px, plane_height_px: Band-plane size.
        pixel_pitch_m, focal_m: Pinhole geometry.
        ifov_band_deg_per_px: Band IFOV for the smear cap.
    """

    cfg: ControllerConfig
    gimbal: GimbalConfig
    eph: EphemerisConfig
    preprocessing: PreprocessingConfig
    arbiter: GimbalArbiter
    residual_filt: ResidualFilter
    plane_width_px: int
    plane_height_px: int
    pixel_pitch_m: float
    focal_m: float
    ifov_band_deg_per_px: float

    @staticmethod
    def from_config(
        cfg: ControllerConfig,
        sensor: SensorConfig,
        gimbal: GimbalConfig,
        eph: EphemerisConfig | None = None,
        preprocessing: PreprocessingConfig | None = None,
    ) -> PayloadController:
        """Build the immutable controller from typed config slices.

        Inputs:
            cfg: Controller tuning.
            sensor: Mosaic geometry and optics.
            gimbal: Plant, envelopes, encoder.
            eph: WGS-84 and Earth-rate constants; defaults to EphemerisConfig().
            preprocessing: Smear pixel budget; defaults to PreprocessingConfig().

        Outputs:
            PayloadController: Fully constructed pure core.
        """
        eph_cfg = eph if eph is not None else EphemerisConfig()
        prep = preprocessing if preprocessing is not None else PreprocessingConfig()
        return PayloadController(
            cfg=cfg,
            gimbal=gimbal,
            eph=eph_cfg,
            preprocessing=prep,
            arbiter=GimbalArbiter(cfg.arbiter, gimbal),
            residual_filt=ResidualFilter.from_config(cfg.residual, cfg.outer.dt_s),
            plane_width_px=sensor.width_px // 2,
            plane_height_px=sensor.height_px // 2,
            pixel_pitch_m=2.0 * sensor.pixel_um * 1.0e-6,
            focal_m=sensor.focal_length_mm * 1.0e-3,
            ifov_band_deg_per_px=sensor.ifov_band_deg_per_px,
        )

    def initial_state(self) -> ControlState:
        """Cold TRACKING arbiter, zero residual, empty encoder ring, r=0.

        Outputs:
            ControlState: Starting state. last_inner_s and last_outer_s are None.
        """
        return ControlState(
            arbiter=ArbiterState(
                gimbal_state=GimbalState.TRACKING,
                tracked_blobs=(),
                current_target_id=None,
                miss_count=0,
            ),
            residual=self.residual_filt.initial_state(),
            residual_history=self.residual_filt.initial_history(),
            encoder_ring=(),
            encoder_timestamp_ring=(),
            integrator=0.0,
            r_cog_ecef_m=None,
            r_rad_s=0.0,
            y_m=0.0,
            last_inner_s=None,
            last_outer_s=None,
            last_theta_enc_rad=None,
            integrity_freeze_strikes=0,
            integrity_lock_strikes=0,
            lock_theta_ref_rad=None,
            lock_ref_s=None,
            last_exposure_us=0.0,
            pose_mode=None,
            pose_el_deg=0.0,
            last_tau_nm=0.0,
            last_theta_los=0.0,
            last_omega_t_nom=0.0,
            last_omega_az_nom=0.0,
            last_omega_scene_el=0.0,
            last_e_az=0.0,
        )

    def ingest_inference(
        self,
        state: ControlState,
        result: InferenceResultMsg,
        t_s: float,
        exposure_us: float,
        iss: IssSample | None = None,
        theta_g_rad: float | None = None,
    ) -> tuple[ControlState, VisionSample]:
        """Gate and match blobs; build a vision sample. Does not step the loops.

        Inputs:
            state: Current control state (tracked blobs for IoU).
            result: Detector output.
            t_s: Monotonic shutter time.
            exposure_us: Live exposure.
            iss: ISS state at shutter, or None.

        Outputs:
            tuple[ControlState, VisionSample]: State with updated tracked-blob
            ancestry only in the sample; the arbiter still owns mode.
        """
        gated = apply_confidence_gate(result.blobs, self.cfg.vision.confidence_gate)
        gated = apply_min_area_gate(gated, self.cfg.vision.min_blob_area_px)
        matched = match_blobs(
            state.arbiter.tracked_blobs, tuple(gated), self.cfg.vision.blob_iou_match_threshold
        )
        z_v: float | None = None
        p_cog: tuple[float, float] | None = None
        e_az = state.last_e_az
        if matched:
            # All accepted components form one visible-plume aggregate.  Do not
            # select a component by ID or order: each component centroid is
            # weighted by its accepted pixel area, which is equivalent to the
            # union-pixel centroid for disjoint connected components.
            total_area = sum(blob.pixel_area for blob in matched)
            p_cog = (
                sum(blob.pixel_area * blob.centroid_raw[0] for blob in matched) / total_area,
                sum(blob.pixel_area * blob.centroid_raw[1] for blob in matched) / total_area,
            )
            e_az, z_v = pinhole_error_rad(
                p_cog,
                self.plane_width_px,
                self.plane_height_px,
                self.pixel_pitch_m,
                self.focal_m,
            )
        sample = VisionSample(
            t_s=t_s,
            frame_id=str(result.frame_id),
            z_v=z_v,
            p_cog=p_cog,
            exposure_us=exposure_us,
            blobs=matched,
            mode_flags=result.mode_flags,
            iss=iss,
            theta_g_rad=theta_g_rad,
        )
        return replace(state, last_e_az=e_az), sample

    def inner_step(
        self,
        state: ControlState,
        now: float,
        theta_enc_rad: float,
        dt_s: float | None = None,
        encoder_timestamp_s: float | None = None,
        locked: bool = False,
        safe_latched: bool = False,
    ) -> InnerTick:
        """One inner tick: push encoder, fit y_m, PI + computed torque.

        Inputs:
            state: Current control state.
            now: Monotonic seconds of this tick.
            theta_enc_rad: Encoder elevation, radians.
            dt_s: Inner period; defaults to cfg.inner.dt_s.
            locked: Launch lock engaged (freeze I; caller writes τ=0).
            safe_latched: Use the stow position loop instead of tracking r.

        Outputs:
            InnerTick: Updated state and torque.
        """
        dt = self.cfg.inner.dt_s if dt_s is None else dt_s
        sample_s = now if encoder_timestamp_s is None else encoder_timestamp_s
        ring = state.encoder_ring + (theta_enc_rad,)
        timestamp_ring = state.encoder_timestamp_ring + (sample_s,)
        max_n = self.cfg.inner.rate_fit_n
        if len(ring) > max_n:
            ring = ring[-max_n:]
            timestamp_ring = timestamp_ring[-max_n:]
        y_m = fit_rate_timed(
            ring, timestamp_ring, self.cfg.inner.rate_fit_n, self.cfg.inner.rate_fit_degree
        )
        el_deg = math.degrees(theta_enc_rad)
        at_sci_min = el_deg <= self.gimbal.el_science_min_deg + 1e-9
        at_sci_max = el_deg >= self.gimbal.el_science_max_deg - 1e-9
        stopped = (
            el_deg <= self.gimbal.el_hw_min_deg + 1e-9 or el_deg >= self.gimbal.el_hw_max_deg - 1e-9
        )
        if safe_latched or state.pose_mode is not None:
            pose_el = state.pose_el_deg if state.pose_mode is not None else self.gimbal.stow_el_deg
            r = position_rate(
                math.radians(pose_el),
                theta_enc_rad,
                self.cfg.position.K_pos,
                math.radians(self.cfg.position.r_max_deg_per_s),
            )
            if safe_latched:
                r = position_rate(
                    math.radians(self.gimbal.stow_el_deg),
                    theta_enc_rad,
                    self.cfg.position.K_pos,
                    math.radians(self.cfg.position.r_max_deg_per_s),
                )
        else:
            r = state.r_rad_s
            max_decel = self.gimbal.tau_max_nm / self.gimbal.J_kg_m2
            guard = math.radians(self.cfg.integrity.science_boundary_guard_deg)
            if r > 0.0:
                remaining = max(
                    0.0,
                    math.radians(self.gimbal.el_science_max_deg) - guard - theta_enc_rad,
                )
                r = min(
                    r,
                    math.sqrt(2.0 * max_decel * remaining),
                    self.cfg.inner.kp * remaining,
                )
            elif r < 0.0:
                remaining = max(
                    0.0,
                    theta_enc_rad - math.radians(self.gimbal.el_science_min_deg) - guard,
                )
                r = max(
                    r,
                    -math.sqrt(2.0 * max_decel * remaining),
                    -self.cfg.inner.kp * remaining,
                )
        if locked:
            r = 0.0
        at_bound = (at_sci_min and r < 0.0) or (at_sci_max and r > 0.0)
        result = inner_step(
            r,
            y_m,
            state.integrator,
            dt,
            self.gimbal.J_kg_m2,
            self.gimbal.B_nms_per_rad,
            self.cfg.inner.kp,
            self.cfg.inner.ki,
            self.gimbal.tau_max_nm,
            stopped or at_bound,
            locked=locked,
        )
        new_state = replace(
            state,
            encoder_ring=ring,
            encoder_timestamp_ring=timestamp_ring,
            integrator=result.integrator,
            r_rad_s=r,
            y_m=y_m,
            last_inner_s=now,
            last_theta_enc_rad=theta_enc_rad,
            last_tau_nm=result.tau_nm,
        )
        return InnerTick(state=new_state, tau_nm=0.0 if locked else result.tau_nm)

    def outer_step(
        self,
        state: ControlState,
        now: float,
        encoder: EncoderSample,
        vision: VisionSample | None,
        iss: IssSample | None,
        safe_commanded: bool,
        safe_cleared: bool,
        dt_s: float | None = None,
        timestamp_utc: str = "",
        reference_change: PredictorReferenceChange | None = None,
        detailed_plant: bool = True,
    ) -> OuterTick:
        """One outer tick: timestamped encoder history, predictor, and rate reference.

        Inputs:
            state: Current control state.
            now: Monotonic seconds of this tick.
            encoder: Valid timestamped, unwrapped encoder sample for this tick.
            vision: Dequeued vision sample, or None on coast.
            iss: ISS ECI state, or None (omega_t_nom = 0).
            safe_commanded, safe_cleared: FDIR SAFE flags.
            dt_s: Outer period; defaults to cfg.outer.dt_s.
            timestamp_utc: ISO stamp for pointing telemetry (empty skips the event).
            reference_change: Explicit predictor-reference replacement, if any.

        Outputs:
            OuterTick: Updated state, optional STOW request, telemetry.
        """
        del dt_s  # Detailed-plant cadence is retained by inner_step only.
        theta_g_rad = encoder.angle_rad
        el_deg = math.degrees(theta_g_rad)
        blobs = vision.blobs if vision is not None else ()
        mode_flags = vision.mode_flags if vision is not None else 0
        new_arbiter, request, events = self.arbiter.step(
            state.arbiter,
            blobs,
            now,
            safe_commanded,
            safe_cleared,
            el_deg,
            mode_flags,
            vision_updated=vision is not None,
            observation_t_s=vision.t_s if vision is not None else None,
            timestamp_utc=timestamp_utc,
        )

        pose_mode = state.pose_mode
        pose_el = state.pose_el_deg
        if request is not None:
            pose_mode = request.mode
            if request.mode is GimbalCommandMode.STOW:
                pose_el = self.gimbal.stow_el_deg
            elif request.mode is GimbalCommandMode.HOME:
                pose_el = self.gimbal.home_el_deg
            else:
                pose_el = request.el_deg
        if new_arbiter.gimbal_state is GimbalState.SAFE:
            pose_mode = GimbalCommandMode.STOW
            pose_el = self.gimbal.stow_el_deg
        elif pose_mode is not None and safe_cleared:
            pose_mode = None

        residual = state.residual
        history = state.residual_history
        vision_disposition = "none"
        omega_az = state.last_omega_az_nom
        if new_arbiter.gimbal_state is GimbalState.SAFE:
            if vision is not None and vision.exposure_us > 0.0:
                exposure_us = vision.exposure_us
            else:
                exposure_us = state.last_exposure_us
            r_cog = state.r_cog_ecef_m
            omega_t_nom = 0.0
            theta_los = state.last_theta_los
        else:
            if safe_cleared:
                residual = self.residual_filt.initial_state()
                history = self.residual_filt.initial_history(
                    t_s=encoder.t_s,
                    encoder_angle_rad=encoder.angle_rad,
                    encoder_endpoint_variance_rad2=encoder.angle_variance_rad2,
                )
            if vision is not None and vision.exposure_us > 0.0:
                exposure_us = vision.exposure_us
            else:
                exposure_us = state.last_exposure_us
            r_cog = state.r_cog_ecef_m
            shutter_iss = vision.iss if vision is not None else None
            shutter_theta = vision.theta_g_rad if vision is not None else None
            if (
                vision is not None
                and vision.p_cog is not None
                and shutter_iss is not None
                and shutter_theta is not None
            ):
                inter = intersect_cog(
                    vision.p_cog,
                    shutter_theta,
                    shutter_iss.r_m,
                    shutter_iss.v_m_s,
                    shutter_iss.utc_s,
                    self.eph.epoch_utc_s,
                    self.eph.omega_earth_rad_s,
                    self.eph.wgs84_a_m,
                    self.eph.wgs84_f,
                    self.plane_width_px,
                    self.plane_height_px,
                    self.pixel_pitch_m,
                    self.focal_m,
                    r_cog,
                    self.cfg.predictor.cog_height_m,
                )
                if inter.hit and inter.r_cog_ecef_m is not None:
                    r_cog = inter.r_cog_ecef_m

            omega_t_nom = 0.0
            theta_los = 0.0
            omega_az = 0.0
            if r_cog is not None and iss is not None:
                theta_los, omega_t_nom, omega_az = predict_los(
                    iss.utc_s,
                    iss.r_m,
                    iss.v_m_s,
                    r_cog,
                    self.eph.omega_earth_rad_s,
                    self.eph.epoch_utc_s,
                )

            history, _ = submit_event(history, encoder, now_s=now)
            nominal = NominalRateSample(
                sample_id=f"nominal:{encoder.sample_id}",
                t_s=encoder.t_s,
                rate_rad_s=omega_t_nom,
            )
            history, _ = submit_event(history, nominal, now_s=now)
            if reference_change is not None:
                history, _ = submit_event(history, reference_change, now_s=now)
            if vision is not None and vision.z_v is not None:
                observation = VisionObservation(
                    frame_id=vision.frame_id,
                    t_s=vision.t_s,
                    error_rad=vision.z_v,
                    measurement_variance_rad2=self.residual_filt.r_v,
                )
                history, _ = submit_event(history, observation, now_s=now)
            estimate = estimate_at(history, self.residual_filt, encoder.t_s)
            residual = estimate.state
            history = estimate.history
            if vision is not None:
                matching = [
                    item.disposition.value
                    for item in estimate.dispositions
                    if item.event_id == vision.frame_id
                ]
                if matching:
                    vision_disposition = matching[-1]

        # Vision establishes aggregate liveness; navigation only contributes an
        # optional nominal-rate prediction. This permits startup and tracking
        # through an ephemeris outage.
        live = new_arbiter.aggregate_live
        e_hat = float(residual.x[0])
        omega_res = float(residual.x[1])
        omega_scene_el = omega_t_nom + omega_res

        if pose_mode is not None:
            r = position_rate(
                math.radians(pose_el),
                theta_g_rad,
                self.cfg.position.K_pos,
                math.radians(self.cfg.position.r_max_deg_per_s),
            )
        else:
            max_decel = self.gimbal.tau_max_nm / self.gimbal.J_kg_m2 if detailed_plant else math.inf
            rate_loop_bandwidth = self.cfg.inner.kp if detailed_plant else math.inf
            r = outer_rate(
                omega_t_nom,
                omega_res,
                e_hat,
                self.cfg.outer.Kp,
                new_arbiter.gimbal_state,
                live,
                theta_g_rad,
                math.radians(self.gimbal.el_science_max_deg),
                math.radians(self.gimbal.max_hw_slew_rate_deg_per_s),
                exposure_us,
                self.preprocessing.max_motion_smear_px,
                self.ifov_band_deg_per_px,
                math.radians(self.gimbal.el_science_min_deg),
                max_decel,
                rate_loop_bandwidth,
            )

        new_state = replace(
            state,
            arbiter=new_arbiter,
            residual=residual,
            residual_history=history,
            r_cog_ecef_m=r_cog,
            r_rad_s=r,
            last_outer_s=now,
            last_exposure_us=exposure_us,
            pose_mode=pose_mode,
            pose_el_deg=pose_el,
            last_theta_los=theta_los,
            last_omega_t_nom=omega_t_nom,
            last_omega_az_nom=omega_az,
            last_omega_scene_el=omega_scene_el,
        )
        if timestamp_utc:
            checkpoint = history.checkpoint
            span_s = max(0.0, encoder.t_s - checkpoint.t_s)
            anchor_angle = checkpoint.encoder_angle_rad
            net_displacement = 0.0 if anchor_angle is None else encoder.angle_rad - anchor_angle
            events.append(
                TelemetryEventMsg(
                    msg_type=MessageType.TELEMETRY_EVENT,
                    timestamp_utc=timestamp_utc,
                    subsystem="payload",
                    event_name="pointing",
                    payload={
                        "e": e_hat,
                        "r": r,
                        "tau": state.last_tau_nm,
                        "omega_t_nom": omega_t_nom,
                        "omega_az": omega_az,
                        "omega_t_res": omega_res,
                        "omega_t_total": omega_t_nom + omega_res,
                        "y_m": state.y_m,
                        "P00": float(residual.P[0, 0]),
                        "P01": float(residual.P[0, 1]),
                        "P10": float(residual.P[1, 0]),
                        "P11": float(residual.P[1, 1]),
                        "encoder_anchor_t_s": checkpoint.t_s,
                        "encoder_endpoint_id": encoder.sample_id,
                        "encoder_endpoint_t_s": encoder.t_s,
                        "encoder_net_displacement_rad": net_displacement,
                        "encoder_sample_age_s": max(0.0, now - encoder.t_s),
                        "encoder_span_s": span_s,
                        "process_q_e_rad2": self.residual_filt.q_a * span_s**3 / 3.0,
                        "process_q_rate_rad2_s2": self.residual_filt.q_a * span_s,
                        "encoder_endpoint_covariance_rad2": (
                            checkpoint.encoder_endpoint_variance_rad2 + encoder.angle_variance_rad2
                        ),
                        "reversal_covariance_per_event_rad2": (
                            self.residual_filt.reversal_variance_rad2
                        ),
                        "vision_event_id": vision.frame_id if vision is not None else "",
                        "vision_shutter_s": vision.t_s if vision is not None else -1.0,
                        "vision_arrival_s": now if vision is not None else -1.0,
                        "vision_disposition": vision_disposition,
                    },
                )
            )
        return OuterTick(state=new_state, request=request, telemetry=events, fault=None)
