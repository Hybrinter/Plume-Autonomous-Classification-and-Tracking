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
    fit_rate,
    inner_step,
    intersect_cog,
    outer_rate,
    pinhole_error_rad,
    position_rate,
    predict_los,
)
from flight.payload.tracking import (
    ResidualFilter,
    ResidualSnapshot,
    ResidualState,
    match_blobs,
    predict as residual_predict,
    push_snapshot,
    rewind_update,
)


@dataclass(frozen=True, slots=True)
class VisionSample:
    """One vision packet for the outer loop (shell queue payload).

    Attributes:
        t_s: Monotonic shutter time.
        z_v: Elevation boresight error in radians, or None when no blob.
        p_cog: Band-plane centroid, or None when no blob.
        exposure_us: Live frame exposure.
        blobs: Gated, matched blobs (empty on a miss).
        mode_flags: Inference mode_flags for SAFE latching.
    """

    t_s: float
    z_v: float | None
    p_cog: tuple[float, float] | None
    exposure_us: float
    blobs: tuple[BlobMeta, ...]
    mode_flags: int


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
        residual: Two-state residual Kalman state.
        encoder_ring: Encoder elevations in radians, oldest to newest.
        integrator: Inner PI integrator.
        r_cog_ecef_m: Last good CoG Earth point, ECEF meters.
        r_rad_s: Last rate reference.
        y_m: Last encoder-rate estimate, rad/s.
        snapshots: Residual rewind ring, oldest first.
        last_inner_s: Monotonic time of the last inner step.
        last_outer_s: Monotonic time of the last outer step.
        last_exposure_us: Last live exposure (REWIND smear cap).
        pose_mode: STOW/HOME/ABSOLUTE while the position loop is active.
        pose_el_deg: Position-loop target elevation, degrees.
        last_tau_nm: Last inner torque, N·m.
        last_theta_los: Last predictor elevation, rad.
        last_omega_t_nom: Last co-rotating rate, rad/s.
        last_e_az: Unactuated optical azimuth error, rad (telemetry only).
    """

    arbiter: ArbiterState
    residual: ResidualState
    encoder_ring: tuple[float, ...]
    integrator: float
    r_cog_ecef_m: tuple[float, float, float] | None
    r_rad_s: float
    y_m: float
    snapshots: tuple[ResidualSnapshot, ...]
    last_inner_s: float
    last_outer_s: float
    last_exposure_us: float
    pose_mode: GimbalCommandMode | None
    pose_el_deg: float
    last_tau_nm: float
    last_theta_los: float
    last_omega_t_nom: float
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
        fault: Unused (no runaway); always None.
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
            arbiter=GimbalArbiter(cfg, gimbal),
            residual_filt=ResidualFilter.from_config(cfg),
            plane_width_px=sensor.width_px // 2,
            plane_height_px=sensor.height_px // 2,
            pixel_pitch_m=2.0 * sensor.pixel_um * 1.0e-6,
            focal_m=sensor.focal_length_mm * 1.0e-3,
            ifov_band_deg_per_px=sensor.ifov_band_deg_per_px,
        )

    def initial_state(self) -> ControlState:
        """Cold TRACKING arbiter, zero residual, empty encoder ring, r=0.

        Outputs:
            ControlState: Starting state. last_inner_s and last_outer_s are 0.0.
        """
        return ControlState(
            arbiter=ArbiterState(
                gimbal_state=GimbalState.TRACKING,
                tracked_blobs=(),
                current_target_id=None,
                miss_count=0,
            ),
            residual=self.residual_filt.initial_state(),
            encoder_ring=(),
            integrator=0.0,
            r_cog_ecef_m=None,
            r_rad_s=0.0,
            y_m=0.0,
            snapshots=(),
            last_inner_s=0.0,
            last_outer_s=0.0,
            last_exposure_us=0.0,
            pose_mode=None,
            pose_el_deg=0.0,
            last_tau_nm=0.0,
            last_theta_los=0.0,
            last_omega_t_nom=0.0,
            last_e_az=0.0,
        )

    def ingest_inference(
        self,
        state: ControlState,
        result: InferenceResultMsg,
        t_s: float,
        exposure_us: float,
    ) -> tuple[ControlState, VisionSample]:
        """Gate and match blobs; build a vision sample. Does not step the loops.

        Inputs:
            state: Current control state (tracked blobs for IoU).
            result: Detector output.
            t_s: Monotonic shutter time.
            exposure_us: Live exposure.

        Outputs:
            tuple[ControlState, VisionSample]: State with updated tracked-blob
            ancestry only in the sample; the arbiter still owns mode.
        """
        gated = apply_confidence_gate(result.blobs, self.cfg.confidence_gate)
        gated = apply_min_area_gate(gated, self.cfg.min_blob_area_px)
        matched = match_blobs(
            state.arbiter.tracked_blobs, tuple(gated), self.cfg.blob_iou_match_threshold
        )
        z_v: float | None = None
        p_cog: tuple[float, float] | None = None
        e_az = state.last_e_az
        if matched:
            p_cog = matched[0].centroid_raw
            e_az, z_v = pinhole_error_rad(
                p_cog,
                self.plane_width_px,
                self.plane_height_px,
                self.pixel_pitch_m,
                self.focal_m,
            )
        sample = VisionSample(
            t_s=t_s,
            z_v=z_v,
            p_cog=p_cog,
            exposure_us=exposure_us,
            blobs=matched,
            mode_flags=result.mode_flags,
        )
        return replace(state, last_e_az=e_az), sample

    def inner_step(
        self,
        state: ControlState,
        now: float,
        theta_enc_rad: float,
        dt_s: float | None = None,
    ) -> InnerTick:
        """One inner tick: push encoder, fit y_m, PI + computed torque.

        Inputs:
            state: Current control state.
            now: Monotonic seconds of this tick.
            theta_enc_rad: Encoder elevation, radians.
            dt_s: Inner period; defaults to cfg.dt_inner_s.

        Outputs:
            InnerTick: Updated state and torque.
        """
        dt = self.cfg.dt_inner_s if dt_s is None else dt_s
        ring = state.encoder_ring + (theta_enc_rad,)
        max_n = self.cfg.rate_fit_n
        if len(ring) > max_n:
            ring = ring[-max_n:]
        y_m = fit_rate(ring, dt, self.cfg.rate_fit_n, self.cfg.rate_fit_degree)
        el_deg = math.degrees(theta_enc_rad)
        stopped = (
            el_deg <= self.gimbal.el_hw_min_deg + 1e-9 or el_deg >= self.gimbal.el_hw_max_deg - 1e-9
        )
        result = inner_step(
            state.r_rad_s,
            y_m,
            state.integrator,
            dt,
            self.gimbal.J_kg_m2,
            self.gimbal.B_nms_per_rad,
            self.cfg.kp,
            self.cfg.ki,
            self.gimbal.tau_max_nm,
            stopped,
        )
        new_state = replace(
            state,
            encoder_ring=ring,
            integrator=result.integrator,
            y_m=y_m,
            last_inner_s=now,
            last_tau_nm=result.tau_nm,
        )
        return InnerTick(state=new_state, tau_nm=result.tau_nm)

    def outer_step(
        self,
        state: ControlState,
        now: float,
        theta_g_rad: float,
        vision: VisionSample | None,
        iss: IssSample | None,
        safe_commanded: bool,
        safe_cleared: bool,
        dt_s: float | None = None,
        timestamp_utc: str = "",
    ) -> OuterTick:
        """One outer tick: arbiter, predictor, residual KF, rate reference.

        Inputs:
            state: Current control state.
            now: Monotonic seconds of this tick.
            theta_g_rad: Current elevation, radians.
            vision: Dequeued vision sample, or None on coast.
            iss: ISS ECI state, or None (omega_t_nom = 0).
            safe_commanded, safe_cleared: FDIR SAFE flags.
            dt_s: Outer period; defaults to cfg.dt_outer_s.
            timestamp_utc: ISO stamp for pointing telemetry (empty skips the event).

        Outputs:
            OuterTick: Updated state, optional STOW request, telemetry.
        """
        dt = self.cfg.dt_outer_s if dt_s is None else dt_s
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
        )

        pose_mode = state.pose_mode
        pose_el = state.pose_el_deg
        if request is not None and request.mode is GimbalCommandMode.STOW:
            pose_mode = GimbalCommandMode.STOW
            pose_el = self.gimbal.stow_el_deg
        if new_arbiter.gimbal_state is GimbalState.SAFE:
            pose_mode = GimbalCommandMode.STOW
            pose_el = self.gimbal.stow_el_deg
        elif (
            new_arbiter.gimbal_state is not GimbalState.SAFE and pose_mode is GimbalCommandMode.STOW
        ):
            if safe_cleared:
                pose_mode = None

        if vision is not None and vision.exposure_us > 0.0:
            exposure_us = vision.exposure_us
        else:
            exposure_us = state.last_exposure_us
        r_cog = state.r_cog_ecef_m
        if vision is not None and vision.p_cog is not None and iss is not None:
            inter = intersect_cog(
                vision.p_cog,
                theta_g_rad,
                iss.r_m,
                iss.v_m_s,
                iss.utc_s,
                self.eph.epoch_utc_s,
                self.eph.omega_earth_rad_s,
                self.eph.wgs84_a_m,
                self.eph.wgs84_f,
                self.plane_width_px,
                self.plane_height_px,
                self.pixel_pitch_m,
                self.focal_m,
                r_cog,
            )
            if inter.hit and inter.r_cog_ecef_m is not None:
                r_cog = inter.r_cog_ecef_m

        omega_t_nom = 0.0
        theta_los = 0.0
        if r_cog is not None and iss is not None:
            theta_los, omega_t_nom = predict_los(
                iss.utc_s,
                iss.r_m,
                iss.v_m_s,
                r_cog,
                self.eph.omega_earth_rad_s,
                self.eph.epoch_utc_s,
            )

        residual = residual_predict(self.residual_filt, state.residual, dt, omega_t_nom, state.y_m)
        snap = ResidualSnapshot(
            t_s=now,
            state=residual,
            dt_s=dt,
            omega_t_nom=omega_t_nom,
            y_m=state.y_m,
        )
        snapshots = push_snapshot(state.snapshots, snap, self.cfg.rewind_snapshots)

        if vision is not None and vision.z_v is not None:
            residual = rewind_update(
                self.residual_filt,
                snapshots,
                residual,
                now,
                vision.t_s,
                vision.z_v,
                self.cfg.rewind_horizon_s,
            )

        live = bool(residual.has_measurement)
        e_hat = float(residual.x[0])
        omega_res = float(residual.x[1])

        if pose_mode is not None:
            r = position_rate(
                math.radians(pose_el),
                theta_g_rad,
                self.cfg.K_pos,
                math.radians(self.cfg.r_max_stow_deg_per_s),
            )
        else:
            r = outer_rate(
                omega_t_nom,
                omega_res,
                e_hat,
                self.cfg.Kp,
                new_arbiter.gimbal_state,
                live,
                theta_g_rad,
                math.radians(self.gimbal.el_science_max_deg),
                math.radians(self.gimbal.max_hw_slew_rate_deg_per_s),
                exposure_us,
                self.preprocessing.max_motion_smear_px,
                self.ifov_band_deg_per_px,
            )

        new_state = replace(
            state,
            arbiter=new_arbiter,
            residual=residual,
            r_cog_ecef_m=r_cog,
            r_rad_s=r,
            snapshots=snapshots,
            last_outer_s=now,
            last_exposure_us=exposure_us,
            pose_mode=pose_mode,
            pose_el_deg=pose_el,
            last_theta_los=theta_los,
            last_omega_t_nom=omega_t_nom,
        )
        if timestamp_utc:
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
                        "omega_t_res": omega_res,
                        "y_m": state.y_m,
                    },
                )
            )
        return OuterTick(state=new_state, request=request, telemetry=events, fault=None)
