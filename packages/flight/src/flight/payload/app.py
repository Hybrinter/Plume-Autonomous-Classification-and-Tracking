"""Payload application shell: binds HAL, vision queue, and the cascaded controller.

Per frame: acquire a raw mosaic, preprocess co-located, detect, enqueue a vision
sample. The outer loop dequeues vision, reads ephemeris, and writes r. The inner
loop reads the encoder and writes torque. Catch-up methods advance in T_in / T_out
steps so a ManualClock jump still moves the plant.

Satisfies: REQ-AIML-COMP-001, REQ-AIML-COMP-002, REQ-OPER-HIGH-002.
"""

from __future__ import annotations

# stdlib
import math
import threading
from collections import deque
from dataclasses import dataclass, field, replace

# third-party
import numpy as np

# internal
from flight.hal.interfaces import (
    GimbalActuator,
    GimbalHealth,
    GimbalPosition,
    GimbalRateActuator,
    GimbalRateCommand,
    ImagingSensor,
    IssEphemeris,
    StorageWriter,
)
from flight.libs.bus import MessageBus, Subscription
from flight.libs.config import (
    FaultConfig,
    InferenceConfig,
    PactConfig,
    PreprocessingConfig,
    SensorConfig,
)
from flight.libs.messages import (
    CommandAckMsg,
    FaultEventMsg,
    GimbalCommandMsg,
    HeartbeatMsg,
    InferenceResultMsg,
    LaunchLockStateMsg,
    ModeChangeMsg,
    ProcessedFrameMsg,
    ProductRefMsg,
    RoutedCommandMsg,
    TelemetryEventMsg,
)
from flight.libs.time import Clock
from flight.libs.types import (
    AckStatus,
    Band,
    DownlinkPriority,
    Err,
    FaultCode,
    GimbalCommandMode,
    GimbalState,
    LaunchLockState,
    MessageType,
    MosaicFrame,
    Ok,
    Result,
    SystemMode,
)
from flight.payload.control import ControlState, IssSample, PayloadController, VisionSample
from flight.payload.gimbal.integrity import check_integrity, lock_hold_rate
from flight.payload.gimbal.request import GimbalRequest
from flight.payload.inference import DetectorBackend
from flight.payload.preprocess import (
    MosaicCalibration,
    SmearRateSource,
    calibrate_mosaic,
    compute_quality_flags,
    normalize_dn,
    select_bands,
    separate_bands,
)
from flight.payload.tracking import EncoderSample


@dataclass(frozen=True, slots=True)
class TickOutcome:
    """Summary of one payload cycle, returned by process_frame for telemetry/testing.

    Attributes:
        frame_id: The frame_id of the processed raw frame.
        fault: FaultCode if preprocessing or detection failed this frame, else None.
        command_issued: True if a GimbalCommandMsg was published this cycle.
        gimbal_state: The arbiter GimbalState after this frame.
    """

    frame_id: int
    fault: FaultCode | None
    command_issued: bool
    gimbal_state: GimbalState


@dataclass(slots=True)
class LockGate:
    """Mutable launch-lock view. Fail-closed: defaults engaged until RELEASED is seen.

    UNKNOWN is treated as engaged. Tests without a mechanical app must publish RELEASED
    or set engaged=False.
    """

    engaged: bool = True


@dataclass(slots=True)
class SafeLatch:
    """SAFE visible to the inner thread without waiting on detect."""

    commanded: bool = False


@dataclass(slots=True)
class StowGate:
    """Re-issue STOW after lock release if SAFE entry was blocked."""

    pending: bool = False


@dataclass(slots=True)
class PoseIntent:
    """Ground pose waiting to apply on the next outer tick."""

    mode: GimbalCommandMode | None = None
    el_deg: float = 0.0


@dataclass(slots=True)
class ActuatorSafety:
    """Mutable shell state for driver evidence and bounded recovery accounting."""

    last_feedback_s: float | None = None
    recovery_window_start_s: float | None = None
    recovery_attempts: int = 0
    latched_fault: bool = False
    recovery_pending_reset: bool = False
    last_health_publish_s: float | None = None


@dataclass(slots=True)
class EncoderStream:
    """Timestamped encoder samples shared by frame association and outer control."""

    samples: deque[EncoderSample] = field(default_factory=lambda: deque(maxlen=4096))
    consumed_ids: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class PayloadApp:
    """Payload subsystem app: imperative shell around the cascaded pointing loops.

    Attributes:
        sensor: ImagingSensor driver.
        gimbal: GimbalActuator driver.
        ephemeris: IssEphemeris driver.
        detector: DetectorBackend.
        controller: Pure PayloadController.
        bus: Typed MessageBus.
        clock: Injected Clock.
        calib: MosaicCalibration.
        storage: StorageWriter.
        sensor_cfg, inference_cfg, preprocessing_cfg, fault_cfg: Config slices.
        mode_sub, lock_sub, cmd_sub: Bus subscriptions.
        lock_gate: Launch-lock inhibit (fail-closed).
        safe_latch: SAFE flag the inner thread reads every T_in.
        stow_gate: Re-issue STOW after lock release if still SAFE.
        pose_intent: Ground STOW/HOME/GOTO waiting for the next outer tick.
        vision_queue: In-process vision samples (not the MessageBus).
        inner_lock: Short lock around ControlState copies only.
    """

    sensor: ImagingSensor
    gimbal: GimbalActuator
    ephemeris: IssEphemeris
    detector: DetectorBackend
    controller: PayloadController
    bus: MessageBus
    clock: Clock
    calib: MosaicCalibration
    storage: StorageWriter
    sensor_cfg: SensorConfig
    inference_cfg: InferenceConfig
    preprocessing_cfg: PreprocessingConfig
    fault_cfg: FaultConfig
    mode_sub: Subscription[ModeChangeMsg]
    lock_sub: Subscription[LaunchLockStateMsg]
    cmd_sub: Subscription[RoutedCommandMsg]
    lock_gate: LockGate = field(default_factory=LockGate)
    safe_latch: SafeLatch = field(default_factory=SafeLatch)
    stow_gate: StowGate = field(default_factory=StowGate)
    pose_intent: PoseIntent = field(default_factory=PoseIntent)
    vision_queue: deque[VisionSample] = field(default_factory=lambda: deque(maxlen=4))
    inner_lock: threading.Lock = field(default_factory=threading.Lock)
    actuator_io_lock: threading.Lock = field(default_factory=threading.Lock)
    actuator_safety: ActuatorSafety = field(default_factory=ActuatorSafety)
    encoder_stream: EncoderStream = field(default_factory=EncoderStream)

    @staticmethod
    def from_config(
        cfg: PactConfig,
        sensor: ImagingSensor,
        gimbal: GimbalActuator,
        ephemeris: IssEphemeris,
        detector: DetectorBackend,
        bus: MessageBus,
        clock: Clock,
        calib: MosaicCalibration,
        storage: StorageWriter,
    ) -> PayloadApp:
        """Assemble a PayloadApp from a PactConfig and injected services.

        Raises:
            ValueError: Invalid sensor mosaic or inference geometry.
        """
        if cfg.sensor.width_px % 2 or cfg.sensor.height_px % 2:
            raise ValueError("sensor mosaic dimensions must be even")
        plane_h, plane_w = cfg.sensor.height_px // 2, cfg.sensor.width_px // 2
        if plane_h != cfg.inference.input_height_px or plane_w != cfg.inference.input_width_px:
            raise ValueError("band plane must equal the inference input size")
        if sorted(cfg.sensor.mosaic_layout) != sorted(b.value for b in Band):
            raise ValueError("mosaic_layout must name each Band exactly once")
        if any(b not in cfg.sensor.mosaic_layout for b in cfg.inference.input_bands):
            raise ValueError("input_bands must be a subset of mosaic_layout")
        return PayloadApp(
            sensor=sensor,
            gimbal=gimbal,
            ephemeris=ephemeris,
            detector=detector,
            controller=PayloadController.from_config(
                cfg.controller, cfg.sensor, cfg.gimbal, cfg.ephemeris, cfg.preprocessing
            ),
            bus=bus,
            clock=clock,
            calib=calib,
            storage=storage,
            sensor_cfg=cfg.sensor,
            inference_cfg=cfg.inference,
            preprocessing_cfg=cfg.preprocessing,
            fault_cfg=cfg.fault,
            mode_sub=bus.subscribe(ModeChangeMsg),
            lock_sub=bus.subscribe(LaunchLockStateMsg),
            cmd_sub=bus.subscribe(RoutedCommandMsg),
            lock_gate=LockGate(),
            safe_latch=SafeLatch(),
            stow_gate=StowGate(),
            pose_intent=PoseIntent(),
            vision_queue=deque(maxlen=cfg.controller.vision.queue_depth),
            inner_lock=threading.Lock(),
            actuator_io_lock=threading.Lock(),
            actuator_safety=ActuatorSafety(),
        )

    def poll_mode_changes(self) -> tuple[bool, bool]:
        """Drain pending ModeChangeMsg; return (safe_commanded, safe_cleared)."""
        safe_commanded = False
        safe_cleared = False
        while not self.mode_sub.empty():
            msg = self.mode_sub.get_nowait()
            if msg.new_mode is SystemMode.SAFE:
                safe_commanded = True
                self.safe_latch.commanded = True
            else:
                safe_cleared = self._clear_actuator_fault_for_ground()
                self.safe_latch.commanded = not safe_cleared
        return safe_commanded, safe_cleared

    def poll_lock_state(self) -> None:
        """Drain pending LaunchLockStateMsg. UNKNOWN and ENGAGED both inhibit."""
        while not self.lock_sub.empty():
            state = self.lock_sub.get_nowait().state
            self.lock_gate.engaged = state is not LaunchLockState.RELEASED

    def handle_commands(self) -> None:
        """Apply routed STOW / HOME / GOTO into the pose intent and ack."""
        while not self.cmd_sub.empty():
            command = self.cmd_sub.get_nowait()
            if command.target != "payload":
                continue
            if command.command_id == "GIMBAL_STOW":
                self.pose_intent.mode = GimbalCommandMode.STOW
                self.pose_intent.el_deg = self.controller.gimbal.stow_el_deg
                self._ack_command(command, AckStatus.ACCEPTED, FaultCode.NONE, "stow latched")
            elif command.command_id == "GIMBAL_HOME":
                self.pose_intent.mode = GimbalCommandMode.HOME
                self.pose_intent.el_deg = self.controller.gimbal.home_el_deg
                self._ack_command(command, AckStatus.ACCEPTED, FaultCode.NONE, "home latched")
            elif command.command_id == "GIMBAL_GOTO":
                self.pose_intent.mode = GimbalCommandMode.ABSOLUTE
                self.pose_intent.el_deg = float(command.params["el_deg"])
                self._ack_command(command, AckStatus.ACCEPTED, FaultCode.NONE, "goto latched")
            else:
                self._ack_command(
                    command, AckStatus.REJECTED, FaultCode.COMMAND_INVALID, "unsupported command"
                )

    def _rate_mode(self) -> bool:
        """Return whether the injected actuator exposes the production rate path."""
        return isinstance(self.gimbal, GimbalRateActuator)

    def _record_encoder(self, position: GimbalPosition) -> EncoderSample:
        """Record one valid feedback frame and return its estimator-domain sample."""
        sample_id = (
            f"encoder:{position.sequence}"
            if position.sequence > 0
            else f"encoder:t:{position.timestamp_s:.9f}"
        )
        sample = EncoderSample(
            sample_id=sample_id,
            t_s=position.timestamp_s,
            angle_rad=math.radians(position.el_deg),
            angle_variance_rad2=self.controller.cfg.residual.encoder_variance_rad2,
        )
        if all(existing.sample_id != sample.sample_id for existing in self.encoder_stream.samples):
            self.encoder_stream.samples.append(sample)
        return sample

    def _encoder_for_tick(self, tick_s: float) -> EncoderSample | None:
        """Return the newest unconsumed sample whose device time belongs to this tick."""
        candidates = [
            sample
            for sample in self.encoder_stream.samples
            if sample.sample_id not in self.encoder_stream.consumed_ids
            and sample.t_s <= tick_s + 1.0e-12
        ]
        if not candidates:
            return None
        selected = max(candidates, key=lambda sample: (sample.t_s, sample.sample_id))
        self.encoder_stream.consumed_ids.add(selected.sample_id)
        return selected

    def _encoder_angle_at(self, t_s: float, *, max_span_s: float | None = None) -> float | None:
        """Interpolate a shutter angle only from a valid bounded sample bracket."""
        ordered = sorted(self.encoder_stream.samples, key=lambda sample: sample.t_s)
        exact = [sample for sample in ordered if abs(sample.t_s - t_s) <= 1.0e-12]
        if exact:
            return exact[0].angle_rad
        before = [sample for sample in ordered if sample.t_s < t_s]
        after = [sample for sample in ordered if sample.t_s > t_s]
        if not before or not after:
            return None
        left = before[-1]
        right = after[0]
        span = right.t_s - left.t_s
        span_limit = (
            self.controller.cfg.residual.interpolation_span_max_s
            if max_span_s is None
            else max_span_s
        )
        if span <= 0.0 or span > span_limit:
            return None
        alpha = (t_s - left.t_s) / span
        return left.angle_rad + alpha * (right.angle_rad - left.angle_rad)

    def _encoder_rate_over_exposure_deg_per_s(self, raw: MosaicFrame) -> float | None:
        """Mean elevation rate across the exposure from encoder brackets, or None."""
        dt_exp_s = raw.exposure_us * 1.0e-6
        if dt_exp_s <= 0.0:
            return None
        t_end = raw.timestamp_s
        t_start = t_end - dt_exp_s
        theta_end = self._encoder_angle_at(t_end, max_span_s=math.inf)
        theta_start = self._encoder_angle_at(t_start, max_span_s=math.inf)
        if theta_end is None or theta_start is None:
            return None
        return math.degrees((theta_end - theta_start) / dt_exp_s)

    def _smear_gimbal_rate_deg_per_s(
        self,
        raw: MosaicFrame,
        state: ControlState,
        measured_rate_deg_per_s: float | None,
    ) -> tuple[float, SmearRateSource]:
        """Elevation rate for MOTION_SMEAR: measured, else encoder, else command.

        ``0.0`` is a valid measured, encoder, or commanded rate. Unknown is a
        missing measured value plus a failed encoder bracket, which falls back
        to the commanded rate and labels it COMMANDED.
        """
        if measured_rate_deg_per_s is not None:
            return measured_rate_deg_per_s, SmearRateSource.MEASURED
        encoder_rate = self._encoder_rate_over_exposure_deg_per_s(raw)
        if encoder_rate is not None:
            return encoder_rate, SmearRateSource.ENCODER
        return math.degrees(state.commanded_rate_rad_s), SmearRateSource.COMMANDED

    @staticmethod
    def _invalidate_encoder_state(state: ControlState) -> ControlState:
        """Remove motion authority and detailed-plant encoder baseline after a read fault."""
        return replace(
            state,
            encoder=replace(
                state.encoder,
                samples=(),
                last_theta_enc_rad=None,
                measured_rate_rad_s=0.0,
            ),
            commanded_rate_rad_s=0.0,
        )

    def _fresh_recovery_state(self, state: ControlState, sample: EncoderSample) -> ControlState:
        """Start a new residual checkpoint after encoder/actuator recovery."""
        return replace(
            self._fresh_inner_state(state),
            residual=self.controller.residual_filt.initial_state(),
            residual_history=self.controller.residual_filt.initial_history(
                t_s=sample.t_s,
                encoder_angle_rad=sample.angle_rad,
                encoder_endpoint_variance_rad2=sample.angle_variance_rad2,
            ),
        )

    def process_frame(
        self,
        raw: MosaicFrame,
        state: ControlState,
        now: float,
        slew_rate_deg_per_s: float | None = None,
        gimbal_pos: GimbalPosition | None = None,
        safe_commanded: bool = False,
        safe_cleared: bool = False,
    ) -> tuple[ControlState, TickOutcome]:
        """Preprocess, detect, and enqueue a vision sample. Does not write torque.

        SAFE flags are accepted for call-site compatibility; the outer loop applies them.
        ``slew_rate_deg_per_s`` is a measured elevation rate. ``0.0`` is stationary.
        ``None`` uses encoder motion over the exposure, then the commanded rate.
        """
        del safe_commanded, safe_cleared
        mosaic = np.asarray(raw.mosaic, dtype=np.float32)

        calibrated = calibrate_mosaic(mosaic, self.calib)
        if isinstance(calibrated, Err):
            self._publish_fault(calibrated.error, f"calibration failed frame_id={raw.frame_id}")
            return state, self._fault_outcome(raw.frame_id, calibrated.error, state)

        planes = separate_bands(calibrated.value)
        if isinstance(planes, Err):
            self._publish_fault(planes.error, f"demosaic failed frame_id={raw.frame_id}")
            return state, self._fault_outcome(raw.frame_id, planes.error, state)

        normalized = normalize_dn(planes.value, self.sensor_cfg.bit_depth)
        selected = select_bands(
            normalized, self.sensor_cfg.mosaic_layout, self.inference_cfg.input_bands
        )
        if isinstance(selected, Err):
            self._publish_fault(selected.error, f"band select failed frame_id={raw.frame_id}")
            return state, self._fault_outcome(raw.frame_id, selected.error, state)

        if gimbal_pos is None:
            position = self._read_position()
            if isinstance(position, Ok):
                gimbal_pos = position.value
            else:
                state = self._invalidate_encoder_state(state)
        else:
            self._record_encoder(gimbal_pos)

        gimbal_rate_deg_per_s, _ = self._smear_gimbal_rate_deg_per_s(
            raw, state, slew_rate_deg_per_s
        )
        omega_scene_el_deg_per_s = math.degrees(state.target.last_omega_scene_el)
        quality_flags = compute_quality_flags(
            selected.value,
            raw.exposure_us,
            gimbal_rate_deg_per_s,
            self.sensor_cfg.ifov_band_deg_per_px,
            raw.timestamp_utc,
            self.preprocessing_cfg,
            omega_scene_el_deg_per_s=omega_scene_el_deg_per_s,
        )

        processed = ProcessedFrameMsg(
            msg_type=MessageType.PROCESSED_FRAME,
            timestamp_utc=raw.timestamp_utc,
            frame_id=raw.frame_id,
            tensor=selected.value,
            quality_flags=quality_flags,
        )

        detect_result = self.detector.detect(processed)
        if isinstance(detect_result, Err):
            self._publish_fault(detect_result.error, f"detection failed frame_id={raw.frame_id}")
            return state, self._fault_outcome(raw.frame_id, detect_result.error, state)
        inference = detect_result.value
        self.bus.publish(inference)
        self._store_mask_product(inference)

        iss, eph_err = self._read_iss_at(raw.timestamp_s if raw.timestamp_s else now)
        if eph_err is not None:
            self._publish_fault(eph_err, "ephemeris read failed")
        new_state, sample = self.controller.ingest_inference(
            state,
            inference,
            raw.timestamp_s,
            raw.exposure_us,
            iss,
            theta_g_rad=self._encoder_angle_at(raw.timestamp_s),
        )
        self.vision_queue.append(sample)
        outcome = TickOutcome(
            frame_id=raw.frame_id,
            fault=None,
            command_issued=False,
            gimbal_state=new_state.arbiter.gimbal_state,
        )
        return new_state, outcome

    def advance_outer(
        self,
        state: ControlState,
        now: float,
        safe_commanded: bool = False,
        safe_cleared: bool = False,
    ) -> tuple[ControlState, TickOutcome]:
        """Catch up the outer loop to `now` in T_out steps.

        Dequeues at most one due vision sample per outer tick (shutter time <= tick
        time; oldest first). Reads ephemeris each tick. Publishes pointing telemetry
        and pose GimbalCommandMsg.
        """
        dt = self.controller.cfg.outer.dt_s
        current = state
        command_issued = False
        safe_commanded = safe_commanded or self.safe_latch.commanded
        if self.pose_intent.mode is not None:
            pose_mode = self.pose_intent.mode
            pose_el = self.pose_intent.el_deg
            current = replace(
                current,
                pose=replace(current.pose, pose_mode=pose_mode, pose_el_deg=pose_el),
            )
            issued = self._actuate_pose(
                GimbalRequest(mode=pose_mode, el_deg=pose_el, reason="ground_pose"),
                current,
                frame_id=0,
            )
            command_issued = command_issued or issued
            self.pose_intent.mode = None
        if current.last_outer_s is None:
            origin = min(self.clock.monotonic_s(), now)
            current = replace(current, last_outer_s=origin)
            t = origin
        else:
            t = current.last_outer_s
        gap = now - t
        cap = self.controller.cfg.integrity.catchup_max_s
        if gap > cap:
            self._publish_fault(FaultCode.GIMBAL_FAULT, "outer catch-up cap exceeded")
            t = now - cap
            current = replace(current, last_outer_s=t)
        while t + dt <= now + 1e-12:
            t = t + dt
            encoder = self._encoder_for_tick(t)
            if encoder is None:
                # A catch-up tick may not consume a current encoder sample and
                # relabel it with historical time. Keep the committed cursor at
                # the last processed tick; a later call may then replay this
                # historical interval when the physical sample arrives.
                if safe_commanded or self.safe_latch.commanded:
                    current = self._invalidate_encoder_state(current)
                    current = replace(
                        current,
                        arbiter=replace(current.arbiter, gimbal_state=GimbalState.SAFE),
                        pose=replace(
                            current.pose,
                            pose_mode=GimbalCommandMode.STOW,
                            pose_el_deg=self.controller.gimbal.stow_el_deg,
                        ),
                    )
                safe_commanded = False
                safe_cleared = False
                continue
            vision: VisionSample | None = None
            if self.vision_queue:
                sample_t = self.vision_queue[0].t_s
                if sample_t <= t + 1e-9:
                    vision = self.vision_queue.popleft()
            iss, eph_err = self._read_iss_at(t)
            if eph_err is not None:
                self._publish_fault(eph_err, "ephemeris read failed")
            theta = encoder.angle_rad
            current = replace(
                current,
                encoder=replace(current.encoder, last_theta_enc_rad=theta),
            )
            if self.actuator_safety.recovery_pending_reset:
                current = self._fresh_recovery_state(current, encoder)
            tick = self.controller.outer_step(
                current,
                t,
                encoder,
                vision,
                iss,
                safe_commanded,
                safe_cleared,
                dt,
                timestamp_utc=self.clock.wall_clock_iso(),
                detailed_plant=not self._rate_mode(),
            )
            current = tick.state
            if self.lock_gate.engaged:
                current = replace(current, commanded_rate_rad_s=0.0)
            for event in tick.telemetry:
                if (
                    self.lock_gate.engaged
                    and event.subsystem == "payload"
                    and event.event_name == "pointing"
                ):
                    payload = dict(event.payload)
                    payload["r"] = 0.0
                    event = replace(event, payload=payload)
                self.bus.publish(event)
            if tick.request is not None:
                issued = self._actuate_pose(tick.request, current, frame_id=0)
                command_issued = command_issued or issued
                if not issued and self.lock_gate.engaged:
                    self.stow_gate.pending = True
            if self._rate_mode():
                assert isinstance(self.gimbal, GimbalRateActuator)
                if (
                    not self.lock_gate.engaged
                    and current.arbiter.gimbal_state is GimbalState.SAFE
                    and current.pose.pose_mode is GimbalCommandMode.STOW
                ):
                    with self.actuator_io_lock:
                        stow_result = self.gimbal.stow_reference_step(t)
                    if isinstance(stow_result, Err):
                        self._record_actuator_failure(stow_result.error, t)
                        self._publish_fault(stow_result.error, "bounded gimbal stow failed")
                        self._inhibit_motion("bounded stow failed")
                else:
                    self._write_rate(
                        math.degrees(current.commanded_rate_rad_s), t, self.lock_gate.engaged
                    )
            safe_commanded = False
            safe_cleared = False
        if (
            self.stow_gate.pending
            and not self.lock_gate.engaged
            and current.arbiter.gimbal_state is GimbalState.SAFE
        ):
            issued = self._actuate_pose(
                GimbalRequest(
                    mode=GimbalCommandMode.STOW,
                    el_deg=self.controller.gimbal.stow_el_deg,
                    reason="safe_stow_after_lock_release",
                ),
                current,
                frame_id=0,
            )
            command_issued = command_issued or issued
            self.stow_gate.pending = not issued
        outcome = TickOutcome(
            frame_id=0,
            fault=None,
            command_issued=command_issued,
            gimbal_state=current.arbiter.gimbal_state,
        )
        return current, outcome

    def advance_inner(self, state: ControlState, now: float) -> ControlState:
        """Catch up the inner loop to `now` in T_in steps and write torque."""
        if self._rate_mode():
            # Production Xeryon control is rate-commanded at outer cadence;
            # never read/fit the detailed-plant inner loop on that path.
            return state
        dt = self.controller.cfg.inner.dt_s
        current = state
        self.poll_lock_state()
        if current.inner.last_inner_s is None:
            origin = min(self.clock.monotonic_s(), now)
            current = replace(current, inner=replace(current.inner, last_inner_s=origin))
        t = current.inner.last_inner_s
        assert t is not None
        gap = now - t
        cap = self.controller.cfg.integrity.catchup_max_s
        if gap > cap:
            self._publish_fault(FaultCode.GIMBAL_FAULT, "inner catch-up cap exceeded")
            t = now - cap
            current = replace(current, inner=replace(current.inner, last_inner_s=t))
        while t + dt <= now + 1e-12:
            t = t + dt
            pos = self._read_position()
            if isinstance(pos, Ok):
                theta = math.radians(pos.value.el_deg)
                encoder_timestamp_s = pos.value.timestamp_s
            else:
                self._publish_fault(pos.error, "encoder unavailable")
                current = replace(current, commanded_rate_rad_s=0.0)
                break
            if self.actuator_safety.recovery_pending_reset:
                current = self._fresh_inner_state(current)
            enc_rate = 0.0
            if current.encoder.last_theta_enc_rad is not None and current.encoder.samples:
                prior_s = current.encoder.samples[-1].t_s
                measured_dt_s = encoder_timestamp_s - prior_s
                if measured_dt_s > 0.0:
                    enc_rate = (theta - current.encoder.last_theta_enc_rad) / measured_dt_s
            locked = self.lock_gate.engaged
            tick = self.controller.inner_step(
                current,
                t,
                theta,
                dt,
                encoder_timestamp_s=encoder_timestamp_s,
                locked=locked,
                safe_latched=self.safe_latch.commanded,
            )
            current, integrity_fault = self._apply_integrity(
                current, tick.state, tick.tau_nm, theta, t, enc_rate, locked
            )
            if integrity_fault is not None:
                self._publish_fault(integrity_fault, "pointing integrity trip")
                self.safe_latch.commanded = True
            if integrity_fault is not None:
                self._inhibit_motion("pointing integrity trip")
            else:
                self._write_torque(tick.tau_nm, t, locked)
        return current

    def _apply_integrity(
        self,
        prior: ControlState,
        tick_state: ControlState,
        tau_nm: float,
        theta: float,
        now: float,
        enc_rate: float,
        locked: bool,
    ) -> tuple[ControlState, FaultCode | None]:
        """Latch lock-hold pose, run the detector, and stamp strikes onto tick state."""
        motion, ref_th, ref_t = lock_hold_rate(
            locked, theta, now, prior.integrity.lock_theta_ref_rad, prior.integrity.lock_ref_s
        )
        integrity = check_integrity(
            self.controller.cfg.integrity,
            tick_state.commanded_rate_rad_s,
            tick_state.encoder.measured_rate_rad_s,
            tau_nm,
            enc_rate,
            locked,
            prior.integrity.freeze_strikes,
            prior.integrity.lock_strikes,
            motion,
        )
        updated = replace(
            tick_state,
            integrity=replace(
                tick_state.integrity,
                freeze_strikes=integrity.freeze_strikes,
                lock_strikes=integrity.lock_fight_strikes,
                lock_theta_ref_rad=ref_th,
                lock_ref_s=ref_t,
            ),
        )
        return updated, integrity.fault

    def _read_position(self) -> Result[GimbalPosition, FaultCode]:
        """Read encoder feedback under the sole driver-I/O lock.

        A failed read immediately latches drive containment before the next torque
        write.  The pure controller receives the hardware's sample timestamp
        separately, so no wall-clock arrival time is mistaken for an encoder time.
        """
        with self.actuator_io_lock:
            result = self.gimbal.read_position()
        if isinstance(result, Ok):
            self._record_encoder(result.value)
            self.actuator_safety.last_feedback_s = result.value.timestamp_s
        else:
            self._record_actuator_failure(result.error, self.clock.monotonic_s())
            self._inhibit_motion("encoder unavailable")
            if self.actuator_safety.latched_fault:
                self.safe_latch.commanded = True
        return result

    def _clear_actuator_fault_for_ground(self) -> bool:
        """Clear a latched actuator fault only from confirmed inhibited health."""
        if not self.actuator_safety.latched_fault:
            return True
        with self.actuator_io_lock:
            health = self.gimbal.read_health()
        if (
            isinstance(health, Err)
            or not health.value.feedback_valid
            or not health.value.inhibit_confirmed
        ):
            self._publish_fault(FaultCode.GIMBAL_FAULT, "actuator fault clear rejected")
            return False
        self.actuator_safety.latched_fault = False
        self.actuator_safety.recovery_pending_reset = True
        self.actuator_safety.recovery_attempts = 0
        self.actuator_safety.recovery_window_start_s = None
        self._publish_actuator_recovery("ground_cleared")
        return True

    def _inhibit_motion(self, reason: str) -> bool:
        """Request driver-side containment and fault if it cannot be confirmed."""
        with self.actuator_io_lock:
            result = self.gimbal.inhibit(reason)
        if isinstance(result, Err):
            self._publish_fault(result.error, f"gimbal inhibit unconfirmed: {reason}")
            self.actuator_safety.latched_fault = True
            return False
        if not result.value.inhibit_confirmed:
            self._publish_fault(FaultCode.GIMBAL_FAULT, f"gimbal inhibit unconfirmed: {reason}")
            self.actuator_safety.latched_fault = True
            return False
        self._publish_actuator_health(result.value, force=True)
        return True

    def _write_torque(self, tau_nm: float, now: float, locked: bool) -> bool:
        """Write a leased torque command only while feedback and integrity are healthy."""
        if locked or self.actuator_safety.latched_fault:
            return self._inhibit_motion("motion inhibited by lock or actuator fault")
        with self.actuator_io_lock:
            health = self.gimbal.read_health()
        if isinstance(health, Err) or not health.value.feedback_valid:
            self.actuator_safety.latched_fault = True
            self.safe_latch.commanded = True
            self._inhibit_motion("invalid actuator feedback")
            self._publish_fault(FaultCode.GIMBAL_FAULT, "invalid actuator feedback")
            return False
        authority_now = self.clock.monotonic_s()
        last_feedback_s = health.value.last_feedback_s
        if (
            last_feedback_s is None
            or authority_now - last_feedback_s > self.controller.cfg.integrity.feedback_max_age_s
        ):
            self.actuator_safety.latched_fault = True
            self.safe_latch.commanded = True
            self._inhibit_motion("stale actuator feedback")
            self._publish_fault(FaultCode.GIMBAL_FAULT, "stale actuator feedback")
            return False
        self._publish_actuator_health(health.value)
        valid_until_s = authority_now + self.controller.cfg.integrity.command_authority_s
        with self.actuator_io_lock:
            send = self.gimbal.set_torque(tau_nm, valid_until_s)
        if isinstance(send, Err):
            self._record_actuator_failure(send.error, now)
            self._inhibit_motion("torque command failed")
            if self.actuator_safety.latched_fault:
                self.safe_latch.commanded = True
            return False
        if self.actuator_safety.recovery_pending_reset:
            self._publish_actuator_recovery("recovered")
        self.actuator_safety.recovery_pending_reset = False
        self.actuator_safety.recovery_attempts = 0
        self.actuator_safety.recovery_window_start_s = None
        return True

    def _write_rate(self, rate_deg_per_s: float, now: float, locked: bool) -> bool:
        """Write a leased production rate command after feedback/safety checks."""
        if not self._rate_mode():
            return False
        if locked or self.actuator_safety.latched_fault:
            return self._inhibit_motion("motion inhibited by lock or actuator fault")
        with self.actuator_io_lock:
            health = self.gimbal.read_health()
        if isinstance(health, Err) or not health.value.feedback_valid:
            self.actuator_safety.latched_fault = True
            self.safe_latch.commanded = True
            self._inhibit_motion("invalid actuator feedback")
            self._publish_fault(FaultCode.GIMBAL_FAULT, "invalid actuator feedback")
            return False
        authority_now = self.clock.monotonic_s()
        last_feedback_s = health.value.last_feedback_s
        if (
            last_feedback_s is None
            or authority_now - last_feedback_s > self.controller.cfg.integrity.feedback_max_age_s
        ):
            self.actuator_safety.latched_fault = True
            self.safe_latch.commanded = True
            self._inhibit_motion("stale actuator feedback")
            self._publish_fault(FaultCode.GIMBAL_STALE_FEEDBACK, "stale actuator feedback")
            return False
        self._publish_actuator_health(health.value)
        command = GimbalRateCommand(
            rate_deg_per_s=rate_deg_per_s,
            valid_until_s=authority_now + self.controller.cfg.integrity.command_authority_s,
        )
        with self.actuator_io_lock:
            if not isinstance(self.gimbal, GimbalRateActuator):
                return False
            send = self.gimbal.set_rate(command)
        if isinstance(send, Err):
            self._record_actuator_failure(send.error, now)
            self._inhibit_motion("rate command failed")
            self.safe_latch.commanded = self.actuator_safety.latched_fault
            return False
        if self.actuator_safety.recovery_pending_reset:
            self._publish_actuator_recovery("recovered")
        self.actuator_safety.recovery_pending_reset = False
        self.actuator_safety.recovery_attempts = 0
        self.actuator_safety.recovery_window_start_s = None
        return True

    def _record_actuator_failure(self, code: FaultCode, now: float) -> None:
        """Classify failures; only transient availability faults receive a retry budget."""
        transient = code is FaultCode.COMM_TIMEOUT
        safety = self.actuator_safety
        cfg = self.controller.cfg.integrity
        if not transient:
            safety.latched_fault = True
            self._publish_fault(code, "unclassified actuator failure")
            return
        if (
            safety.recovery_window_start_s is None
            or now - safety.recovery_window_start_s > cfg.recovery_window_s
        ):
            safety.recovery_window_start_s = now
            safety.recovery_attempts = 0
        safety.recovery_attempts += 1
        safety.recovery_pending_reset = True
        self._publish_actuator_recovery("retrying")
        if safety.recovery_attempts > cfg.recovery_max_attempts:
            safety.latched_fault = True
            self._publish_fault(code, "actuator transient recovery budget exhausted")

    @staticmethod
    def _fresh_inner_state(state: ControlState) -> ControlState:
        """Discard dynamic controller memory before resuming after an I/O outage."""
        return replace(
            state,
            encoder=replace(
                state.encoder,
                samples=(),
                last_theta_enc_rad=None,
                measured_rate_rad_s=0.0,
            ),
            inner=replace(state.inner, integrator=0.0, last_tau_nm=0.0),
            integrity=replace(state.integrity, freeze_strikes=0, lock_strikes=0),
        )

    def _publish_actuator_recovery(self, state: str) -> None:
        """Expose bounded transient-recovery state without treating it as integrity."""
        self.bus.publish(
            TelemetryEventMsg(
                msg_type=MessageType.TELEMETRY_EVENT,
                timestamp_utc=self.clock.wall_clock_iso(),
                subsystem="payload",
                event_name="gimbal_actuator_recovery",
                payload={
                    "state": state,
                    "attempt": self.actuator_safety.recovery_attempts,
                },
            )
        )

    def _publish_actuator_health(self, health: GimbalHealth, force: bool = False) -> None:
        """Publish compact, persistent actuator-health evidence for FDIR and ground."""
        now = self.clock.monotonic_s()
        last = self.actuator_safety.last_health_publish_s
        if not force and last is not None and now - last < self.fault_cfg.watchdog_interval_s:
            return
        self.actuator_safety.last_health_publish_s = now
        self.bus.publish(
            TelemetryEventMsg(
                msg_type=MessageType.TELEMETRY_EVENT,
                timestamp_utc=self.clock.wall_clock_iso(),
                subsystem="payload",
                event_name="gimbal_actuator_health",
                payload={
                    "feedback_valid": health.feedback_valid,
                    "last_feedback_s": (
                        math.nan if health.last_feedback_s is None else health.last_feedback_s
                    ),
                    "command_valid_until_s": (
                        math.nan
                        if health.command_valid_until_s is None
                        else health.command_valid_until_s
                    ),
                    "inhibited": health.inhibited,
                    "inhibit_confirmed": health.inhibit_confirmed,
                    "controller_status_bits": health.controller_status_bits,
                    "motor_on": health.motor_on,
                    "closed_loop": health.closed_loop,
                    "duty_credit_s": (
                        -1.0 if health.duty_credit_s is None else health.duty_credit_s
                    ),
                    "duty_locked_out": health.duty_locked_out,
                    "watchdog_gate_confirmed": health.watchdog_gate_confirmed,
                    "time_mapping_valid": health.time_mapping_valid,
                    "requested_rate_deg_per_s": (
                        math.nan
                        if health.requested_rate_deg_per_s is None
                        else health.requested_rate_deg_per_s
                    ),
                    "quantized_rate_deg_per_s": (
                        math.nan
                        if health.quantized_rate_deg_per_s is None
                        else health.quantized_rate_deg_per_s
                    ),
                },
            )
        )

    def _read_iss_at(self, monotonic_t: float) -> tuple[IssSample | None, FaultCode | None]:
        """Read ISS ECI at the UTC corresponding to monotonic_t. Err is published by caller."""
        utc = self.clock.utc_s() + (monotonic_t - self.clock.monotonic_s())
        result = self.ephemeris.read_state(utc)
        if isinstance(result, Err):
            return None, result.error
        value = result.value
        return IssSample(r_m=value.r_m, v_m_s=value.v_m_s, utc_s=value.epoch_utc_s), None

    def _actuate_pose(self, request: GimbalRequest, state: ControlState, frame_id: int) -> bool:
        """Map a pose GimbalRequest onto HAL and publish GimbalCommandMsg."""
        if self.lock_gate.engaged:
            self._inhibit_motion("launch lock engaged")
            self.bus.publish(
                TelemetryEventMsg(
                    msg_type=MessageType.TELEMETRY_EVENT,
                    timestamp_utc=self.clock.wall_clock_iso(),
                    subsystem="payload",
                    event_name="gimbal_motion_inhibited",
                    payload={"reason": "launch_lock_engaged", "mode": request.mode.value},
                )
            )
            return False
        with self.actuator_io_lock:
            if request.mode is GimbalCommandMode.STOW:
                send_result = self.gimbal.stow()
            elif request.mode is GimbalCommandMode.HOME:
                send_result = self.gimbal.home()
            else:
                send_result = self.gimbal.goto_angle(request.el_deg)
        if isinstance(send_result, Err):
            self._publish_fault(send_result.error, "gimbal pose actuation failed")
            return False
        self.bus.publish(
            GimbalCommandMsg(
                msg_type=MessageType.GIMBAL_COMMAND,
                timestamp_utc=self.clock.wall_clock_iso(),
                frame_id=frame_id,
                mode=request.mode,
                el_value_deg=request.el_deg,
                state=state.arbiter.gimbal_state,
                reason=request.reason,
            )
        )
        return True

    def run(self, stop_event: threading.Event) -> None:
        """Run acquisition + outer loop; spawn the inner torque thread."""
        self.sensor.start_acquisition()
        holder: dict[str, ControlState] = {"state": self.controller.initial_state()}
        heartbeat_seq = 0
        last_heartbeat = self.clock.monotonic_s()

        def inner_loop() -> None:
            """One inner_step + set_torque per T_in after origin init."""
            dt = self.controller.cfg.inner.dt_s
            while not stop_event.is_set():
                now_inner = self.clock.monotonic_s()
                self.poll_lock_state()
                with self.inner_lock:
                    snap = holder["state"]
                    locked = self.lock_gate.engaged
                    safe = self.safe_latch.commanded
                    if snap.inner.last_inner_s is None:
                        holder["state"] = replace(
                            snap, inner=replace(snap.inner, last_inner_s=now_inner)
                        )
                        snap = holder["state"]
                if snap.inner.last_inner_s == now_inner:
                    stop_event.wait(timeout=dt)
                    continue
                pos = self._read_position()
                if isinstance(pos, Ok):
                    theta = math.radians(pos.value.el_deg)
                    encoder_timestamp_s = pos.value.timestamp_s
                else:
                    self._publish_fault(pos.error, "encoder unavailable")
                    with self.inner_lock:
                        latest = holder["state"]
                        holder["state"] = replace(latest, commanded_rate_rad_s=0.0)
                    stop_event.wait(timeout=dt)
                    continue
                if self.actuator_safety.recovery_pending_reset:
                    snap = self._fresh_inner_state(snap)
                enc_rate = 0.0
                if snap.encoder.last_theta_enc_rad is not None and snap.encoder.samples:
                    measured_dt_s = encoder_timestamp_s - snap.encoder.samples[-1].t_s
                    if measured_dt_s > 0.0:
                        enc_rate = (theta - snap.encoder.last_theta_enc_rad) / measured_dt_s
                tick = self.controller.inner_step(
                    snap,
                    now_inner,
                    theta,
                    dt,
                    encoder_timestamp_s=encoder_timestamp_s,
                    locked=locked,
                    safe_latched=safe,
                )
                stamped, integrity_fault = self._apply_integrity(
                    snap, tick.state, tick.tau_nm, theta, now_inner, enc_rate, locked
                )
                if integrity_fault is not None:
                    self._publish_fault(integrity_fault, "pointing integrity trip")
                    self.safe_latch.commanded = True
                if integrity_fault is not None:
                    self._inhibit_motion("pointing integrity trip")
                else:
                    self._write_torque(tick.tau_nm, now_inner, locked)
                with self.inner_lock:
                    latest = holder["state"]
                    merged_r = (
                        stamped.commanded_rate_rad_s
                        if (locked or safe or latest.pose.pose_mode is not None)
                        else latest.commanded_rate_rad_s
                    )
                    holder["state"] = replace(
                        latest,
                        encoder=stamped.encoder,
                        inner=replace(
                            stamped.inner,
                            last_inner_s=now_inner,
                            last_tau_nm=tick.tau_nm,
                        ),
                        commanded_rate_rad_s=merged_r,
                        integrity=stamped.integrity,
                    )
                stop_event.wait(timeout=dt)

        inner_thread: threading.Thread | None = None
        if not self._rate_mode():
            inner_thread = threading.Thread(target=inner_loop, name="payload-inner", daemon=True)
            inner_thread.start()
        try:
            while not stop_event.is_set():
                now = self.clock.monotonic_s()
                if now - last_heartbeat >= self.fault_cfg.watchdog_interval_s:
                    self.bus.publish(
                        HeartbeatMsg(
                            msg_type=MessageType.HEARTBEAT,
                            timestamp_utc=self.clock.wall_clock_iso(),
                            subsystem="payload",
                            sequence=heartbeat_seq,
                        )
                    )
                    heartbeat_seq += 1
                    last_heartbeat = now
                safe_commanded, safe_cleared = self.poll_mode_changes()
                self.poll_lock_state()
                self.handle_commands()
                acq = self.sensor.acquire_frame()
                if isinstance(acq, Ok):
                    pos_res = self._read_position()
                    pos: GimbalPosition | None = pos_res.value if isinstance(pos_res, Ok) else None
                    with self.inner_lock:
                        current = holder["state"]
                    current, _outcome = self.process_frame(acq.value, current, now, gimbal_pos=pos)
                    with self.inner_lock:
                        latest = holder["state"]
                        holder["state"] = replace(latest, last_e_az=current.last_e_az)
                else:
                    self._publish_fault(acq.error, "imaging sensor stall")
                    if safe_commanded and not self.lock_gate.engaged:
                        self._actuate_pose(
                            GimbalRequest(
                                mode=GimbalCommandMode.STOW,
                                el_deg=self.controller.gimbal.stow_el_deg,
                                reason="safe_sensor_fault",
                            ),
                            current,
                            frame_id=0,
                        )
                with self.inner_lock:
                    current = holder["state"]
                current, _outer = self.advance_outer(current, now, safe_commanded, safe_cleared)
                with self.inner_lock:
                    latest = holder["state"]
                    holder["state"] = replace(
                        current,
                        encoder=latest.encoder,
                        inner=latest.inner,
                        integrity=latest.integrity,
                    )
                stop_event.wait(timeout=self.controller.cfg.outer.dt_s)
        finally:
            stop_event.set()
            if inner_thread is not None:
                inner_thread.join(timeout=1.0)
            if isinstance(self.gimbal, GimbalRateActuator):
                # The production adapter owns the vendor transport and must
                # confirm inhibit before closing it.  Detailed-plant SIL has
                # no shutdown surface and keeps its existing teardown path.
                self._inhibit_motion("payload app stopping")
                self.gimbal.shutdown()
            self.sensor.stop_acquisition()

    def _store_mask_product(self, inference: InferenceResultMsg) -> None:
        """Persist a compact uint8 thumbnail of the segmentation mask as a science product."""
        mask = np.asarray(inference.mask, dtype=np.float32)
        if mask.ndim != 2 or mask.size == 0:
            return
        step = max(1, mask.shape[0] // 32, mask.shape[1] // 32)
        thumb = (np.clip(mask[::step, ::step], 0.0, 1.0) * 255.0).astype(np.uint8)
        data = thumb.tobytes()
        item_id = f"mask_thumb_{inference.frame_id}"
        result = self.storage.store(item_id, data, DownlinkPriority.SCIENCE_PRODUCT)
        if isinstance(result, Ok):
            self.bus.publish(
                ProductRefMsg(
                    msg_type=MessageType.PRODUCT_REF,
                    timestamp_utc=self.clock.wall_clock_iso(),
                    entry_id=result.value,
                    priority=DownlinkPriority.SCIENCE_PRODUCT,
                    item_id=item_id,
                    byte_len=len(data),
                )
            )

    def _ack_command(
        self, command: RoutedCommandMsg, status: AckStatus, fault: FaultCode, detail: str
    ) -> None:
        """Publish an execution CommandAckMsg for a routed payload command."""
        self.bus.publish(
            CommandAckMsg(
                msg_type=MessageType.COMMAND_ACK,
                timestamp_utc=self.clock.wall_clock_iso(),
                status=status,
                command_id=command.command_id,
                source=command.source,
                seq=command.seq,
                fault_code=fault,
                detail=detail,
            )
        )

    def _publish_fault(self, code: FaultCode, detail: str) -> None:
        """Publish a FaultEventMsg from the payload subsystem onto the bus."""
        self.bus.publish(
            FaultEventMsg(
                msg_type=MessageType.FAULT_EVENT,
                timestamp_utc=self.clock.wall_clock_iso(),
                fault_code=code,
                subsystem="payload",
                detail=detail,
            )
        )

    def _fault_outcome(self, frame_id: int, code: FaultCode, state: ControlState) -> TickOutcome:
        """Build a TickOutcome for a frame that faulted before control ran."""
        return TickOutcome(
            frame_id=frame_id,
            fault=code,
            command_issued=False,
            gimbal_state=state.arbiter.gimbal_state,
        )
