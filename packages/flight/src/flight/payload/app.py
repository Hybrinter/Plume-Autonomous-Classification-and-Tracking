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
    GimbalPosition,
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
    SystemMode,
)
from flight.payload.control import ControlState, IssSample, PayloadController, VisionSample
from flight.payload.gimbal.integrity import check_integrity
from flight.payload.gimbal.request import GimbalRequest
from flight.payload.inference import DetectorBackend
from flight.payload.preprocess import (
    MosaicCalibration,
    calibrate_mosaic,
    compute_quality_flags,
    normalize_dn,
    select_bands,
    separate_bands,
)


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
                safe_cleared = True
                self.safe_latch.commanded = False
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

    def process_frame(
        self,
        raw: MosaicFrame,
        state: ControlState,
        now: float,
        slew_rate_deg_per_s: float = 0.0,
        gimbal_pos: GimbalPosition | None = None,
        safe_commanded: bool = False,
        safe_cleared: bool = False,
    ) -> tuple[ControlState, TickOutcome]:
        """Preprocess, detect, and enqueue a vision sample. Does not write torque.

        SAFE flags are accepted for call-site compatibility; the outer loop applies them.
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

        quality_flags = compute_quality_flags(
            selected.value,
            raw.exposure_us,
            slew_rate_deg_per_s,
            self.sensor_cfg.ifov_band_deg_per_px,
            raw.timestamp_utc,
            self.preprocessing_cfg,
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

        if gimbal_pos is not None:
            theta_g = math.radians(gimbal_pos.el_deg)
        elif state.last_theta_enc_rad is not None:
            theta_g = state.last_theta_enc_rad
        else:
            theta_g = 0.0
        iss, eph_err = self._read_iss_at(raw.timestamp_s if raw.timestamp_s else now)
        if eph_err is not None:
            self._publish_fault(eph_err, "ephemeris read failed")
        new_state, sample = self.controller.ingest_inference(
            state, inference, raw.timestamp_s, raw.exposure_us, theta_g, iss
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
        if self.pose_intent.mode is not None:
            pose_mode = self.pose_intent.mode
            pose_el = self.pose_intent.el_deg
            current = replace(current, pose_mode=pose_mode, pose_el_deg=pose_el)
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
            vision: VisionSample | None = None
            if self.vision_queue:
                sample_t = self.vision_queue[0].t_s
                if sample_t <= t + 1e-9:
                    vision = self.vision_queue.popleft()
            iss, eph_err = self._read_iss_at(t)
            if eph_err is not None:
                self._publish_fault(eph_err, "ephemeris read failed")
            pos = self.gimbal.read_position()
            if isinstance(pos, Ok):
                theta = math.radians(pos.value.el_deg)
                current = replace(current, last_theta_enc_rad=theta)
            elif current.last_theta_enc_rad is not None:
                theta = current.last_theta_enc_rad
            else:
                self._publish_fault(FaultCode.GIMBAL_FAULT, "encoder unavailable")
                current = replace(current, r_rad_s=0.0)
                break
            tick = self.controller.outer_step(
                current,
                t,
                theta,
                vision,
                iss,
                safe_commanded,
                safe_cleared,
                dt,
                timestamp_utc=self.clock.wall_clock_iso(),
            )
            current = tick.state
            if self.lock_gate.engaged:
                current = replace(current, r_rad_s=0.0)
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
        dt = self.controller.cfg.inner.dt_s
        current = state
        self.poll_lock_state()
        if current.last_inner_s is None:
            origin = min(self.clock.monotonic_s(), now)
            current = replace(current, last_inner_s=origin)
        t = current.last_inner_s
        assert t is not None
        gap = now - t
        cap = self.controller.cfg.integrity.catchup_max_s
        if gap > cap:
            self._publish_fault(FaultCode.GIMBAL_FAULT, "inner catch-up cap exceeded")
            t = now - cap
            current = replace(current, last_inner_s=t)
        while t + dt <= now + 1e-12:
            t = t + dt
            pos = self.gimbal.read_position()
            if isinstance(pos, Ok):
                theta = math.radians(pos.value.el_deg)
            elif current.last_theta_enc_rad is not None:
                theta = current.last_theta_enc_rad
            else:
                self._publish_fault(FaultCode.GIMBAL_FAULT, "encoder unavailable")
                current = replace(current, r_rad_s=0.0)
                send = self.gimbal.set_torque(0.0)
                if isinstance(send, Err):
                    self._publish_fault(send.error, "gimbal torque failed")
                break
            enc_rate = 0.0
            if current.last_theta_enc_rad is not None and dt > 0.0:
                enc_rate = (theta - current.last_theta_enc_rad) / dt
            locked = self.lock_gate.engaged
            tick = self.controller.inner_step(
                current, t, theta, dt, locked=locked, safe_latched=self.safe_latch.commanded
            )
            current = tick.state
            integrity = check_integrity(
                self.controller.cfg.integrity,
                current.r_rad_s,
                current.y_m,
                tick.tau_nm,
                enc_rate,
                locked,
                current.integrity_freeze_strikes,
                current.integrity_lock_strikes,
            )
            current = replace(
                current,
                integrity_freeze_strikes=integrity.freeze_strikes,
                integrity_lock_strikes=integrity.lock_fight_strikes,
            )
            if integrity.fault is not None:
                self._publish_fault(integrity.fault, "pointing integrity trip")
                self.safe_latch.commanded = True
            tau = 0.0 if locked else tick.tau_nm
            send = self.gimbal.set_torque(tau)
            if isinstance(send, Err):
                self._publish_fault(send.error, "gimbal torque failed")
        return current

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
        prev_pos: GimbalPosition | None = None
        prev_pos_now = 0.0

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
                    if snap.last_inner_s is None:
                        holder["state"] = replace(snap, last_inner_s=now_inner)
                        snap = holder["state"]
                if snap.last_inner_s == now_inner:
                    stop_event.wait(timeout=dt)
                    continue
                pos = self.gimbal.read_position()
                if isinstance(pos, Ok):
                    theta = math.radians(pos.value.el_deg)
                elif snap.last_theta_enc_rad is not None:
                    theta = snap.last_theta_enc_rad
                else:
                    self._publish_fault(FaultCode.GIMBAL_FAULT, "encoder unavailable")
                    send = self.gimbal.set_torque(0.0)
                    if isinstance(send, Err):
                        self._publish_fault(send.error, "gimbal torque failed")
                    with self.inner_lock:
                        latest = holder["state"]
                        holder["state"] = replace(latest, r_rad_s=0.0)
                    stop_event.wait(timeout=dt)
                    continue
                enc_rate = 0.0
                if snap.last_theta_enc_rad is not None and dt > 0.0:
                    enc_rate = (theta - snap.last_theta_enc_rad) / dt
                tick = self.controller.inner_step(
                    snap, now_inner, theta, dt, locked=locked, safe_latched=safe
                )
                integrity = check_integrity(
                    self.controller.cfg.integrity,
                    tick.state.r_rad_s,
                    tick.state.y_m,
                    tick.tau_nm,
                    enc_rate,
                    locked,
                    snap.integrity_freeze_strikes,
                    snap.integrity_lock_strikes,
                )
                if integrity.fault is not None:
                    self._publish_fault(integrity.fault, "pointing integrity trip")
                    self.safe_latch.commanded = True
                tau = 0.0 if locked else tick.tau_nm
                send = self.gimbal.set_torque(tau)
                if isinstance(send, Err):
                    self._publish_fault(send.error, "gimbal torque failed")
                with self.inner_lock:
                    latest = holder["state"]
                    merged_r = (
                        tick.state.r_rad_s
                        if (locked or safe or latest.pose_mode is not None)
                        else latest.r_rad_s
                    )
                    holder["state"] = replace(
                        latest,
                        encoder_ring=tick.state.encoder_ring,
                        integrator=tick.state.integrator,
                        y_m=tick.state.y_m,
                        r_rad_s=merged_r,
                        last_inner_s=now_inner,
                        last_theta_enc_rad=theta,
                        last_tau_nm=tau,
                        integrity_freeze_strikes=integrity.freeze_strikes,
                        integrity_lock_strikes=integrity.lock_fight_strikes,
                    )
                stop_event.wait(timeout=dt)

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
                    slew_rate = 0.0
                    pos_res = self.gimbal.read_position()
                    pos: GimbalPosition | None = None
                    if isinstance(pos_res, Ok):
                        pos = pos_res.value
                        if prev_pos is not None and now > prev_pos_now:
                            slew_rate = abs(pos.el_deg - prev_pos.el_deg) / (now - prev_pos_now)
                        prev_pos = pos
                        prev_pos_now = now
                    with self.inner_lock:
                        current = holder["state"]
                    current, _outcome = self.process_frame(acq.value, current, now, slew_rate, pos)
                    with self.inner_lock:
                        latest = holder["state"]
                        holder["state"] = replace(latest, last_e_az=current.last_e_az)
                else:
                    self._publish_fault(acq.error, "imaging sensor stall")
                    if safe_commanded and not self.lock_gate.engaged:
                        self.gimbal.stow()
                with self.inner_lock:
                    current = holder["state"]
                current, _outer = self.advance_outer(current, now, safe_commanded, safe_cleared)
                with self.inner_lock:
                    latest = holder["state"]
                    holder["state"] = replace(
                        current,
                        encoder_ring=latest.encoder_ring,
                        integrator=latest.integrator,
                        y_m=latest.y_m,
                        last_inner_s=latest.last_inner_s,
                        last_theta_enc_rad=latest.last_theta_enc_rad,
                        last_tau_nm=latest.last_tau_nm,
                        integrity_freeze_strikes=latest.integrity_freeze_strikes,
                        integrity_lock_strikes=latest.integrity_lock_strikes,
                    )
                stop_event.wait(timeout=self.controller.cfg.outer.dt_s)
        finally:
            stop_event.set()
            inner_thread.join(timeout=1.0)
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
