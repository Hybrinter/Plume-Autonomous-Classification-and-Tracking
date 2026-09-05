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
    FaultEventMsg,
    GimbalCommandMsg,
    HeartbeatMsg,
    InferenceResultMsg,
    LaunchLockStateMsg,
    ModeChangeMsg,
    ProcessedFrameMsg,
    ProductRefMsg,
    TelemetryEventMsg,
)
from flight.libs.time import Clock
from flight.libs.types import (
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
    """Mutable launch-lock view the payload uses to inhibit gimbal motion while ENGAGED.

    Fields:
        engaged: True when the latest LaunchLockStateMsg reported ENGAGED. Defaults False so a
            payload that never hears about a launch lock (e.g. a unit test with no mechanical
            app) does not gate motion; in the full system the mechanical app publishes the lock
            state every cycle, so the gate tracks the real ENGAGED/RELEASED transitions.
    """

    engaged: bool = False


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
        mode_sub, lock_sub: Bus subscriptions.
        lock_gate: Launch-lock inhibit.
        vision_queue: In-process vision samples (not the MessageBus).
        inner_lock: Serializes ControlState between the inner thread and the outer path.
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
    lock_gate: LockGate = field(default_factory=LockGate)
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
            lock_gate=LockGate(),
            vision_queue=deque(maxlen=cfg.controller.vision_queue_depth),
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
            else:
                safe_cleared = True
        return safe_commanded, safe_cleared

    def poll_lock_state(self) -> None:
        """Drain pending LaunchLockStateMsg and update the launch-lock inhibit gate."""
        while not self.lock_sub.empty():
            self.lock_gate.engaged = self.lock_sub.get_nowait().state is LaunchLockState.ENGAGED

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
        del now, gimbal_pos, safe_commanded, safe_cleared
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

        new_state, sample = self.controller.ingest_inference(
            state, inference, raw.timestamp_s, raw.exposure_us
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
        dt = self.controller.cfg.dt_outer_s
        current = state
        command_issued = False
        t = current.last_outer_s
        while t + dt <= now + 1e-12:
            t = t + dt
            vision: VisionSample | None = None
            if self.vision_queue:
                sample_t = self.vision_queue[0].t_s
                last_tick = t + dt > now + 1e-12
                if sample_t <= t + 1e-9 or (last_tick and sample_t <= now + 1e-9):
                    vision = self.vision_queue.popleft()
            iss = self._read_iss()
            pos = self.gimbal.read_position()
            theta = math.radians(pos.value.el_deg) if isinstance(pos, Ok) else 0.0
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
            safe_commanded = False
            safe_cleared = False
        outcome = TickOutcome(
            frame_id=0,
            fault=None,
            command_issued=command_issued,
            gimbal_state=current.arbiter.gimbal_state,
        )
        return current, outcome

    def advance_inner(self, state: ControlState, now: float) -> ControlState:
        """Catch up the inner loop to `now` in T_in steps and write torque."""
        dt = self.controller.cfg.dt_inner_s
        current = state
        t = current.last_inner_s
        while t + dt <= now + 1e-12:
            t = t + dt
            pos = self.gimbal.read_position()
            theta = math.radians(pos.value.el_deg) if isinstance(pos, Ok) else 0.0
            tick = self.controller.inner_step(current, t, theta, dt)
            current = tick.state
            if not self.lock_gate.engaged:
                send = self.gimbal.set_torque(tick.tau_nm)
                if isinstance(send, Err):
                    self._publish_fault(send.error, "gimbal torque failed")
        return current

    def _read_iss(self) -> IssSample | None:
        """Read ISS ECI state; None on Err (predictor then uses omega_t_nom=0)."""
        result = self.ephemeris.read_state(self.clock.utc_s())
        if isinstance(result, Err):
            return None
        value = result.value
        return IssSample(r_m=value.r_m, v_m_s=value.v_m_s, utc_s=value.epoch_utc_s)

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
            """Inner rate loop: catch up and write torque until stop."""
            while not stop_event.is_set():
                now_inner = self.clock.monotonic_s()
                with self.inner_lock:
                    holder["state"] = self.advance_inner(holder["state"], now_inner)
                stop_event.wait(timeout=self.controller.cfg.dt_inner_s)

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
                acq = self.sensor.acquire_frame()
                with self.inner_lock:
                    current = holder["state"]
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
                        current, _outcome = self.process_frame(
                            acq.value, current, now, slew_rate, pos
                        )
                    else:
                        self._publish_fault(acq.error, "imaging sensor stall")
                        if safe_commanded and not self.lock_gate.engaged:
                            self.gimbal.stow()
                    current, _outer = self.advance_outer(current, now, safe_commanded, safe_cleared)
                    holder["state"] = current
                stop_event.wait(timeout=self.controller.cfg.dt_outer_s)
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
