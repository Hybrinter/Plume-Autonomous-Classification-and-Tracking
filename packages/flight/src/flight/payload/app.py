"""Payload application shell: binds the HAL and the pure payload core into one loop.

Collapses the legacy imaging + inference + controller processes into a single
in-process payload app. Per frame: acquire a raw mosaic frame from the imaging sensor,
preprocess it co-located (calibrate the raw mosaic plane -> CFA-separate into band planes
-> normalize -> select bands -> quality flags; no queue round-trip, honoring the
preprocessing co-location invariant), run the swappable detector, and ingest vision.
A control thread (or SIL apply_control) runs inner PI torque ticks and outer LQG ticks
and drives the gimbal HAL. All decision logic lives in PayloadController; this module
owns only I/O, sequencing, and message construction.

Contains:
  - TickOutcome: per-frame result summary (frame id, fault code, command-issued flag,
    resulting gimbal state) used for telemetry and testing.
  - PayloadApp: frozen holder of injected services, including the MosaicCalibration and
    the SensorConfig geometry. from_config() assembles it from a PactConfig, concrete
    drivers, and an injected MosaicCalibration, validating sensor/inference geometry at
    startup; process_frame() ingests vision; apply_control() runs outer plus inner ticks;
    run() is the imaging loop plus a control thread (emits heartbeats, publishes a fault
    on camera stall).

Non-obvious notes:
  - The arbiter `now` is sourced from Clock.monotonic_s() (it consumes `now` only as
    interval/rate-limit deltas); message timestamps use Clock.wall_clock_iso().
  - The inference tensor is the full demosaiced band plane (crop_origin_px=(0, 0),
    scale_factor=1). process_frame ingests vision only. A control thread (or SIL
    apply_control) runs inner torque ticks and outer LQG ticks.
  - The MOTION_SMEAR quality gate consumes a slew rate; run() derives it from consecutive
    gimbal encoder reads and degrades to 0.0 (never-flag) on the first frame or a failed
    read.

Satisfies: REQ-AIML-COMP-001, REQ-AIML-COMP-002 (payload process orchestration),
           REQ-OPER-HIGH-002 (subsystem app loop).
"""

from __future__ import annotations

# stdlib
import math
import threading
from dataclasses import dataclass, field, replace

# third-party
import numpy as np

# internal
from flight.hal.interfaces import GimbalActuator, GimbalPosition, ImagingSensor, StorageWriter
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
from flight.payload.control import ControlState, PayloadController
from flight.payload.gimbal import GimbalRequest, reset_servo
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
        command_issued: True if a GimbalCommandMsg was sent to the gimbal this frame.
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
    """Payload subsystem app: imperative shell around the pure payload core.

    Holds the injected HAL drivers, detector, pure controller, bus, clock, mosaic
    calibration, and the config slices needed for preprocessing, sensor geometry, and
    heartbeats. Frozen to prevent field reassignment; the held services are themselves
    mutable (consistent with the composition-root injection pattern).

    Attributes:
        sensor: ImagingSensor driver (sim or real), acquire-only mosaic contract.
        gimbal: GimbalActuator driver (sim or real).
        detector: DetectorBackend (ScriptedDetector or OnnxDetector composer).
        controller: The pure PayloadController.
        bus: The typed MessageBus to publish onto.
        clock: Injected Clock (RealClock in flight, ManualClock in tests).
        calib: MosaicCalibration applied to the raw mosaic plane (identity in SIL).
        sensor_cfg: SensorConfig (mosaic geometry, bit depth, IFOV).
        inference_cfg: InferenceConfig (band selection + input geometry).
        preprocessing_cfg: PreprocessingConfig (quality thresholds).
        fault_cfg: FaultConfig (heartbeat interval).
        mode_sub: Subscription to ModeChangeMsg; drained each frame for SAFE entry/exit.
    """

    sensor: ImagingSensor
    gimbal: GimbalActuator
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

    @staticmethod
    def from_config(
        cfg: PactConfig,
        sensor: ImagingSensor,
        gimbal: GimbalActuator,
        detector: DetectorBackend,
        bus: MessageBus,
        clock: Clock,
        calib: MosaicCalibration,
        storage: StorageWriter,
    ) -> PayloadApp:
        """Assemble a PayloadApp from a PactConfig, injected services, and a calibration.

        Builds the pure PayloadController from cfg.controller and carries cfg.sensor,
        cfg.inference, cfg.preprocessing, and cfg.fault for the loop. The drivers,
        detector, bus, clock, and MosaicCalibration are injected by the caller (the
        composition root chooses real vs sim and loads/identity-builds the calibration).

        Inputs:
            cfg (PactConfig): Top-level configuration.
            sensor (ImagingSensor): Imaging sensor driver (sim or real).
            gimbal (GimbalActuator): Gimbal actuator driver (sim or real).
            detector (DetectorBackend): Detector backend (scripted or ONNX).
            bus (MessageBus): The typed bus to publish onto.
            clock (Clock): Injected clock (RealClock in flight, ManualClock in tests).
            calib (MosaicCalibration): Per-pixel mosaic calibration (identity in SIL).

        Outputs:
            PayloadApp: A fully constructed payload app.

        Raises:
            ValueError: If the sensor mosaic dimensions are odd, the band plane is not
                equal to the inference input on both axes, the mosaic_layout does not name
                each Band exactly once, or input_bands is not a subset of mosaic_layout.
                Raising is correct here: composition-root startup is the one place a bad
                config is unrecoverable.
        """
        if cfg.sensor.width_px % 2 or cfg.sensor.height_px % 2:
            raise ValueError("sensor mosaic dimensions must be even")
        plane_h, plane_w = cfg.sensor.height_px // 2, cfg.sensor.width_px // 2
        if plane_h != cfg.inference.input_height_px or plane_w != cfg.inference.input_width_px:
            raise ValueError("band plane size must equal the inference input")
        if sorted(cfg.sensor.mosaic_layout) != sorted(b.value for b in Band):
            raise ValueError("mosaic_layout must name each Band exactly once")
        if any(b not in cfg.sensor.mosaic_layout for b in cfg.inference.input_bands):
            raise ValueError("input_bands must be a subset of mosaic_layout")
        return PayloadApp(
            sensor=sensor,
            gimbal=gimbal,
            detector=detector,
            controller=PayloadController.from_config(cfg.controller, cfg.sensor, cfg.gimbal),
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
        )

    def poll_mode_changes(self) -> tuple[bool, bool]:
        """Drain pending ModeChangeMsg; return (safe_commanded, safe_cleared).

        SAFE requests latch the payload via the arbiter; any non-SAFE mode message is the
        ground-commanded recovery signal. Both may be True in one drain (last writer wins
        downstream: the arbiter applies safe_commanded first).

        Outputs:
            tuple[bool, bool]: (safe_commanded, safe_cleared) over all drained messages.
        """
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
        """Drain pending LaunchLockStateMsg and update the launch-lock motion-inhibit gate.

        The gate is set ENGAGED only by an explicit ENGAGED state and cleared by any other
        state (RELEASED/UNKNOWN -> motion permitted, since the lock is no longer holding). The
        latest message wins; with no messages the gate is unchanged (sticky).

        Outputs:
            None.
        """
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
        """Process one raw mosaic frame: preprocess, detect, ingest vision.

        Runs the co-located preprocessing pipeline (calibrate the raw mosaic plane ->
        CFA-separate -> normalize -> select bands -> quality flags), then the detector,
        then PayloadController.ingest_vision. The full demosaiced plane is the inference
        tensor (crop origin (0, 0), scale 1). Inner and outer control ticks run in
        apply_control / the control thread, not here.

        Inputs:
            raw (MosaicFrame): Raw mosaic frame; raw.mosaic must match the calibration
                shape (sensor height_px x width_px).
            state (ControlState): Control state carried from the previous frame.
            now (float): Monotonic seconds used as shutter time when capture_monotonic_s
                is 0.
            slew_rate_deg_per_s (float): Gimbal slew rate over the exposure for the
                MOTION_SMEAR gate; defaults to 0.0 (never-flag).
            gimbal_pos (GimbalPosition | None): Unused on the ingest path; kept for call
                compatibility.
            safe_commanded (bool): Unused on the ingest path; kept for call compatibility.
            safe_cleared (bool): Unused on the ingest path; kept for call compatibility.

        Outputs:
            tuple[ControlState, TickOutcome]: (new_state, outcome). new_state is unchanged
            on a fault before ingest. command_issued is always False here.
        """
        mosaic = np.asarray(raw.mosaic, dtype=np.float32)  # np.ndarray[float32, (H, W)]

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
            self.sensor_cfg.ifov_deg_per_px,
            raw.timestamp_utc,
            self.preprocessing_cfg,
        )

        plane_h = selected.value.shape[1]
        plane_w = selected.value.shape[2]
        in_h = self.inference_cfg.input_height_px
        in_w = self.inference_cfg.input_width_px
        if plane_h != in_h or plane_w != in_w:
            self._publish_fault(
                FaultCode.FRAME_MALFORMED, f"plane size mismatch frame_id={raw.frame_id}"
            )
            return state, self._fault_outcome(raw.frame_id, FaultCode.FRAME_MALFORMED, state)
        tensor = selected.value
        crop_origin = (0, 0)
        scale = 1.0

        processed = ProcessedFrameMsg(
            msg_type=MessageType.PROCESSED_FRAME,
            timestamp_utc=raw.timestamp_utc,
            frame_id=raw.frame_id,
            tensor=tensor,  # np.ndarray[float32, (len(input_bands), input_h, input_w)]
            quality_flags=quality_flags,
            crop_origin_px=crop_origin,
            scale_factor=scale,
        )

        detect_result = self.detector.detect(processed)
        if isinstance(detect_result, Err):
            self._publish_fault(detect_result.error, f"detection failed frame_id={raw.frame_id}")
            return state, self._fault_outcome(raw.frame_id, detect_result.error, state)
        inference = detect_result.value
        self.bus.publish(inference)
        self._store_mask_product(inference)

        shutter = raw.capture_monotonic_s if raw.capture_monotonic_s > 0.0 else now
        new_state = self.controller.ingest_vision(state, inference, shutter)
        del gimbal_pos, safe_commanded, safe_cleared

        outcome = TickOutcome(
            frame_id=raw.frame_id,
            fault=None,
            command_issued=False,
            gimbal_state=new_state.arbiter.gimbal_state,
        )
        return new_state, outcome

    def apply_control(
        self,
        state: ControlState,
        now: float,
        dt: float,
        safe_commanded: bool = False,
        safe_cleared: bool = False,
    ) -> tuple[ControlState, TickOutcome]:
        """Run one outer LQG tick and inner rate-PI ticks; actuate torque or stow.

        Inputs:
            state (ControlState): Control state after ingest (or the previous tick).
            now (float): Monotonic seconds at the end of this control interval.
            dt (float): Interval length in seconds. Inner ticks chunk at dt_inner_max_s.
            safe_commanded (bool): True to latch SAFE and stow.
            safe_cleared (bool): True to exit SAFE to IDLE.

        Outputs:
            tuple[ControlState, TickOutcome]: Updated state and whether a command issued.
        """
        pos: GimbalPosition | None = None
        pos_res = self.gimbal.read_position()
        if isinstance(pos_res, Ok):
            pos = pos_res.value
        outer_dt = dt if dt > 0.0 else self.controller.cfg.dt_outer_min_s
        new_state, request, telemetry, _fault = self.controller.step_outer(
            state, pos, now, outer_dt, safe_commanded, safe_cleared
        )
        for event in telemetry:
            self.bus.publish(event)

        actuated = False
        if new_state.arbiter.gimbal_state is GimbalState.SAFE:
            if request is not None and request.mode is GimbalCommandMode.STOW:
                send_result = self.gimbal.stow()
                if isinstance(send_result, Err):
                    self._publish_fault(send_result.error, "gimbal stow failed")
                self._publish_gimbal_command(request, new_state, 0)
                actuated = True
        elif self.lock_gate.engaged:
            if request is not None:
                self.bus.publish(
                    TelemetryEventMsg(
                        msg_type=MessageType.TELEMETRY_EVENT,
                        timestamp_utc=self.clock.wall_clock_iso(),
                        subsystem="payload",
                        event_name="gimbal_motion_inhibited",
                        payload={"reason": "launch_lock_engaged", "mode": request.mode.value},
                    )
                )
        else:
            inner_remaining = dt if dt > 0.0 else self.controller.cfg.dt_inner_min_s
            max_inner = self.controller.cfg.dt_inner_max_s
            tau = (0.0, 0.0)
            while inner_remaining > 0.0:
                chunk = inner_remaining if inner_remaining <= max_inner else max_inner
                pos_res = self.gimbal.read_position()
                pos = pos_res.value if isinstance(pos_res, Ok) else pos
                new_state, tau = self.controller.step_inner(new_state, pos, chunk)
                inner_remaining -= chunk
            send_result = self.gimbal.set_torque(tau[0], tau[1])
            if isinstance(send_result, Err):
                self._publish_fault(send_result.error, "gimbal torque failed")
            if request is not None:
                self._publish_gimbal_command(request, new_state, 0)
                actuated = True

        outcome = TickOutcome(
            frame_id=0,
            fault=None,
            command_issued=actuated,
            gimbal_state=new_state.arbiter.gimbal_state,
        )
        return new_state, outcome

    def _publish_gimbal_command(
        self,
        request: GimbalRequest,
        state: ControlState,
        frame_id: int,
    ) -> None:
        """Publish a GimbalCommandMsg telemetry record for an issued request."""
        self.bus.publish(
            GimbalCommandMsg(
                msg_type=MessageType.GIMBAL_COMMAND,
                timestamp_utc=self.clock.wall_clock_iso(),
                frame_id=frame_id,
                mode=request.mode,
                az_value_deg=request.az_deg,
                el_value_deg=request.el_deg,
                state=state.arbiter.gimbal_state,
                reason=request.reason,
            )
        )

    def run(self, stop_event: threading.Event) -> None:
        """Run the payload acquisition loop until stop_event is set.

        Starts acquisition and a control thread, then repeatedly: emits a HeartbeatMsg
        every fault_cfg.watchdog_interval_s, acquires a frame, computes the gimbal slew
        rate from consecutive encoder reads, and ingests the frame. The control thread
        waits dt_inner_min_s, reads the encoder, steps the inner PI onto set_torque, and
        periodically steps the outer LQG. Stops acquisition on exit.

        Inputs:
            stop_event (threading.Event): The loop exits cleanly once it is set.

        Outputs:
            None.

        Notes:
            The slew rate is the angular speed between the previous and current gimbal
            positions divided by the elapsed monotonic seconds; it is 0.0 on the first
            frame, when no time has elapsed, or when the position read fails, so the
            MOTION_SMEAR gate degrades gracefully. SAFE/recovery mode messages are drained
            each imaging iteration and stored for the next outer tick. If SAFE is commanded
            while frame acquisition fails, stow() is called and the PI integrator is reset.
        """
        self.sensor.start_acquisition()
        shared: list[ControlState] = [self.controller.initial_state()]
        state_lock = threading.Lock()
        safe_flags = {"commanded": False, "cleared": False}
        heartbeat_seq = 0
        last_heartbeat = self.clock.monotonic_s()
        prev_pos: GimbalPosition | None = None
        prev_pos_now = 0.0
        inner_wait = self.controller.cfg.dt_inner_min_s
        outer_period = self.controller.cfg.dt_outer_min_s

        def control_loop() -> None:
            """Inner PI plus periodic outer LQG on a dedicated thread."""
            last_outer = self.clock.monotonic_s()
            last_inner = last_outer
            while not stop_event.wait(timeout=inner_wait):
                now_c = self.clock.monotonic_s()
                self.poll_lock_state()
                dt_inner = now_c - last_inner
                last_inner = now_c
                if dt_inner <= 0.0:
                    dt_inner = inner_wait
                with state_lock:
                    local = shared[0]
                    commanded = safe_flags["commanded"]
                    cleared = safe_flags["cleared"]
                    safe_flags["commanded"] = False
                    safe_flags["cleared"] = False
                    do_outer = commanded or cleared or (now_c - last_outer) >= outer_period
                    if do_outer:
                        outer_dt = now_c - last_outer
                        last_outer = now_c
                        if outer_dt <= 0.0:
                            outer_dt = outer_period
                        pos_res = self.gimbal.read_position()
                        pos = pos_res.value if isinstance(pos_res, Ok) else None
                        local, request, telemetry, _fault = self.controller.step_outer(
                            local, pos, now_c, outer_dt, commanded, cleared
                        )
                        for event in telemetry:
                            self.bus.publish(event)
                        if local.arbiter.gimbal_state is GimbalState.SAFE:
                            if request is not None and request.mode is GimbalCommandMode.STOW:
                                send_result = self.gimbal.stow()
                                if isinstance(send_result, Err):
                                    self._publish_fault(send_result.error, "gimbal stow failed")
                                self._publish_gimbal_command(request, local, 0)
                            shared[0] = local
                            continue
                        if self.lock_gate.engaged:
                            if request is not None:
                                self.bus.publish(
                                    TelemetryEventMsg(
                                        msg_type=MessageType.TELEMETRY_EVENT,
                                        timestamp_utc=self.clock.wall_clock_iso(),
                                        subsystem="payload",
                                        event_name="gimbal_motion_inhibited",
                                        payload={
                                            "reason": "launch_lock_engaged",
                                            "mode": request.mode.value,
                                        },
                                    )
                                )
                            shared[0] = local
                            continue
                        if request is not None:
                            self._publish_gimbal_command(request, local, 0)
                    if (
                        local.arbiter.gimbal_state is not GimbalState.SAFE
                        and not self.lock_gate.engaged
                    ):
                        pos_res = self.gimbal.read_position()
                        pos = pos_res.value if isinstance(pos_res, Ok) else None
                        local, tau = self.controller.step_inner(local, pos, dt_inner)
                        send_result = self.gimbal.set_torque(tau[0], tau[1])
                        if isinstance(send_result, Err):
                            self._publish_fault(send_result.error, "gimbal torque failed")
                    shared[0] = local

        control_thread = threading.Thread(target=control_loop, name="payload-control", daemon=True)
        control_thread.start()
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
                if safe_commanded or safe_cleared:
                    with state_lock:
                        safe_flags["commanded"] = safe_flags["commanded"] or safe_commanded
                        safe_flags["cleared"] = safe_flags["cleared"] or safe_cleared
                acq = self.sensor.acquire_frame()
                if isinstance(acq, Ok):
                    slew_rate = 0.0
                    pos: GimbalPosition | None = None
                    pos_res = self.gimbal.read_position()
                    if isinstance(pos_res, Ok):
                        pos = pos_res.value
                        if prev_pos is not None and now > prev_pos_now:
                            d_az = pos_res.value.az_deg - prev_pos.az_deg
                            d_el = pos_res.value.el_deg - prev_pos.el_deg
                            slew_rate = math.hypot(d_az, d_el) / (now - prev_pos_now)
                        prev_pos = pos_res.value
                        prev_pos_now = now
                    with state_lock:
                        ingested, _outcome = self.process_frame(
                            acq.value,
                            shared[0],
                            now,
                            slew_rate,
                            pos,
                            False,
                            False,
                        )
                        shared[0] = ingested
                else:
                    self._publish_fault(acq.error, "imaging sensor stall")
                    if safe_commanded:
                        self.gimbal.stow()
                        with state_lock:
                            shared[0] = replace(shared[0], servo=reset_servo(shared[0].servo))
        finally:
            stop_event.set()
            control_thread.join(timeout=1.0)
            self.sensor.stop_acquisition()

    def _store_mask_product(self, inference: InferenceResultMsg) -> None:
        """Persist a compact uint8 thumbnail of the segmentation mask as a science product.

        The mask is a science product (spec Section 4): it is decimated to at most 32x32 and
        quantized to bytes, stored via the injected StorageWriter (bypassing the bus -- the
        large-artifact invariant), and advertised on the bus as a compact ProductRefMsg the
        downlink manager can prioritize. A storage failure is swallowed here (the StorageWriter
        already published a STORAGE_FULL fault); the frame loop continues.

        Inputs:
            inference (InferenceResultMsg): The detection result whose mask is stored.

        Outputs:
            None.
        """
        mask = np.asarray(inference.mask, dtype=np.float32)  # np.ndarray[float32, (H, W)]
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
        """Publish a FaultEventMsg from the payload subsystem onto the bus.

        Inputs:
            code (FaultCode): The fault code to report.
            detail (str): Human-readable detail string for logging/telemetry.

        Outputs:
            None.
        """
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
        """Build a TickOutcome for a frame that faulted before control ran.

        Inputs:
            frame_id (int): The frame_id that faulted.
            code (FaultCode): The fault code raised.
            state (ControlState): The unchanged control state (its arbiter state is
                reported).

        Outputs:
            TickOutcome: With command_issued=False and the prior gimbal state.
        """
        return TickOutcome(
            frame_id=frame_id,
            fault=code,
            command_issued=False,
            gimbal_state=state.arbiter.gimbal_state,
        )
