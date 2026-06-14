"""Payload application shell: binds the HAL and the pure payload core into one loop.

Collapses the legacy imaging + inference + controller processes into a single
in-process payload app. Per frame: acquire a raw mosaic frame from the imaging sensor,
preprocess it co-located (calibrate the raw mosaic plane -> CFA-separate into band planes
-> normalize -> select bands -> quality flags; no queue round-trip, honoring the
preprocessing co-location invariant), run the swappable detector, step the pure
PayloadController, then drive the gimbal HAL and publish results onto the typed bus. All
decision logic lives in PayloadController; this module owns only I/O, sequencing, and
message construction.

Contains:
  - TickOutcome: per-frame result summary (frame id, fault code, command-issued flag,
    resulting gimbal state) used for telemetry and testing.
  - PayloadApp: frozen holder of injected services, including the MosaicCalibration and
    the SensorConfig geometry. from_config() assembles it from a PactConfig, concrete
    drivers, and an injected MosaicCalibration, validating sensor/inference geometry at
    startup; process_frame() runs one frame end-to-end; run() is the acquisition loop
    (emits heartbeats, computes the slew rate from gimbal reads, publishes a fault on
    camera stall).

Non-obvious notes:
  - The arbiter `now` is sourced from Clock.monotonic_s() (it consumes `now` only as
    interval/rate-limit deltas); message timestamps use Clock.wall_clock_iso().
  - The inference ROI is mode-dependent: outside TRACKING the full band plane is
    decimated to the inference input size (crop_origin_px=(0, 0), scale_factor=1/factor);
    in TRACKING with an initialized estimator a full-resolution ROI is cropped around the
    Kalman-estimated target (scale_factor=1.0). Quality flags always run on the full
    plane before the ROI is taken.
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
from dataclasses import dataclass

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
    ModeChangeMsg,
    ProcessedFrameMsg,
    ProductRefMsg,
)
from flight.libs.time import Clock
from flight.libs.types import (
    Band,
    DownlinkPriority,
    Err,
    FaultCode,
    GimbalCommandMode,
    GimbalState,
    MessageType,
    MosaicFrame,
    Ok,
    SystemMode,
)
from flight.payload.control import ControlState, PayloadController
from flight.payload.model import DetectorBackend
from flight.payload.preprocess import (
    MosaicCalibration,
    calibrate_mosaic,
    compute_quality_flags,
    crop_to_roi,
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
        detector: DetectorBackend (ScriptedDetector or OnnxDetector).
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
            ValueError: If the sensor mosaic dimensions are odd, the band plane is smaller
                than the inference input, the plane is not an equal integer multiple of the
                inference input on both axes (required for uniform search-mode decimation),
                the mosaic_layout does not name each Band exactly once, or input_bands is
                not a subset of mosaic_layout. Raising is correct here: composition-root
                startup is the one place a bad config is unrecoverable.
        """
        if cfg.sensor.width_px % 2 or cfg.sensor.height_px % 2:
            raise ValueError("sensor mosaic dimensions must be even")
        plane_h, plane_w = cfg.sensor.height_px // 2, cfg.sensor.width_px // 2
        if plane_h < cfg.inference.input_height_px or plane_w < cfg.inference.input_width_px:
            raise ValueError("band plane must be at least the inference input size")
        if (
            plane_h % cfg.inference.input_height_px
            or plane_w % cfg.inference.input_width_px
            or plane_h // cfg.inference.input_height_px != plane_w // cfg.inference.input_width_px
        ):
            raise ValueError("plane size must be an integer multiple of the inference input")
        if sorted(cfg.sensor.mosaic_layout) != sorted(b.value for b in Band):
            raise ValueError("mosaic_layout must name each Band exactly once")
        if any(b not in cfg.sensor.mosaic_layout for b in cfg.inference.input_bands):
            raise ValueError("input_bands must be a subset of mosaic_layout")
        return PayloadApp(
            sensor=sensor,
            gimbal=gimbal,
            detector=detector,
            controller=PayloadController.from_config(cfg.controller, cfg.sensor),
            bus=bus,
            clock=clock,
            calib=calib,
            storage=storage,
            sensor_cfg=cfg.sensor,
            inference_cfg=cfg.inference,
            preprocessing_cfg=cfg.preprocessing,
            fault_cfg=cfg.fault,
            mode_sub=bus.subscribe(ModeChangeMsg),
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
        """Process one raw mosaic frame end-to-end: preprocess -> detect -> control -> actuate.

        Runs the co-located preprocessing pipeline (calibrate the raw mosaic plane ->
        CFA-separate -> normalize -> select bands -> quality flags -> mode-dependent ROI:
        decimated full plane in search, full-resolution Kalman-centered crop in TRACKING),
        then the detector, then the pure PayloadController. Publishes InferenceResultMsg
        and each arbiter TelemetryEventMsg; when a request is issued it is mapped onto
        the GimbalActuator HAL (set_rate/goto_angle/stow/home by mode) and a
        GimbalCommandMsg telemetry record is published. A control fault (deadband strike
        or encoder runaway) publishes a FaultEventMsg. On a preprocessing or detection
        fault the state is returned unchanged, a FaultEventMsg is published, and
        outcome.fault is set.

        Inputs:
            raw (MosaicFrame): Raw mosaic frame; raw.mosaic must match the calibration
                shape (sensor height_px x width_px).
            state (ControlState): Control state carried from the previous frame.
            now (float): Monotonic seconds for the arbiter (interval/rate-limit deltas).
            slew_rate_deg_per_s (float): Gimbal slew rate over the exposure for the
                MOTION_SMEAR gate; defaults to 0.0 (never-flag).
            gimbal_pos (GimbalPosition | None): Latest encoder read for the runaway monitor.
            safe_commanded (bool): True to latch SAFE and stow this frame.
            safe_cleared (bool): True to exit SAFE to IDLE this frame.

        Outputs:
            tuple[ControlState, TickOutcome]: (new_state, outcome). new_state is unchanged
            on a fault before control.
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

        plane_h, plane_w = selected.value.shape[1], selected.value.shape[2]
        in_tracking = state.arbiter.gimbal_state is GimbalState.TRACKING and state.ema.initialized
        if in_tracking:
            # Full-resolution ROI centered on the Kalman-estimated boresight-error target.
            est_az = float(state.kalman.x[0])
            est_el = float(state.kalman.x[1])
            center_x = int(plane_w / 2 + est_az / self.sensor_cfg.ifov_deg_per_px)
            center_y = int(plane_h / 2 - est_el / self.sensor_cfg.ifov_deg_per_px)
            tensor, crop_origin = crop_to_roi(
                selected.value,
                (center_x, center_y),
                (self.inference_cfg.input_height_px, self.inference_cfg.input_width_px),
            )
            scale = 1.0
        else:
            # Decimated full-plane search mode.
            factor = plane_h // self.inference_cfg.input_height_px
            tensor = selected.value[:, ::factor, ::factor]
            crop_origin = (0, 0)
            scale = 1.0 / factor

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

        new_state, request, telemetry, ctrl_fault = self.controller.step(
            state, inference, now, gimbal_pos, safe_commanded, safe_cleared
        )
        for event in telemetry:
            self.bus.publish(event)
        if ctrl_fault is not None:
            self._publish_fault(ctrl_fault, f"control fault frame_id={raw.frame_id}")

        if request is not None:
            if request.mode is GimbalCommandMode.RATE:
                send_result = self.gimbal.set_rate(request.az_deg, request.el_deg)
            elif request.mode is GimbalCommandMode.ABSOLUTE:
                send_result = self.gimbal.goto_angle(request.az_deg, request.el_deg)
            elif request.mode is GimbalCommandMode.STOW:
                send_result = self.gimbal.stow()
            else:
                send_result = self.gimbal.home()
            if isinstance(send_result, Err):
                self._publish_fault(
                    send_result.error, f"gimbal actuation failed frame_id={raw.frame_id}"
                )
            self.bus.publish(
                GimbalCommandMsg(
                    msg_type=MessageType.GIMBAL_COMMAND,
                    timestamp_utc=self.clock.wall_clock_iso(),
                    frame_id=raw.frame_id,
                    mode=request.mode,
                    az_value_deg=request.az_deg,
                    el_value_deg=request.el_deg,
                    state=new_state.arbiter.gimbal_state,
                    reason=request.reason,
                )
            )

        outcome = TickOutcome(
            frame_id=raw.frame_id,
            fault=None,
            command_issued=request is not None,
            gimbal_state=new_state.arbiter.gimbal_state,
        )
        return new_state, outcome

    def run(self, stop_event: threading.Event) -> None:
        """Run the payload acquisition loop until stop_event is set.

        Starts acquisition, then repeatedly: emits a HeartbeatMsg every
        fault_cfg.watchdog_interval_s, acquires a frame, computes the gimbal slew rate
        from consecutive encoder reads, and processes the frame (publishing a
        FaultEventMsg on a camera stall). Stops acquisition on exit. Control state is
        threaded internally, starting from controller.initial_state().

        Inputs:
            stop_event (threading.Event): The loop exits cleanly once it is set.

        Outputs:
            None.

        Notes:
            The slew rate is the angular speed between the previous and current gimbal
            positions divided by the elapsed monotonic seconds; it is 0.0 on the first
            frame, when no time has elapsed, or when the position read fails, so the
            MOTION_SMEAR gate degrades gracefully. SAFE/recovery mode messages are drained
            each iteration via poll_mode_changes and threaded into process_frame. As a
            shell-level safety fallback, if SAFE is commanded while frame acquisition fails,
            stow() is called directly so a stalled camera cannot prevent mechanical safing.
        """
        self.sensor.start_acquisition()
        state = self.controller.initial_state()
        heartbeat_seq = 0
        last_heartbeat = self.clock.monotonic_s()
        prev_pos: GimbalPosition | None = None
        prev_pos_now = 0.0
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
                    state, _outcome = self.process_frame(
                        acq.value, state, now, slew_rate, pos, safe_commanded, safe_cleared
                    )
                else:
                    self._publish_fault(acq.error, "imaging sensor stall")
                    if safe_commanded:
                        self.gimbal.stow()
        finally:
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
