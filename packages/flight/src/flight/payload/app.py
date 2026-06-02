"""Payload application shell: binds the HAL and the pure payload core into one loop.

Collapses the legacy imaging + inference + controller processes into a single
in-process payload app. Per frame: acquire a raw frame from the imaging sensor,
preprocess it co-located (radiometric correction -> band selection -> quality flags;
no queue round-trip, honoring the preprocessing co-location invariant), run the
swappable detector, step the pure PayloadController, then drive the gimbal HAL and
publish results onto the typed bus. All decision logic lives in PayloadController;
this module owns only I/O, sequencing, and message construction.

Contains:
  - build_identity_calibration: identity RadiometricCalibration sized from InferenceConfig
    (zero dark frame, unit flat field) -- a placeholder until real sensor characterization.
  - TickOutcome: per-frame result summary (frame id, fault code, command-issued flag,
    resulting gimbal state) used for telemetry and testing.
  - PayloadApp: frozen holder of injected services. from_config() assembles it from a
    PactConfig and concrete drivers; process_frame() runs one frame end-to-end; run()
    is the acquisition loop (emits heartbeats, publishes a fault on camera stall).

Non-obvious notes:
  - The arbiter `now` is sourced from Clock.monotonic_s() (it consumes `now` only as
    interval/rate-limit deltas); message timestamps use Clock.wall_clock_iso().
  - No crop is applied (crop_origin_px=(0, 0), scale_factor=1.0), matching the legacy
    inference process; raw frames must match the identity calibration shape.

Satisfies: REQ-AIML-COMP-001, REQ-AIML-COMP-002 (payload process orchestration),
           REQ-OPER-HIGH-002 (subsystem app loop).
"""

from __future__ import annotations

# stdlib
import threading
from dataclasses import dataclass

# third-party
import numpy as np

# internal
from flight.hal.interfaces import GimbalActuator, ImagingSensor
from flight.libs.bus import MessageBus
from flight.libs.config import FaultConfig, InferenceConfig, PactConfig, PreprocessingConfig
from flight.libs.messages import (
    FaultEventMsg,
    HeartbeatMsg,
    ProcessedFrameMsg,
    RawFrameMsg,
)
from flight.libs.time import Clock
from flight.libs.types import Err, FaultCode, GimbalState, MessageType, Ok
from flight.payload.control import ControlState, PayloadController
from flight.payload.model import DetectorBackend
from flight.payload.preprocess import (
    RadiometricCalibration,
    apply_calibration,
    compute_quality_flags,
    select_bands,
)


def build_identity_calibration(cfg: InferenceConfig) -> RadiometricCalibration:
    """Build an identity radiometric calibration sized to the inference config.

    Produces a zero dark frame and a unit flat field of shape
    (len(input_bands), input_height_px, input_width_px), so apply_calibration is a
    no-op. Placeholder until real per-pixel dark/flat frames are available from
    sensor characterization; raw frames fed to the payload must match this shape.

    Args:
        cfg: InferenceConfig supplying band count and input height/width in pixels.

    Returns:
        RadiometricCalibration with zero dark_frame and unit flat_field (float32).
    """
    channels = len(cfg.input_bands)
    shape = (channels, cfg.input_height_px, cfg.input_width_px)
    return RadiometricCalibration(
        dark_frame=np.zeros(shape, dtype=np.float32),  # np.ndarray[float32, (C, H, W)]
        flat_field=np.ones(shape, dtype=np.float32),  # np.ndarray[float32, (C, H, W)]
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

    Holds the injected HAL drivers, detector, pure controller, bus, clock,
    calibration, and the config slices needed for preprocessing and heartbeats.
    Frozen to prevent field reassignment; the held services are themselves mutable
    (consistent with the composition-root injection pattern).
    """

    sensor: ImagingSensor
    gimbal: GimbalActuator
    detector: DetectorBackend
    controller: PayloadController
    bus: MessageBus
    clock: Clock
    calib: RadiometricCalibration
    inference_cfg: InferenceConfig
    preprocessing_cfg: PreprocessingConfig
    fault_cfg: FaultConfig

    @staticmethod
    def from_config(
        cfg: PactConfig,
        sensor: ImagingSensor,
        gimbal: GimbalActuator,
        detector: DetectorBackend,
        bus: MessageBus,
        clock: Clock,
    ) -> PayloadApp:
        """Assemble a PayloadApp from a PactConfig and concrete injected services.

        Builds the pure PayloadController from cfg.controller and an identity
        calibration from cfg.inference; carries cfg.preprocessing and cfg.fault for
        the loop. The drivers, detector, bus, and clock are injected by the caller
        (the composition root chooses real vs sim).

        Args:
            cfg: Top-level PactConfig.
            sensor: ImagingSensor driver (sim or real).
            gimbal: GimbalActuator driver (sim or real).
            detector: DetectorBackend (ScriptedDetector or OnnxDetector).
            bus: The typed MessageBus to publish onto.
            clock: Injected Clock (RealClock in flight, ManualClock in tests).

        Returns:
            A fully constructed PayloadApp.
        """
        return PayloadApp(
            sensor=sensor,
            gimbal=gimbal,
            detector=detector,
            controller=PayloadController.from_config(cfg.controller),
            bus=bus,
            clock=clock,
            calib=build_identity_calibration(cfg.inference),
            inference_cfg=cfg.inference,
            preprocessing_cfg=cfg.preprocessing,
            fault_cfg=cfg.fault,
        )

    def process_frame(
        self,
        raw: RawFrameMsg,
        state: ControlState,
        now: float,
    ) -> tuple[ControlState, TickOutcome]:
        """Process one raw frame end-to-end: preprocess -> detect -> control -> actuate.

        Runs preprocessing co-located (no queue), then the detector, then the pure
        PayloadController. Publishes InferenceResultMsg and each arbiter
        TelemetryEventMsg; when a command is issued it is both sent to the gimbal HAL
        and published. On a preprocessing or detection fault the state is returned
        unchanged, a FaultEventMsg is published, and outcome.fault is set.

        Args:
            raw: Raw frame to process. raw.raw_bands must match the calibration shape
                (len(input_bands), input_height_px, input_width_px).
            state: ControlState carried from the previous frame.
            now: Monotonic seconds for the arbiter (interval/rate-limit deltas only).

        Returns:
            (new_state, outcome). new_state is unchanged on a fault before control.
        """
        raw_bands = np.asarray(raw.raw_bands, dtype=np.float32)  # np.ndarray[float32, (C, H, W)]

        calib_result = apply_calibration(raw_bands, self.calib)
        if isinstance(calib_result, Err):
            self._publish_fault(
                calib_result.error, f"radiometric calibration failed frame_id={raw.frame_id}"
            )
            return state, self._fault_outcome(raw.frame_id, calib_result.error, state)
        cal_bands = calib_result.value

        selected = select_bands(cal_bands, self.inference_cfg.input_bands)  # (4, H, W) float32
        quality_flags = compute_quality_flags(
            selected, raw.exposure_us, raw.timestamp_utc, self.preprocessing_cfg
        )
        processed = ProcessedFrameMsg(
            msg_type=MessageType.PROCESSED_FRAME,
            timestamp_utc=raw.timestamp_utc,
            frame_id=raw.frame_id,
            tensor=selected,
            quality_flags=quality_flags,
            crop_origin_px=(0, 0),
            scale_factor=1.0,
        )

        detect_result = self.detector.detect(processed)
        if isinstance(detect_result, Err):
            self._publish_fault(detect_result.error, f"detection failed frame_id={raw.frame_id}")
            return state, self._fault_outcome(raw.frame_id, detect_result.error, state)
        inference = detect_result.value
        self.bus.publish(inference)

        new_state, command, telemetry = self.controller.step(state, inference, now)
        for event in telemetry:
            self.bus.publish(event)

        if command is not None:
            send_result = self.gimbal.send_command(command)
            if isinstance(send_result, Err):
                self._publish_fault(
                    send_result.error, f"gimbal command failed frame_id={raw.frame_id}"
                )
            self.bus.publish(command)

        outcome = TickOutcome(
            frame_id=raw.frame_id,
            fault=None,
            command_issued=command is not None,
            gimbal_state=new_state.arbiter.gimbal_state,
        )
        return new_state, outcome

    def run(self, stop_event: threading.Event) -> None:
        """Run the payload acquisition loop until stop_event is set.

        Starts acquisition, then repeatedly: emits a HeartbeatMsg every
        fault_cfg.watchdog_interval_s, acquires a frame, and processes it (publishing
        a FaultEventMsg on a camera stall). Stops acquisition on exit. Control state
        is threaded internally, starting from controller.initial_state().

        Args:
            stop_event: threading.Event; the loop exits cleanly once it is set.
        """
        self.sensor.start_acquisition()
        state = self.controller.initial_state()
        heartbeat_seq = 0
        last_heartbeat = self.clock.monotonic_s()
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
                acq = self.sensor.acquire_frame()
                if isinstance(acq, Ok):
                    state, _outcome = self.process_frame(acq.value, state, now)
                else:
                    self._publish_fault(acq.error, "imaging sensor stall")
        finally:
            self.sensor.stop_acquisition()

    def _publish_fault(self, code: FaultCode, detail: str) -> None:
        """Publish a FaultEventMsg from the payload subsystem onto the bus.

        Args:
            code: The FaultCode to report.
            detail: Human-readable detail string for logging/telemetry.
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

        Args:
            frame_id: The frame_id that faulted.
            code: The FaultCode raised.
            state: The unchanged control state (its arbiter state is reported).

        Returns:
            A TickOutcome with command_issued=False and the prior gimbal state.
        """
        return TickOutcome(
            frame_id=frame_id,
            fault=code,
            command_issued=False,
            gimbal_state=state.arbiter.gimbal_state,
        )
