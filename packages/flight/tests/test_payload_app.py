"""Integration tests for the payload application shell (acquire->...->command)."""

import threading

import numpy as np
from flight.hal.drivers_sim import SimGimbal, SimSensor
from flight.libs.bus import MessageBus
from flight.libs.config import PactConfig
from flight.libs.messages import GimbalCommandMsg, InferenceResultMsg, ProcessedFrameMsg
from flight.libs.time import ManualClock
from flight.libs.types import FaultCode, GimbalState, MosaicFrame, Ok, Result
from flight.payload.app import PayloadApp, TickOutcome
from flight.payload.calibration_io import build_identity_calibration
from flight.payload.model import DetectorBackend, ScriptedDetector


def _mosaic_frame(frame_id: int) -> MosaicFrame:
    """Build a zeroed (512, 512) uint16 mosaic frame matching the default sensor geometry."""
    mosaic = np.zeros((512, 512), dtype=np.uint16)  # np.ndarray[uint16, (H, W)]
    return MosaicFrame(
        timestamp_utc="2026-06-01T00:00:00.000Z",
        frame_id=frame_id,
        mosaic=mosaic,
        exposure_us=1000.0,
        gain_db=0.0,
    )


def _plume_detector() -> ScriptedDetector:
    """Scripted detector whose mask yields one strong, stable central blob each frame."""
    mask = np.zeros((256, 256), dtype=np.float32)  # np.ndarray[float32, (H, W)]
    mask[100:150, 100:150] = 1.0
    return ScriptedDetector(mask, confidence_gate=0.55, min_blob_area_px=15)


def _build_app(detector: DetectorBackend) -> tuple[PayloadApp, MessageBus, SimGimbal]:
    """Assemble a PayloadApp over sim drivers, the given detector, and a fresh bus."""
    cfg = PactConfig()
    bus = MessageBus()
    clock = ManualClock()
    gimbal = SimGimbal(clock=clock)
    sensor = SimSensor([])  # frames are fed directly to process_frame in these tests
    calib = build_identity_calibration(cfg.sensor.height_px, cfg.sensor.width_px)
    app = PayloadApp.from_config(cfg, sensor, gimbal, detector, bus, clock, calib)
    return app, bus, gimbal


def test_process_frame_demosaics_to_half_resolution() -> None:
    """A 512x512 mosaic yields a (4, 256, 256) tensor for the detector."""
    captured: list[tuple[int, ...]] = []

    class _CapturingDetector:
        """Records the tensor shape it receives, then delegates to the plume detector."""

        def __init__(self) -> None:
            self._inner = _plume_detector()

        def detect(self, frame: ProcessedFrameMsg) -> Result[InferenceResultMsg, FaultCode]:
            """Capture the band tensor shape, then run the wrapped detector."""
            captured.append(np.asarray(frame.tensor).shape)
            return self._inner.detect(frame)

    app, _bus, _gimbal = _build_app(_CapturingDetector())
    _state, outcome = app.process_frame(_mosaic_frame(1), app.controller.initial_state(), now=1.0)

    assert outcome.fault is None
    assert captured == [(4, 256, 256)]  # 4 demosaicked bands at sensor size / 2


def test_persistent_plume_drives_gimbal_through_app() -> None:
    """A stable plume across frames drives the app to TRACKING and moves the gimbal."""
    app, bus, gimbal = _build_app(_plume_detector())
    cmd_sub = bus.subscribe(GimbalCommandMsg)
    inf_sub = bus.subscribe(InferenceResultMsg)

    state = app.controller.initial_state()
    outcomes: list[TickOutcome] = []
    now = 0.0
    for frame_id in range(1, 9):
        now += 1.0
        state, outcome = app.process_frame(_mosaic_frame(frame_id), state, now)
        outcomes.append(outcome)

    assert state.arbiter.gimbal_state is GimbalState.TRACKING
    assert any(o.command_issued for o in outcomes)
    assert not cmd_sub.empty()  # at least one gimbal command was published

    position = gimbal.read_position()
    assert isinstance(position, Ok)
    assert (position.value.az_deg, position.value.el_deg) != (0.0, 0.0)  # gimbal moved

    inference_count = 0
    while not inf_sub.empty():
        inf_sub.get_nowait()
        inference_count += 1
    assert inference_count == 8  # one InferenceResultMsg published per frame


def test_no_detection_publishes_inference_but_no_command() -> None:
    """With an empty mask, frames are inferred and published but no command is issued."""
    empty_detector = ScriptedDetector(
        np.zeros((256, 256), dtype=np.float32), confidence_gate=0.55, min_blob_area_px=15
    )
    app, bus, _gimbal = _build_app(empty_detector)
    cmd_sub = bus.subscribe(GimbalCommandMsg)

    state = app.controller.initial_state()
    now = 0.0
    for frame_id in range(1, 6):
        now += 1.0
        state, outcome = app.process_frame(_mosaic_frame(frame_id), state, now)
        assert outcome.command_issued is False

    assert state.arbiter.gimbal_state is GimbalState.IDLE
    assert cmd_sub.empty()


def test_run_loop_starts_and_stops_cleanly() -> None:
    """run() returns promptly when stop_event is pre-set, exercising acquisition glue."""
    app, bus, _gimbal = _build_app(_plume_detector())
    cmd_sub = bus.subscribe(GimbalCommandMsg)

    stop = threading.Event()
    stop.set()
    app.run(stop)  # start + stop acquisition, no frame processed

    assert cmd_sub.empty()
