"""Integration tests for the payload application shell (acquire->...->command)."""

import threading

import numpy as np
from flight.hal.drivers_sim import SimGimbal, SimSensor
from flight.libs.bus import MessageBus
from flight.libs.config import PactConfig
from flight.libs.messages import GimbalCommandMsg, InferenceResultMsg, RawFrameMsg
from flight.libs.time import ManualClock
from flight.libs.types import GimbalState, MessageType, Ok
from flight.payload.app import PayloadApp, TickOutcome
from flight.payload.model import ScriptedDetector


def _raw_frame(frame_id: int) -> RawFrameMsg:
    """Build a (4, 256, 256) zero-band raw frame matching the identity calibration."""
    raw_bands = np.zeros((4, 256, 256), dtype=np.float32)  # np.ndarray[float32, (C, H, W)]
    return RawFrameMsg(
        msg_type=MessageType.RAW_FRAME,
        timestamp_utc="2026-06-01T00:00:00.000Z",
        frame_id=frame_id,
        raw_bands=raw_bands,
        exposure_us=1000.0,
        gain_db=0.0,
        gimbal_az_deg=0.0,
        gimbal_el_deg=0.0,
    )


def _plume_detector() -> ScriptedDetector:
    """Scripted detector whose mask yields one strong, stable central blob each frame."""
    mask = np.zeros((256, 256), dtype=np.float32)  # np.ndarray[float32, (H, W)]
    mask[100:150, 100:150] = 1.0
    return ScriptedDetector(mask, confidence_gate=0.55, min_blob_area_px=15)


def _build_app(detector: ScriptedDetector) -> tuple[PayloadApp, MessageBus, SimGimbal]:
    """Assemble a PayloadApp over sim drivers, the given detector, and a fresh bus."""
    cfg = PactConfig()
    bus = MessageBus()
    gimbal = SimGimbal()
    sensor = SimSensor([])  # frames are fed directly to process_frame in these tests
    app = PayloadApp.from_config(cfg, sensor, gimbal, detector, bus, ManualClock())
    return app, bus, gimbal


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
        state, outcome = app.process_frame(_raw_frame(frame_id), state, now)
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
        state, outcome = app.process_frame(_raw_frame(frame_id), state, now)
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
