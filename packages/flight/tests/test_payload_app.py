"""Integration tests for the payload application shell (acquire->...->actuate)."""

import threading

import numpy as np
from flight.hal.drivers_sim import SimGimbal, SimSensor
from flight.libs.bus import MessageBus
from flight.libs.config import PactConfig
from flight.libs.messages import (
    GimbalCommandMsg,
    InferenceResultMsg,
    ModeChangeMsg,
    ProcessedFrameMsg,
)
from flight.libs.time import ManualClock
from flight.libs.types import (
    FaultCode,
    GimbalCommandMode,
    GimbalState,
    MessageType,
    MosaicFrame,
    Ok,
    Result,
    SystemMode,
)
from flight.payload.app import PayloadApp, TickOutcome
from flight.payload.calibration_io import build_identity_calibration
from flight.payload.model import DetectorBackend, ScriptedDetector


def _mosaic_frame(frame_id: int) -> MosaicFrame:
    """Build a zeroed (1024, 1024) uint16 mosaic frame matching the default sensor geometry."""
    mosaic = np.zeros((1024, 1024), dtype=np.uint16)  # np.ndarray[uint16, (H, W)]
    return MosaicFrame(
        timestamp_utc="2026-06-01T00:00:00.000Z",
        frame_id=frame_id,
        mosaic=mosaic,
        exposure_us=1000.0,
        gain_db=0.0,
    )


def _plume_detector() -> ScriptedDetector:
    """Scripted detector whose mask yields one strong, stable off-boresight blob each frame.

    The blob centroid (~169.5, ~169.5 in tensor space) back-projects in decimated search
    mode (scale 0.5) to full-plane (~339, ~339): ~117 px off the 512-plane boresight (256,
    256), clearing the minimum deadband so TRACKING issues RATE commands. In TRACKING ROI
    mode (scale 1.0, crop clamped at the plane edge) the displacement stays below the
    maximum deadband, so commands keep flowing.
    """
    mask = np.zeros((256, 256), dtype=np.float32)  # np.ndarray[float32, (H, W)]
    mask[145:195, 145:195] = 1.0  # centroid ~ (169.5, 169.5) in tensor space
    return ScriptedDetector(mask, confidence_gate=0.55, min_blob_area_px=15)


def _build_app(detector: DetectorBackend) -> tuple[PayloadApp, MessageBus, SimGimbal, ManualClock]:
    """Assemble a PayloadApp over sim drivers, the given detector, and a fresh bus."""
    cfg = PactConfig()
    bus = MessageBus()
    clock = ManualClock()
    gimbal = SimGimbal(clock=clock)
    sensor = SimSensor([])  # frames are fed directly to process_frame in these tests
    calib = build_identity_calibration(cfg.sensor.height_px, cfg.sensor.width_px)
    app = PayloadApp.from_config(cfg, sensor, gimbal, detector, bus, clock, calib)
    return app, bus, gimbal, clock


def test_process_frame_demosaics_to_half_resolution() -> None:
    """A 1024x1024 mosaic demosaics to 512 band planes, decimated to (4, 256, 256) in search."""
    captured: list[tuple[int, ...]] = []

    class _CapturingDetector:
        """Records the tensor shape it receives, then delegates to the plume detector."""

        def __init__(self) -> None:
            self._inner = _plume_detector()

        def detect(self, frame: ProcessedFrameMsg) -> Result[InferenceResultMsg, FaultCode]:
            """Capture the band tensor shape, then run the wrapped detector."""
            captured.append(np.asarray(frame.tensor).shape)
            return self._inner.detect(frame)

    app, _bus, _gimbal, _clock = _build_app(_CapturingDetector())
    _state, outcome = app.process_frame(_mosaic_frame(1), app.controller.initial_state(), now=1.0)

    assert outcome.fault is None
    assert captured == [(4, 256, 256)]  # 4 bands, 512 planes decimated 2x in search mode


def test_search_mode_decimates_full_plane() -> None:
    """Outside TRACKING the model sees the decimated full plane (scale 0.5, no crop)."""
    app, bus, _gimbal, _clock = _build_app(_plume_detector())
    inf_sub = bus.subscribe(InferenceResultMsg)

    app.process_frame(_mosaic_frame(1), app.controller.initial_state(), now=1.0)

    msg = inf_sub.get_nowait()
    assert msg.scale_factor == 0.5
    assert msg.crop_origin_px == (0, 0)


def test_tracking_mode_crops_full_resolution_roi() -> None:
    """In TRACKING with an initialized estimator, a 256x256 scale-1.0 ROI is cropped."""
    app, bus, _gimbal, clock = _build_app(_plume_detector())

    state = app.controller.initial_state()
    now = 0.0
    for frame_id in range(1, 9):
        now += 1.0
        clock.advance(1.0)
        state, _outcome = app.process_frame(_mosaic_frame(frame_id), state, now)
    assert state.arbiter.gimbal_state is GimbalState.TRACKING

    inf_sub = bus.subscribe(InferenceResultMsg)
    now += 1.0
    clock.advance(1.0)
    state, _outcome = app.process_frame(_mosaic_frame(9), state, now)

    msg = inf_sub.get_nowait()
    assert msg.scale_factor == 1.0
    assert msg.crop_origin_px != (0, 0)


def test_persistent_plume_drives_gimbal_through_app() -> None:
    """A stable plume across frames drives the app to TRACKING and moves the gimbal."""
    app, bus, gimbal, clock = _build_app(_plume_detector())
    cmd_sub = bus.subscribe(GimbalCommandMsg)
    inf_sub = bus.subscribe(InferenceResultMsg)

    state = app.controller.initial_state()
    outcomes: list[TickOutcome] = []
    now = 0.0
    for frame_id in range(1, 9):
        now += 1.0
        clock.advance(1.0)  # let SimGimbal integrate commanded motion between frames
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
    app, bus, _gimbal, _clock = _build_app(empty_detector)
    cmd_sub = bus.subscribe(GimbalCommandMsg)

    state = app.controller.initial_state()
    now = 0.0
    for frame_id in range(1, 6):
        now += 1.0
        state, outcome = app.process_frame(_mosaic_frame(frame_id), state, now)
        assert outcome.command_issued is False

    assert state.arbiter.gimbal_state is GimbalState.IDLE
    assert cmd_sub.empty()


def test_mode_change_safe_issues_stow_actuation() -> None:
    """A ModeChangeMsg(SAFE) on the bus makes the next frame issue a STOW actuation."""
    app, bus, gimbal, clock = _build_app(_plume_detector())
    cmd_sub = bus.subscribe(GimbalCommandMsg)

    bus.publish(
        ModeChangeMsg(
            msg_type=MessageType.MODE_CHANGE,
            timestamp_utc="2026-06-01T00:00:00.000Z",
            new_mode=SystemMode.SAFE,
            requested_by="ground",
        )
    )
    safe_commanded, safe_cleared = app.poll_mode_changes()
    assert safe_commanded is True
    assert safe_cleared is False

    state = app.controller.initial_state()
    state, outcome = app.process_frame(
        _mosaic_frame(1), state, now=1.0, safe_commanded=safe_commanded
    )
    assert outcome.command_issued is True
    assert state.arbiter.gimbal_state is GimbalState.SAFE

    published = cmd_sub.get_nowait()
    assert published.mode is GimbalCommandMode.STOW

    clock.advance(60.0)  # let the gimbal reach the stow pose
    switch = gimbal.read_stow_switch()
    assert isinstance(switch, Ok)
    assert switch.value is True


def test_run_loop_starts_and_stops_cleanly() -> None:
    """run() returns promptly when stop_event is pre-set, exercising acquisition glue."""
    app, bus, _gimbal, _clock = _build_app(_plume_detector())
    cmd_sub = bus.subscribe(GimbalCommandMsg)

    stop = threading.Event()
    stop.set()
    app.run(stop)  # start + stop acquisition, no frame processed

    assert cmd_sub.empty()
