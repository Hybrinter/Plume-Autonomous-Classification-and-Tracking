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
    DownlinkPriority,
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
from flight.payload.inference import DetectorBackend, ScriptedDetector


class _MemStorage:
    """In-memory StorageWriter double for payload tests (records stored products)."""

    def __init__(self) -> None:
        """Start with an empty store and a zeroed entry counter."""
        self.items: dict[str, bytes] = {}
        self._n = 0

    def store(
        self, item_id: str, data: bytes, priority: DownlinkPriority
    ) -> Result[str, FaultCode]:
        """Record data under a fresh entry id and return it."""
        entry_id = f"{self._n:08d}_{item_id}"
        self._n += 1
        self.items[entry_id] = data
        return Ok(entry_id)


def _mosaic_frame(frame_id: int) -> MosaicFrame:
    """Build a zeroed (2048, 2448) uint16 mosaic frame matching the default sensor geometry."""
    mosaic = np.zeros((2048, 2448), dtype=np.uint16)  # np.ndarray[uint16, (H, W)]
    return MosaicFrame(
        timestamp_utc="2026-06-01T00:00:00.000Z",
        frame_id=frame_id,
        mosaic=mosaic,
        exposure_us=1000.0,
        gain_db=0.0,
    )


def _plume_detector() -> ScriptedDetector:
    """Scripted detector whose mask yields one strong, stable below-boresight blob each frame.

    The blob centroid (~611.5, ~899.5) sits ~388 px below the 1024x1224-plane boresight
    (612, 512). TRACKING issues a negative elevation RATE. Drivers pin azimuth at 0.
    """
    mask = np.zeros((1024, 1224), dtype=np.float32)  # np.ndarray[float32, (H, W)]
    mask[875:925, 587:637] = 1.0  # centroid ~ (611.5, 899.5) in tensor / band-plane space
    return ScriptedDetector(mask, confidence_gate=0.55, min_blob_area_px=15)


def _build_app(detector: DetectorBackend) -> tuple[PayloadApp, MessageBus, SimGimbal, ManualClock]:
    """Assemble a PayloadApp over sim drivers, the given detector, and a fresh bus."""
    cfg = PactConfig()
    bus = MessageBus()
    clock = ManualClock()
    gimbal = SimGimbal(clock=clock)
    sensor = SimSensor([])  # frames are fed directly to process_frame in these tests
    calib = build_identity_calibration(cfg.sensor.height_px, cfg.sensor.width_px)
    app = PayloadApp.from_config(cfg, sensor, gimbal, detector, bus, clock, calib, _MemStorage())
    return app, bus, gimbal, clock


def test_process_frame_passes_full_band_plane() -> None:
    """A 2048x2448 mosaic demosaics to (4, 1024, 1224) and is passed to detect() uncropped."""
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
    assert captured == [(4, 1024, 1224)]


def test_persistent_plume_drives_gimbal_through_app() -> None:
    """A stable plume across frames drives the app to TRACKING and moves elevation."""
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
    assert position.value.el_deg < -0.1  # plume below boresight -> negative elevation
    assert abs(position.value.az_deg) < 0.1  # azimuth is pinned at 0

    inference_count = 0
    while not inf_sub.empty():
        inf_sub.get_nowait()
        inference_count += 1
    assert inference_count == 8  # one InferenceResultMsg published per frame


def test_no_detection_publishes_inference_but_no_command() -> None:
    """With an empty mask, frames are inferred and published but no command is issued."""
    empty_detector = ScriptedDetector(
        np.zeros((1024, 1224), dtype=np.float32), confidence_gate=0.55, min_blob_area_px=15
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
