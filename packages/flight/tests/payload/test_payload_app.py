"""Integration tests for the payload application shell (acquire->ingest, then control)."""

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
from flight.payload.control import ControlState
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
    """Build a zeroed (1024, 1024) uint16 mosaic frame matching the default sensor geometry."""
    mosaic = np.zeros((1024, 1024), dtype=np.uint16)  # np.ndarray[uint16, (H, W)]
    return MosaicFrame(
        timestamp_utc="2026-06-01T00:00:00.000Z",
        frame_id=frame_id,
        mosaic=mosaic,
        exposure_us=1000.0,
        gain_db=0.0,
        capture_monotonic_s=float(frame_id),
    )


def _plume_detector() -> ScriptedDetector:
    """Scripted detector with a 50x50 blob at band-plane (340, 340)."""
    mask = np.zeros((512, 512), dtype=np.float32)  # np.ndarray[float32, (H, W)]
    mask[315:365, 315:365] = 1.0
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


def _ingest_and_control(
    app: PayloadApp,
    clock: ManualClock,
    frame_id: int,
    state: ControlState,
    now: float,
    dt: float,
    *,
    safe_commanded: bool = False,
) -> tuple[ControlState, TickOutcome]:
    """Ingest one frame then run one control interval (tests stand in for the control thread)."""
    clock.advance(dt)
    state, _ingest = app.process_frame(_mosaic_frame(frame_id), state, now)
    state, outcome = app.apply_control(state, now, dt, safe_commanded=safe_commanded)
    return state, outcome


def test_process_frame_demosaics_to_full_plane() -> None:
    """A 1024x1024 mosaic demosaics to a (4, 512, 512) inference tensor."""
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
    assert captured == [(4, 512, 512)]


def test_full_plane_identity_crop() -> None:
    """Every frame uses identity crop origin and scale 1."""
    app, bus, _gimbal, _clock = _build_app(_plume_detector())
    inf_sub = bus.subscribe(InferenceResultMsg)

    app.process_frame(_mosaic_frame(1), app.controller.initial_state(), now=1.0)

    msg = inf_sub.get_nowait()
    assert msg.scale_factor == 1.0
    assert msg.crop_origin_px == (0, 0)


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
        state, outcome = _ingest_and_control(app, clock, frame_id, state, now, 1.0)
        outcomes.append(outcome)

    assert state.arbiter.gimbal_state is GimbalState.TRACKING
    assert any(o.command_issued for o in outcomes)
    assert not cmd_sub.empty()

    position = gimbal.read_position()
    assert isinstance(position, Ok)
    assert (position.value.az_deg, position.value.el_deg) != (0.0, 0.0)

    inference_count = 0
    while not inf_sub.empty():
        inf_sub.get_nowait()
        inference_count += 1
    assert inference_count == 8


def test_no_detection_publishes_inference_but_no_command() -> None:
    """With an empty mask, frames are inferred and published but no command is issued."""
    empty_detector = ScriptedDetector(
        np.zeros((512, 512), dtype=np.float32), confidence_gate=0.55, min_blob_area_px=15
    )
    app, bus, _gimbal, clock = _build_app(empty_detector)
    cmd_sub = bus.subscribe(GimbalCommandMsg)

    state = app.controller.initial_state()
    now = 0.0
    for frame_id in range(1, 6):
        now += 1.0
        state, outcome = _ingest_and_control(app, clock, frame_id, state, now, 1.0)
        assert outcome.command_issued is False

    assert state.arbiter.gimbal_state is GimbalState.IDLE
    assert cmd_sub.empty()


def test_mode_change_safe_issues_stow_actuation() -> None:
    """A ModeChangeMsg(SAFE) on the bus makes the next control tick issue a STOW actuation."""
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
    state, outcome = _ingest_and_control(
        app, clock, 1, state, 1.0, 1.0, safe_commanded=safe_commanded
    )
    assert outcome.command_issued is True
    assert state.arbiter.gimbal_state is GimbalState.SAFE

    published = cmd_sub.get_nowait()
    assert published.mode is GimbalCommandMode.STOW

    clock.advance(60.0)
    switch = gimbal.read_stow_switch()
    assert isinstance(switch, Ok)
    assert switch.value is True


def test_run_loop_starts_and_stops_cleanly() -> None:
    """run() returns promptly when stop_event is pre-set, exercising acquisition glue."""
    app, bus, _gimbal, _clock = _build_app(_plume_detector())
    cmd_sub = bus.subscribe(GimbalCommandMsg)

    stop = threading.Event()
    stop.set()
    app.run(stop)

    assert cmd_sub.empty()
