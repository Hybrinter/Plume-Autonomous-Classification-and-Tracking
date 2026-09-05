"""Integration tests for the payload application shell."""

import threading

import numpy as np
from flight.hal.drivers_sim import SimGimbal, SimIssEphemeris, SimSensor
from flight.libs.bus import MessageBus
from flight.libs.config import PactConfig
from flight.libs.messages import (
    GimbalCommandMsg,
    InferenceResultMsg,
    ModeChangeMsg,
    ProcessedFrameMsg,
    TelemetryEventMsg,
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
from flight.payload.app import PayloadApp
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
    mosaic = np.zeros((2048, 2448), dtype=np.uint16)
    return MosaicFrame(
        timestamp_utc="2026-06-01T00:00:00.000Z",
        timestamp_s=float(frame_id),
        frame_id=frame_id,
        mosaic=mosaic,
        exposure_us=1000.0,
        gain_db=0.0,
    )


def _plume_detector() -> ScriptedDetector:
    """Scripted detector whose mask yields one strong below-boresight blob each frame."""
    mask = np.zeros((1024, 1224), dtype=np.float32)
    mask[875:925, 587:637] = 1.0
    return ScriptedDetector(mask, confidence_gate=0.55, min_blob_area_px=15)


def _build_app(detector: DetectorBackend) -> tuple[PayloadApp, MessageBus, SimGimbal, ManualClock]:
    """Assemble a PayloadApp over sim drivers, the given detector, and a fresh bus."""
    cfg = PactConfig()
    bus = MessageBus()
    clock = ManualClock()
    gimbal = SimGimbal(clock=clock, cfg=cfg.gimbal, inner_dt_s=cfg.controller.inner.dt_s)
    sensor = SimSensor([])
    eph = SimIssEphemeris(clock=clock, cfg=cfg.ephemeris)
    calib = build_identity_calibration(cfg.sensor.height_px, cfg.sensor.width_px)
    app = PayloadApp.from_config(
        cfg, sensor, gimbal, eph, detector, bus, clock, calib, _MemStorage()
    )
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
    """A stable plume drives TRACKING and moves elevation through the catch-up loops."""
    app, bus, gimbal, clock = _build_app(_plume_detector())
    telem_sub = bus.subscribe(TelemetryEventMsg)
    inf_sub = bus.subscribe(InferenceResultMsg)

    state = app.controller.initial_state()
    now = 0.0
    for frame_id in range(1, 9):
        now += 1.0
        state, _outcome = app.process_frame(_mosaic_frame(frame_id), state, now)
        state, _outer = app.advance_outer(state, now)
        state = app.advance_inner(state, now)
        clock.advance(1.0)

    assert state.arbiter.gimbal_state is GimbalState.TRACKING
    pointing = [
        m
        for m in _drain_telem(telem_sub)
        if m.subsystem == "payload" and m.event_name == "pointing"
    ]
    assert pointing
    assert any(float(m.payload["r"]) != 0.0 for m in pointing)

    position = gimbal.read_position()
    assert isinstance(position, Ok)
    assert position.value.el_deg < -0.1
    assert not hasattr(position.value, "az_deg")

    inference_count = 0
    while not inf_sub.empty():
        inf_sub.get_nowait()
        inference_count += 1
    assert inference_count == 8


def test_no_detection_publishes_inference_but_no_pose_command() -> None:
    """Empty masks publish inference and do not issue pose GimbalCommandMsg."""
    empty_detector = ScriptedDetector(
        np.zeros((1024, 1224), dtype=np.float32), confidence_gate=0.55, min_blob_area_px=15
    )
    app, bus, _gimbal, clock = _build_app(empty_detector)
    cmd_sub = bus.subscribe(GimbalCommandMsg)

    state = app.controller.initial_state()
    now = 0.0
    for frame_id in range(1, 5):
        now += 1.0
        state, outcome = app.process_frame(_mosaic_frame(frame_id), state, now)
        state, outer = app.advance_outer(state, now)
        state = app.advance_inner(state, now)
        clock.advance(1.0)
        assert outcome.command_issued is False
        assert outer.command_issued is False

    assert state.arbiter.gimbal_state is GimbalState.TRACKING
    assert cmd_sub.empty()


def test_mode_change_safe_issues_stow_actuation() -> None:
    """A ModeChangeMsg(SAFE) makes advance_outer issue STOW and the position loop stow."""
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
    state, _proc = app.process_frame(_mosaic_frame(1), state, now=1.0)
    state, outcome = app.advance_outer(state, now=1.0, safe_commanded=safe_commanded)
    assert outcome.command_issued is True
    assert state.arbiter.gimbal_state is GimbalState.SAFE
    published = cmd_sub.get_nowait()
    assert published.mode is GimbalCommandMode.STOW

    state = app.advance_inner(state, now=7.0)
    clock.advance(7.0)
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


def _drain_telem(subscription: object) -> list[TelemetryEventMsg]:
    """Drain TelemetryEventMsg values from a subscription."""
    out: list[TelemetryEventMsg] = []
    while not subscription.empty():  # type: ignore[attr-defined]
        out.append(subscription.get_nowait())  # type: ignore[attr-defined]
    return out
