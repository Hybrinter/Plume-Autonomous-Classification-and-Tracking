"""Integration tests for the payload application shell."""

import math
import threading
from dataclasses import replace

import numpy as np
import pytest
from flight.hal.drivers_sim import SimGimbal, SimIssEphemeris, SimSensor
from flight.hal.interfaces import GimbalPosition
from flight.libs.bus import MessageBus
from flight.libs.config import PactConfig
from flight.libs.messages import (
    FaultEventMsg,
    GimbalCommandMsg,
    InferenceResultMsg,
    LaunchLockStateMsg,
    ModeChangeMsg,
    ProcessedFrameMsg,
    RoutedCommandMsg,
    TelemetryEventMsg,
)
from flight.libs.time import ManualClock
from flight.libs.types import (
    DownlinkPriority,
    Err,
    FaultCode,
    FrameUsabilityTag,
    GimbalCommandMode,
    GimbalState,
    LaunchLockState,
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
    """Scripted detector whose mask yields one strong above-boresight blob each frame."""
    mask = np.zeros((1024, 1224), dtype=np.float32)
    mask[99:149, 587:637] = 1.0
    return ScriptedDetector(mask, confidence_gate=0.55, min_blob_area_px=15)


class _FlagDetector:
    """Scripted detector that records quality flags from each processed frame."""

    def __init__(self) -> None:
        self.flags: list[frozenset[FrameUsabilityTag]] = []
        self._inner = _plume_detector()

    def detect(self, frame: ProcessedFrameMsg) -> Result[InferenceResultMsg, FaultCode]:
        """Capture quality flags, then run the wrapped plume detector."""
        self.flags.append(frame.quality_flags)
        return self._inner.detect(frame)


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
    app.lock_gate.engaged = False
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
        position = gimbal.read_position()
        assert isinstance(position, Ok)
        shutter = replace(position.value, timestamp_s=now)
        state, _outcome = app.process_frame(_mosaic_frame(frame_id), state, now, gimbal_pos=shutter)
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
    assert position.value.el_deg > 0.1
    assert position.value.el_deg <= app.controller.gimbal.el_science_max_deg
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
    """SAFE issues STOW, but a stopped host expires torque instead of driving on."""
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
    health = gimbal.read_health()
    assert isinstance(health, Ok)
    assert health.value.inhibit_confirmed is True


def test_run_loop_starts_and_stops_cleanly() -> None:
    """run() returns promptly when stop_event is pre-set, exercising acquisition glue."""
    app, bus, _gimbal, _clock = _build_app(_plume_detector())
    cmd_sub = bus.subscribe(GimbalCommandMsg)

    stop = threading.Event()
    stop.set()
    app.run(stop)

    assert cmd_sub.empty()


def test_clock_origin_does_not_replay_from_zero() -> None:
    """A late monotonic origin stamps last_inner_s and does not catch up from 0."""
    app, bus, _gimbal, clock = _build_app(_plume_detector())
    clock.advance(3600.0)
    fault_sub = bus.subscribe(FaultEventMsg)
    state = app.advance_inner(app.controller.initial_state(), now=3600.0)
    assert state.last_inner_s == 3600.0
    faults = []
    while not fault_sub.empty():
        faults.append(fault_sub.get_nowait())
    assert not any("catch-up" in f.detail for f in faults)


def test_lock_engaged_writes_zero_torque() -> None:
    """Fail-closed lock writes tau=0 and freezes the commanded rate."""
    app, bus, gimbal, _clock = _build_app(_plume_detector())
    app.lock_gate.engaged = True
    bus.publish(
        LaunchLockStateMsg(
            msg_type=MessageType.LAUNCH_LOCK_STATE,
            timestamp_utc="t",
            state=LaunchLockState.ENGAGED,
        )
    )
    state = replace(app.controller.initial_state(), r_rad_s=0.1)
    state = app.advance_inner(state, now=1.0)
    assert gimbal._tau_nm == 0.0
    assert state.r_rad_s == 0.0


def test_safe_latch_replaces_tracking_torque_with_stow_control() -> None:
    """Healthy SAFE cannot continue a stale outward tracking command."""
    app, _bus, gimbal, _clock = _build_app(_plume_detector())
    assert isinstance(gimbal.set_torque(0.2, valid_until_s=1.0), Ok)
    app.safe_latch.commanded = True
    state = replace(app.controller.initial_state(), last_inner_s=0.0, r_rad_s=0.1)
    app.advance_inner(state, now=0.001)
    assert gimbal._tau_nm <= 0.0


def test_encoder_failure_contains_motion_and_commands_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cached encoder state cannot authorize torque after a feedback failure."""
    app, _bus, gimbal, _clock = _build_app(_plume_detector())
    assert isinstance(gimbal.set_torque(0.2, valid_until_s=1.0), Ok)
    monkeypatch.setattr(
        gimbal,
        "read_position",
        lambda: Err(FaultCode.GIMBAL_FAULT),
    )
    state = replace(
        app.controller.initial_state(),
        last_inner_s=0.0,
        last_theta_enc_rad=0.1,
        r_rad_s=0.1,
    )
    state = app.advance_inner(state, now=0.001)
    assert gimbal._tau_nm == 0.0
    assert app.safe_latch.commanded is True
    state, _outcome = app.advance_outer(state, now=0.021)
    assert state.arbiter.gimbal_state is GimbalState.SAFE


def test_ground_goto_latches_pose_mode() -> None:
    """A routed GIMBAL_GOTO sets pose_mode and publishes a pose command."""
    app, bus, _gimbal, _clock = _build_app(_plume_detector())
    cmd_sub = bus.subscribe(GimbalCommandMsg)
    bus.publish(
        RoutedCommandMsg(
            msg_type=MessageType.ROUTED_COMMAND,
            timestamp_utc="t",
            target="payload",
            command_id="GIMBAL_GOTO",
            params={"el_deg": 20.0},
            source="ground",
            seq=1,
        )
    )
    app.handle_commands()
    state, outcome = app.advance_outer(app.controller.initial_state(), now=1.0)
    assert outcome.command_issued is True
    assert state.pose_mode is GimbalCommandMode.ABSOLUTE
    assert state.pose_el_deg == 20.0
    published = cmd_sub.get_nowait()
    assert published.mode is GimbalCommandMode.ABSOLUTE
    assert published.el_value_deg == 20.0


def test_process_frame_preserves_measured_zero_slew() -> None:
    """A stationary measured gimbal rate flags MOTION_SMEAR against a moving scene."""
    detector = _FlagDetector()
    app, _bus, _gimbal, _clock = _build_app(detector)
    state = replace(
        app.controller.initial_state(),
        r_rad_s=math.radians(1.0),
        last_omega_scene_el=math.radians(1.0),
    )
    raw = replace(_mosaic_frame(1), exposure_us=100_000.0)
    _state, outcome = app.process_frame(raw, state, now=1.0, slew_rate_deg_per_s=0.0)
    assert outcome.fault is None
    assert detector.flags
    assert FrameUsabilityTag.MOTION_SMEAR in detector.flags[0]


def test_process_frame_uses_encoder_when_command_and_motion_disagree() -> None:
    """Missing measured slew uses encoder motion, not the commanded rate."""
    detector = _FlagDetector()
    app, _bus, _gimbal, _clock = _build_app(detector)
    app._record_encoder(GimbalPosition(el_deg=10.0, timestamp_s=0.9, sequence=1))
    pos = GimbalPosition(el_deg=10.0, timestamp_s=1.0, sequence=2)
    state = replace(
        app.controller.initial_state(),
        r_rad_s=math.radians(1.0),
        last_omega_scene_el=math.radians(1.0),
    )
    raw = replace(_mosaic_frame(1), timestamp_s=1.0, exposure_us=100_000.0)
    _state, outcome = app.process_frame(raw, state, now=1.0, gimbal_pos=pos)
    assert outcome.fault is None
    assert detector.flags
    assert FrameUsabilityTag.MOTION_SMEAR in detector.flags[0]


def _drain_telem(subscription: object) -> list[TelemetryEventMsg]:
    """Drain TelemetryEventMsg values from a subscription."""
    out: list[TelemetryEventMsg] = []
    while not subscription.empty():  # type: ignore[attr-defined]
        out.append(subscription.get_nowait())  # type: ignore[attr-defined]
    return out
