"""Integration tests for the payload application shell."""

import math
import threading
from dataclasses import replace

import numpy as np
import pytest
from flight.hal.drivers_sim import SimGimbal, SimIssEphemeris, SimSensor
from flight.hal.interfaces import GimbalHealth, GimbalPosition, GimbalRateCommand, IssState
from flight.libs.bus import MessageBus, Subscription
from flight.libs.config import PactConfig
from flight.libs.messages import (
    CommandAckMsg,
    FaultEventMsg,
    GimbalCommandMsg,
    InferenceResultMsg,
    ProcessedFrameMsg,
    RoutedCommandMsg,
    SystemModeActivatedMsg,
    TelemetryEventMsg,
)
from flight.libs.time import Clock, ManualClock, RealClock
from flight.libs.types import (
    AckStatus,
    DownlinkPriority,
    Err,
    FaultCode,
    FrameUsabilityTag,
    GimbalCommandMode,
    MessageType,
    MosaicFrame,
    Ok,
    Result,
    SystemMode,
)
from flight.payload.app import PayloadApp
from flight.payload.calibration_io import build_identity_calibration
from flight.payload.imaging import CaptureDecision, plan_capture
from flight.payload.inference import DetectorBackend, InferenceRuntime, ScriptedDetector
from flight.payload.preprocess import SmearRateSource
from flight.payload.state import PayloadState, graph_name_of, node_name_of

_EPOCH = "epoch-payload-test"


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
    """Build a zeroed (3, 1544, 2064) uint16 prism buffer matching the default sensor."""
    mosaic = np.zeros((3, 1544, 2064), dtype=np.uint16)
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
    mask = np.zeros((1544, 2064), dtype=np.float32)
    mask[149:225, 990:1074] = 1.0
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


class _CaptureGsdDetector:
    """Capture the local GSD metadata presented before detector execution."""

    def __init__(self) -> None:
        self.tile_gsd_m: np.ndarray | None = None
        self.quality_flags: frozenset[FrameUsabilityTag] = frozenset()
        self._inner = _plume_detector()

    def detect(self, frame: ProcessedFrameMsg) -> Result[InferenceResultMsg, FaultCode]:
        """Store GSD metadata then return the normal scripted result."""
        self.tile_gsd_m = np.asarray(frame.tile_gsd_m) if frame.tile_gsd_m is not None else None
        self.quality_flags = frame.quality_flags
        return self._inner.detect(frame)


class _FailedEphemeris:
    """Ephemeris source that exercises nominal GSD fallback while reporting its fault."""

    def read_state(self, now_utc_s: float) -> Result[IssState, FaultCode]:
        del now_utc_s
        return Err(FaultCode.EPHEMERIS_FAULT)


def _build_app(detector: DetectorBackend) -> tuple[PayloadApp, MessageBus, SimGimbal, ManualClock]:
    """Assemble a PayloadApp over sim drivers, the given detector, and a fresh bus."""
    base = PactConfig()
    cfg = replace(
        base,
        inference=replace(base.inference, tile_rows=1, tile_cols=1),
        gimbal=replace(
            base.gimbal,
            simulation=replace(base.gimbal.simulation, encoder_noise_deg=0.0),
        ),
    )
    bus = MessageBus()
    clock = ManualClock()
    gimbal = SimGimbal(clock=clock, cfg=cfg.gimbal, inner_dt_s=cfg.controller.inner.dt_s)
    sensor = SimSensor([])
    eph = SimIssEphemeris(clock=clock, cfg=cfg.ephemeris)
    calib = build_identity_calibration(cfg.sensor.height_px, cfg.sensor.width_px)
    app = PayloadApp.from_config(
        cfg,
        sensor,
        gimbal,
        eph,
        InferenceRuntime.from_scripted(detector),
        bus,
        clock,
        calib,
        _MemStorage(),
        _EPOCH,
    )
    return app, bus, gimbal, clock


def _publish_mode(
    bus: MessageBus,
    mode: SystemMode,
    sequence: int,
    *,
    epoch: str = _EPOCH,
    previous_mode: SystemMode | None = None,
    request_id: str | None = None,
    recovery_authorized: bool = False,
) -> None:
    """Publish an authority SystemModeActivatedMsg onto the bus."""
    bus.publish(
        SystemModeActivatedMsg(
            msg_type=MessageType.SYSTEM_MODE_ACTIVATED,
            timestamp_utc="t",
            epoch=epoch,
            sequence=sequence,
            previous_mode=previous_mode,
            active_mode=mode,
            reason="test",
            request_id=request_id,
            recovery_authorized=recovery_authorized,
        )
    )


def _activate(
    app: PayloadApp, bus: MessageBus, state: PayloadState, mode: SystemMode, seq: int = 1
) -> PayloadState:
    """Publish one activation and drain it into the payload state."""
    _publish_mode(bus, mode, seq)
    return app.poll_activations(state, now=0.0)


def _operate(app: PayloadApp, bus: MessageBus, gimbal: SimGimbal, now: float = 0.0) -> PayloadState:
    """Boot state + OPERATE activation + one encoder sample (ready to capture)."""
    state = app.initial_state()
    pos = gimbal.read_position()
    assert isinstance(pos, Ok)
    app.note_gimbal_feedback(pos.value)
    _publish_mode(bus, SystemMode.OPERATE, 1)
    return app.poll_activations(state, now)


class _RateGimbal:
    """Minimal GimbalRateActuator that stamps encoder samples on each read."""

    def __init__(self, dt_s: float, clock: Clock | None = None) -> None:
        self.dt_s = dt_s
        self.clock = clock
        self.reads = 0
        self._last_feedback_s: float | None = None
        self._inhibited = False

    def set_torque(
        self, tau_nm: float, valid_until_s: float | None = None
    ) -> Result[None, FaultCode]:
        del tau_nm, valid_until_s
        return Err(FaultCode.GIMBAL_FAULT)

    def inhibit(self, reason: str) -> Result[GimbalHealth, FaultCode]:
        del reason
        self._inhibited = True
        return Ok(self._health())

    def read_health(self) -> Result[GimbalHealth, FaultCode]:
        return Ok(self._health())

    def goto_angle(self, el_deg: float) -> Result[None, FaultCode]:
        del el_deg
        return Ok(None)

    def home(self) -> Result[None, FaultCode]:
        return Ok(None)

    def stow(self) -> Result[None, FaultCode]:
        return Ok(None)

    def read_position(self) -> Result[GimbalPosition, FaultCode]:
        self.reads += 1
        timestamp_s = self.clock.monotonic_s() if self.clock is not None else self.reads * self.dt_s
        self._last_feedback_s = timestamp_s
        return Ok(GimbalPosition(el_deg=0.0, timestamp_s=timestamp_s, sequence=self.reads))

    def read_stow_switch(self) -> Result[bool, FaultCode]:
        return Ok(False)

    def set_rate(self, command: GimbalRateCommand) -> Result[None, FaultCode]:
        del command
        self._inhibited = False
        return Ok(None)

    def stow_reference_step(self, now_s: float | None = None) -> Result[bool, FaultCode]:
        del now_s
        return Ok(False)

    def shutdown(self) -> Result[None, FaultCode]:
        return Ok(None)

    def _health(self) -> GimbalHealth:
        return GimbalHealth(
            feedback_valid=True,
            last_feedback_s=self._last_feedback_s,
            command_valid_until_s=None,
            inhibited=self._inhibited,
            inhibit_confirmed=self._inhibited,
        )


def _rate_app(detector: DetectorBackend) -> tuple[PayloadApp, MessageBus, _RateGimbal]:
    """Assemble a PayloadApp over a rate-mode gimbal double."""
    base = PactConfig()
    cfg = replace(base, inference=replace(base.inference, tile_rows=1, tile_cols=1))
    bus = MessageBus()
    clock = ManualClock()
    gimbal = _RateGimbal(cfg.controller.outer.dt_s)
    sensor = SimSensor([])
    eph = SimIssEphemeris(clock=clock, cfg=cfg.ephemeris)
    calib = build_identity_calibration(cfg.sensor.height_px, cfg.sensor.width_px)
    app = PayloadApp.from_config(
        cfg,
        sensor,
        gimbal,
        eph,
        InferenceRuntime.from_scripted(detector),
        bus,
        clock,
        calib,
        _MemStorage(),
        _EPOCH,
    )
    return app, bus, gimbal


def test_rate_mode_inner_catchup_samples_encoder_before_outer() -> None:
    """Rate-mode catch-up records encoder samples so outer ticks keep T_out cadence."""
    base = PactConfig()
    cfg = replace(base, inference=replace(base.inference, tile_rows=1, tile_cols=1))
    dt_out = cfg.controller.outer.dt_s
    app, _bus, gimbal = _rate_app(_plume_detector())
    state = app.initial_state()
    now = 0.1
    state = app.advance_inner(state, now)
    state, _ = app.advance_outer(state, now)
    state = app.advance_inner(state, now)
    now = 0.2
    t_out = state.last_outer_s
    assert t_out is not None
    while t_out + dt_out <= now + 1e-12:
        t_out = t_out + dt_out
        state = app.advance_inner(state, t_out)
        state, _ = app.advance_outer(state, t_out)
    state = app.advance_inner(state, now)
    assert gimbal.reads >= 5
    assert state.last_outer_s is not None
    assert state.last_outer_s == pytest.approx(0.2, abs=dt_out + 1e-12)
    assert len(app.encoder_stream.samples) == gimbal.reads


def test_process_frame_passes_nchw_tensor() -> None:
    """A prism buffer is published as NCHW (1, 3, 1544, 2064) with no crop."""
    captured: list[tuple[int, ...]] = []

    class _CapturingDetector:
        """Records the tensor shape it receives, then delegates to the plume detector."""

        def __init__(self) -> None:
            self._inner = _plume_detector()

        def detect(self, frame: ProcessedFrameMsg) -> Result[InferenceResultMsg, FaultCode]:
            """Capture the band tensor shape, then run the wrapped detector."""
            captured.append(np.asarray(frame.tensor).shape)
            return self._inner.detect(frame)

    app, bus, gimbal, _clock = _build_app(_CapturingDetector())
    state = _operate(app, bus, gimbal)
    _state, outcome = app.process_frame(_mosaic_frame(1), state, now=1.0)

    assert outcome.fault is None
    assert captured == [(1, 3, 1544, 2064)]


def test_process_frame_computes_measured_tile_gsd_before_detection() -> None:
    """Detector receives finite physical tile GSD without the nominal fallback tag."""
    detector = _CaptureGsdDetector()
    app, bus, gimbal, _clock = _build_app(detector)
    position = gimbal.read_position()
    assert isinstance(position, Ok)
    shutter = replace(position.value, timestamp_s=1.0)

    state = _operate(app, bus, gimbal)
    state, outcome = app.process_frame(_mosaic_frame(1), state, now=1.0, gimbal_pos=shutter)

    assert outcome.fault is None
    assert detector.tile_gsd_m is not None
    assert detector.tile_gsd_m.shape == (1, 2)
    assert np.isfinite(detector.tile_gsd_m).all()
    assert np.all(detector.tile_gsd_m > 0)
    assert FrameUsabilityTag.GSD_NOMINAL not in detector.quality_flags


def test_ephemeris_failure_uses_tagged_gsd_fallback_and_keeps_inference_live() -> None:
    """Missing ephemeris degrades GSD metadata and publishes a fault without stopping inference."""
    detector = _CaptureGsdDetector()
    app, bus, gimbal, _clock = _build_app(detector)
    app = replace(app, ephemeris=_FailedEphemeris())
    fault_sub = bus.subscribe(FaultEventMsg)
    inference_sub = bus.subscribe(InferenceResultMsg)
    position = gimbal.read_position()
    assert isinstance(position, Ok)
    shutter = replace(position.value, timestamp_s=1.0)

    state = _operate(app, bus, gimbal)
    state, outcome = app.process_frame(_mosaic_frame(1), state, now=1.0, gimbal_pos=shutter)

    assert outcome.fault is None
    assert detector.tile_gsd_m is not None
    assert detector.tile_gsd_m.shape == (1, 2)
    assert np.isfinite(detector.tile_gsd_m).all()
    assert FrameUsabilityTag.GSD_NOMINAL in detector.quality_flags
    assert fault_sub.get_nowait().fault_code is FaultCode.EPHEMERIS_FAULT
    assert inference_sub.get_nowait().frame_id == 1
    state, outer = app.advance_outer(state, now=1.0)
    assert outer.fault is None
    assert state.last_outer_s == pytest.approx(1.0)
    assert graph_name_of(state) == "operate"
    assert node_name_of(state) == "tracking"


def test_imaging_duty_limits_tensors_and_keeps_gimbal_steps() -> None:
    """N due opportunities publish floor(N * duty) tensors and still step the gimbal."""
    app, bus, gimbal, clock = _build_app(_plume_detector())
    inf_sub = bus.subscribe(InferenceResultMsg)
    state = _operate(app, bus, gimbal)
    n = 8
    now = 0.0
    for frame_id in range(1, n + 1):
        now += 1.0
        position = gimbal.read_position()
        assert isinstance(position, Ok)
        shutter = replace(position.value, timestamp_s=now)
        context = app._capture_context(app._latest(state))
        assert context is not None
        plan = plan_capture(
            app.capture_shell.schedule, state.policy, context, now, app.params.policy_limits
        )
        assert isinstance(plan, Ok)
        app.capture_shell.schedule = plan.value.schedule
        if plan.value.decision is CaptureDecision.CAPTURE:
            state, outcome = app.process_frame(
                _mosaic_frame(frame_id), state, now, gimbal_pos=shutter
            )
            assert outcome.fault is None
        else:
            app.note_gimbal_feedback(shutter)
        state, _outer = app.advance_outer(state, now)
        state = app.advance_inner(state, now)
        clock.advance(1.0)
    duty = app.sensor_cfg.capture.duty_cycle
    inference_count = 0
    while not inf_sub.empty():
        inf_sub.get_nowait()
        inference_count += 1
    assert duty == 0.5
    assert inference_count == math.floor(n * duty)
    assert state.last_outer_s == pytest.approx(now)


def test_persistent_plume_drives_gimbal_through_app() -> None:
    """A stable plume keeps TRACKING and moves elevation through the catch-up loops."""
    app, bus, gimbal, clock = _build_app(_plume_detector())
    inf_sub = bus.subscribe(InferenceResultMsg)
    telem_sub = bus.subscribe(TelemetryEventMsg)

    state = _operate(app, bus, gimbal)
    now = 0.0
    for frame_id in range(1, 9):
        now += 1.0
        clock.advance(1.0)
        position = gimbal.read_position()
        assert isinstance(position, Ok)
        shutter = replace(position.value, timestamp_s=now)
        app.note_gimbal_feedback(shutter)
        state, _outcome = app.process_frame(_mosaic_frame(frame_id), state, now, gimbal_pos=shutter)
        state = app.advance_inner(state, now)
        state, _outer = app.advance_outer(state, now)

    assert graph_name_of(state) == "operate"
    assert node_name_of(state) == "tracking"

    position = gimbal.read_position()
    assert isinstance(position, Ok)
    assert position.value.el_deg > 0.1
    assert position.value.el_deg <= app.servo.gimbal.el_science_max_deg
    assert not hasattr(position.value, "az_deg")

    inference_count = 0
    while not inf_sub.empty():
        inf_sub.get_nowait()
        inference_count += 1
    assert inference_count == 8
    pointing = [t.payload for t in _drain_telem(telem_sub) if t.event_name == "pointing"]
    assert any(p["r"] not in (None, 0.0) for p in pointing)


def test_no_detection_publishes_inference_but_no_pose_command() -> None:
    """Empty masks publish inference and do not issue pose GimbalCommandMsg."""
    empty_detector = ScriptedDetector(
        np.zeros((1544, 2064), dtype=np.float32), confidence_gate=0.55, min_blob_area_px=15
    )
    app, bus, gimbal, clock = _build_app(empty_detector)
    cmd_sub = bus.subscribe(GimbalCommandMsg)

    state = _operate(app, bus, gimbal)
    now = 0.0
    for frame_id in range(1, 5):
        now += 1.0
        state, outcome = app.process_frame(_mosaic_frame(frame_id), state, now)
        state, outer = app.advance_outer(state, now)
        state = app.advance_inner(state, now)
        clock.advance(1.0)
        assert outcome.command_issued is False
        assert outer.command_issued is False

    assert graph_name_of(state) == "operate"
    assert node_name_of(state) == "tracking"
    assert cmd_sub.empty()


def test_safe_activation_inhibits_immediately() -> None:
    """An accepted SAFE activation inhibits the actuator and selects the safe graph."""
    app, bus, gimbal, clock = _build_app(_plume_detector())
    cmd_sub = bus.subscribe(GimbalCommandMsg)

    state = _operate(app, bus, gimbal)
    _publish_mode(bus, SystemMode.SAFE, 2)
    state = app.poll_activations(state, now=1.0)
    assert graph_name_of(state) == "safe"
    assert app.containment.local_latched is True

    health = gimbal.read_health()
    assert isinstance(health, Ok)
    assert health.value.inhibit_confirmed is True
    assert cmd_sub.empty()


def test_queued_safe_then_operate_keeps_inhibit() -> None:
    """A queued SAFE followed by OPERATE never skips the SAFE inhibit trace."""
    app, bus, gimbal, _clock = _build_app(_plume_detector())
    state = _operate(app, bus, gimbal)
    _publish_mode(bus, SystemMode.SAFE, 2)
    _publish_mode(bus, SystemMode.OPERATE, 3)
    state = app.poll_activations(state, now=1.0)
    assert graph_name_of(state) == "operate"
    assert app.containment.local_latched is True
    health = gimbal.read_health()
    assert isinstance(health, Ok)
    assert health.value.inhibit_confirmed is True


def test_stale_and_duplicate_activation_do_not_reenter() -> None:
    """Duplicate/stale activation records neither reenter nor reset the graph."""
    app, bus, gimbal, _clock = _build_app(_plume_detector())
    state = _operate(app, bus, gimbal)
    _publish_mode(bus, SystemMode.OPERATE, 1)
    _publish_mode(bus, SystemMode.IDLE, 0)
    state = app.poll_activations(state, now=1.0)
    assert graph_name_of(state) == "operate"
    assert state.control_revision == 1


def test_wrong_epoch_activation_faults_and_contains() -> None:
    """An activation under the wrong epoch raises a sync fault and contains locally."""
    app, bus, gimbal, _clock = _build_app(_plume_detector())
    fault_sub = bus.subscribe(FaultEventMsg)
    state = app.initial_state()
    _publish_mode(bus, SystemMode.OPERATE, 1, epoch="other-epoch")
    state = app.poll_activations(state, now=1.0)
    assert state.graph is None
    assert app.containment.local_latched is True
    faults = [f.fault_code for f in _drain_faults(fault_sub)]
    assert FaultCode.GIMBAL_FAULT in faults


class _DutySensor:
    """ImagingSensor double that counts acquire and drain calls, then stops the loop."""

    def __init__(self, stop: threading.Event, opportunities: int) -> None:
        self._stop = stop
        self._opportunities = opportunities
        self.acquires = 0
        self.drains = 0
        self.starts = 0
        self.stops = 0
        self.drain_error: FaultCode | None = None

    def _finish_opportunity(self) -> None:
        if self.acquires + self.drains >= self._opportunities:
            self._stop.set()

    def acquire_frame(self) -> Result[MosaicFrame, FaultCode]:
        """Count a capture opportunity and return a valid scripted frame."""
        self.acquires += 1
        self._finish_opportunity()
        return Ok(_mosaic_frame(self.acquires))

    def drain_frame(self) -> Result[None, FaultCode]:
        """Count an off-duty opportunity and optionally fail the release."""
        self.drains += 1
        self._finish_opportunity()
        if self.drain_error is not None:
            return Err(self.drain_error)
        return Ok(None)

    def set_exposure_us(self, exposure: float) -> Result[None, FaultCode]:
        """Accept exposure writes without a camera."""
        del exposure
        return Ok(None)

    def set_gain_db(self, gain: float) -> Result[None, FaultCode]:
        """Accept gain writes without a camera."""
        del gain
        return Ok(None)

    def start_acquisition(self) -> Result[None, FaultCode]:
        """Record that the payload loop started the stream."""
        self.starts += 1
        return Ok(None)

    def stop_acquisition(self) -> Result[None, FaultCode]:
        """Record that the payload loop stopped the stream."""
        self.stops += 1
        return Ok(None)


def test_run_drains_camera_on_skipped_opportunities() -> None:
    """Off-duty loop ticks release a waiting image and do not acquire it."""
    base = PactConfig()
    cfg = replace(base, inference=replace(base.inference, tile_rows=1, tile_cols=1))
    bus = MessageBus()
    clock = RealClock()
    stop = threading.Event()
    sensor = _DutySensor(stop, opportunities=4)
    gimbal = _RateGimbal(cfg.controller.outer.dt_s, clock)
    eph = SimIssEphemeris(clock=clock, cfg=cfg.ephemeris)
    calib = build_identity_calibration(cfg.sensor.height_px, cfg.sensor.width_px)
    app = PayloadApp.from_config(
        cfg,
        sensor,
        gimbal,
        eph,
        InferenceRuntime.from_scripted(_plume_detector()),
        bus,
        clock,
        calib,
        _MemStorage(),
        _EPOCH,
    )
    _publish_mode(bus, SystemMode.OPERATE, 1)
    app.run(stop)
    assert sensor.starts == 1
    assert sensor.stops == 1
    assert app.capture_shell.schedule.opportunities == 4
    assert sensor.drains == 2
    assert sensor.acquires == 2


def test_run_publishes_fault_when_camera_drain_fails() -> None:
    """A failed off-duty release publishes CAMERA_STALL and keeps the control loop moving."""
    base = PactConfig()
    cfg = replace(base, inference=replace(base.inference, tile_rows=1, tile_cols=1))
    bus = MessageBus()
    clock = RealClock()
    stop = threading.Event()
    sensor = _DutySensor(stop, opportunities=1)
    sensor.drain_error = FaultCode.CAMERA_STALL
    fault_sub = bus.subscribe(FaultEventMsg)
    gimbal = _RateGimbal(cfg.controller.outer.dt_s, clock)
    eph = SimIssEphemeris(clock=clock, cfg=cfg.ephemeris)
    calib = build_identity_calibration(cfg.sensor.height_px, cfg.sensor.width_px)
    app = PayloadApp.from_config(
        cfg,
        sensor,
        gimbal,
        eph,
        InferenceRuntime.from_scripted(_plume_detector()),
        bus,
        clock,
        calib,
        _MemStorage(),
        _EPOCH,
    )
    _publish_mode(bus, SystemMode.OPERATE, 1)
    app.run(stop)
    assert sensor.drains == 1
    assert sensor.acquires == 0
    assert gimbal.reads >= 1
    faults = []
    while not fault_sub.empty():
        faults.append(fault_sub.get_nowait())
    assert any(
        fault.fault_code is FaultCode.CAMERA_STALL and fault.detail == "imaging sensor buffer drain"
        for fault in faults
    )


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
    state = app.advance_inner(app.initial_state(), now=3600.0)
    assert state.servo.inner.last_inner_s == 3600.0
    faults = []
    while not fault_sub.empty():
        faults.append(fault_sub.get_nowait())
    assert not any("catch-up" in f.detail for f in faults)


def test_containment_replaces_tracking_torque_with_inhibit() -> None:
    """A latched containment cannot continue a stale outward tracking command."""
    app, _bus, gimbal, _clock = _build_app(_plume_detector())
    assert isinstance(gimbal.set_torque(0.2, valid_until_s=1.0), Ok)
    app.containment.local_latched = True
    initial = app.initial_state()
    state = replace(
        initial,
        servo=replace(
            initial.servo,
            inner=replace(initial.servo.inner, last_inner_s=0.0),
            commanded_rate_rad_s=-0.1,
        ),
    )
    app.advance_inner(state, now=0.001)
    health = gimbal.read_health()
    assert isinstance(health, Ok)
    assert health.value.inhibit_confirmed is True


def test_encoder_failure_contains_motion(
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
    initial = app.initial_state()
    state = replace(
        initial,
        servo=replace(
            initial.servo,
            inner=replace(initial.servo.inner, last_inner_s=0.0),
            encoder=replace(initial.servo.encoder, last_theta_enc_rad=0.1),
            commanded_rate_rad_s=0.1,
        ),
    )
    app.advance_inner(state, now=0.001)
    assert app.containment.local_latched is True
    health = gimbal.read_health()
    assert isinstance(health, Ok)
    assert health.value.inhibit_confirmed is True


def test_commands_nack_until_activated() -> None:
    """Routed payload commands get one correlated NACK while no graph is active."""
    app, bus, _gimbal, _clock = _build_app(_plume_detector())
    ack_sub = bus.subscribe(CommandAckMsg)
    for command_id in ("GIMBAL_HOLD", "GIMBAL_RESUME"):
        bus.publish(
            RoutedCommandMsg(
                msg_type=MessageType.ROUTED_COMMAND,
                timestamp_utc="t",
                target="payload",
                command_id=command_id,
                params={},
                source="ground",
                seq=7,
            )
        )
    state = app.handle_commands(app.initial_state(), now=0.0)
    assert state.graph is None
    first = ack_sub.get_nowait()
    second = ack_sub.get_nowait()
    assert first.status is AckStatus.REJECTED
    assert first.fault_code is FaultCode.COMMAND_INVALID
    assert second.status is AckStatus.REJECTED


def _routed(
    command_id: str, seq: int, params: dict[str, str | int | float | bool] | None = None
) -> RoutedCommandMsg:
    """Build one routed payload command envelope."""
    return RoutedCommandMsg(
        msg_type=MessageType.ROUTED_COMMAND,
        timestamp_utc="t",
        target="payload",
        command_id=command_id,
        params=params or {},
        source="ground",
        seq=seq,
    )


def _fresh_encoder(app: PayloadApp, gimbal: SimGimbal, now: float) -> None:
    """Record an encoder sample stamped at the command tick."""
    pos = gimbal.read_position()
    assert isinstance(pos, Ok)
    app.note_gimbal_feedback(replace(pos.value, timestamp_s=now))


def test_ground_goto_commits_hold_target() -> None:
    """A routed GIMBAL_GOTO under OPERATE commits the manual hold target."""
    app, bus, gimbal, _clock = _build_app(_plume_detector())
    ack_sub = bus.subscribe(CommandAckMsg)
    cmd_sub = bus.subscribe(GimbalCommandMsg)
    state = _operate(app, bus, gimbal)
    _fresh_encoder(app, gimbal, 1.0)
    bus.publish(_routed("GIMBAL_HOLD", 1))
    state = app.handle_commands(state, now=1.0)
    assert ack_sub.get_nowait().status is AckStatus.ACCEPTED
    assert node_name_of(state) == "hold"
    _fresh_encoder(app, gimbal, 1.5)
    bus.publish(_routed("GIMBAL_GOTO", 2, {"el_deg": 20.0}))
    state = app.handle_commands(state, now=1.5)
    ack = ack_sub.get_nowait()
    assert ack.status is AckStatus.ACCEPTED
    assert node_name_of(state) == "hold"
    published = _drain_cmds(cmd_sub)
    assert published
    assert published[-1].mode is GimbalCommandMode.ABSOLUTE
    assert published[-1].el_value_deg == 20.0
    assert published[-1].payload_graph == "operate"
    assert published[-1].activation_epoch == _EPOCH


def test_duplicate_command_never_reexecutes() -> None:
    """A repeated (source, seq, command_id) emits no competing execution ack."""
    app, bus, gimbal, _clock = _build_app(_plume_detector())
    ack_sub = bus.subscribe(CommandAckMsg)
    state = _operate(app, bus, gimbal)
    _fresh_encoder(app, gimbal, 1.0)
    command = _routed("GIMBAL_HOLD", 5)
    bus.publish(command)
    bus.publish(command)
    state = app.handle_commands(state, now=1.0)
    acks = _drain_acks(ack_sub)
    assert len(acks) == 1
    assert acks[0].status is AckStatus.ACCEPTED
    assert node_name_of(state) == "hold"


def test_process_frame_preserves_measured_zero_slew() -> None:
    """A stationary measured gimbal rate is kept even when the commanded rate is nonzero."""
    detector = _FlagDetector()
    app, bus, gimbal, _clock = _build_app(detector)
    state = _operate(app, bus, gimbal)
    state = replace(
        state,
        servo=replace(state.servo, commanded_rate_rad_s=math.radians(1.0)),
    )
    raw = replace(_mosaic_frame(1), exposure_us=100_000.0)
    rate_deg_per_s, source = app._smear_gimbal_rate_deg_per_s(raw, state, 0.0)
    assert rate_deg_per_s == 0.0
    assert source is SmearRateSource.MEASURED
    _state, outcome = app.process_frame(raw, state, now=1.0)
    assert outcome.fault is None
    assert detector.flags
    assert FrameUsabilityTag.MOTION_SMEAR not in detector.flags[0]


def test_process_frame_uses_encoder_when_command_and_motion_disagree() -> None:
    """Missing measured slew uses encoder motion, not the commanded rate."""
    detector = _FlagDetector()
    app, bus, _gimbal, _clock = _build_app(detector)
    app._record_encoder(GimbalPosition(el_deg=10.0, timestamp_s=0.9, sequence=1))
    pos = GimbalPosition(el_deg=10.0, timestamp_s=1.0, sequence=2)
    app._record_encoder(pos)
    initial = app.initial_state()
    state = replace(
        initial,
        servo=replace(initial.servo, commanded_rate_rad_s=math.radians(1.0)),
    )
    _publish_mode(bus, SystemMode.OPERATE, 1)
    state = app.poll_activations(state, now=0.0)
    raw = replace(_mosaic_frame(1), timestamp_s=1.0, exposure_us=100_000.0)
    rate_deg_per_s, source = app._smear_gimbal_rate_deg_per_s(raw, state, None)
    assert source is SmearRateSource.ENCODER
    assert abs(rate_deg_per_s) < 1e-12
    _state, outcome = app.process_frame(raw, state, now=1.0, gimbal_pos=pos)
    assert outcome.fault is None
    assert detector.flags
    assert FrameUsabilityTag.MOTION_SMEAR not in detector.flags[0]


def _drain_telem(subscription: Subscription[TelemetryEventMsg]) -> list[TelemetryEventMsg]:
    """Drain TelemetryEventMsg values from a subscription."""
    out: list[TelemetryEventMsg] = []
    while not subscription.empty():
        out.append(subscription.get_nowait())
    return out


def _drain_cmds(subscription: Subscription[GimbalCommandMsg]) -> list[GimbalCommandMsg]:
    """Drain GimbalCommandMsg values from a subscription."""
    out: list[GimbalCommandMsg] = []
    while not subscription.empty():
        out.append(subscription.get_nowait())
    return out


def _drain_acks(subscription: Subscription[CommandAckMsg]) -> list[CommandAckMsg]:
    """Drain CommandAckMsg values from a subscription."""
    out: list[CommandAckMsg] = []
    while not subscription.empty():
        out.append(subscription.get_nowait())
    return out


def _drain_faults(subscription: Subscription[FaultEventMsg]) -> list[FaultEventMsg]:
    """Drain FaultEventMsg values from a subscription."""
    out: list[FaultEventMsg] = []
    while not subscription.empty():
        out.append(subscription.get_nowait())
    return out
