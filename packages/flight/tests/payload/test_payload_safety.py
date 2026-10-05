"""PR5 runtime-safety regressions for the payload application shell.

Covers: actuator-spy no-motion paths, capture staleness under blocked
acquire/detect/store, recovery-evidence ordering and the release matrix,
activation bookkeeping, flagged-vision ordering, and stop/shutdown fault
observability.
"""

import math
import threading
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
from flight.hal.drivers_sim import SimGimbal, SimIssEphemeris, SimSensor
from flight.hal.interfaces import (
    GimbalActuator,
    GimbalHealth,
    GimbalPosition,
    GimbalRateCommand,
    ImagingSensor,
    StorageWriter,
)
from flight.libs.bus import MessageBus, Subscription
from flight.libs.config import PactConfig
from flight.libs.messages import (
    CommandAckMsg,
    FaultEventMsg,
    GimbalCommandMsg,
    InferenceResultMsg,
    ProcessedFrameMsg,
    ProductRefMsg,
    RoutedCommandMsg,
    SafetyStateMsg,
    SystemModeActivatedMsg,
    SystemModeRequestMsg,
    SystemModeTransitionMsg,
    TelemetryEventMsg,
)
from flight.libs.time import ManualClock
from flight.libs.types import (
    AckStatus,
    DownlinkPriority,
    Err,
    FaultCode,
    GimbalCommandMode,
    MessageType,
    ModeTransitionDecision,
    MosaicFrame,
    Ok,
    Result,
    SystemMode,
)
from flight.payload.app import PayloadApp, TickOutcome
from flight.payload.calibration_io import build_identity_calibration
from flight.payload.gimbal.request import (
    InhibitReference,
    PoseReference,
    StowReference,
    TravelEnvelope,
)
from flight.payload.graphs import operate
from flight.payload.inference import DetectorBackend, ScriptedDetector
from flight.payload.records import CaptureContext, CapturedVision, VisionSample
from flight.payload.state import PayloadState, graph_name_of, node_name_of
from flight.payload.tracking import EncoderSample, ResidualState

_EPOCH = "epoch-safety-test"


class _MemStorage:
    """In-memory StorageWriter double (records stored products)."""

    def __init__(self) -> None:
        self.items: dict[str, bytes] = {}
        self._n = 0

    def store(
        self, item_id: str, data: bytes, priority: DownlinkPriority
    ) -> Result[str, FaultCode]:
        entry_id = f"{self._n:08d}_{item_id}"
        self._n += 1
        self.items[entry_id] = data
        return Ok(entry_id)


def _mosaic_frame(frame_id: int) -> MosaicFrame:
    """Zeroed (3, 1544, 2064) uint16 prism buffer matching the default sensor."""
    return MosaicFrame(
        timestamp_utc="2026-06-01T00:00:00.000Z",
        timestamp_s=float(frame_id),
        frame_id=frame_id,
        mosaic=np.zeros((3, 1544, 2064), dtype=np.uint16),
        exposure_us=1000.0,
        gain_db=0.0,
    )


def _plume_detector() -> ScriptedDetector:
    """Scripted detector whose mask yields one strong above-boresight blob."""
    mask = np.zeros((1544, 2064), dtype=np.float32)
    mask[149:225, 990:1074] = 1.0
    return ScriptedDetector(mask, confidence_gate=0.55, min_blob_area_px=15)


def _build_app(
    detector: DetectorBackend,
    *,
    gimbal: GimbalActuator | None = None,
    sensor: ImagingSensor | None = None,
    storage: StorageWriter | None = None,
) -> tuple[PayloadApp, MessageBus, GimbalActuator, ManualClock]:
    """Assemble a PayloadApp over injected doubles and a fresh bus."""
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
    if gimbal is None:
        gimbal = SimGimbal(clock=clock, cfg=cfg.gimbal, inner_dt_s=cfg.controller.inner.dt_s)
    if sensor is None:
        sensor = SimSensor([])
    eph = SimIssEphemeris(clock=clock, cfg=cfg.ephemeris)
    calib = build_identity_calibration(cfg.sensor.height_px, cfg.sensor.width_px)
    app = PayloadApp.from_config(
        cfg,
        sensor,
        gimbal,
        eph,
        detector,
        bus,
        clock,
        calib,
        storage if storage is not None else _MemStorage(),
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
    reason: str = "test",
    request_id: str | None = None,
    recovery_authorized: bool = False,
) -> None:
    """Publish one authority activation record."""
    bus.publish(
        SystemModeActivatedMsg(
            msg_type=MessageType.SYSTEM_MODE_ACTIVATED,
            timestamp_utc="t",
            epoch=epoch,
            sequence=sequence,
            previous_mode=previous_mode,
            active_mode=mode,
            reason=reason,
            request_id=request_id,
            recovery_authorized=recovery_authorized,
        )
    )


def _publish_safety(
    bus: MessageBus,
    *,
    sequence: int,
    observed_s: float,
    epoch: str = _EPOCH,
    safe_latched: bool = False,
    active_faults: tuple[FaultCode, ...] = (),
    recovery_request_id: str | None = None,
) -> None:
    """Publish one fault-owned safety evidence record."""
    bus.publish(
        SafetyStateMsg(
            msg_type=MessageType.SAFETY_STATE,
            timestamp_utc="t",
            active_faults=active_faults,
            safe_latched=safe_latched,
            safe_reason=active_faults[0] if active_faults else FaultCode.NONE,
            evidence_epoch=epoch,
            evidence_sequence=sequence,
            observed_s=observed_s,
            recovery_request_id=recovery_request_id,
        )
    )


def _operate(
    app: PayloadApp, bus: MessageBus, gimbal: GimbalActuator, now: float = 0.0
) -> PayloadState:
    """Boot state + OPERATE activation + one encoder sample."""
    state = app.initial_state()
    pos = gimbal.read_position()
    assert isinstance(pos, Ok)
    app.note_gimbal_feedback(pos.value)
    _publish_mode(bus, SystemMode.OPERATE, 1)
    return app.poll_activations(state, now)


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


def _drain[T](subscription: Subscription[T]) -> list[T]:
    """Drain all queued messages from a subscription."""
    out: list[T] = []
    while not subscription.empty():
        out.append(subscription.get_nowait())
    return out


class _SpyGimbalBase:
    """Shared actuator spy surface; subclasses pick rate or torque mode."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._inhibited = False
        self._last_feedback_s: float | None = None
        self._seq = 0
        self.inhibit_failures = 0

    def inhibit(self, reason: str) -> Result[GimbalHealth, FaultCode]:
        self.calls.append(f"inhibit:{reason}")
        if self.inhibit_failures > 0:
            self.inhibit_failures -= 1
            self._inhibited = False
            return Err(FaultCode.GIMBAL_FAULT)
        self._inhibited = True
        return Ok(self._health())

    def read_health(self) -> Result[GimbalHealth, FaultCode]:
        self.calls.append("read_health")
        return Ok(self._health())

    def goto_angle(self, el_deg: float) -> Result[None, FaultCode]:
        self.calls.append("goto_angle")
        return Ok(None)

    def home(self) -> Result[None, FaultCode]:
        self.calls.append("home")
        return Ok(None)

    def stow(self) -> Result[None, FaultCode]:
        self.calls.append("stow")
        return Ok(None)

    def read_position(self) -> Result[GimbalPosition, FaultCode]:
        self.calls.append("read_position")
        self._last_feedback_s = 0.0
        self._seq += 1
        return Ok(GimbalPosition(el_deg=0.0, timestamp_s=0.0, sequence=self._seq))

    def read_stow_switch(self) -> Result[bool, FaultCode]:
        self.calls.append("read_stow_switch")
        return Ok(False)

    def _health(self) -> GimbalHealth:
        return GimbalHealth(
            feedback_valid=True,
            last_feedback_s=self._last_feedback_s,
            command_valid_until_s=None,
            inhibited=self._inhibited,
            inhibit_confirmed=self._inhibited,
        )


class _SpyTorqueGimbal(_SpyGimbalBase):
    """Detailed-plant spy: no rate surface, so it is not a GimbalRateActuator."""

    def set_torque(
        self, tau_nm: float, valid_until_s: float | None = None
    ) -> Result[None, FaultCode]:
        self.calls.append("set_torque")
        return Ok(None)


class _SpyRateGimbal(_SpyGimbalBase):
    """GimbalRateActuator spy recording the exact HAL call list."""

    def set_torque(
        self, tau_nm: float, valid_until_s: float | None = None
    ) -> Result[None, FaultCode]:
        self.calls.append("set_torque")
        return Err(FaultCode.GIMBAL_FAULT)

    def set_rate(self, command: GimbalRateCommand) -> Result[None, FaultCode]:
        self.calls.append("set_rate")
        self._inhibited = False
        return Ok(None)

    def stow_reference_step(self, now_s: float | None = None) -> Result[bool, FaultCode]:
        self.calls.append("stow_reference_step")
        return Ok(False)

    def shutdown(self) -> Result[None, FaultCode]:
        self.calls.append("shutdown")
        return Ok(None)


_MOTION_CALLS = {"set_torque", "set_rate", "goto_angle", "home", "stow", "stow_reference_step"}


def _motion_calls(gimbal: _SpyRateGimbal | _SpyTorqueGimbal) -> list[str]:
    """Return the recorded motion-authority calls only."""
    return [call for call in gimbal.calls if call.split(":")[0] in _MOTION_CALLS]


class _FailStowGimbal(_SpyRateGimbal):
    """Rate gimbal whose stow metadata operation reports a driver failure."""

    def __init__(self, code: FaultCode) -> None:
        super().__init__()
        self._code = code

    def stow(self) -> Result[None, FaultCode]:
        self.calls.append("stow")
        return Err(self._code)


class _FailGotoGimbal(_SpyTorqueGimbal):
    """Detailed-plant gimbal whose goto metadata reports a driver failure.

    fail_after earlier goto_angle calls succeed so a committed HOLD edge can
    precede the failing routed GOTO under test.
    """

    def __init__(self, code: FaultCode, fail_after: int = 0) -> None:
        super().__init__()
        self._code = code
        self._fail_after = fail_after
        self._goto_calls = 0

    def goto_angle(self, el_deg: float) -> Result[None, FaultCode]:
        self.calls.append("goto_angle")
        self._goto_calls += 1
        if self._goto_calls > self._fail_after:
            return Err(self._code)
        return Ok(None)


class _AlwaysFailInhibitGimbal(_SpyTorqueGimbal):
    """Torque gimbal whose inhibit can never be confirmed."""

    def inhibit(self, reason: str) -> Result[GimbalHealth, FaultCode]:
        self.calls.append(f"inhibit:{reason}")
        self._inhibited = False
        return Err(FaultCode.GIMBAL_FAULT)


class _ScriptedPositionGimbal(_SpyTorqueGimbal):
    """Torque gimbal returning scripted GimbalPosition values per read."""

    def __init__(self, positions: list[GimbalPosition]) -> None:
        super().__init__()
        self._positions = positions
        self._idx = 0

    def read_position(self) -> Result[GimbalPosition, FaultCode]:
        self.calls.append("read_position")
        position = self._positions[min(self._idx, len(self._positions) - 1)]
        self._idx += 1
        if math.isfinite(position.timestamp_s):
            self._last_feedback_s = position.timestamp_s
        return Ok(position)


_ENVELOPE = TravelEnvelope(theta_min_rad=0.0, theta_max_rad=1.0, omega_max_rad_s=0.5)


@pytest.mark.parametrize(
    ("el_deg", "timestamp_s"),
    [
        (float("nan"), 1.0),
        (float("inf"), 1.0),
        (0.0, float("nan")),
        (0.0, float("inf")),
        (0.0, float("-inf")),
    ],
    ids=["nan-angle", "inf-angle", "nan-time", "inf-time", "neg-inf-time"],
)
def test_nonfinite_encoder_sample_rejected_before_history(
    el_deg: float, timestamp_s: float
) -> None:
    """A NaN/Inf angle or timestamp never enters the encoder history.

    The read returns Err(GIMBAL_ENCODER_INVALID), containment latches with a
    confirmed inhibit, no motion write occurs, and the next finite sample is
    recorded cleanly so a later recovery is not poisoned by retained NaN.
    """
    gimbal = _ScriptedPositionGimbal(
        [
            GimbalPosition(el_deg=0.0, timestamp_s=0.0, sequence=1),
            GimbalPosition(el_deg=el_deg, timestamp_s=timestamp_s, sequence=2),
            GimbalPosition(el_deg=0.0, timestamp_s=1.0, sequence=3),
        ]
    )
    app, bus, _g, _clock = _build_app(_plume_detector(), gimbal=gimbal)
    fault_sub = bus.subscribe(FaultEventMsg)
    _operate(app, bus, gimbal)
    recorded = len(app.encoder_stream.samples)
    result = app.sample_feedback()
    assert isinstance(result, Err)
    assert result.error is FaultCode.GIMBAL_ENCODER_INVALID
    assert len(app.encoder_stream.samples) == recorded
    assert all(math.isfinite(s.angle_rad) for s in app.encoder_stream.samples)
    assert app.containment.local_latched is True
    assert any(c.startswith("inhibit") for c in gimbal.calls)
    assert _motion_calls(gimbal) == []
    assert FaultCode.GIMBAL_ENCODER_INVALID in [f.fault_code for f in _drain(fault_sub)]
    recovered = app.sample_feedback()
    assert isinstance(recovered, Ok)
    assert all(math.isfinite(s.angle_rad) for s in app.encoder_stream.samples)
    assert len(app.encoder_stream.samples) == recorded + 1


@pytest.mark.parametrize("code", [FaultCode.GIMBAL_FAULT, FaultCode.COMM_TIMEOUT])
def test_stow_metadata_failure_latches_without_audit(code: FaultCode) -> None:
    """A failed stow() arm emits no audit command and latches + inhibits."""
    gimbal = _FailStowGimbal(code)
    app, bus, _g, _clock = _build_app(_plume_detector(), gimbal=gimbal)
    cmd_sub = bus.subscribe(GimbalCommandMsg)
    fault_sub = bus.subscribe(FaultEventMsg)
    state = _operate(app, bus, gimbal)
    commit = app._commit_reference(
        state, StowReference(target_rad=0.5, envelope=_ENVELOPE, timeout_s=30.0), now=1.0
    )
    assert commit.command_issued is False
    assert commit.fault is code
    assert cmd_sub.empty()
    assert app.containment.local_latched is True
    assert "stow" in gimbal.calls
    assert any(c.startswith("inhibit") for c in gimbal.calls)
    assert _motion_calls(gimbal) == ["stow"]
    assert code in [f.fault_code for f in _drain(fault_sub)]


@pytest.mark.parametrize("code", [FaultCode.GIMBAL_FAULT, FaultCode.COMM_TIMEOUT])
def test_goto_metadata_failure_latches_without_audit(code: FaultCode) -> None:
    """A failed goto_angle emits no audit command and latches + inhibits."""
    gimbal = _FailGotoGimbal(code)
    app, bus, _g, _clock = _build_app(_plume_detector(), gimbal=gimbal)
    cmd_sub = bus.subscribe(GimbalCommandMsg)
    fault_sub = bus.subscribe(FaultEventMsg)
    state = _operate(app, bus, gimbal)
    commit = app._commit_reference(
        state, PoseReference(target_rad=0.5, envelope=_ENVELOPE), now=1.0
    )
    assert commit.command_issued is False
    assert commit.fault is code
    assert cmd_sub.empty()
    assert app.containment.local_latched is True
    assert "goto_angle" in gimbal.calls
    assert any(c.startswith("inhibit") for c in gimbal.calls)
    assert _motion_calls(gimbal) == ["goto_angle"]
    assert code in [f.fault_code for f in _drain(fault_sub)]


def _fresh_encoder(app: PayloadApp, gimbal: _SpyGimbalBase, now: float) -> None:
    """Record an encoder sample stamped at the command tick."""
    pos = gimbal.read_position()
    assert isinstance(pos, Ok)
    app.note_gimbal_feedback(replace(pos.value, timestamp_s=now))


def test_routed_goto_metadata_failure_rejects_and_retains_hold() -> None:
    """A routed GOTO whose pose metadata fails acks REJECTED, keeps HOLD.

    The original graph node and hold target survive, the control reference is
    inhibited, and no node_transition telemetry is emitted for the failed edge.
    """
    gimbal = _FailGotoGimbal(FaultCode.GIMBAL_FAULT, fail_after=1)
    app, bus, _g, _clock = _build_app(_plume_detector(), gimbal=gimbal)
    ack_sub = bus.subscribe(CommandAckMsg)
    cmd_sub = bus.subscribe(GimbalCommandMsg)
    telem_sub = bus.subscribe(TelemetryEventMsg)
    state = _operate(app, bus, gimbal)
    _fresh_encoder(app, gimbal, 1.0)
    bus.publish(_routed("GIMBAL_HOLD", 1))
    state = app.handle_commands(state, now=1.0)
    assert ack_sub.get_nowait().status is AckStatus.ACCEPTED
    assert node_name_of(state) == "hold"
    assert isinstance(state.graph, operate.State)
    target_rad = state.graph.hold.target_rad
    _drain(telem_sub)
    _drain(cmd_sub)
    _fresh_encoder(app, gimbal, 1.5)
    bus.publish(_routed("GIMBAL_GOTO", 2, {"el_deg": 20.0}))
    state = app.handle_commands(state, now=1.5)
    acks = _drain(ack_sub)
    assert len(acks) == 1
    assert acks[0].status is AckStatus.REJECTED
    assert acks[0].fault_code is FaultCode.GIMBAL_FAULT
    assert node_name_of(state) == "hold"
    assert isinstance(state.graph, operate.State)
    assert state.graph.hold.target_rad == target_rad
    assert isinstance(state.reference, InhibitReference)
    assert app.containment.local_latched is True
    assert not any(e.event_name == "node_transition" for e in _drain(telem_sub))
    assert not any(m.mode is GimbalCommandMode.ABSOLUTE for m in _drain(cmd_sub))


@pytest.mark.parametrize("spy", [_SpyRateGimbal, _SpyTorqueGimbal], ids=["rate", "torque"])
def test_boot_and_unactivated_issue_no_motion(spy: type[_SpyRateGimbal | _SpyTorqueGimbal]) -> None:
    """Boot/unactivated control ticks call inhibit but never a motion command."""
    gimbal = spy()
    app, _bus, _g, _clock = _build_app(_plume_detector(), gimbal=gimbal)
    app._inhibit_motion("boot")
    state = app.initial_state()
    state = app.advance_inner(state, now=0.0)
    state, _out = app.advance_outer(state, now=0.0)
    assert any(call.startswith("inhibit") for call in gimbal.calls)
    assert _motion_calls(gimbal) == []


@pytest.mark.parametrize("spy", [_SpyRateGimbal, _SpyTorqueGimbal], ids=["rate", "torque"])
def test_fault_evidence_without_safe_activation_inhibits(
    spy: type[_SpyRateGimbal | _SpyTorqueGimbal],
) -> None:
    """Latched SAFE evidence inhibits locally even without a SAFE activation."""
    gimbal = spy()
    app, bus, _g, _clock = _build_app(_plume_detector(), gimbal=gimbal)
    state = _operate(app, bus, gimbal)
    _publish_safety(bus, sequence=1, observed_s=0.0, safe_latched=True)
    state = app.poll_activations(state, now=1.0)
    assert app.containment.local_latched is True
    assert graph_name_of(state) == "operate"
    assert _motion_calls(gimbal) == []


@pytest.mark.parametrize("spy", [_SpyRateGimbal, _SpyTorqueGimbal], ids=["rate", "torque"])
def test_safe_then_ordinary_activation_never_permits_motion(
    spy: type[_SpyRateGimbal | _SpyTorqueGimbal],
) -> None:
    """Once SAFE latched, ordinary activations cannot re-arm motion."""
    gimbal = spy()
    app, bus, _g, _clock = _build_app(_plume_detector(), gimbal=gimbal)
    state = _operate(app, bus, gimbal)
    _publish_mode(bus, SystemMode.SAFE, 2, previous_mode=SystemMode.OPERATE)
    state = app.poll_activations(state, now=1.0)
    assert app.containment.local_latched is True
    _publish_mode(bus, SystemMode.IDLE, 3, previous_mode=SystemMode.SAFE)
    _publish_mode(bus, SystemMode.OPERATE, 4, previous_mode=SystemMode.IDLE)
    state = app.poll_activations(state, now=1.0)
    state = app.advance_inner(state, now=1.0)
    state, _out = app.advance_outer(state, now=1.0)
    assert app.containment.local_latched is True
    assert _motion_calls(gimbal) == []


def test_deliberate_inhibit_reference_no_latch() -> None:
    """A graph InhibitReference inhibits the driver without fault-latching."""
    gimbal = _SpyRateGimbal()
    app, bus, _g, _clock = _build_app(_plume_detector(), gimbal=gimbal)
    state = app.initial_state()
    pos = gimbal.read_position()
    assert isinstance(pos, Ok)
    app.note_gimbal_feedback(pos.value)
    _publish_mode(bus, SystemMode.IDLE, 1)
    state = app.poll_activations(state, now=0.0)
    state = app.advance_inner(state, now=0.0)
    state, _out = app.advance_outer(state, now=0.0)
    assert graph_name_of(state) == "idle"
    assert any(call.startswith("inhibit") for call in gimbal.calls)
    assert app.containment.local_latched is False


def test_failed_inhibit_retries_on_next_tick() -> None:
    """A failed inhibit leaves inhibit_applied False so the next tick retries."""
    gimbal = _SpyRateGimbal()
    app, bus, _g, _clock = _build_app(_plume_detector(), gimbal=gimbal)
    state = _operate(app, bus, gimbal)
    gimbal.inhibit_failures = 1
    _publish_mode(bus, SystemMode.SAFE, 2, previous_mode=SystemMode.OPERATE)
    state = app.poll_activations(state, now=1.0)
    assert app.containment.local_latched is True
    assert app.containment.inhibit_applied is False
    dt = app.params.config.controller.outer.dt_s
    app.note_gimbal_feedback(GimbalPosition(el_deg=0.0, timestamp_s=1.0 + dt, sequence=2))
    state, _out = app.advance_outer(state, now=1.0 + dt)
    inhibits = [call for call in gimbal.calls if call.startswith("inhibit")]
    assert len(inhibits) >= 2
    health = gimbal.read_health()
    assert isinstance(health, Ok)
    assert health.value.inhibit_confirmed is True


def test_same_mode_newer_sequence_reinhibits_old_drive() -> None:
    """A same-mode activation at a newer sequence inhibits before the cold graph."""
    gimbal = _SpyRateGimbal()
    app, bus, _g, _clock = _build_app(_plume_detector(), gimbal=gimbal)
    state = _operate(app, bus, gimbal)
    inhibits = [call for call in gimbal.calls if call.startswith("inhibit")]
    _publish_mode(bus, SystemMode.OPERATE, 2, previous_mode=SystemMode.OPERATE)
    state = app.poll_activations(state, now=1.0)
    new_inhibits = [call for call in gimbal.calls if call.startswith("inhibit")]
    assert len(new_inhibits) > len(inhibits)
    assert graph_name_of(state) == "operate"


# -- capture staleness under blocking acquire/detect/store ---------------------


class _BlockingAcquireSensor:
    """ImagingSensor double whose acquire_frame blocks until released."""

    def __init__(self) -> None:
        self.acquired = threading.Event()
        self.release = threading.Event()
        self._frame = _mosaic_frame(7)

    def acquire_frame(self) -> Result[MosaicFrame, FaultCode]:
        self.acquired.set()
        self.release.wait(timeout=5.0)
        return Ok(self._frame)

    def drain_frame(self) -> Result[None, FaultCode]:
        return Ok(None)

    def set_exposure_us(self, exposure: float) -> Result[None, FaultCode]:
        return Ok(None)

    def set_gain_db(self, gain: float) -> Result[None, FaultCode]:
        return Ok(None)

    def start_acquisition(self) -> Result[None, FaultCode]:
        return Ok(None)

    def stop_acquisition(self) -> Result[None, FaultCode]:
        return Ok(None)


class _CountingDetector:
    """Detector that records calls and optionally blocks inside detect."""

    def __init__(self) -> None:
        self.calls = 0
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block = False
        self._inner = _plume_detector()

    def detect(self, frame: ProcessedFrameMsg) -> Result[InferenceResultMsg, FaultCode]:
        self.calls += 1
        if self.block:
            self.entered.set()
            self.release.wait(timeout=5.0)
        return self._inner.detect(frame)


class _ErrDetector:
    """Detector returning a fixed fault, optionally blocking until released."""

    def __init__(self, code: FaultCode, block: bool = False) -> None:
        self.code = code
        self.block = block
        self.entered = threading.Event()
        self.release = threading.Event()

    def detect(self, frame: ProcessedFrameMsg) -> Result[InferenceResultMsg, FaultCode]:
        if self.block:
            self.entered.set()
            self.release.wait(timeout=5.0)
        return Err(self.code)


class _BlockingStorage(_MemStorage):
    """StorageWriter double whose store blocks until released."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def store(
        self, item_id: str, data: bytes, priority: DownlinkPriority
    ) -> Result[str, FaultCode]:
        self.entered.set()
        self.release.wait(timeout=5.0)
        return super().store(item_id, data, priority)


def _run_thread(target: Callable[[], object]) -> threading.Thread:
    """Start a daemon thread around a callable."""
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread


@pytest.mark.parametrize("mode", [SystemMode.OPERATE, SystemMode.IDLE])
def test_blocked_acquire_under_new_activation_skips_detector_and_publish(
    mode: SystemMode,
) -> None:
    """An activation while acquire blocks drops the stale frame entirely."""
    sensor = _BlockingAcquireSensor()
    detector = _CountingDetector()
    app, bus, gimbal, _clock = _build_app(detector, sensor=sensor)
    inf_sub = bus.subscribe(InferenceResultMsg)
    state = _operate(app, bus, gimbal)
    app.capture_this_opportunity()
    outcome_box: list[object] = []
    done = threading.Event()

    def capture() -> None:
        outcome_box.append(app.capture_once(state, 1.0))
        done.set()

    thread = _run_thread(capture)
    try:
        assert sensor.acquired.wait(timeout=5.0)
        _publish_mode(bus, mode, 2, previous_mode=SystemMode.OPERATE)
        new_state = app.poll_activations(state, now=1.0)
        assert new_state.control_revision > state.control_revision
        sensor.release.set()
        thread.join(timeout=5.0)
    finally:
        sensor.release.set()
        thread.join(timeout=5.0)
    assert done.is_set()
    assert detector.calls == 0
    assert inf_sub.empty()
    assert len(app.vision_queue) == 0


@pytest.mark.parametrize("mode", [SystemMode.SAFE, SystemMode.IDLE])
def test_blocked_detector_stale_completion_publishes_nothing(mode: SystemMode) -> None:
    """An activation while detect blocks suppresses pub, product, and vision."""
    detector = _CountingDetector()
    detector.block = True
    app, bus, gimbal, _clock = _build_app(detector)
    inf_sub = bus.subscribe(InferenceResultMsg)
    product_sub = bus.subscribe(ProductRefMsg)
    state = _operate(app, bus, gimbal)
    done = threading.Event()

    def process() -> None:
        app.process_frame(_mosaic_frame(1), state, 1.0)
        done.set()

    thread = _run_thread(process)
    try:
        assert detector.entered.wait(timeout=5.0)
        _publish_mode(bus, mode, 2, previous_mode=SystemMode.OPERATE)
        safe_state = app.poll_activations(state, now=1.0)
        assert graph_name_of(safe_state) == mode.value.lower()
        if mode is SystemMode.SAFE:
            assert app.containment.local_latched is True
        detector.release.set()
        thread.join(timeout=5.0)
    finally:
        detector.release.set()
        thread.join(timeout=5.0)
    assert done.is_set()
    assert inf_sub.empty()
    assert product_sub.empty()
    assert len(app.vision_queue) == 0
    shell = app.runtime_shell.state
    assert shell is not None and graph_name_of(shell) == mode.value.lower()


def test_local_inference_nan_latches_next_control_poll() -> None:
    """A payload-origin containing fault latches at the next control poll.

    The Err still publishes its FaultEventMsg on the live activation, and the
    next poll_activations confirms a driver inhibit with the graph identity
    unchanged; no SAFE activation is needed for local containment.
    """
    detector = _ErrDetector(FaultCode.INFERENCE_NAN)
    gimbal = _SpyTorqueGimbal()
    app, bus, _g, _clock = _build_app(detector, gimbal=gimbal)
    fault_sub = bus.subscribe(FaultEventMsg)
    state = _operate(app, bus, gimbal)
    node_before = node_name_of(state)
    app.process_frame(_mosaic_frame(1), state, 1.0)
    assert FaultCode.INFERENCE_NAN in [f.fault_code for f in _drain(fault_sub)]
    assert app.containment.local_latched is False
    state = app.poll_activations(state, now=1.0)
    assert app.containment.local_latched is True
    assert any(c.startswith("inhibit") for c in gimbal.calls)
    assert node_name_of(state) == node_before


def test_stale_detector_err_publishes_nothing() -> None:
    """An Err returned after a superseding activation drops as stale.

    No fault is published, so the local safety path cannot latch on work from
    a dead activation; the deliberate activation inhibit stays a non-latch.
    """
    detector = _ErrDetector(FaultCode.INFERENCE_NAN, block=True)
    app, bus, gimbal, _clock = _build_app(detector)
    fault_sub = bus.subscribe(FaultEventMsg)
    state = _operate(app, bus, gimbal)
    _drain(fault_sub)
    done = threading.Event()
    outcome_box: list[tuple[PayloadState, TickOutcome]] = []

    def process() -> None:
        outcome_box.append(app.process_frame(_mosaic_frame(1), state, 1.0))
        done.set()

    thread = _run_thread(process)
    try:
        assert detector.entered.wait(timeout=5.0)
        _publish_mode(bus, SystemMode.OPERATE, 2, previous_mode=SystemMode.OPERATE)
        app.poll_activations(state, now=1.0)
        detector.release.set()
        thread.join(timeout=5.0)
    finally:
        detector.release.set()
        thread.join(timeout=5.0)
    assert done.is_set()
    assert fault_sub.empty()
    assert app.containment.local_latched is False
    assert len(outcome_box) == 1
    assert outcome_box[0][1].fault is None


def test_local_fault_poll_bounded_when_inhibit_always_fails() -> None:
    """A self-published inhibit fault cannot spin the local-fault drain.

    The subscription drains to a finite batch before any HAL call, so the
    GIMBAL_FAULT a failed inhibit republishes is handled on the next poll:
    each poll retries the unconfirmed inhibit exactly once and returns.
    """
    gimbal = _AlwaysFailInhibitGimbal()
    app, bus, _g, _clock = _build_app(_plume_detector(), gimbal=gimbal)
    state = _operate(app, bus, gimbal)
    baseline = sum(1 for c in gimbal.calls if c.startswith("inhibit"))
    bus.publish(
        FaultEventMsg(
            msg_type=MessageType.FAULT_EVENT,
            timestamp_utc="t",
            fault_code=FaultCode.INFERENCE_NAN,
            subsystem="payload",
            detail="local nan",
        )
    )
    done = threading.Event()

    def poll() -> None:
        app.poll_activations(state, now=1.0)
        done.set()

    thread = _run_thread(poll)
    thread.join(timeout=5.0)
    assert done.is_set()
    assert app.containment.local_latched is True
    inhibits = [c for c in gimbal.calls if c.startswith("inhibit")]
    assert len(inhibits) == baseline + 1
    app.poll_activations(state, now=2.0)
    inhibits = [c for c in gimbal.calls if c.startswith("inhibit")]
    assert len(inhibits) == baseline + 2


def test_commit_inhibit_reference_inhibits_immediately() -> None:
    """An InhibitReference commit reaches the driver before the next inner tick."""
    gimbal = _SpyTorqueGimbal()
    app, bus, _g, _clock = _build_app(_plume_detector(), gimbal=gimbal)
    state = _operate(app, bus, gimbal)
    inhibits_before = [c for c in gimbal.calls if c.startswith("inhibit")]
    commit = app._commit_reference(state, InhibitReference("stale feedback"), 0.0)
    assert commit.fault is None
    assert commit.command_issued is False
    inhibits_after = [c for c in gimbal.calls if c.startswith("inhibit")]
    assert inhibits_after == inhibits_before + ["inhibit:stale feedback"]
    assert app.containment.local_latched is False


def test_commit_inhibit_reference_failure_records_fault() -> None:
    """An unconfirmed deliberate inhibit latches and reports GIMBAL_FAULT."""
    gimbal = _SpyTorqueGimbal()
    app, bus, _g, _clock = _build_app(_plume_detector(), gimbal=gimbal)
    state = _operate(app, bus, gimbal)
    assert app.containment.local_latched is False
    gimbal.inhibit_failures = 1
    commit = app._commit_reference(state, InhibitReference("stale feedback"), 0.0)
    assert commit.fault is FaultCode.GIMBAL_FAULT
    assert commit.command_issued is False
    assert "inhibit:stale feedback" in gimbal.calls
    assert app.containment.local_latched is True


def test_commit_inhibit_reference_while_latched_records_only() -> None:
    """Under an existing latch the reference is recorded without a HAL call."""
    gimbal = _AlwaysFailInhibitGimbal()
    app, bus, _g, _clock = _build_app(_plume_detector(), gimbal=gimbal)
    state = _operate(app, bus, gimbal)
    assert app.containment.local_latched is True
    commit = app._commit_reference(state, InhibitReference("stale feedback"), 0.0)
    assert commit.fault is None
    assert not any(c == "inhibit:stale feedback" for c in gimbal.calls)


def test_local_fault_same_drain_blocks_release() -> None:
    """Local unsafe evidence vetoes an otherwise matching recovery release."""
    gimbal = _SpyTorqueGimbal()
    app, bus, _g, _clock = _build_app(_plume_detector(), gimbal=gimbal)
    state = _operate(app, bus, gimbal)
    _publish_mode(bus, SystemMode.SAFE, 2, previous_mode=SystemMode.OPERATE)
    _publish_mode(
        bus,
        SystemMode.IDLE,
        3,
        previous_mode=SystemMode.SAFE,
        request_id="req-rec",
        recovery_authorized=True,
    )
    _publish_safety(
        bus,
        sequence=1,
        observed_s=1.0,
        recovery_request_id="req-rec",
    )
    bus.publish(
        FaultEventMsg(
            msg_type=MessageType.FAULT_EVENT,
            timestamp_utc="t",
            fault_code=FaultCode.INFERENCE_NAN,
            subsystem="payload",
            detail="local nan",
        )
    )
    state = app.poll_activations(state, now=1.0)
    assert app.containment.local_latched is True


def test_blocked_storage_suppresses_all_outputs() -> None:
    """An activation while storage blocks suppresses product, inference, vision."""
    storage = _BlockingStorage()
    app, bus, gimbal, _clock = _build_app(_plume_detector(), storage=storage)
    inf_sub = bus.subscribe(InferenceResultMsg)
    product_sub = bus.subscribe(ProductRefMsg)
    state = _operate(app, bus, gimbal)
    done = threading.Event()

    def process() -> None:
        app.process_frame(_mosaic_frame(1), state, 1.0)
        done.set()

    thread = _run_thread(process)
    try:
        assert storage.entered.wait(timeout=5.0)
        _publish_mode(bus, SystemMode.OPERATE, 2, previous_mode=SystemMode.OPERATE)
        app.poll_activations(state, now=1.0)
        storage.release.set()
        thread.join(timeout=5.0)
    finally:
        storage.release.set()
        thread.join(timeout=5.0)
    assert done.is_set()
    assert inf_sub.empty()
    assert product_sub.empty()
    assert len(app.vision_queue) == 0


def test_blocked_detector_policy_revision_change_suppresses_outputs() -> None:
    """A policy revision bump while detect blocks drops the stale completion."""
    detector = _CountingDetector()
    detector.block = True
    app, bus, gimbal, _clock = _build_app(detector)
    inf_sub = bus.subscribe(InferenceResultMsg)
    product_sub = bus.subscribe(ProductRefMsg)
    state = _operate(app, bus, gimbal)
    done = threading.Event()

    def process() -> None:
        app.process_frame(_mosaic_frame(1), state, 1.0)
        done.set()

    thread = _run_thread(process)
    try:
        assert detector.entered.wait(timeout=5.0)
        app._install(replace(state, policy_revision=state.policy_revision + 1))
        detector.release.set()
        thread.join(timeout=5.0)
    finally:
        detector.release.set()
        thread.join(timeout=5.0)
    assert done.is_set()
    assert inf_sub.empty()
    assert product_sub.empty()
    assert len(app.vision_queue) == 0


def test_blocked_detector_containment_latch_inhibits_and_suppresses() -> None:
    """A containment latch while detect blocks inhibits the gimbal before release."""
    detector = _CountingDetector()
    detector.block = True
    gimbal = _SpyRateGimbal()
    app, bus, g, _clock = _build_app(detector, gimbal=gimbal)
    inf_sub = bus.subscribe(InferenceResultMsg)
    product_sub = bus.subscribe(ProductRefMsg)
    state = _operate(app, bus, g)
    inhibits_before = [c for c in gimbal.calls if c.startswith("inhibit")]
    done = threading.Event()

    def process() -> None:
        app.process_frame(_mosaic_frame(1), state, 1.0)
        done.set()

    thread = _run_thread(process)
    try:
        assert detector.entered.wait(timeout=5.0)
        app._latch_containment("test")
        inhibits_after = [c for c in gimbal.calls if c.startswith("inhibit")]
        assert len(inhibits_after) > len(inhibits_before)
        detector.release.set()
        thread.join(timeout=5.0)
    finally:
        detector.release.set()
        thread.join(timeout=5.0)
    assert done.is_set()
    assert app.containment.local_latched is True
    assert inf_sub.empty()
    assert product_sub.empty()
    assert len(app.vision_queue) == 0


def test_process_frame_gimbal_pos_leaves_control_state_untouched() -> None:
    """Injected gimbal_pos is association-only: no encoder or shell mutation."""
    app, bus, gimbal, _clock = _build_app(_plume_detector())
    state = _operate(app, bus, gimbal)
    app._install(state)
    before_samples = tuple(app.encoder_stream.samples)
    before_shell = app.runtime_shell.state
    pos = GimbalPosition(el_deg=5.0, timestamp_s=0.5, sequence=99)
    app.process_frame(_mosaic_frame(1), state, 1.0, gimbal_pos=pos)
    assert tuple(app.encoder_stream.samples) == before_samples
    assert app.runtime_shell.state is before_shell


# -- activation bookkeeping ----------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "graph"),
    [
        (SystemMode.IDLE, "idle"),
        (SystemMode.STOW, "stow"),
        (SystemMode.SAFE, "safe"),
        (SystemMode.INIT, "init"),
        (SystemMode.OPERATE, "operate"),
    ],
)
def test_all_five_activation_modes_select_their_graph(mode: SystemMode, graph: str) -> None:
    """Each authority mode maps to its payload graph via the closed mapping."""
    app, bus, _gimbal, _clock = _build_app(_plume_detector())
    state = app.initial_state()
    _publish_mode(bus, mode, 1)
    state = app.poll_activations(state, now=0.0)
    assert graph_name_of(state) == graph


def test_request_and_transition_messages_never_select_graphs() -> None:
    """Requests/audit records are not authority: they never install a graph."""
    app, bus, _gimbal, _clock = _build_app(_plume_detector())
    bus.publish(
        SystemModeRequestMsg(
            msg_type=MessageType.SYSTEM_MODE_REQUEST,
            timestamp_utc="t",
            request_id="r1",
            requested_mode=SystemMode.OPERATE,
            requested_by="payload",
            reason="test",
        )
    )
    bus.publish(
        SystemModeTransitionMsg(
            msg_type=MessageType.SYSTEM_MODE_TRANSITION,
            timestamp_utc="t",
            transition_id="t1",
            request_id="r1",
            epoch=_EPOCH,
            previous_mode=None,
            requested_mode=SystemMode.OPERATE,
            resulting_mode=SystemMode.OPERATE,
            decision=ModeTransitionDecision.ACCEPTED,
            reason="test",
            activation_sequence=1,
        )
    )
    state = app.poll_activations(app.initial_state(), now=0.0)
    assert state.graph is None


def test_same_key_changed_contents_conflict_and_contain() -> None:
    """A repeated (epoch, seq) key with different contents faults and contains."""
    app, bus, gimbal, _clock = _build_app(_plume_detector())
    fault_sub = bus.subscribe(FaultEventMsg)
    state = _operate(app, bus, gimbal)
    _publish_mode(bus, SystemMode.OPERATE, 1, reason="different-reason")
    state = app.poll_activations(state, now=1.0)
    assert app.containment.local_latched is True
    faults = [f.fault_code for f in _drain(fault_sub)]
    assert FaultCode.GIMBAL_FAULT in faults


def test_same_key_different_previous_mode_conflicts() -> None:
    """Same activation key carrying a different previous_mode is a conflict."""
    app, bus, _gimbal, _clock = _build_app(_plume_detector())
    state = app.initial_state()
    _publish_mode(bus, SystemMode.OPERATE, 1, previous_mode=SystemMode.IDLE)
    state = app.poll_activations(state, now=0.0)
    assert graph_name_of(state) == "operate"
    _publish_mode(bus, SystemMode.OPERATE, 1, previous_mode=SystemMode.SAFE)
    state = app.poll_activations(state, now=1.0)
    assert app.containment.local_latched is True


def test_activation_gap_is_accepted_with_telemetry() -> None:
    """A forward sequence gap reenters normally and reports the gap."""
    app, bus, gimbal, _clock = _build_app(_plume_detector())
    telem_sub = bus.subscribe(TelemetryEventMsg)
    state = _operate(app, bus, gimbal)
    _publish_mode(bus, SystemMode.IDLE, 5, previous_mode=SystemMode.OPERATE)
    state = app.poll_activations(state, now=1.0)
    assert graph_name_of(state) == "idle"
    names = [t.event_name for t in _drain(telem_sub)]
    assert "activation_gap" in names


def test_newer_same_mode_activation_reenters_cold_graph() -> None:
    """A same-mode newer sequence is an accepted reentry, not a duplicate."""
    app, bus, gimbal, _clock = _build_app(_plume_detector())
    state = _operate(app, bus, gimbal)
    _publish_mode(bus, SystemMode.OPERATE, 2, previous_mode=SystemMode.OPERATE)
    state = app.poll_activations(state, now=1.0)
    assert state.activation.last is not None
    assert state.activation.last.key.sequence == 2
    assert graph_name_of(state) == "operate"


# -- flagged vision ordering ----------------------------------------------------


def _flagged_vision(state: PayloadState, t_s: float) -> CapturedVision:
    """One due flagged vision sample stamped under the current activation."""
    last = state.activation.last
    assert last is not None
    return CapturedVision(
        context=CaptureContext(
            activation_key=last.key,
            policy_revision=state.policy_revision,
            model_version="test",
            containment_generation=0,
        ),
        sample=VisionSample(
            t_s=t_s,
            frame_id="flagged-1",
            z_v=None,
            p_cog=None,
            exposure_us=1000.0,
            blobs=(),
            mode_flags=1,
            iss=None,
            theta_g_rad=0.0,
        ),
    )


@pytest.mark.parametrize(
    ("command_id", "params"),
    [
        ("GIMBAL_HOLD", {}),
        ("GIMBAL_GOTO", {"el_deg": 10.0}),
        ("GIMBAL_HOME", {}),
        ("GIMBAL_RESUME", {}),
    ],
)
def test_flagged_vision_nacks_commands(
    command_id: str, params: dict[str, str | int | float | bool]
) -> None:
    """A due flagged sample inhibits through the graph before commands commit."""
    gimbal = _SpyRateGimbal()
    app, bus, _g, _clock = _build_app(_plume_detector(), gimbal=gimbal)
    ack_sub = bus.subscribe(CommandAckMsg)
    request_sub = bus.subscribe(SystemModeRequestMsg)
    fault_sub = bus.subscribe(FaultEventMsg)
    state = _operate(app, bus, gimbal)
    t = app.params.config.controller.outer.dt_s
    app.note_gimbal_feedback(GimbalPosition(el_deg=0.0, timestamp_s=t, sequence=2))
    app.vision_queue.append(_flagged_vision(state, t))
    bus.publish(_routed(command_id, 1, params))
    state, _out = app.advance_outer(state, now=t)
    acks = [a for a in _drain(ack_sub)]
    assert acks and acks[0].status is AckStatus.REJECTED
    requests = _drain(request_sub)
    assert any(r.requested_mode is SystemMode.SAFE for r in requests)
    faults = [f.fault_code for f in _drain(fault_sub)]
    assert FaultCode.INFERENCE_NAN in faults
    assert _motion_calls(gimbal) == []


# -- recovery / release matrix --------------------------------------------------


def _latched(app: PayloadApp, bus: MessageBus, gimbal: GimbalActuator) -> PayloadState:
    """OPERATE then SAFE: returns the latched SAFE state."""
    state = _operate(app, bus, gimbal)
    _publish_mode(bus, SystemMode.SAFE, 2, previous_mode=SystemMode.OPERATE)
    state = app.poll_activations(state, now=1.0)
    assert app.containment.local_latched is True
    return state


def _authorize_idle(bus: MessageBus, request_id: str, seq: int = 3) -> None:
    """Publish the authorized SAFE->IDLE recovery activation."""
    _publish_mode(
        bus,
        SystemMode.IDLE,
        seq,
        previous_mode=SystemMode.SAFE,
        request_id=request_id,
        recovery_authorized=True,
    )


def _prime_release_health(
    app: PayloadApp, gimbal: GimbalActuator, clock: ManualClock, at_s: float
) -> None:
    """Advance the clock and record a fresh bounded encoder sample at `at_s`."""
    clock.advance(at_s - clock.monotonic_s())
    pos = gimbal.read_position()
    assert isinstance(pos, Ok)
    app.note_gimbal_feedback(pos.value)


def test_ordinary_idle_activation_never_releases() -> None:
    """An IDLE activation without recovery_authorized stays latched."""
    app, bus, gimbal, _clock = _build_app(_plume_detector())
    state = _latched(app, bus, gimbal)
    _publish_mode(bus, SystemMode.IDLE, 3, previous_mode=SystemMode.SAFE)
    state = app.poll_activations(state, now=2.0)
    assert app.containment.local_latched is True


def test_authorized_idle_without_evidence_stays_pending_no_fault() -> None:
    """Missing release evidence is pending, not a fault."""
    app, bus, gimbal, _clock = _build_app(_plume_detector())
    fault_sub = bus.subscribe(FaultEventMsg)
    state = _latched(app, bus, gimbal)
    _drain(fault_sub)
    _authorize_idle(bus, "rec-1")
    state = app.poll_activations(state, now=2.0)
    assert app.containment.local_latched is True
    assert _drain(fault_sub) == []


def test_fresh_matching_evidence_releases_once() -> None:
    """Fresh fault-owned release evidence under a matching request_id releases."""
    app, bus, gimbal, clock = _build_app(_plume_detector())
    state = _latched(app, bus, gimbal)
    _authorize_idle(bus, "rec-1")
    state = app.poll_activations(state, now=2.0)
    assert app.containment.local_latched is True
    _prime_release_health(app, gimbal, clock, 2.0)
    _publish_safety(
        bus, sequence=7, observed_s=0.0, safe_latched=False, recovery_request_id="rec-1"
    )
    state = app.poll_activations(state, now=2.0)
    assert app.containment.local_latched is False


@pytest.mark.parametrize(
    ("epoch", "observed_s", "label"),
    [
        ("other", 0.0, "wrong epoch"),
        (_EPOCH, 100.0, "future observation"),
        (_EPOCH, math.nan, "nonfinite observation"),
        (_EPOCH, -100.0, "stale observation"),
    ],
)
def test_malformed_release_evidence_stays_pending(
    epoch: str, observed_s: float, label: str
) -> None:
    """Wrong-epoch, future, nonfinite, or stale evidence never releases."""
    app, bus, gimbal, clock = _build_app(_plume_detector())
    fault_sub = bus.subscribe(FaultEventMsg)
    state = _latched(app, bus, gimbal)
    _authorize_idle(bus, "rec-1")
    state = app.poll_activations(state, now=2.0)
    _drain(fault_sub)
    _prime_release_health(app, gimbal, clock, 2.0)
    _publish_safety(
        bus,
        sequence=7,
        observed_s=observed_s,
        epoch=epoch,
        recovery_request_id="rec-1",
    )
    state = app.poll_activations(state, now=2.0)
    assert app.containment.local_latched is True, label
    assert _drain(fault_sub) == []


def test_stale_evidence_sequence_is_ignored() -> None:
    """A nonincreasing evidence sequence cannot replace the latest record."""
    app, bus, gimbal, clock = _build_app(_plume_detector())
    state = _latched(app, bus, gimbal)
    _authorize_idle(bus, "rec-1")
    state = app.poll_activations(state, now=2.0)
    _prime_release_health(app, gimbal, clock, 2.0)
    _publish_safety(bus, sequence=9, observed_s=0.0, recovery_request_id="other")
    state = app.poll_activations(state, now=2.0)
    _publish_safety(bus, sequence=8, observed_s=0.0, recovery_request_id="rec-1")
    state = app.poll_activations(state, now=2.0)
    assert app.containment.local_latched is True


def test_unsafe_evidence_before_clear_in_same_drain_cannot_release() -> None:
    """Unsafe evidence anywhere in the drain blocks release that tick."""
    app, bus, gimbal, clock = _build_app(_plume_detector())
    state = _latched(app, bus, gimbal)
    _authorize_idle(bus, "rec-1")
    state = app.poll_activations(state, now=2.0)
    _prime_release_health(app, gimbal, clock, 2.0)
    _publish_safety(
        bus,
        sequence=8,
        observed_s=0.0,
        safe_latched=False,
        active_faults=(FaultCode.GIMBAL_FAULT,),
    )
    _publish_safety(
        bus, sequence=9, observed_s=0.0, safe_latched=False, recovery_request_id="rec-1"
    )
    state = app.poll_activations(state, now=2.0)
    assert app.containment.local_latched is True


def test_replayed_request_id_cannot_release_new_latch() -> None:
    """A consumed recovery request_id cannot release a later latch."""
    app, bus, gimbal, clock = _build_app(_plume_detector())
    state = _latched(app, bus, gimbal)
    _authorize_idle(bus, "rec-1")
    state = app.poll_activations(state, now=2.0)
    _prime_release_health(app, gimbal, clock, 2.0)
    _publish_safety(bus, sequence=7, observed_s=0.0, recovery_request_id="rec-1")
    state = app.poll_activations(state, now=2.0)
    assert app.containment.local_latched is False
    _publish_mode(bus, SystemMode.SAFE, 4, previous_mode=SystemMode.IDLE)
    state = app.poll_activations(state, now=3.0)
    assert app.containment.local_latched is True
    _authorize_idle(bus, "rec-1", seq=5)
    _prime_release_health(app, gimbal, clock, 3.0)
    _publish_safety(bus, sequence=8, observed_s=0.0, recovery_request_id="rec-1")
    state = app.poll_activations(state, now=3.0)
    assert app.containment.local_latched is True


def test_unconfirmed_driver_prevents_release() -> None:
    """Release evidence without a confirmed driver inhibit cannot release."""
    gimbal = _SpyRateGimbal()
    app, bus, _g, _clock = _build_app(_plume_detector(), gimbal=gimbal)
    fault_sub = bus.subscribe(FaultEventMsg)
    state = _operate(app, bus, gimbal)
    gimbal.inhibit_failures = 99  # every inhibit attempt reports failure
    app.containment.local_latched = True
    _authorize_idle(bus, "rec-1")
    state = app.poll_activations(state, now=2.0)
    _publish_safety(bus, sequence=7, observed_s=0.0, recovery_request_id="rec-1")
    state = app.poll_activations(state, now=2.0)
    assert app.containment.local_latched is True
    faults = [f.fault_code for f in _drain(fault_sub)]
    assert FaultCode.GIMBAL_FAULT in faults


def test_missing_encoder_sample_prevents_release() -> None:
    """Release evidence without a valid encoder sample cannot release."""
    app, bus, gimbal, clock = _build_app(_plume_detector())
    state = _latched(app, bus, gimbal)
    _authorize_idle(bus, "rec-1")
    state = app.poll_activations(state, now=2.0)
    clock.advance(2.0)
    pos = gimbal.read_position()  # refresh driver feedback only, not the stream
    assert isinstance(pos, Ok)
    app.encoder_stream.samples.clear()
    _publish_safety(bus, sequence=7, observed_s=0.0, recovery_request_id="rec-1")
    state = app.poll_activations(state, now=2.0)
    assert app.containment.local_latched is True


# -- stop/shutdown fault observability -----------------------------------------


class _FailingStopSensor(_BlockingAcquireSensor):
    """Sensor whose stop_acquisition reports a driver failure."""

    def acquire_frame(self) -> Result[MosaicFrame, FaultCode]:
        return Err(FaultCode.CAMERA_STALL)

    def stop_acquisition(self) -> Result[None, FaultCode]:
        return Err(FaultCode.CAMERA_STALL)


def test_capture_off_stop_failure_publishes_fault_and_keeps_state() -> None:
    """A failed stop leaves acquisition_on/applied set and emits a fault."""
    app, bus, _gimbal, _clock = _build_app(_plume_detector(), sensor=_FailingStopSensor())
    fault_sub = bus.subscribe(FaultEventMsg)
    app.capture_shell.acquisition_on = True
    app.capture_shell.applied_capture = (1000.0, 0.0)
    state = app.initial_state()
    state, outcome = app.capture_once(state, 0.0)
    assert outcome.fault is FaultCode.CAMERA_STALL
    assert app.capture_shell.acquisition_on is True
    assert app.capture_shell.applied_capture == (1000.0, 0.0)
    faults = [f.fault_code for f in _drain(fault_sub)]
    assert FaultCode.CAMERA_STALL in faults


class _FailShutdownGimbal(_SpyRateGimbal):
    """Rate gimbal whose shutdown reports a driver failure."""

    def shutdown(self) -> Result[None, FaultCode]:
        self.calls.append("shutdown")
        return Err(FaultCode.GIMBAL_FAULT)


def test_run_shutdown_failures_publish_faults() -> None:
    """run()'s finally publishes faults for failed gimbal shutdown/stop."""
    sensor = _FailingStopSensor()
    gimbal = _FailShutdownGimbal()
    app, bus, _g, clock = _build_app(_plume_detector(), gimbal=gimbal, sensor=sensor)
    app.capture_shell.acquisition_on = True
    fault_sub = bus.subscribe(FaultEventMsg)
    stop = threading.Event()

    thread = _run_thread(lambda: app.run(stop))
    try:
        clock.advance(0.05)
        stop.set()
        thread.join(timeout=5.0)
    finally:
        stop.set()
        thread.join(timeout=5.0)
    faults = [f.fault_code for f in _drain(fault_sub)]
    assert FaultCode.GIMBAL_FAULT in faults
    assert FaultCode.CAMERA_STALL in faults
    assert "shutdown" in gimbal.calls


# -- one edge per outer tick ----------------------------------------------------


def test_two_queued_commands_commit_across_two_outer_ticks() -> None:
    """Two distinct queued commands execute on consecutive outer ticks."""
    app, bus, gimbal, _clock = _build_app(_plume_detector())
    ack_sub = bus.subscribe(CommandAckMsg)
    state = _operate(app, bus, gimbal)
    dt = app.params.config.controller.outer.dt_s
    pos = gimbal.read_position()
    assert isinstance(pos, Ok)
    app.note_gimbal_feedback(replace(pos.value, timestamp_s=dt, sequence=2))
    bus.publish(_routed("GIMBAL_HOLD", 1))
    bus.publish(_routed("GIMBAL_RESUME", 2))
    state, _out = app.advance_outer(state, now=dt)
    acks = _drain(ack_sub)
    assert len(acks) == 1
    assert node_name_of(state) == "hold"
    app.note_gimbal_feedback(replace(pos.value, timestamp_s=2 * dt, sequence=3))
    state, _out = app.advance_outer(state, now=2 * dt)
    acks = _drain(ack_sub)
    assert len(acks) == 1


# -- pointing telemetry ---------------------------------------------------------


def test_pointing_telemetry_exact_fields() -> None:
    """TRACKING pointing telemetry reports lead-authored exact field values."""
    app, bus, gimbal, _clock = _build_app(_plume_detector())
    telem_sub = bus.subscribe(TelemetryEventMsg)
    state = _operate(app, bus, gimbal)
    graph = state.graph
    assert isinstance(graph, operate.State)
    graph = replace(
        graph,
        node=operate.OperateNode.TRACKING,
        residual=ResidualState(x=np.array([0.01, 0.02]), P=np.eye(2), has_measurement=True),
        residual_history=replace(
            graph.residual_history,
            checkpoint=replace(
                graph.residual_history.checkpoint,
                t_s=1.0,
                encoder_angle_rad=0.2,
                encoder_endpoint_variance_rad2=0.004,
            ),
        ),
        target=replace(graph.target, last_omega_t_nom=0.03),
    )
    state = replace(state, graph=graph)
    encoder = EncoderSample(sample_id="e1", t_s=2.0, angle_rad=0.3, angle_variance_rad2=0.006)
    app._publish_pointing(state, 2.0, encoder, None)
    msg = next(t for t in _drain(telem_sub) if t.event_name == "pointing")
    payload = msg.payload
    residual_cfg = PactConfig().controller.residual
    assert payload["omega_t_total"] == pytest.approx(0.05)
    assert payload["encoder_span_s"] == pytest.approx(1.0)
    assert payload["encoder_net_displacement_rad"] == pytest.approx(0.1)
    assert payload["encoder_endpoint_covariance_rad2"] == pytest.approx(0.01)
    assert payload["process_q_e_rad2"] == pytest.approx(residual_cfg.q_accel_rad2_s3 / 3.0)
    assert payload["process_q_rate_rad2_s2"] == pytest.approx(residual_cfg.q_accel_rad2_s3)
    last = state.activation.last
    assert last is not None
    assert payload["payload_graph"] == "operate"
    assert payload["payload_node"] == "tracking"
    assert payload["activation_epoch"] == last.key.epoch
    assert payload["activation_sequence"] == last.key.sequence


def test_pointing_telemetry_safe_idle_fields_none() -> None:
    """Outside TRACKING, residual/rate/chrono fields are None, not zeros."""
    app, bus, _gimbal, _clock = _build_app(_plume_detector())
    telem_sub = bus.subscribe(TelemetryEventMsg)
    state = app.initial_state()
    _publish_mode(bus, SystemMode.SAFE, 1)
    state = app.poll_activations(state, now=0.0)
    encoder = EncoderSample(sample_id="e1", t_s=1.0, angle_rad=0.0, angle_variance_rad2=0.0)
    app._publish_pointing(state, 1.0, encoder, None)
    payload = next(t for t in _drain(telem_sub) if t.event_name == "pointing").payload
    assert payload["payload_graph"] == "safe"
    for field_name in (
        "e",
        "omega_t_nom",
        "omega_t_res",
        "omega_t_total",
        "P00",
        "encoder_span_s",
        "process_q_e_rad2",
        "process_q_rate_rad2_s2",
        "encoder_net_displacement_rad",
        "encoder_endpoint_covariance_rad2",
    ):
        assert payload[field_name] is None, field_name


def _valid_vision(state: PayloadState, t_s: float, frame_id: str = "v1") -> CapturedVision:
    """One due unflagged vision sample stamped under the current activation."""
    last = state.activation.last
    assert last is not None
    return CapturedVision(
        context=CaptureContext(
            activation_key=last.key,
            policy_revision=state.policy_revision,
            model_version="test",
        ),
        sample=VisionSample(
            t_s=t_s,
            frame_id=frame_id,
            z_v=None,
            p_cog=None,
            exposure_us=1000.0,
            blobs=(),
            mode_flags=0,
            iss=None,
            theta_g_rad=0.0,
        ),
    )


def test_same_tick_command_and_vision_commit_one_edge() -> None:
    """A valid command + due vision in one outer tick commit exactly one edge."""
    app, bus, gimbal, _clock = _build_app(_plume_detector())
    ack_sub = bus.subscribe(CommandAckMsg)
    state = _operate(app, bus, gimbal)
    dt = app.params.config.controller.outer.dt_s
    app.note_gimbal_feedback(GimbalPosition(el_deg=0.0, timestamp_s=dt, sequence=2))
    app.vision_queue.append(_valid_vision(state, dt))
    bus.publish(_routed("GIMBAL_HOLD", 1))
    state, _out = app.advance_outer(state, now=dt)
    acks = _drain(ack_sub)
    assert len(acks) == 1
    assert acks[0].status is AckStatus.ACCEPTED
    assert node_name_of(state) == "hold"
    graph = state.graph
    assert isinstance(graph, operate.State)
    assert len([frame for frame, _t in graph.seen_vision]) == 1
