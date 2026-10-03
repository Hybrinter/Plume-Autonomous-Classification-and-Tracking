"""Tests for planned capture execution and configurable imaging policy.

Pure schedule tests cover plan_capture deadlines, duty selection, context
resets, and record_capture decimation. App tests use a counting spy sensor to
prove HAL ordering, idempotence, failure containment through the control
owner, and the decimation/product gates. Startup tests reject invalid policy
config before any HAL call.
"""

import math
import threading
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
from flight.hal.drivers_sim import SimGimbal, SimIssEphemeris
from flight.libs.bus import MessageBus, Subscription
from flight.libs.config import (
    PactConfig,
    PayloadPolicyConfig,
    PayloadPolicyOverrideConfig,
)
from flight.libs.messages import (
    FaultEventMsg,
    InferenceResultMsg,
    ProcessedFrameMsg,
    ProductRefMsg,
    RoutedCommandMsg,
    SystemModeActivatedMsg,
    TelemetryEventMsg,
)
from flight.libs.time import ManualClock
from flight.libs.types import (
    ActivationKey,
    DownlinkPriority,
    Err,
    FaultCode,
    MessageType,
    MosaicFrame,
    Ok,
    Result,
    SystemMode,
)
from flight.payload.app import PayloadApp, TickOutcome
from flight.payload.calibration_io import build_identity_calibration
from flight.payload.graphs import operate
from flight.payload.graphs.base import EffectivePolicy, SystemRequestIntent, TickInputs
from flight.payload.graphs.parameters import GraphParameters
from flight.payload.imaging import (
    CaptureDecision,
    CaptureSchedule,
    capture_wait_s,
    plan_capture,
    record_capture,
)
from flight.payload.inference import DetectorBackend, ScriptedDetector
from flight.payload.records import CaptureContext, HealthSample
from flight.payload.state import PayloadState, node_name_of
from flight.payload.tracking import EncoderSample

_EPOCH = "epoch-imaging-test"
_PARAMS = GraphParameters(config=PactConfig())
_LIMITS = _PARAMS.policy_limits
_KEY = ActivationKey(epoch=_EPOCH, sequence=1)


def _context(revision: int = 0, sequence: int = 1) -> CaptureContext:
    """One capture context for pure schedule tests."""
    return CaptureContext(
        activation_key=ActivationKey(epoch=_EPOCH, sequence=sequence),
        policy_revision=revision,
        model_version="",
    )


def _policy(
    capture_interval_s: float | None = None,
    duty_cycle: float | None = None,
) -> EffectivePolicy:
    """The enabled default policy with selected imaging fields replaced."""
    policy = _PARAMS.default_policy(enabled=True)
    imaging = policy.imaging
    if capture_interval_s is not None:
        imaging = replace(imaging, capture_interval_s=capture_interval_s)
    if duty_cycle is not None:
        imaging = replace(imaging, duty_cycle=duty_cycle)
    return replace(policy, imaging=imaging)


def _tick(now_s: float, key: ActivationKey, *, health: HealthSample | None = None) -> TickInputs:
    """Minimal TickInputs for graph-level policy checks."""
    return TickInputs(
        now_s=now_s,
        timestamp_utc="",
        activation_key=key,
        encoder=EncoderSample(sample_id="e", t_s=now_s, angle_rad=0.0, angle_variance_rad2=0.0),
        navigation=None,
        vision=None,
        health=health
        or HealthSample(feedback_valid=True, inhibit_confirmed=False, contained=False),
    )


def test_plan_capture_first_opportunity_due_immediately() -> None:
    """A fresh schedule spends its first opportunity at the first call."""
    plan = plan_capture(CaptureSchedule(), _policy(), _context(), 0.0, _LIMITS)
    assert isinstance(plan, Ok)
    assert plan.value.decision is CaptureDecision.DRAIN
    assert plan.value.schedule.opportunities == 1
    assert plan.value.schedule.next_opportunity_s == pytest.approx(
        0.0 + _policy().imaging.capture_interval_s
    )


def test_plan_capture_waits_without_spending_duty() -> None:
    """Interval 0.5: calls at .49/.99 wait; .5/1 spend opportunities (3 total)."""
    schedule = CaptureSchedule()
    policy = _policy(capture_interval_s=0.5)
    context = _context()
    decisions: list[CaptureDecision] = []
    for now in (0.0, 0.49, 0.5, 0.99, 1.0):
        plan = plan_capture(schedule, policy, context, now, _LIMITS)
        assert isinstance(plan, Ok)
        schedule = plan.value.schedule
        decisions.append(plan.value.decision)
    assert schedule.opportunities == 3
    assert decisions[1] is CaptureDecision.WAIT
    assert decisions[3] is CaptureDecision.WAIT


def test_capture_wait_hits_configured_period_inside_outer_dt() -> None:
    """Shipped 35 Hz period is not aliased to two outer steps (40 ms)."""
    cfg = PactConfig()
    interval = GraphParameters(config=cfg).default_policy(True).imaging.capture_interval_s
    outer_dt = cfg.controller.outer.dt_s
    assert interval == pytest.approx(1.0 / cfg.sensor.capture.max_frame_rate_hz)
    assert interval > outer_dt
    first = capture_wait_s(interval, 0.0, 0.0, outer_dt)
    second = capture_wait_s(interval, first, first, outer_dt)
    assert first == pytest.approx(outer_dt)
    assert second == pytest.approx(interval - outer_dt)
    assert first + second == pytest.approx(interval)
    # A second full outer sleep would step past the armed deadline.
    assert first + outer_dt > interval


def test_capture_wait_bounds_idle_and_returns_immediately_on_overrun() -> None:
    """No armed future deadline uses the policy wake; an overrun does not sleep."""
    outer_dt = 0.020
    assert capture_wait_s(None, 0.0, 0.0, outer_dt) == pytest.approx(outer_dt)
    # Deadline already due at the call: bounded wake, no spin.
    assert capture_wait_s(0.5, 1.0, 1.0, outer_dt) == pytest.approx(outer_dt)
    # Deadline armed at the call, processing finished after it: run now.
    assert capture_wait_s(1.0 / 35.0, 0.0, 0.040, outer_dt) == 0.0
    # A far deadline stays capped at the policy wake.
    assert capture_wait_s(5.0, 0.0, 0.0, outer_dt) == pytest.approx(outer_dt)
    assert capture_wait_s(None, math.nan, math.nan, outer_dt) == pytest.approx(outer_dt)
    assert capture_wait_s(1.0, 0.0, 0.0, math.nan) == 0.0
    assert capture_wait_s(1.0, 0.0, 0.0, 0.0) == 0.0


def test_run_capture_sleep_reaches_deadline_before_two_outer_periods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run() sleeps to the armed deadline, not another full outer period."""
    app, _bus, _gimbal, clock, _spy, _storage = _build()
    interval = app.params.default_policy(True).imaging.capture_interval_s
    outer_dt = app.params.config.controller.outer.dt_s
    stop = threading.Event()
    waits: list[float] = []
    original_wait = threading.Event.wait

    def fake_capture(self: PayloadApp, state: object, now: float) -> tuple[object, TickOutcome]:
        deadline = self.capture_shell.schedule.next_opportunity_s
        if deadline is None or now + 1e-12 >= deadline:
            self.capture_shell.schedule = replace(
                self.capture_shell.schedule,
                next_opportunity_s=now + interval,
            )
        return state, TickOutcome(0, None, False)

    def fake_wait(self: threading.Event, timeout: float | None = None) -> bool:
        if self is not stop or threading.current_thread().name == "payload-control":
            return original_wait(self, timeout)
        assert timeout is not None
        waits.append(timeout)
        clock.advance(timeout)
        if clock.monotonic_s() + 1e-12 >= interval or len(waits) > 4:
            self.set()
        return self.is_set()

    monkeypatch.setattr(PayloadApp, "capture_once", fake_capture)
    monkeypatch.setattr(threading.Event, "wait", fake_wait)
    app.run(stop)

    assert waits[0] == pytest.approx(outer_dt)
    assert waits[1] == pytest.approx(interval - outer_dt)
    assert sum(waits) == pytest.approx(interval)
    assert len(waits) == 2


def test_plan_capture_late_call_spends_one_opportunity() -> None:
    """A jump past many deadlines spends exactly one opportunity, no catchup."""
    schedule = CaptureSchedule()
    policy = _policy(capture_interval_s=0.5)
    context = _context()
    for now in (0.0, 10.0):
        plan = plan_capture(schedule, policy, context, now, _LIMITS)
        assert isinstance(plan, Ok)
        schedule = plan.value.schedule
    assert schedule.opportunities == 2
    assert schedule.next_opportunity_s == pytest.approx(10.5)


def test_plan_capture_context_change_resets_phase() -> None:
    """A new policy revision or activation restarts the schedule phase."""
    schedule = CaptureSchedule()
    policy = _policy()
    plan = plan_capture(schedule, policy, _context(revision=0), 0.0, _LIMITS)
    assert isinstance(plan, Ok)
    schedule = plan.value.schedule
    # Early call under a new revision is due immediately with fresh counters.
    plan = plan_capture(schedule, policy, _context(revision=1), 0.001, _LIMITS)
    assert isinstance(plan, Ok)
    assert plan.value.schedule.opportunities == 1
    assert plan.value.schedule.captured_frames == 0
    # A new activation under the same revision also resets.
    plan = plan_capture(
        plan.value.schedule, policy, _context(revision=1, sequence=2), 0.002, _LIMITS
    )
    assert isinstance(plan, Ok)
    assert plan.value.schedule.opportunities == 1


def test_plan_capture_rejects_nonfinite_time() -> None:
    """NaN/Inf call times are rejected, never planned."""
    for bad in (math.nan, math.inf, -math.inf):
        plan = plan_capture(CaptureSchedule(), _policy(), _context(), bad, _LIMITS)
        assert isinstance(plan, Err)
        assert plan.error is FaultCode.COMMAND_INVALID


def test_plan_capture_rejects_invalid_policy() -> None:
    """Enabled inference with disabled acquisition is rejected, not planned."""
    policy = _PARAMS.default_policy(enabled=True)
    bad = replace(
        policy,
        imaging=replace(policy.imaging, acquisition_enabled=False),
    )
    plan = plan_capture(CaptureSchedule(), bad, _context(), 0.0, _LIMITS)
    assert isinstance(plan, Err)
    assert plan.error is FaultCode.COMMAND_INVALID


@pytest.mark.parametrize(
    ("duty", "captures"),
    [(0.0, set()), (0.5, {2, 4}), (1.0, {1, 2, 3, 4})],
)
def test_plan_capture_duty_floor_selects_opportunities(duty: float, captures: set[int]) -> None:
    """Duty floor: index starts at 1; 0.5 captures even opportunities."""
    schedule = CaptureSchedule()
    interval = 1.0 / _LIMITS.max_frame_rate_hz
    policy = _policy(capture_interval_s=interval, duty_cycle=duty)
    if duty == 0.0:
        policy = replace(policy, inference=replace(policy.inference, enabled=False))
    context = _context()
    captured: set[int] = set()
    for index in range(1, 5):
        plan = plan_capture(schedule, policy, context, index * interval, _LIMITS)
        assert isinstance(plan, Ok)
        schedule = plan.value.schedule
        if plan.value.decision is CaptureDecision.CAPTURE:
            captured.add(schedule.opportunities)
    assert captured == captures


def test_record_capture_decimates_inference_after_first_success() -> None:
    """every_n_frames=3 runs inference on successful captures 1, 4, 7."""
    schedule = CaptureSchedule()
    inference = replace(_PARAMS.default_policy(True).inference, every_n_frames=3)
    context = _context()
    due: list[bool] = []
    for _ in range(7):
        schedule, run = record_capture(schedule, context, inference)
        due.append(run)
    assert due == [True, False, False, True, False, False, True]
    assert schedule.captured_frames == 7


def test_record_capture_disabled_never_runs_and_resets_on_context() -> None:
    """Disabled inference never runs; a context change restarts the count."""
    schedule = CaptureSchedule()
    disabled = replace(_PARAMS.default_policy(True).inference, enabled=False)
    schedule, run = record_capture(schedule, _context(), disabled)
    assert run is False
    assert schedule.captured_frames == 1
    enabled = replace(_PARAMS.default_policy(True).inference, every_n_frames=3)
    schedule, _ = record_capture(schedule, _context(), enabled)
    schedule, run = record_capture(schedule, _context(revision=1), enabled)
    assert run is True
    assert schedule.captured_frames == 1


def test_operating_policy_defaults_equal_enabled_base() -> None:
    """Empty overrides resolve to the unchanged enabled base policy."""
    params = _PARAMS
    resolved = params.operating_policy()
    assert isinstance(resolved, Ok)
    assert resolved.value == params.default_policy(enabled=True)


def test_operating_policy_applies_graph_then_node_override() -> None:
    """Node fields override the operate table; unset fields inherit downward."""
    base = PactConfig()
    cfg = replace(
        base,
        payload_policy=PayloadPolicyConfig(
            operate=PayloadPolicyOverrideConfig(capture_interval_s=2.0, duty_cycle=1.0),
            tracking=PayloadPolicyOverrideConfig(exposure_us=100.0),
            hold=PayloadPolicyOverrideConfig(acquisition_enabled=False),
        ),
    )
    params = GraphParameters(config=cfg)
    tracking = params.operating_policy(cfg.payload_policy.tracking)
    assert isinstance(tracking, Ok)
    assert tracking.value.imaging.exposure_us == 100.0
    assert tracking.value.imaging.capture_interval_s == 2.0
    assert tracking.value.imaging.duty_cycle == 1.0
    rewind = params.operating_policy(cfg.payload_policy.rewind)
    assert isinstance(rewind, Ok)
    assert rewind.value.imaging.exposure_us == cfg.sensor.capture.initial_exposure_us
    fast_rewind = params.operating_policy(cfg.payload_policy.fast_rewind)
    assert isinstance(fast_rewind, Ok)
    assert fast_rewind.value.imaging.capture_interval_s == 2.0
    hold = params.operating_policy(cfg.payload_policy.hold)
    assert isinstance(hold, Err)


def test_operating_policy_rejects_invalid_resolved_combinations() -> None:
    """Invalid resolved combos return Err: zero duty + inference, bad exposure."""
    base = PactConfig()
    cfg = replace(
        base,
        payload_policy=PayloadPolicyConfig(
            operate=PayloadPolicyOverrideConfig(duty_cycle=0.0),
        ),
    )
    params = GraphParameters(config=cfg)
    assert isinstance(params.operating_policy(), Err)
    cfg = replace(
        base,
        payload_policy=PayloadPolicyConfig(
            tracking=PayloadPolicyOverrideConfig(exposure_us=9_000_000.0),
        ),
    )
    params = GraphParameters(config=cfg)
    assert isinstance(params.operating_policy(cfg.payload_policy.tracking), Err)


def test_policy_override_field_ranges_reject() -> None:
    """Pydantic ranges reject out-of-range/nonfinite/loose-typed overrides."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PayloadPolicyOverrideConfig(duty_cycle=1.5)
    with pytest.raises(ValidationError):
        PayloadPolicyOverrideConfig(capture_interval_s=0.0)
    with pytest.raises(ValidationError):
        PayloadPolicyOverrideConfig(capture_interval_s=math.nan)
    with pytest.raises(ValidationError):
        PayloadPolicyOverrideConfig(exposure_us=math.inf)
    with pytest.raises(ValidationError):
        PayloadPolicyOverrideConfig(every_n_frames=0)
    with pytest.raises(ValidationError):
        PayloadPolicyOverrideConfig(every_n_frames=True)


def test_stale_feedback_inhibit_uses_node_configured_policy() -> None:
    """Stale-feedback inhibit resolves the live node's configured policy."""
    base = PactConfig()
    cfg = replace(
        base,
        payload_policy=PayloadPolicyConfig(
            tracking=PayloadPolicyOverrideConfig(exposure_us=250.0),
        ),
    )
    params = GraphParameters(config=cfg)
    key = ActivationKey(epoch="e", sequence=1)
    state = operate.initial_state(_tick(0.0, key), params)
    _state, outcome = operate.step(
        state,
        _tick(
            0.02,
            key,
            health=HealthSample(feedback_valid=False, inhibit_confirmed=False, contained=False),
        ),
        params,
    )
    assert outcome.outcome.policy.imaging.acquisition_enabled is True
    assert outcome.outcome.policy.imaging.exposure_us == 250.0


def test_stale_feedback_inhibit_resolver_err_faults_and_requests_safe() -> None:
    """An invalid node override under stale feedback fails closed with SAFE."""
    base = PactConfig()
    cfg = replace(
        base,
        payload_policy=PayloadPolicyConfig(
            tracking=PayloadPolicyOverrideConfig(exposure_us=9_000_000.0),
        ),
    )
    params = GraphParameters(config=cfg)
    key = ActivationKey(epoch="e", sequence=1)
    state = operate.initial_state(_tick(0.0, key), params)
    _state, outcome = operate.step(
        state,
        _tick(
            0.02,
            key,
            health=HealthSample(feedback_valid=False, inhibit_confirmed=False, contained=False),
        ),
        params,
    )
    assert outcome.outcome.policy.imaging.acquisition_enabled is False
    assert outcome.outcome.faults == (FaultCode.COMMAND_INVALID,)
    assert outcome.outcome.system_request is SystemRequestIntent.SAFE


# ---------------------------------------------------------------------------
# App-level spy-sensor tests
# ---------------------------------------------------------------------------


class _MemStorage:
    """In-memory StorageWriter double (records stored products)."""

    def __init__(self) -> None:
        self.items: dict[str, bytes] = {}

    def store(
        self, item_id: str, data: bytes, priority: DownlinkPriority
    ) -> Result[str, FaultCode]:
        entry_id = f"{len(self.items):08d}_{item_id}"
        self.items[entry_id] = data
        return Ok(entry_id)


class _SpySensor:
    """ImagingSensor double recording HAL order, values, and error injection.

    `calls` holds stage names in order; `values` holds the last accepted
    setter float so tests assert exact policy values reach the HAL. The
    `on_<stage>` hooks run synchronously inside the stage so a test can land
    an activation or latch containment while a HAL call is in flight.
    """

    def __init__(self, frames: list[MosaicFrame] | None = None) -> None:
        self.calls: list[str] = []
        self.frames = list(frames or [])
        self.errors: dict[str, FaultCode] = {}
        self.values: dict[str, float] = {}
        self.on_exposure: Callable[[], None] | None = None
        self.on_gain: Callable[[], None] | None = None
        self.on_start: Callable[[], None] | None = None
        self.on_acquire: Callable[[], None] | None = None
        self.on_drain: Callable[[], None] | None = None

    def _fail(self, stage: str) -> FaultCode | None:
        self.calls.append(stage)
        return self.errors.get(stage)

    def set_exposure_us(self, exposure: float) -> Result[None, FaultCode]:
        code = self._fail("set_exposure_us")
        if code is not None:
            return Err(code)
        self.values["exposure_us"] = exposure
        if self.on_exposure is not None:
            self.on_exposure()
        return Ok(None)

    def set_gain_db(self, gain: float) -> Result[None, FaultCode]:
        code = self._fail("set_gain_db")
        if code is not None:
            return Err(code)
        self.values["gain_db"] = gain
        if self.on_gain is not None:
            self.on_gain()
        return Ok(None)

    def start_acquisition(self) -> Result[None, FaultCode]:
        code = self._fail("start_acquisition")
        if code is not None:
            return Err(code)
        if self.on_start is not None:
            self.on_start()
        return Ok(None)

    def stop_acquisition(self) -> Result[None, FaultCode]:
        code = self._fail("stop_acquisition")
        return Err(code) if code is not None else Ok(None)

    def drain_frame(self) -> Result[None, FaultCode]:
        code = self._fail("drain_frame")
        # The hook models the world changing while the device call is in
        # flight, so it runs before the outcome is returned.
        if self.on_drain is not None:
            self.on_drain()
        if code is not None:
            return Err(code)
        if self.frames:
            self.frames.pop(0)
        return Ok(None)

    def acquire_frame(self) -> Result[MosaicFrame, FaultCode]:
        code = self._fail("acquire_frame")
        if self.on_acquire is not None:
            self.on_acquire()
        if code is not None:
            return Err(code)
        if not self.frames:
            return Err(FaultCode.CAMERA_STALL)
        return Ok(self.frames.pop(0))


class _CountingDetector:
    """Detector double counting detect() calls; delegates to a scripted mask."""

    def __init__(self, inner: DetectorBackend) -> None:
        self.inner = inner
        self.calls = 0

    def detect(self, frame: ProcessedFrameMsg) -> Result[InferenceResultMsg, FaultCode]:
        self.calls += 1
        return self.inner.detect(frame)


def _mosaic(frame_id: int, exposure_us: float = 1000.0) -> MosaicFrame:
    """A zeroed default-geometry mosaic with the given exposure."""
    return MosaicFrame(
        timestamp_utc="2026-06-01T00:00:00.000Z",
        timestamp_s=float(frame_id),
        frame_id=frame_id,
        mosaic=np.zeros((3, 1544, 2064), dtype=np.uint16),
        exposure_us=exposure_us,
        gain_db=0.0,
    )


def _plume_detector() -> ScriptedDetector:
    """Scripted detector whose mask yields one strong off-boresight blob."""
    mask = np.zeros((1544, 2064), dtype=np.float32)
    mask[149:225, 990:1074] = 1.0
    return ScriptedDetector(mask, confidence_gate=0.55, min_blob_area_px=15)


def _build(
    cfg: PactConfig | None = None,
    sensor: _SpySensor | None = None,
    detector: DetectorBackend | None = None,
) -> tuple[PayloadApp, MessageBus, SimGimbal, ManualClock, _SpySensor, _MemStorage]:
    """Assemble a PayloadApp over a spy sensor and sim drivers."""
    base = cfg or PactConfig()
    if cfg is None:
        base = replace(
            base,
            inference=replace(base.inference, tile_rows=1, tile_cols=1),
            gimbal=replace(
                base.gimbal,
                simulation=replace(base.gimbal.simulation, encoder_noise_deg=0.0),
            ),
        )
    bus = MessageBus()
    clock = ManualClock()
    gimbal = SimGimbal(clock=clock, cfg=base.gimbal, inner_dt_s=base.controller.inner.dt_s)
    spy = sensor or _SpySensor()
    eph = SimIssEphemeris(clock=clock, cfg=base.ephemeris)
    calib = build_identity_calibration(base.sensor.height_px, base.sensor.width_px)
    storage = _MemStorage()
    app = PayloadApp.from_config(
        base,
        spy,
        gimbal,
        eph,
        detector or _plume_detector(),
        bus,
        clock,
        calib,
        storage,
        _EPOCH,
    )
    return app, bus, gimbal, clock, spy, storage


def _operate(app: PayloadApp, bus: MessageBus, gimbal: SimGimbal) -> PayloadState:
    """Boot, one encoder sample, OPERATE activation drained and installed."""
    state = app.initial_state()
    pos = gimbal.read_position()
    assert isinstance(pos, Ok)
    app.note_gimbal_feedback(pos.value)
    bus.publish(
        SystemModeActivatedMsg(
            msg_type=MessageType.SYSTEM_MODE_ACTIVATED,
            timestamp_utc="t",
            epoch=_EPOCH,
            sequence=1,
            previous_mode=None,
            active_mode=SystemMode.OPERATE,
            reason="test",
            request_id=None,
            recovery_authorized=False,
        )
    )
    return app.poll_activations(state, now=0.0)


def test_capture_once_unactivated_does_no_imaging_io() -> None:
    """Without an activation the capture loop touches no camera HAL at all."""
    app, bus, gimbal, _clock, spy, _storage = _build()
    state = app.initial_state()
    _state, outcome = app.capture_once(state, now=0.0)
    assert outcome.fault is None
    assert spy.calls == []


def test_capture_once_off_policy_stops_stream_and_skips_drain() -> None:
    """Contained policy forces the stream off with no acquire/drain/detect."""
    app, bus, gimbal, _clock, spy, _storage = _build()
    detector = _CountingDetector(_plume_detector())
    app = replace(app, detector=detector)
    telem = bus.subscribe(TelemetryEventMsg)
    state = _operate(app, bus, gimbal)
    app.capture_once(state, now=0.0)
    assert spy.calls[-1] == "drain_frame"  # first opportunity drains at duty 0.5
    _drain(telem)
    app.containment.local_latched = True
    _state, outcome = app.capture_once(state, now=0.0)
    assert outcome.fault is None
    assert spy.calls[-1] == "stop_acquisition"
    assert detector.calls == 0
    # A forced stop of a still-enabled policy records no applied-policy event.
    assert not [e for e in _drain(telem) if e.event_name == "imaging_policy"]


def test_capture_once_applies_settings_then_starts_once() -> None:
    """Startup order is exposure, gain, start; an unchanged policy repeats nothing."""
    app, bus, gimbal, _clock, spy, _storage = _build()
    state = _operate(app, bus, gimbal)
    app.capture_once(state, now=0.0)
    assert spy.calls == [
        "set_exposure_us",
        "set_gain_db",
        "start_acquisition",
        "drain_frame",
    ]
    # Early call: WAIT, no HAL at all.
    interval = state.policy.imaging.capture_interval_s
    _state, outcome = app.capture_once(state, now=0.0)
    assert outcome.fault is None
    assert len(spy.calls) == 4
    app.capture_once(state, now=interval)
    assert spy.calls[-1] == "acquire_frame"
    assert spy.calls.count("start_acquisition") == 1


def test_capture_once_changed_settings_stop_first_then_apply() -> None:
    """A new exposure/gain stops the stream, then sets exposure, gain, start."""
    app, bus, gimbal, _clock, spy, _storage = _build()
    state = _operate(app, bus, gimbal)
    app.capture_once(state, now=0.0)
    spy.calls.clear()
    policy = state.policy
    changed = replace(
        policy,
        imaging=replace(policy.imaging, exposure_us=500.0, gain_db=3.0),
    )
    state = replace(state, policy=changed, policy_revision=state.policy_revision + 1)
    app._install(state)
    _state, outcome = app.capture_once(state, now=0.0)
    assert outcome.fault is None
    assert spy.calls == [
        "stop_acquisition",
        "set_exposure_us",
        "set_gain_db",
        "start_acquisition",
        "drain_frame",
    ]
    assert app.capture_shell.applied_capture == (500.0, 3.0)


def test_capture_once_policy_only_change_resets_without_camera_cycle() -> None:
    """A cadence-only revision resets the schedule but does not cycle the camera."""
    app, bus, gimbal, _clock, spy, _storage = _build()
    state = _operate(app, bus, gimbal)
    app.capture_once(state, now=0.0)
    spy.calls.clear()
    policy = state.policy
    changed = replace(
        policy,
        imaging=replace(policy.imaging, capture_interval_s=5.0),
    )
    state = replace(state, policy=changed, policy_revision=state.policy_revision + 1)
    app._install(state)
    app.capture_once(state, now=0.0)
    assert spy.calls == ["drain_frame"]  # fresh schedule, first opportunity due
    assert spy.calls.count("start_acquisition") == 0
    # The reset schedule waits until the new deadline.
    app.capture_once(state, now=1.0)
    assert app.capture_shell.schedule.next_opportunity_s == pytest.approx(5.0)


@pytest.mark.parametrize(
    "stage",
    ["set_exposure_us", "set_gain_db", "start_acquisition", "drain_frame", "acquire_frame"],
)
def test_capture_hal_failures_queue_control_owned_containment(stage: str) -> None:
    """Each HAL stage failure publishes the fault; the next control poll latches.

    The capture loop itself never calls the gimbal: pending_fault is consumed
    by _poll_local_faults, which engages containment on the control owner.
    """
    app, bus, gimbal, _clock, spy, _storage = _build()
    fault_sub = bus.subscribe(FaultEventMsg)
    spy.errors[stage] = FaultCode.CAMERA_STALL
    state = _operate(app, bus, gimbal)
    app.capture_once(state, now=0.0)  # opportunity 1 drains
    if stage != "drain_frame":
        app.capture_once(state, now=1.0)  # opportunity 2 captures -> stage hits
        if stage == "acquire_frame":
            assert spy.calls[-1] == "acquire_frame"
    assert app.containment.local_latched is False
    faults = [f.fault_code for f in _drain(fault_sub)]
    assert FaultCode.CAMERA_STALL in faults
    unsafe = app._poll_local_faults()
    assert unsafe is True
    assert app.containment.local_latched is True
    health = gimbal.read_health()
    assert isinstance(health, Ok)
    assert health.value.inhibit_confirmed is True


def test_failed_stop_preserves_applied_state_and_continues() -> None:
    """A failed stop keeps acquisition_on/applied_capture and emits the fault."""
    app, bus, gimbal, _clock, spy, _storage = _build()
    fault_sub = bus.subscribe(FaultEventMsg)
    state = _operate(app, bus, gimbal)
    app.capture_once(state, now=0.0)
    assert app.capture_shell.acquisition_on is True
    applied = app.capture_shell.applied_capture
    spy.errors["stop_acquisition"] = FaultCode.CAMERA_STALL
    # Off policy forces a stop attempt.
    off = replace(state, policy=app.params.default_policy(False))
    app._install(off)
    _state, outcome = app.capture_once(off, now=0.0)
    assert outcome.fault is FaultCode.CAMERA_STALL
    assert app.capture_shell.acquisition_on is True
    assert app.capture_shell.applied_capture == applied
    assert FaultCode.CAMERA_STALL in [f.fault_code for f in _drain(fault_sub)]
    app._poll_local_faults()
    assert app.containment.local_latched is True


def test_from_config_rejects_invalid_policy_before_hal() -> None:
    """An invalid resolved operate/node policy is a startup failure, no HAL."""
    base = PactConfig()
    cfg = replace(
        base,
        payload_policy=PayloadPolicyConfig(
            operate=PayloadPolicyOverrideConfig(acquisition_enabled=False, inference_enabled=True),
        ),
    )
    spy = _SpySensor()
    bus = MessageBus()
    clock = ManualClock()
    gimbal = SimGimbal(clock=clock, cfg=cfg.gimbal, inner_dt_s=cfg.controller.inner.dt_s)
    eph = SimIssEphemeris(clock=clock, cfg=cfg.ephemeris)
    calib = build_identity_calibration(cfg.sensor.height_px, cfg.sensor.width_px)
    with pytest.raises(ValueError, match="invalid payload policy"):
        PayloadApp.from_config(
            cfg, spy, gimbal, eph, _plume_detector(), bus, clock, calib, _MemStorage(), _EPOCH
        )
    assert spy.calls == []


def test_inference_decimation_skips_vision_and_products() -> None:
    """every_n_frames=2 publishes inference only on captures 1 and 3; a skipped
    frame produces no vision, no product, and no miss bookkeeping change."""
    base = PactConfig()
    cfg = replace(
        base,
        payload_policy=PayloadPolicyConfig(
            operate=PayloadPolicyOverrideConfig(every_n_frames=2),
        ),
        inference=replace(base.inference, tile_rows=1, tile_cols=1),
        gimbal=replace(
            base.gimbal,
            simulation=replace(base.gimbal.simulation, encoder_noise_deg=0.0),
        ),
    )
    spy = _SpySensor(frames=[_mosaic(i) for i in range(1, 8)])
    app, bus, gimbal, _clock, spy, storage = _build(cfg=cfg, sensor=spy)
    detector = _CountingDetector(_plume_detector())
    app = replace(app, detector=detector)
    inf_sub = bus.subscribe(InferenceResultMsg)
    prod_sub = bus.subscribe(ProductRefMsg)
    state = _operate(app, bus, gimbal)
    # Duty 0.5 captures opportunities 2,4,6 -> successful captures 1,2,3.
    for index in range(1, 7):
        _state, _out = app.capture_once(state, now=index * 0.03)
    assert detector.calls == 2  # captures 1 and 3 ran inference
    inference = _drain(inf_sub)
    assert [m.frame_id for m in inference] == [2, 6]
    assert len(app.vision_queue) == 2
    assert len(_drain(prod_sub)) == 2
    # A skipped capture never became empty vision: graph miss_count stays 0.
    graph = state.graph
    assert isinstance(graph, operate.State)
    assert graph.miss_count == 0


def test_publish_products_false_skips_storage_only() -> None:
    """publish_products=False keeps inference but stores and references nothing."""
    base = PactConfig()
    cfg = replace(
        base,
        payload_policy=PayloadPolicyConfig(
            operate=PayloadPolicyOverrideConfig(publish_products=False),
        ),
        inference=replace(base.inference, tile_rows=1, tile_cols=1),
        gimbal=replace(
            base.gimbal,
            simulation=replace(base.gimbal.simulation, encoder_noise_deg=0.0),
        ),
    )
    spy = _SpySensor(frames=[_mosaic(1), _mosaic(2)])
    app, bus, gimbal, _clock, spy, storage = _build(cfg=cfg, sensor=spy)
    inf_sub = bus.subscribe(InferenceResultMsg)
    prod_sub = bus.subscribe(ProductRefMsg)
    state = _operate(app, bus, gimbal)
    app.capture_once(state, now=0.0)
    _state, outcome = app.capture_once(state, now=0.03)
    assert outcome.fault is None
    assert _drain(inf_sub)
    assert prod_sub.empty()
    assert storage.items == {}


def test_new_activation_resets_capture_phase() -> None:
    """A same-mode newer activation resets counters: first capture re-runs inference."""
    base = PactConfig()
    cfg = replace(
        base,
        payload_policy=PayloadPolicyConfig(
            operate=PayloadPolicyOverrideConfig(every_n_frames=3),
        ),
        inference=replace(base.inference, tile_rows=1, tile_cols=1),
        gimbal=replace(
            base.gimbal,
            simulation=replace(base.gimbal.simulation, encoder_noise_deg=0.0),
        ),
    )
    spy = _SpySensor(frames=[_mosaic(i) for i in range(1, 9)])
    app, bus, gimbal, _clock, spy, _storage = _build(cfg=cfg, sensor=spy)
    detector = _CountingDetector(_plume_detector())
    app = replace(app, detector=detector)
    state = _operate(app, bus, gimbal)
    # Captures at opportunities 2 and 4 -> successful captures 1 and 2.
    for index in range(1, 5):
        state, _ = app.capture_once(state, now=index * 0.03)
    assert detector.calls == 1  # only the first successful capture inferred
    bus.publish(
        SystemModeActivatedMsg(
            msg_type=MessageType.SYSTEM_MODE_ACTIVATED,
            timestamp_utc="t",
            epoch=_EPOCH,
            sequence=2,
            previous_mode=SystemMode.OPERATE,
            active_mode=SystemMode.OPERATE,
            reason="re",
            request_id=None,
            recovery_authorized=False,
        )
    )
    state = app.poll_activations(state, now=0.2)
    for index in range(1, 5):
        state, _ = app.capture_once(state, now=0.2 + index * 0.03)
    assert detector.calls == 2  # reset: first capture under the new context infers
    assert app.capture_shell.schedule.captured_frames == 2


def _reactivate(bus: MessageBus, app: PayloadApp, state: PayloadState) -> None:
    """Land a same-mode newer activation inside a HAL callback."""
    bus.publish(
        SystemModeActivatedMsg(
            msg_type=MessageType.SYSTEM_MODE_ACTIVATED,
            timestamp_utc="t",
            epoch=_EPOCH,
            sequence=2,
            previous_mode=SystemMode.OPERATE,
            active_mode=SystemMode.OPERATE,
            reason="re",
            request_id=None,
            recovery_authorized=False,
        )
    )
    app.poll_activations(state, now=0.0)


def test_superseded_start_stops_stream_instead_of_capturing() -> None:
    """An activation landing during a blocked start stops, then captures nothing."""
    app, bus, gimbal, _clock, spy, _storage = _build()
    telem = bus.subscribe(TelemetryEventMsg)
    state = _operate(app, bus, gimbal)

    spy.on_start = lambda: _reactivate(bus, app, state)
    _state, outcome = app.capture_once(state, now=0.0)
    assert outcome.fault is None
    assert "acquire_frame" not in spy.calls
    assert spy.calls[-1] == "stop_acquisition"
    assert app.capture_shell.acquisition_on is False
    assert not app.vision_queue
    # The superseded application never emits the old policy record.
    assert not [e for e in _drain(telem) if e.event_name == "imaging_policy"]


@pytest.mark.parametrize(
    ("hook", "expected"),
    [
        ("on_exposure", ["set_exposure_us"]),
        ("on_gain", ["set_exposure_us", "set_gain_db"]),
    ],
)
def test_superseded_settings_never_start_or_acquire(hook: str, expected: list[str]) -> None:
    """An activation inside a setter aborts the apply before any later stage."""
    spy = _SpySensor(frames=[_mosaic(1)])
    app, bus, gimbal, _clock, spy, _storage = _build(sensor=spy)
    telem = bus.subscribe(TelemetryEventMsg)
    state = _operate(app, bus, gimbal)
    setattr(spy, hook, lambda: _reactivate(bus, app, state))
    _state, outcome = app.capture_once(state, now=0.0)
    assert outcome.fault is None
    assert spy.calls == expected
    assert "start_acquisition" not in spy.calls
    assert "acquire_frame" not in spy.calls
    assert app.capture_shell.acquisition_on is False
    assert not [e for e in _drain(telem) if e.event_name == "imaging_policy"]


@pytest.mark.parametrize("stage", ["drain_frame", "acquire_frame"])
def test_stale_frame_scoped_err_publishes_and_latches_nothing(stage: str) -> None:
    """An activation landing during a failed acquire/drain drops the old Err.

    A stale frame-scoped failure returns a fault-free outcome and neither
    publishes nor queues anything, so a dead activation's camera error can
    never contain the newer one.
    """
    spy = _SpySensor(frames=[_mosaic(1), _mosaic(2), _mosaic(3)])
    app, bus, gimbal, _clock, spy, _storage = _build(sensor=spy)
    fault_sub = bus.subscribe(FaultEventMsg)
    inf_sub = bus.subscribe(InferenceResultMsg)
    prod_sub = bus.subscribe(ProductRefMsg)
    state = _operate(app, bus, gimbal)
    hook = "on_acquire" if stage == "acquire_frame" else "on_drain"
    setattr(spy, hook, lambda: _reactivate(bus, app, state))
    spy.errors[stage] = FaultCode.CAMERA_STALL
    if stage == "acquire_frame":
        app.capture_once(state, now=0.0)  # spend opportunity 1 (drain)
        _state, outcome = app.capture_once(state, now=0.03)
    else:
        _state, outcome = app.capture_once(state, now=0.0)
    assert outcome.fault is None
    assert _drain(fault_sub) == []
    assert app.capture_shell.pending_fault is None
    assert app.containment.local_latched is False
    assert not app.vision_queue
    assert inf_sub.empty()
    assert prod_sub.empty()


def _activate(
    bus: MessageBus, mode: SystemMode, sequence: int, previous: SystemMode | None
) -> None:
    """Publish one accepted-epoch activation for the given graph mode."""
    bus.publish(
        SystemModeActivatedMsg(
            msg_type=MessageType.SYSTEM_MODE_ACTIVATED,
            timestamp_utc="t",
            epoch=_EPOCH,
            sequence=sequence,
            previous_mode=previous,
            active_mode=mode,
            reason="test",
            request_id=None,
            recovery_authorized=False,
        )
    )


def test_applied_policy_telemetry_emitted_once_per_revision() -> None:
    """The imaging_policy record publishes on application, then stays quiet."""
    app, bus, gimbal, _clock, spy, _storage = _build()
    telem = bus.subscribe(TelemetryEventMsg)
    state = _operate(app, bus, gimbal)
    app.capture_once(state, now=0.0)
    events = [e for e in _drain(telem) if e.event_name == "imaging_policy"]
    assert len(events) == 1
    payload = events[0].payload
    assert payload["policy_revision"] == state.policy_revision
    assert payload["activation_epoch"] == _EPOCH
    assert payload["acquisition_enabled"] is True
    assert payload["inference_enabled"] is True
    app.capture_once(state, now=0.03)
    assert not [e for e in _drain(telem) if e.event_name == "imaging_policy"]


def test_cadence_only_revision_emits_applied_policy_once() -> None:
    """A cadence/decimation revision emits once with exact values, no cycle."""
    app, bus, gimbal, _clock, spy, _storage = _build()
    telem = bus.subscribe(TelemetryEventMsg)
    state = _operate(app, bus, gimbal)
    app.capture_once(state, now=0.0)
    _drain(telem)
    calls_before = list(spy.calls)
    policy = state.policy
    changed = replace(
        policy,
        imaging=replace(policy.imaging, capture_interval_s=0.5),
        inference=replace(policy.inference, every_n_frames=4),
    )
    state = replace(state, policy=changed, policy_revision=state.policy_revision + 1)
    app._install(state)
    app.capture_once(state, now=0.04)
    events = [e for e in _drain(telem) if e.event_name == "imaging_policy"]
    assert len(events) == 1
    payload = events[0].payload
    assert payload["policy_revision"] == state.policy_revision
    assert payload["capture_interval_s"] == 0.5
    assert payload["every_n_frames"] == 4
    # Settings unchanged: no stop/setter/start cycle ran.
    assert "stop_acquisition" not in spy.calls[len(calls_before) :]
    app.capture_once(state, now=0.5)
    assert not [e for e in _drain(telem) if e.event_name == "imaging_policy"]


def test_same_mode_activation_emits_applied_policy_once() -> None:
    """A same-mode newer activation emits the record without a camera cycle."""
    app, bus, gimbal, _clock, spy, _storage = _build()
    telem = bus.subscribe(TelemetryEventMsg)
    state = _operate(app, bus, gimbal)
    app.capture_once(state, now=0.0)
    _drain(telem)
    _activate(bus, SystemMode.OPERATE, 2, SystemMode.OPERATE)
    state = app.poll_activations(state, now=0.1)
    calls_before = list(spy.calls)
    app.capture_once(state, now=0.2)
    events = [e for e in _drain(telem) if e.event_name == "imaging_policy"]
    assert len(events) == 1
    assert events[0].payload["activation_sequence"] == 2
    assert spy.calls[len(calls_before) :].count("start_acquisition") == 0
    app.capture_once(state, now=0.5)
    assert not [e for e in _drain(telem) if e.event_name == "imaging_policy"]


def test_disabled_graph_emits_off_policy_after_confirmed_stop() -> None:
    """An IDLE activation stops the stream, then records the off policy once."""
    app, bus, gimbal, _clock, spy, _storage = _build()
    telem = bus.subscribe(TelemetryEventMsg)
    state = _operate(app, bus, gimbal)
    app.capture_once(state, now=0.0)
    _drain(telem)
    _activate(bus, SystemMode.IDLE, 2, SystemMode.OPERATE)
    state = app.poll_activations(state, now=0.1)
    _state, outcome = app.capture_once(state, now=0.2)
    assert outcome.fault is None
    assert spy.calls[-1] == "stop_acquisition"
    events = [e for e in _drain(telem) if e.event_name == "imaging_policy"]
    assert len(events) == 1
    assert events[0].payload["acquisition_enabled"] is False
    assert events[0].payload["policy_revision"] == state.policy_revision
    app.capture_once(state, now=0.3)
    assert not [e for e in _drain(telem) if e.event_name == "imaging_policy"]


def test_failed_stop_emits_no_applied_policy_event() -> None:
    """A failed stop is a fault, not an applied off policy."""
    spy = _SpySensor()
    app, bus, gimbal, _clock, spy, _storage = _build(sensor=spy)
    telem = bus.subscribe(TelemetryEventMsg)
    state = _operate(app, bus, gimbal)
    app.capture_once(state, now=0.0)
    _drain(telem)
    _activate(bus, SystemMode.IDLE, 2, SystemMode.OPERATE)
    state = app.poll_activations(state, now=0.1)
    spy.errors["stop_acquisition"] = FaultCode.CAMERA_STALL
    _state, outcome = app.capture_once(state, now=0.2)
    assert outcome.fault is FaultCode.CAMERA_STALL
    assert app.capture_shell.acquisition_on is True
    assert not [e for e in _drain(telem) if e.event_name == "imaging_policy"]


def test_capture_once_nan_now_touches_no_hal() -> None:
    """A nonfinite capture time faults before any setter, start, or acquire."""
    app, bus, gimbal, _clock, spy, _storage = _build()
    state = _operate(app, bus, gimbal)
    _state, outcome = app.capture_once(state, now=math.nan)
    assert outcome.fault is FaultCode.COMMAND_INVALID
    assert spy.calls == []
    assert app.capture_shell.acquisition_on is False
    assert app.capture_shell.schedule.opportunities == 0


def test_plan_capture_off_policy_waits_without_spending() -> None:
    """A validated disabled policy plans WAIT; no opportunity or deadline."""
    policy = _PARAMS.default_policy(enabled=False)
    plan = plan_capture(CaptureSchedule(), policy, _context(), 0.0, _LIMITS)
    assert isinstance(plan, Ok)
    assert plan.value.decision is CaptureDecision.WAIT
    assert plan.value.schedule.opportunities == 0
    assert plan.value.schedule.next_opportunity_s is None


def test_capture_only_policy_acquires_and_counts_without_inference() -> None:
    """duty=1 with inference off acquires and counts frames, no detector run."""
    base = PactConfig()
    cfg = replace(
        base,
        payload_policy=PayloadPolicyConfig(
            operate=PayloadPolicyOverrideConfig(duty_cycle=1.0, inference_enabled=False),
        ),
        inference=replace(base.inference, tile_rows=1, tile_cols=1),
        gimbal=replace(
            base.gimbal,
            simulation=replace(base.gimbal.simulation, encoder_noise_deg=0.0),
        ),
    )
    spy = _SpySensor(frames=[_mosaic(1), _mosaic(2), _mosaic(3)])
    app, bus, gimbal, _clock, spy, _storage = _build(cfg=cfg, sensor=spy)
    detector = _CountingDetector(_plume_detector())
    app = replace(app, detector=detector)
    inf_sub = bus.subscribe(InferenceResultMsg)
    prod_sub = bus.subscribe(ProductRefMsg)
    state = _operate(app, bus, gimbal)
    for index in range(1, 4):
        _state, outcome = app.capture_once(state, now=index * 0.03)
        assert outcome.fault is None
    assert spy.calls.count("acquire_frame") == 3
    assert app.capture_shell.schedule.captured_frames == 3
    assert detector.calls == 0
    assert not app.vision_queue
    assert inf_sub.empty()
    assert prod_sub.empty()
    assert _storage.items == {}


def test_failed_acquire_never_counts_a_capture() -> None:
    """A camera stall queues the fault but never advances captured_frames."""
    spy = _SpySensor(frames=[_mosaic(1)])
    app, bus, gimbal, _clock, spy, _storage = _build(sensor=spy)
    spy.errors["acquire_frame"] = FaultCode.CAMERA_STALL
    state = _operate(app, bus, gimbal)
    app.capture_once(state, now=0.0)  # opportunity 1 drains
    _state, outcome = app.capture_once(state, now=0.03)
    assert outcome.fault is FaultCode.CAMERA_STALL
    assert app.capture_shell.schedule.captured_frames == 0
    assert app.capture_shell.pending_fault is FaultCode.CAMERA_STALL


def test_first_capture_applies_tracking_entry_policy() -> None:
    """A tracking-only override reaches the HAL before any outer tick runs."""
    base = PactConfig()
    cfg = replace(
        base,
        payload_policy=PayloadPolicyConfig(
            tracking=PayloadPolicyOverrideConfig(exposure_us=250.0, gain_db=4.0, duty_cycle=1.0),
        ),
        inference=replace(base.inference, tile_rows=1, tile_cols=1),
        gimbal=replace(
            base.gimbal,
            simulation=replace(base.gimbal.simulation, encoder_noise_deg=0.0),
        ),
    )
    spy = _SpySensor(frames=[_mosaic(1)])
    app, bus, gimbal, _clock, spy, _storage = _build(cfg=cfg, sensor=spy)
    state = _operate(app, bus, gimbal)
    assert state.policy.imaging.exposure_us == 250.0
    assert state.policy.imaging.gain_db == 4.0
    assert state.policy.imaging.duty_cycle == 1.0
    _state, outcome = app.capture_once(state, now=0.0)
    assert outcome.fault is None
    assert spy.values["exposure_us"] == 250.0
    assert spy.values["gain_db"] == 4.0
    assert spy.calls[-1] == "acquire_frame"


def _routed(command_id: str, seq: int) -> RoutedCommandMsg:
    """Build one routed payload command envelope."""
    return RoutedCommandMsg(
        msg_type=MessageType.ROUTED_COMMAND,
        timestamp_utc="t",
        target="payload",
        command_id=command_id,
        params={},
        source="ground",
        seq=seq,
    )


def test_hold_command_transition_updates_policy_and_restarts_schedule() -> None:
    """A real HOLD command edge applies the node override and resets phase."""
    base = PactConfig()
    cfg = replace(
        base,
        payload_policy=PayloadPolicyConfig(
            hold=PayloadPolicyOverrideConfig(duty_cycle=1.0),
        ),
        inference=replace(base.inference, tile_rows=1, tile_cols=1),
        gimbal=replace(
            base.gimbal,
            simulation=replace(base.gimbal.simulation, encoder_noise_deg=0.0),
        ),
    )
    spy = _SpySensor(frames=[_mosaic(1), _mosaic(2)])
    app, bus, gimbal, _clock, spy, _storage = _build(cfg=cfg, sensor=spy)
    state = _operate(app, bus, gimbal)
    app.capture_once(state, now=0.0)
    assert app.capture_shell.schedule.opportunities == 1
    revision_before = state.policy_revision
    pos = gimbal.read_position()
    assert isinstance(pos, Ok)
    app.note_gimbal_feedback(replace(pos.value, timestamp_s=1.0))
    bus.publish(_routed("GIMBAL_HOLD", 1))
    state = app.handle_commands(state, now=1.0)
    assert node_name_of(state) == "hold"
    assert state.policy.imaging.duty_cycle == 1.0
    assert state.policy_revision == revision_before + 1
    _state, outcome = app.capture_once(state, now=1.03)
    assert outcome.fault is None
    # Fresh context: the first due opportunity under duty 1 captures.
    assert app.capture_shell.schedule.opportunities == 1
    assert spy.calls[-1] == "acquire_frame"


def test_frame_exposure_flows_into_vision_sample() -> None:
    """The live frame exposure reaches the vision sample for quality/GSD use."""
    spy = _SpySensor(frames=[_mosaic(1), _mosaic(2, exposure_us=4321.0)])
    app, bus, gimbal, _clock, spy, _storage = _build(sensor=spy)
    state = _operate(app, bus, gimbal)
    app.capture_once(state, now=0.0)
    _state, outcome = app.capture_once(state, now=0.03)
    assert outcome.fault is None
    assert app.vision_queue
    assert app.vision_queue[-1].sample.exposure_us == 4321.0


def _drain[MsgT](sub: Subscription[MsgT]) -> list[MsgT]:
    """Drain a bus subscription into a list."""
    out: list[MsgT] = []
    while not sub.empty():
        out.append(sub.get_nowait())
    return out
