"""End-to-end mode-graph acceptance over the real system-mode authority.

Every activation in this module is produced by the actual SystemModesApp over the
shared bus: boot SAFE, signed TC packets through the ISS uplink and command
router, fault-app SAFE requests, and the payload's keyed INIT completion. The
publish_activation fixture is never used for the nominal path. Commands are
built with build_tc_packet and queued onto SimStationLink, so each runs the
decode -> authenticate -> ingress -> route -> decide -> ack chain for real.

Vision-bearing cases drive the live ecef_column environment through
SilEnvironmentBind with an injected mosaic so shutter stamps track the shared
clock (scripted frame timestamps are fixed indices and go stale once the
system spends cycles outside OPERATE).
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from dataclasses import replace

import numpy as np
import pytest
from flight.libs.bus import Subscription
from flight.libs.commands import build_tc_packet
from flight.libs.config import PactConfig
from flight.libs.messages import (
    CommandAckMsg,
    FaultEventMsg,
    InferenceResultMsg,
    SafetyStateMsg,
    SystemModeActivatedMsg,
    SystemModeRequestMsg,
    SystemModeSyncRequestMsg,
    SystemModeTransitionMsg,
)
from flight.libs.time import ManualClock
from flight.libs.types import (
    AckStatus,
    ActivationKey,
    Err,
    FaultCode,
    MessageType,
    ModeTransitionDecision,
    Ok,
    Result,
    SystemMode,
)
from flight.payload.app import PayloadApp
from flight.payload.graphs.base import InitVerificationResult, InitVerificationStatus
from flight.payload.inference import (
    InferenceRuntime,
    RuntimeSession,
    ScriptedDetector,
    ScriptedRuntimeFactory,
)
from flight.payload.lifecycle import InitializationVerifier, LifecycleObservation
from sim.environment import (
    Environment,
    EnvironmentConfig,
    build_environment,
    camera_from_sensor,
)
from sim.environment.records import DriverFeed, EnvSample, EnvTime, PlumeState, ShutterPose
from sim.scene import build_frames, plume_detector
from sim.sil import (
    SilEnvironmentBind,
    SilHarness,
    SilSystem,
    bind_sil_environment,
    build_sil_system,
    publish_activation,
)

_KEY = b"sil-test-key-0000000000000000000"
_GROUND = "ground"


def _drain[T](subscription: Subscription[T]) -> list[T]:
    """Drain all pending messages from a subscription into a list."""
    result: list[T] = []
    while not subscription.empty():
        result.append(subscription.get_nowait())
    return result


def _config() -> PactConfig:
    """Home-at-origin with noise-free encoders so INIT HOME lands exactly."""
    base = PactConfig()
    return replace(
        base,
        gimbal=replace(
            base.gimbal,
            simulation=replace(base.gimbal.simulation, encoder_noise_deg=0.0),
            home_el_deg=0.0,
        ),
    )


class _CurrentKeyVerifier:
    """Deterministic INIT verifier: VERIFIED on the observed key, with evidence."""

    def __init__(self) -> None:
        self.calls = 0

    def verify(
        self, observation: LifecycleObservation, cancel: threading.Event
    ) -> Result[InitVerificationResult, FaultCode]:
        """Answer VERIFIED on the exact observed activation key."""
        del cancel
        self.calls += 1
        return Ok(
            InitVerificationResult(
                activation_key=observation.inputs.activation_key,
                status=InitVerificationStatus.VERIFIED,
                evidence_id="acceptance-verifier",
            )
        )


class _StaleKeyVerifier:
    """Verifier that lies about the key: VERIFIED but never for the current one."""

    def __init__(self) -> None:
        self.calls = 0
        self._key = ActivationKey(epoch="foreign-epoch", sequence=10_000)

    def verify(
        self, observation: LifecycleObservation, cancel: threading.Event
    ) -> Result[InitVerificationResult, FaultCode]:
        """Answer VERIFIED stamped with a key no live activation can hold."""
        del cancel, observation
        self.calls += 1
        return Ok(
            InitVerificationResult(
                activation_key=self._key,
                status=InitVerificationStatus.VERIFIED,
                evidence_id="stale-verifier",
            )
        )


class _GatedVerifier:
    """Verifier that blocks inside verify() until released, then VERIFIED.

    Holds INIT at READY with verification undecided so the test can inspect the
    runtime before promotion; release answers VERIFIED on the observed key.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.entered = threading.Event()
        self.released = threading.Event()

    def verify(
        self, observation: LifecycleObservation, cancel: threading.Event
    ) -> Result[InitVerificationResult, FaultCode]:
        """Wait for release in bounded slices, then attest VERIFIED."""
        self.calls += 1
        self.entered.set()
        while not self.released.is_set():
            if cancel.is_set():
                return Err(FaultCode.INFERENCE_TIMEOUT)
            self.released.wait(timeout=0.01)
        return Ok(
            InitVerificationResult(
                activation_key=observation.inputs.activation_key,
                status=InitVerificationStatus.VERIFIED,
                evidence_id="acceptance-verifier",
            )
        )


class _BlockedFactory:
    """RuntimeFactory that parks inside load() until released, ignoring cancel.

    Mirrors a model load stuck inside the SDK: the lifecycle worker blocks in
    the call while the control loop, fault path, and authority keep running.
    """

    def __init__(self) -> None:
        self.started = threading.Event()
        self.released = threading.Event()

    def load(self, cancel: threading.Event) -> Result[RuntimeSession, FaultCode]:
        """Block until the test releases the gate; cancel is ignored on purpose."""
        del cancel
        self.started.set()
        self.released.wait(timeout=30.0)
        return Err(FaultCode.INFERENCE_TIMEOUT)


_SYSTEMS: list[SilSystem] = []


@pytest.fixture(autouse=True)
def _shutdown_lifecycle_workers() -> Iterator[None]:
    """Stop lifecycle daemon workers for every system built in this module."""
    yield
    for system in _SYSTEMS:
        system.apps.payload.lifecycle.shutdown(join_timeout_s=0.5)
    _SYSTEMS.clear()


def _build(
    config: PactConfig,
    *,
    verifier: InitializationVerifier | None = None,
    inbound: list[bytes] | None = None,
    detector: ScriptedDetector | None = None,
    inference_runtime: InferenceRuntime | None = None,
) -> SilSystem:
    """Build the wired SIL system: sim drivers, real apps, empty frame queue.

    When inference_runtime is given, the payload app is rebuilt through the
    same from_config composition seam over the system's real drivers, bus,
    clock, calibration, storage, and epoch -- the runtime starts empty so the
    INIT installation gate is exercised for real.
    """
    scripted = detector if detector is not None else plume_detector()
    system = build_sil_system(
        config,
        ManualClock(),
        [],
        scripted,
        inbound_packets=inbound or [],
        thermal_readings=[25.0],
        power_readings=[30.0],
        uplink_key=_KEY,
        initialization_verifier=verifier,
    )
    if inference_runtime is not None:
        payload = system.apps.payload
        replacement = PayloadApp.from_config(
            payload.params.config,
            system.sensor,
            system.gimbal,
            payload.ephemeris,
            inference_runtime,
            system.bus,
            system.clock,
            payload.calib,
            payload.storage,
            payload.activation_epoch,
            verifier=verifier,
        )
        system = replace(system, apps=replace(system.apps, payload=replacement))
    _SYSTEMS.append(system)
    return system


def _bind(config: PactConfig, system: SilSystem, detector: ScriptedDetector) -> SilEnvironmentBind:
    """Bind the live environment so shutter stamps ride the shared clock."""
    built = build_environment(
        EnvironmentConfig(plume="ecef_column"), camera_from_sensor(config.sensor)
    )
    assert isinstance(built, Ok)
    inner = built.value
    mosaic = np.asarray(build_frames(1, seed=0)[0].mosaic)

    class _LiveEnvironment(Environment):
        """Delegate evaluate to the real environment, adding a live mosaic."""

        def __init__(self) -> None:
            """Hold only the delegate; the parent fields are never read."""

        def evaluate(
            self,
            time: EnvTime,
            shutter: ShutterPose,
            rng: np.random.Generator,
            prior_plume: PlumeState | None = None,
        ) -> EnvSample:
            sample = inner.evaluate(time, shutter, rng, prior_plume)
            return EnvSample(
                truth=sample.truth,
                feed=DriverFeed(mosaic=mosaic, mask=sample.feed.mask),
            )

    return bind_sil_environment(
        _LiveEnvironment(),
        system.sensor,
        system.gimbal,
        system.clock,
        config.sensor,
        frames=[],
        detector=detector,
        ephemeris=system.apps.payload.ephemeris,
    )


class _Runner:
    """Small bounded-step driver: advances `now` one second per step."""

    def __init__(self, harness: SilHarness) -> None:
        self.harness = harness
        self.now = 0.0

    def step(self, count: int = 1) -> None:
        """Advance `count` deterministic steps."""
        for _ in range(count):
            self.now += 1.0
            self.harness.step(self.now)

    def until(self, predicate: Callable[[], bool], max_steps: int = 40) -> bool:
        """Step until `predicate()` is true; return whether it fired."""
        for _ in range(max_steps):
            if predicate():
                return True
            self.step()
        return predicate()


def _command(
    system: SilSystem, command_id: str, params: dict[str, str | int | float | bool], seq: int
) -> None:
    """Queue a signed TC packet onto the sim link for the next uplink pump."""
    system.station.enqueue(build_tc_packet(command_id, params, _GROUND, seq, _KEY, apid=1))


def _exit_safe(runner: _Runner, system: SilSystem, seq: int) -> None:
    """Drive one ARM + EXECUTE EXIT_SAFE pair through the signed command path."""
    _command(system, "EXIT_SAFE", {"phase": "ARM"}, seq)
    runner.step()
    _command(system, "EXIT_SAFE", {"phase": "EXECUTE"}, seq + 1)
    runner.step()


def _drive_to_idle(runner: _Runner, system: SilSystem) -> None:
    """Boot SAFE, then ARM/EXECUTE INIT and wait out the verified IDLE handoff."""
    runner.step(2)
    _exit_safe(runner, system, 1)
    assert runner.until(
        lambda: runner.harness.payload_system_mode() is SystemMode.INIT, max_steps=6
    ), "EXIT_SAFE did not reach INIT"
    assert runner.until(
        lambda: runner.harness.payload_system_mode() is SystemMode.IDLE, max_steps=40
    ), "verified INIT never completed to IDLE"


def _drive_to_operate(runner: _Runner, system: SilSystem) -> None:
    """Drive the real path all the way to OPERATE via SET_MODE."""
    _drive_to_idle(runner, system)
    _command(system, "SET_MODE", {"mode": "OPERATE"}, 10)
    assert runner.until(
        lambda: runner.harness.payload_system_mode() is SystemMode.OPERATE, max_steps=6
    ), "SET_MODE OPERATE was not accepted"


def test_default_boot_enters_safe_and_inhibits_motion() -> None:
    """Boot SAFE inhibits the gimbal, blocks inference, and latches containment."""
    config = _config()
    system = _build(config)
    inf_sub = system.bus.subscribe(InferenceResultMsg)
    act_sub = system.bus.subscribe(SystemModeActivatedMsg)
    runner = _Runner(SilHarness(system))
    runner.step(4)

    authority = system.apps.system_modes.state
    assert authority.mode is SystemMode.SAFE
    assert authority.sequence == 1
    activations = _drain(act_sub)
    assert len(activations) == 1
    assert activations[0].active_mode is SystemMode.SAFE
    assert activations[0].sequence == 1
    assert activations[0].previous_mode is None
    assert runner.harness.payload_system_mode() is SystemMode.SAFE
    assert runner.harness.payload_graph() == "safe"
    assert system.apps.payload.containment.local_latched
    health = system.gimbal.read_health()
    assert isinstance(health, Ok)
    assert health.value.inhibit_confirmed
    assert inf_sub.empty()


def test_execute_before_safety_evidence_is_denied() -> None:
    """An EXECUTE decided before any fault evidence exists fails closed."""
    config = _config()
    packets = [
        build_tc_packet("EXIT_SAFE", {"phase": "ARM"}, _GROUND, 1, _KEY, apid=1),
        build_tc_packet("EXIT_SAFE", {"phase": "EXECUTE"}, _GROUND, 2, _KEY, apid=1),
    ]
    system = _build(config, inbound=packets)
    ack_sub = system.bus.subscribe(CommandAckMsg)
    tr_sub = system.bus.subscribe(SystemModeTransitionMsg)
    runner = _Runner(SilHarness(system))
    runner.step(5)

    assert runner.harness.payload_system_mode() is SystemMode.SAFE
    denied = [t for t in _drain(tr_sub) if t.decision is ModeTransitionDecision.DENIED]
    assert any("safety evidence" in t.reason for t in denied)
    acks = _drain(ack_sub)
    assert any(a.status is AckStatus.REJECTED for a in acks)


def test_arm_execute_exit_safe_reaches_init_with_release_evidence() -> None:
    """A grounded ARM/EXECUTE pair yields INIT only with matching release evidence."""
    config = _config()
    system = _build(config)
    safety_sub = system.bus.subscribe(SafetyStateMsg)
    act_sub = system.bus.subscribe(SystemModeActivatedMsg)
    runner = _Runner(SilHarness(system))
    runner.step(2)
    _exit_safe(runner, system, 1)
    assert runner.until(
        lambda: runner.harness.payload_system_mode() is SystemMode.INIT, max_steps=6
    )
    runner.step(4)

    activations = _drain(act_sub)
    init = [a for a in activations if a.active_mode is SystemMode.INIT]
    assert len(init) == 1
    assert init[0].previous_mode is SystemMode.SAFE
    assert init[0].recovery_authorized
    request_id = init[0].request_id
    assert request_id is not None
    evidence = _drain(safety_sub)
    assert any(e.recovery_request_id == request_id for e in evidence)
    assert not system.apps.payload.containment.local_latched
    health = system.gimbal.read_health()
    assert isinstance(health, Ok)
    assert health.value.feedback_valid


def test_default_verifier_init_ready_stays_pending_forever() -> None:
    """With the pending-by-default verifier, READY never completes to IDLE."""
    config = _config()
    system = _build(config)
    req_sub = system.bus.subscribe(SystemModeRequestMsg)
    runner = _Runner(SilHarness(system))
    runner.step(2)
    _exit_safe(runner, system, 1)
    assert runner.until(
        lambda: runner.harness.payload_system_mode() is SystemMode.INIT, max_steps=6
    )
    runner.step(12)

    assert runner.harness.payload_system_mode() is SystemMode.INIT
    assert runner.harness.payload_node() == "ready"
    requests = _drain(req_sub)
    assert not any(r.reason == "graph_intent:init_complete" for r in requests)


def test_verified_recovery_chain_reaches_operate_with_inference_and_motion() -> None:
    """The full real path: SAFE -> INIT installs the runtime -> IDLE -> OPERATE.

    The payload starts on an initially empty InferenceRuntime so the verified
    lifecycle must actually install a session at INIT promotion before IDLE,
    OPERATE, inference, and motion can happen.
    """
    config = _config()
    verifier = _GatedVerifier()
    detector = plume_detector()
    runtime = InferenceRuntime(
        factory=ScriptedRuntimeFactory(detector, identity="acceptance-verified")
    )
    system = _build(config, verifier=verifier, detector=detector, inference_runtime=runtime)
    bind = _bind(config, system, detector)
    req_sub = system.bus.subscribe(SystemModeRequestMsg)
    act_sub = system.bus.subscribe(SystemModeActivatedMsg)
    inf_sub = system.bus.subscribe(InferenceResultMsg)
    assert system.apps.payload.inference.snapshot() is None
    runner = _Runner(SilHarness(system, bind=bind))
    runner.step(2)
    _exit_safe(runner, system, 1)
    assert runner.until(
        lambda: runner.harness.payload_system_mode() is SystemMode.INIT, max_steps=6
    ), "EXIT_SAFE did not reach INIT"
    assert runner.until(lambda: verifier.entered.is_set(), max_steps=40)
    assert verifier.entered.wait(timeout=10.0), "verification never started"
    # Verification undecided: readiness stays false, no session is installed.
    assert system.apps.payload.inference.snapshot() is None
    verifier.released.set()
    assert runner.until(
        lambda: runner.harness.payload_system_mode() is SystemMode.IDLE, max_steps=40
    ), "verified INIT never completed to IDLE"
    _command(system, "SET_MODE", {"mode": "OPERATE"}, 10)
    assert runner.until(
        lambda: runner.harness.payload_system_mode() is SystemMode.OPERATE, max_steps=6
    ), "SET_MODE OPERATE was not accepted"
    runner.step(10)

    completions = [r for r in _drain(req_sub) if r.reason == "graph_intent:init_complete"]
    assert len(completions) == 1
    assert completions[0].requested_by == "payload"
    modes = [a.active_mode for a in _drain(act_sub)]
    assert modes == [
        SystemMode.SAFE,
        SystemMode.INIT,
        SystemMode.IDLE,
        SystemMode.OPERATE,
    ]
    assert verifier.calls == 1
    session = system.apps.payload.inference.snapshot()
    assert session is not None
    assert session.identity == "acceptance-verified"
    assert not inf_sub.empty()
    position = system.gimbal.read_position()
    assert isinstance(position, Ok)
    assert position.value.el_deg > 0.0


def test_set_mode_safe_to_operate_is_denied_and_changes_nothing() -> None:
    """SAFE -> OPERATE via SET_MODE is denied: SAFE only leaves through INIT."""
    config = _config()
    system = _build(config)
    tr_sub = system.bus.subscribe(SystemModeTransitionMsg)
    ack_sub = system.bus.subscribe(CommandAckMsg)
    act_sub = system.bus.subscribe(SystemModeActivatedMsg)
    runner = _Runner(SilHarness(system))
    runner.step(2)
    _command(system, "SET_MODE", {"mode": "OPERATE"}, 1)
    runner.step(3)

    assert runner.harness.payload_system_mode() is SystemMode.SAFE
    assert runner.harness.payload_graph() == "safe"
    denied = [t for t in _drain(tr_sub) if t.decision is ModeTransitionDecision.DENIED]
    assert any(t.requested_mode is SystemMode.OPERATE for t in denied)
    assert any(a.status is AckStatus.REJECTED for a in _drain(ack_sub))
    assert len(_drain(act_sub)) == 1


def test_gimbal_stow_from_operate_reaches_authoritative_stow() -> None:
    """GIMBAL_STOW maps through the authority to a real STOW activation."""
    config = _config()
    detector = plume_detector()
    system = _build(config, verifier=_CurrentKeyVerifier(), detector=detector)
    bind = _bind(config, system, detector)
    act_sub = system.bus.subscribe(SystemModeActivatedMsg)
    ack_sub = system.bus.subscribe(CommandAckMsg)
    runner = _Runner(SilHarness(system, bind=bind))
    _drive_to_operate(runner, system)
    _command(system, "GIMBAL_STOW", {}, 11)
    assert runner.until(
        lambda: runner.harness.payload_system_mode() is SystemMode.STOW, max_steps=6
    )

    activations = _drain(act_sub)
    stow = [a for a in activations if a.active_mode is SystemMode.STOW]
    assert len(stow) == 1
    assert stow[0].previous_mode is SystemMode.OPERATE
    assert any(
        a.status is AckStatus.ACCEPTED and a.command_id == "GIMBAL_STOW" for a in _drain(ack_sub)
    )


def test_local_gimbal_commands_do_not_change_system_activation() -> None:
    """HOLD/HOME/GOTO/RESUME move the operate graph, never the authority."""
    config = _config()
    detector = plume_detector()
    system = _build(config, verifier=_CurrentKeyVerifier(), detector=detector)
    bind = _bind(config, system, detector)
    act_sub = system.bus.subscribe(SystemModeActivatedMsg)
    runner = _Runner(SilHarness(system, bind=bind))
    _drive_to_operate(runner, system)
    seq_before = system.apps.system_modes.state.sequence

    _command(system, "GIMBAL_HOLD", {}, 11)
    runner.step(2)
    assert runner.harness.payload_node() == "hold"
    _command(system, "GIMBAL_HOME", {}, 12)
    _command(system, "GIMBAL_GOTO", {"el_deg": 10.0}, 13)
    runner.step(2)
    _command(system, "GIMBAL_RESUME", {}, 14)
    runner.step(3)

    assert runner.harness.payload_system_mode() is SystemMode.OPERATE
    assert system.apps.system_modes.state.sequence == seq_before
    activations = _drain(act_sub)
    assert activations[-1].active_mode is SystemMode.OPERATE
    assert sum(1 for a in activations if a.active_mode is SystemMode.OPERATE) == 1


def test_hold_survives_vision_until_resume() -> None:
    """In HOLD, fresh plume detections do not move the gimbal; RESUME restores it."""
    config = _config()
    detector = plume_detector()
    system = _build(config, verifier=_CurrentKeyVerifier(), detector=detector)
    bind = _bind(config, system, detector)
    runner = _Runner(SilHarness(system, bind=bind))
    _drive_to_operate(runner, system)

    def _el() -> float:
        pos = system.gimbal.read_position()
        assert isinstance(pos, Ok)
        return pos.value.el_deg

    assert runner.until(lambda: _el() > 0.5, max_steps=15), "tracking never moved"
    _command(system, "GIMBAL_HOLD", {}, 11)
    assert runner.until(lambda: runner.harness.payload_node() == "hold", max_steps=4)
    held = _el()
    runner.step(5)
    frozen = _el()
    assert frozen == pytest.approx(held, abs=1.0e-6)

    _command(system, "GIMBAL_RESUME", {}, 12)
    assert runner.until(lambda: _el() > frozen + 0.05, max_steps=15), (
        "RESUME did not restore tracking motion"
    )


def test_fault_safe_request_reaches_authoritative_safe_immediately() -> None:
    """A containing fault latches local containment first; the authority arbitrates."""
    config = _config()
    detector = plume_detector()
    system = _build(config, verifier=_CurrentKeyVerifier(), detector=detector)
    bind = _bind(config, system, detector)
    act_sub = system.bus.subscribe(SystemModeActivatedMsg)
    runner = _Runner(SilHarness(system, bind=bind))
    _drive_to_operate(runner, system)
    assert runner.until(lambda: runner.harness.payload_node() == "tracking", max_steps=8)
    system.gimbal.freeze_encoder()
    assert runner.until(
        lambda: runner.harness.payload_system_mode() is SystemMode.SAFE, max_steps=8
    )

    activations = _drain(act_sub)
    safe = [a for a in activations if a.active_mode is SystemMode.SAFE]
    assert any(a.previous_mode is SystemMode.OPERATE for a in safe)
    assert system.apps.payload.containment.local_latched
    health = system.gimbal.read_health()
    assert isinstance(health, Ok)
    assert health.value.inhibit_confirmed


def test_containing_fault_during_pending_recovery_reactivates_safe() -> None:
    """A containing fault inside the authorized-but-unreleased INIT window still
    goes through the real fault path: fault-owned SAFE request and authoritative
    SAFE activation, with the payload latch held."""
    config = _config()
    system = _build(config)
    act_sub = system.bus.subscribe(SystemModeActivatedMsg)
    req_sub = system.bus.subscribe(SystemModeRequestMsg)
    runner = _Runner(SilHarness(system))
    runner.step(2)
    _exit_safe(runner, system, 1)
    assert runner.until(
        lambda: runner.harness.payload_system_mode() is SystemMode.INIT, max_steps=6
    )
    # Publish before stepping again: the fault lands while the pending
    # recovery authorization is still unconsumed on the payload side.
    system.bus.publish(
        FaultEventMsg(
            msg_type=MessageType.FAULT_EVENT,
            timestamp_utc=system.clock.wall_clock_iso(),
            fault_code=FaultCode.GIMBAL_RUNAWAY,
            subsystem="test",
            detail="acceptance fault during pending recovery",
        )
    )
    assert runner.until(
        lambda: runner.harness.payload_system_mode() is SystemMode.SAFE, max_steps=6
    )

    requests = _drain(req_sub)
    assert any(r.requested_mode is SystemMode.SAFE and r.requested_by == "fault" for r in requests)
    activations = _drain(act_sub)
    safe = [a for a in activations if a.active_mode is SystemMode.SAFE]
    assert any(a.previous_mode is SystemMode.INIT for a in safe)
    assert system.apps.payload.containment.local_latched


def test_blocked_model_load_cannot_prevent_safe() -> None:
    """A MODEL_LOAD stuck inside the factory never delays a fault-driven SAFE."""
    config = _config()
    blocked = _BlockedFactory()
    system = _build(config, inference_runtime=InferenceRuntime(factory=blocked))
    act_sub = system.bus.subscribe(SystemModeActivatedMsg)
    runner = _Runner(SilHarness(system))
    try:
        runner.step(2)
        _exit_safe(runner, system, 1)
        assert runner.until(
            lambda: runner.harness.payload_system_mode() is SystemMode.INIT, max_steps=6
        )
        assert runner.until(blocked.started.is_set, max_steps=10), (
            "MODEL_LOAD never reached the blocked factory"
        )
        system.bus.publish(
            FaultEventMsg(
                msg_type=MessageType.FAULT_EVENT,
                timestamp_utc=system.clock.wall_clock_iso(),
                fault_code=FaultCode.GIMBAL_RUNAWAY,
                subsystem="test",
                detail="acceptance fault injection",
            )
        )
        assert runner.until(
            lambda: runner.harness.payload_system_mode() is SystemMode.SAFE, max_steps=6
        )
        assert system.apps.payload.containment.local_latched
        activations = _drain(act_sub)
        assert activations[-1].active_mode is SystemMode.SAFE
    finally:
        blocked.released.set()


def test_stale_key_completion_request_is_denied() -> None:
    """A forged init_complete with the wrong key is denied; INIT stays active."""
    config = _config()
    system = _build(config)
    tr_sub = system.bus.subscribe(SystemModeTransitionMsg)
    runner = _Runner(SilHarness(system))
    runner.step(2)
    _exit_safe(runner, system, 1)
    assert runner.until(
        lambda: runner.harness.payload_system_mode() is SystemMode.INIT, max_steps=6
    )
    epoch = system.apps.system_modes.epoch
    system.bus.publish(
        SystemModeRequestMsg(
            msg_type=MessageType.SYSTEM_MODE_REQUEST,
            timestamp_utc=system.clock.wall_clock_iso(),
            request_id="forged-completion",
            requested_mode=SystemMode.IDLE,
            requested_by="payload",
            reason="graph_intent:init_complete",
            activation_key=ActivationKey(epoch=epoch, sequence=99),
        )
    )
    runner.step(3)

    assert runner.harness.payload_system_mode() is SystemMode.INIT
    denied = [t for t in _drain(tr_sub) if t.decision is ModeTransitionDecision.DENIED]
    assert any(t.request_id == "forged-completion" for t in denied)


def test_stale_key_verified_cannot_complete_init() -> None:
    """A VERIFIED result on a foreign key never promotes INIT to IDLE."""
    config = _config()
    verifier = _StaleKeyVerifier()
    system = _build(config, verifier=verifier)
    req_sub = system.bus.subscribe(SystemModeRequestMsg)
    runner = _Runner(SilHarness(system))
    runner.step(2)
    _exit_safe(runner, system, 1)
    assert runner.until(
        lambda: runner.harness.payload_system_mode() is SystemMode.INIT, max_steps=6
    )
    runner.step(12)

    assert runner.harness.payload_system_mode() is SystemMode.INIT
    assert verifier.calls >= 1
    requests = _drain(req_sub)
    assert not any(r.reason == "graph_intent:init_complete" for r in requests)


def test_late_subscriber_sync_replays_current_activation() -> None:
    """A matching-epoch sync replays the activation; a foreign epoch is ignored."""
    config = _config()
    system = _build(config)
    act_sub = system.bus.subscribe(SystemModeActivatedMsg)
    runner = _Runner(SilHarness(system))
    runner.step(2)
    epoch = system.apps.system_modes.epoch

    system.bus.publish(
        SystemModeSyncRequestMsg(
            msg_type=MessageType.SYSTEM_MODE_SYNC_REQUEST,
            timestamp_utc=system.clock.wall_clock_iso(),
            subscriber="late-subscriber",
            expected_epoch="foreign-epoch",
            request_id="sync-wrong",
        )
    )
    system.bus.publish(
        SystemModeSyncRequestMsg(
            msg_type=MessageType.SYSTEM_MODE_SYNC_REQUEST,
            timestamp_utc=system.clock.wall_clock_iso(),
            subscriber="late-subscriber",
            expected_epoch=epoch,
            request_id="sync-ok",
        )
    )
    runner.step(2)

    activations = _drain(act_sub)
    safe_seq1 = [a for a in activations if a.active_mode is SystemMode.SAFE and a.sequence == 1]
    assert len(safe_seq1) == 2
    assert system.apps.system_modes.state.sequence == 1


def test_non_recovery_activation_cannot_recover_bad_hardware() -> None:
    """Bad hardware stays contained: ordinary activations never clear the latch."""
    config = _config()
    detector = plume_detector()
    system = _build(config, verifier=_CurrentKeyVerifier(), detector=detector)
    bind = _bind(config, system, detector)
    tr_sub = system.bus.subscribe(SystemModeTransitionMsg)
    runner = _Runner(SilHarness(system, bind=bind))
    _drive_to_operate(runner, system)
    assert runner.until(lambda: runner.harness.payload_node() == "tracking", max_steps=8)
    system.gimbal.freeze_encoder()
    assert runner.until(
        lambda: runner.harness.payload_system_mode() is SystemMode.SAFE, max_steps=8
    )
    assert system.apps.payload.containment.local_latched

    _exit_safe(runner, system, 20)
    denied = [t for t in _drain(tr_sub) if t.decision is ModeTransitionDecision.DENIED]
    assert denied, "EXIT_SAFE should be denied while the safety latch is set"

    publish_activation(
        system,
        SystemMode.INIT,
        sequence=system.apps.system_modes.state.sequence + 1,
        previous_mode=SystemMode.SAFE,
        request_id="ordinary-activation",
        recovery_authorized=False,
    )
    runner.step(3)

    assert system.apps.payload.containment.local_latched
    health = system.gimbal.read_health()
    assert isinstance(health, Ok)
    assert health.value.inhibit_confirmed
