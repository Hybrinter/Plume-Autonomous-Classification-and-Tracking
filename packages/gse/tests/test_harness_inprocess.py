"""InProcessBackend over the all-sim profile: build, step, collect a capture."""

from dataclasses import replace
from pathlib import Path

import pytest
from flight.libs.messages import SystemModeRequestMsg, SystemModeTransitionMsg
from flight.libs.types import MessageType, ModeTransitionDecision, SystemMode
from gse.harness import InProcessBackend, SocketBackend
from gse.scenario import Scenario, SceneSpec


def _sil_scenario() -> Scenario:
    """A minimal all-sim scenario: a short plume scene, no commands, no assertions."""
    return Scenario(
        name="harness-smoke",
        profile="profiles/sil.toml",
        scene=SceneSpec(num_frames=4, seed=0),
        commands=(),
        assertions=(),
        steps=4,
        dt=1.0,
        initial_mode=SystemMode.OPERATE,
    )


def test_inprocess_backend_builds_steps_and_collects(
    deterministic_profiles: dict[str, Path],
) -> None:
    """Building over profiles/sil.toml then stepping yields a capture with inference results."""
    backend = InProcessBackend()
    backend.build(_sil_scenario(), str(deterministic_profiles["sil"]))
    for i in range(4):
        backend.step(float(i + 1))
    capture = backend.collect()
    backend.shutdown()

    # Duty 0.5 captures even steps: floor(4 * 0.5) == 2. No SAFE in the nominal scene.
    assert capture.inference_count == 2
    assert capture.mode_activations == (SystemMode.OPERATE,)
    assert capture.active_mode is SystemMode.OPERATE
    # The closed loop tracked the off-center plume and moved the gimbal off the origin
    # (off-origin past the 0.1 deg encoder-noise tolerance).
    assert capture.gimbal_moved is True


def test_inprocess_without_fixture_boots_safe_and_no_inference() -> None:
    """With no initial_mode fixture the real authority boots the system SAFE."""
    backend = InProcessBackend()
    backend.build(replace(_sil_scenario(), initial_mode=None), "profiles/sil.toml")
    try:
        for index in range(4):
            backend.step(float(index + 1))
        capture = backend.collect()
        assert capture.inference_count == 0
        assert capture.mode_activations == (SystemMode.SAFE,)
        assert capture.active_mode is SystemMode.SAFE
        assert capture.gimbal_moved is False
    finally:
        backend.shutdown()


def test_subsystem_request_reaches_authority_and_activates() -> None:
    """A bus-carried mode request is decided by the real authority into an activation."""
    backend = InProcessBackend()
    backend.build(replace(_sil_scenario(), initial_mode=SystemMode.IDLE), "profiles/sil.toml")
    try:
        backend.step(1.0)
        system = backend._system
        assert system is not None
        system.bus.publish(
            SystemModeRequestMsg(
                msg_type=MessageType.SYSTEM_MODE_REQUEST,
                timestamp_utc=system.clock.wall_clock_iso(),
                request_id="request",
                requested_mode=SystemMode.SAFE,
                requested_by="test",
                reason="requested only",
            )
        )
        backend.step(2.0)
        capture = backend.collect()
        assert capture.active_mode is SystemMode.SAFE
        assert capture.mode_activations == (SystemMode.IDLE, SystemMode.SAFE)
        assert capture.inference_count == 0
    finally:
        backend.shutdown()


@pytest.mark.parametrize("decision", list(ModeTransitionDecision))
def test_transition_record_alone_does_not_change_collected_mode(
    decision: ModeTransitionDecision,
) -> None:
    """Even an accepted transition record is not a behavioral activation."""
    backend = InProcessBackend()
    backend.build(replace(_sil_scenario(), initial_mode=SystemMode.IDLE), "profiles/sil.toml")
    try:
        backend.step(1.0)
        system = backend._system
        assert system is not None
        system.bus.publish(
            SystemModeTransitionMsg(
                msg_type=MessageType.SYSTEM_MODE_TRANSITION,
                timestamp_utc=system.clock.wall_clock_iso(),
                transition_id="transition",
                request_id="request",
                epoch=system.apps.payload.activation_epoch,
                previous_mode=SystemMode.IDLE,
                requested_mode=SystemMode.SAFE,
                resulting_mode=(
                    SystemMode.SAFE
                    if decision is ModeTransitionDecision.ACCEPTED
                    else SystemMode.IDLE
                ),
                decision=decision,
                reason="audit only",
                activation_sequence=2 if decision is ModeTransitionDecision.ACCEPTED else None,
            )
        )
        backend.step(2.0)
        capture = backend.collect()
        assert capture.active_mode is SystemMode.IDLE
        assert capture.mode_activations == (SystemMode.IDLE,)
        assert capture.inference_count == 0
    finally:
        backend.shutdown()


def test_socket_backend_is_deferred() -> None:
    """SocketBackend is declared but not implemented (PIL/HIL deferred)."""
    backend = SocketBackend()
    try:
        backend.build(_sil_scenario(), "profiles/pil.toml")
    except NotImplementedError as exc:
        assert "deferred" in str(exc)
    else:  # pragma: no cover - guard
        raise AssertionError("SocketBackend.build must raise NotImplementedError")
