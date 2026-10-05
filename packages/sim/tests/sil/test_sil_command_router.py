"""SIL integration: command routing (ingress->route->execute->ack) and SAFE request/recovery."""

from dataclasses import replace

from flight.libs.bus import Subscription
from flight.libs.commands import build_tc_packet
from flight.libs.config import PactConfig
from flight.libs.messages import (
    CommandAckMsg,
    RoutedCommandMsg,
    SystemModeActivatedMsg,
    SystemModeRequestMsg,
)
from flight.libs.time import ManualClock
from flight.libs.types import AckStatus, SystemMode
from sim.scene import build_frames, plume_detector
from sim.sil import SilHarness, build_sil_system, publish_activation

_KEY = b"sil-test-key-0000000000000000000"


def _config() -> PactConfig:
    """Default config with zeroed sim encoder noise (keeps the 0-deg bound fresh)."""
    base = PactConfig()
    return replace(
        base,
        gimbal=replace(
            base.gimbal,
            simulation=replace(base.gimbal.simulation, encoder_noise_deg=0.0),
        ),
    )


def test_command_routed_executed_and_acked() -> None:
    """A signed SET_THERMAL_LIMIT is ingressed, routed to thermal, executed, and exec-acked."""
    pkt = build_tc_packet("SET_THERMAL_LIMIT", {"limit_c": 70.0}, "ground", 1, _KEY, apid=1)
    system = build_sil_system(
        PactConfig(),
        ManualClock(),
        build_frames(3),
        plume_detector(),
        inbound_packets=[pkt],
        thermal_readings=[20.0],
        power_readings=[10.0],
    )
    routed = system.bus.subscribe(RoutedCommandMsg)
    acks = system.bus.subscribe(CommandAckMsg)

    SilHarness(system).run_steps(2)

    routed_thermal = [r for r in _drain(routed) if r.command_id == "SET_THERMAL_LIMIT"]
    assert routed_thermal and routed_thermal[0].target == "thermal"
    exec_acks = [
        a
        for a in _drain(acks)
        if a.command_id == "SET_THERMAL_LIMIT" and a.status is AckStatus.ACCEPTED
    ]
    assert exec_acks  # both ingress + thermal-execution acks are ACCEPTED


def test_safe_request_then_authorized_recovery() -> None:
    """Power over-limit latches containment and raises a SAFE request to the authority.

    The request alone never selects a graph: the payload graph stays OPERATE while
    hardware is inhibited. EXIT_SAFE ARM/EXECUTE route to the authority target, and a
    recovery-authorized IDLE activation releases the latch once fault evidence confirms.
    """
    system = build_sil_system(
        _config(),
        ManualClock(),
        build_frames(20),
        plume_detector(),
        inbound_packets=[],
        thermal_readings=[20.0],
        power_readings=[10.0, 10.0, 80.0, 80.0, 10.0],
    )
    harness = SilHarness(system)
    requests = system.bus.subscribe(SystemModeRequestMsg)
    acks = system.bus.subscribe(CommandAckMsg)
    activations = system.bus.subscribe(SystemModeActivatedMsg)

    publish_activation(system, SystemMode.OPERATE, sequence=1)

    now = 0.0

    def advance(steps: int) -> None:
        nonlocal now
        for _ in range(steps):
            now += 1.0
            harness.step(now)

    advance(4)
    reqs = _drain(requests)
    assert any(r.requested_mode is SystemMode.SAFE and r.requested_by == "fault" for r in reqs)
    assert harness.payload_system_mode() is SystemMode.OPERATE
    assert system.apps.payload.containment.local_latched

    advance(3)

    system.station.enqueue(
        build_tc_packet("EXIT_SAFE", {"phase": "ARM"}, "ground", 1, _KEY, apid=1)
    )
    advance(1)  # router records the ARM
    system.station.enqueue(
        build_tc_packet("EXIT_SAFE", {"phase": "EXECUTE"}, "ground", 2, _KEY, apid=1)
    )
    advance(1)
    rejected = [
        a for a in _drain(acks) if a.command_id == "EXIT_SAFE" and a.status is AckStatus.REJECTED
    ]
    assert not rejected
    assert harness.payload_system_mode() is SystemMode.OPERATE
    assert system.apps.payload.containment.local_latched

    publish_activation(
        system,
        SystemMode.IDLE,
        sequence=2,
        previous_mode=SystemMode.SAFE,
        request_id="rec-1",
        recovery_authorized=True,
    )
    advance(2)

    assert not system.apps.payload.containment.local_latched
    assert harness.payload_system_mode() is SystemMode.IDLE
    accepted = _drain(activations)
    assert accepted and accepted[-1].recovery_authorized


def _drain[T](subscription: Subscription[T]) -> list[T]:
    """Drain all pending messages from a subscription into a list (order-preserving)."""
    out: list[T] = []
    while not subscription.empty():
        out.append(subscription.get_nowait())
    return out
