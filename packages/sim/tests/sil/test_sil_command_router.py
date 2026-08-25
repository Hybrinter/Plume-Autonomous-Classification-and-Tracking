"""SIL integration: command routing (ingress->route->execute->ack) and SAFE enter/exit."""

from flight.libs.commands import build_tc_packet
from flight.libs.config import PactConfig
from flight.libs.messages import CommandAckMsg, ModeChangeMsg, RoutedCommandMsg
from flight.libs.time import ManualClock
from flight.libs.types import AckStatus, GimbalState, SystemMode
from sim.scene import build_frames, plume_detector
from sim.sil import SilHarness, build_sil_system

_KEY = b"sil-test-key-0000000000000000000"


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


def test_safe_entered_then_exited_via_arm_execute() -> None:
    """Thermal over-limit latches SAFE; a ground EXIT_SAFE (ARM->EXECUTE) recovers once cleared."""
    system = build_sil_system(
        PactConfig(),
        ManualClock(),
        build_frames(20),
        plume_detector(),
        inbound_packets=[],
        # spike over 80C (latch SAFE) then hold nominal 20C so EXIT_SAFE's fault-clear gate opens
        thermal_readings=[20.0, 20.0, 95.0, 95.0, 20.0],
        power_readings=[10.0],
    )
    harness = SilHarness(system)
    modes = system.bus.subscribe(ModeChangeMsg)
    acks = system.bus.subscribe(CommandAckMsg)

    now = 0.0

    def advance(steps: int) -> None:
        nonlocal now
        for _ in range(steps):
            now += 1.0
            system.clock.advance(1.0)
            harness.step(now)

    advance(4)  # by step 3-4 thermal is over-limit -> SAFE latched
    assert harness.payload_gimbal_state() is GimbalState.SAFE
    assert SystemMode.SAFE in [m.new_mode for m in _drain(modes)]

    advance(3)  # thermal now reads 20C (cooled) -> the triggering fault clears

    system.station.enqueue(
        build_tc_packet("EXIT_SAFE", {"phase": "ARM"}, "ground", 1, _KEY, apid=1)
    )
    advance(1)  # router records the ARM
    system.station.enqueue(
        build_tc_packet("EXIT_SAFE", {"phase": "EXECUTE"}, "ground", 2, _KEY, apid=1)
    )
    advance(1)  # router routes EXECUTE -> fault app publishes ModeChangeMsg(IDLE)
    advance(1)  # arbiter polls the IDLE mode change at the next cycle and leaves SAFE

    assert harness.payload_gimbal_state() is not GimbalState.SAFE
    exit_acks = [
        a for a in _drain(acks) if a.command_id == "EXIT_SAFE" and a.status is AckStatus.ACCEPTED
    ]
    assert exit_acks  # the fault app emitted an ACCEPTED execution ack for EXIT_SAFE


def _drain(subscription: object) -> list:  # type: ignore[type-arg]
    """Drain all pending messages from a subscription into a list (order-preserving)."""
    out: list = []  # type: ignore[type-arg]
    while not subscription.empty():  # type: ignore[attr-defined]
        out.append(subscription.get_nowait())  # type: ignore[attr-defined]
    return out
