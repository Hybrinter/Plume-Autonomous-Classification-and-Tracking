"""SIL closed-loop integration: the real flight apps over sim drivers via build_apps."""

from flight.libs.bus import Subscription
from flight.libs.commands import build_tc_packet
from flight.libs.config import PactConfig
from flight.libs.messages import (
    CommandAckMsg,
    CommandMsg,
    FaultEventMsg,
    GimbalCommandMsg,
    InferenceResultMsg,
    ModeChangeMsg,
    TelemetryEventMsg,
)
from flight.libs.time import ManualClock
from flight.libs.types import AckStatus, FaultCode, GimbalState, MessageType, Ok, SystemMode
from sim.scene import build_frames, plume_detector
from sim.sil import SilHarness, build_sil_system


def _drain[T](subscription: Subscription[T]) -> list[T]:
    """Drain all pending messages from a subscription into a list."""
    result: list[T] = []
    while not subscription.empty():
        result.append(subscription.get_nowait())
    return result


def test_sil_nominal_closed_loop_tracks_plume() -> None:
    """A plume scene drives payload detection -> gimbal command + telemetry, no SAFE."""
    system = build_sil_system(
        PactConfig(),
        ManualClock(),
        build_frames(8),
        plume_detector(),
        inbound_packets=[],
        thermal_readings=[25.0],
        power_readings=[30.0],
    )
    cmd_sub = system.bus.subscribe(GimbalCommandMsg)
    inf_sub = system.bus.subscribe(InferenceResultMsg)
    telem_sub = system.bus.subscribe(TelemetryEventMsg)
    mode_sub = system.bus.subscribe(ModeChangeMsg)

    SilHarness(system).run_steps(8, dt=1.0)

    # Payload tracked the plume and commanded the gimbal off the origin.
    assert not cmd_sub.empty()
    position = system.gimbal.read_position()
    assert isinstance(position, Ok)
    assert (position.value.az_deg, position.value.el_deg) != (0.0, 0.0)

    # Inference ran once per frame.
    inference_count = 0
    while not inf_sub.empty():
        inf_sub.get_nowait()
        inference_count += 1
    assert inference_count == 8

    # Housekeeping telemetry flowed and the system stayed nominal (no SAFE).
    assert not telem_sub.empty()
    assert mode_sub.empty()


def test_sil_thermal_fault_drives_safe_mode() -> None:
    """A thermal over-limit self-reports a fault that the FDIR app routes to SAFE."""
    system = build_sil_system(
        PactConfig(),
        ManualClock(),
        build_frames(6),
        plume_detector(),
        inbound_packets=[],
        thermal_readings=[25.0, 25.0, 95.0, 95.0, 95.0, 95.0],  # spikes over the 80C limit
        power_readings=[30.0],
    )
    fault_sub = system.bus.subscribe(FaultEventMsg)
    mode_sub = system.bus.subscribe(ModeChangeMsg)

    SilHarness(system).run_steps(6, dt=1.0)

    # Thermal self-reported the over-limit fault and FDIR commanded SAFE.
    assert not fault_sub.empty()
    assert not mode_sub.empty()
    assert mode_sub.get_nowait().new_mode is SystemMode.SAFE


def test_thermal_safe_stows_the_gimbal() -> None:
    """THERMAL_OVER_LIMIT -> FDIR SAFE -> arbiter STOW -> SimGimbal reaches the stow pose."""
    system = build_sil_system(
        PactConfig(),
        ManualClock(),
        build_frames(15),
        plume_detector(),
        inbound_packets=[],
        thermal_readings=[25.0, 95.0],  # spikes over the 80C limit and holds
        power_readings=[30.0],
    )

    # Enough steps for FDIR to route SAFE and the slew-limited dynamics to settle.
    SilHarness(system).run_steps(15, dt=1.0)

    switch = system.gimbal.read_stow_switch()
    assert isinstance(switch, Ok)
    assert switch.value is True


def test_safe_recovery_returns_to_operations() -> None:
    """A ground ModeChangeMsg(non-SAFE) after SAFE un-latches the arbiter."""
    system = build_sil_system(
        PactConfig(),
        ManualClock(),
        build_frames(8),
        plume_detector(),
        inbound_packets=[],
        thermal_readings=[25.0, 95.0, 25.0],  # one over-limit spike, then nominal
        power_readings=[30.0],
    )
    harness = SilHarness(system)
    harness.run_steps(4, dt=1.0)
    assert harness.payload_gimbal_state() is GimbalState.SAFE

    system.bus.publish(
        ModeChangeMsg(
            msg_type=MessageType.MODE_CHANGE,
            timestamp_utc="2026-06-10T00:00:00.000Z",
            new_mode=SystemMode.IDLE,
            requested_by="test_ground_recovery",
        )
    )
    harness.run_steps(2, dt=1.0)

    # The arbiter must have left SAFE (it will re-acquire the scripted plume).
    assert harness.payload_gimbal_state() is not GimbalState.SAFE


def test_tracking_commands_point_toward_the_plume() -> None:
    """RATE commands during TRACKING have the sign of the boresight error and move that way.

    The plume sits at band-plane (340, 340): +x of boresight -> +az error, +y (down) ->
    -el error, so the gimbal must end up at positive azimuth and negative elevation.
    """
    system = build_sil_system(
        PactConfig(),
        ManualClock(),
        build_frames(8),
        plume_detector(),
        inbound_packets=[],
        thermal_readings=[25.0],
        power_readings=[30.0],
    )

    SilHarness(system).run_steps(8, dt=1.0)

    pos = system.gimbal.read_position()
    assert isinstance(pos, Ok)
    assert pos.value.az_deg > 0.5  # plume to the right of boresight
    assert pos.value.el_deg < -0.5  # plume below boresight (image +y)


def test_valid_command_flows_through_to_bus_and_acks() -> None:
    """A signed SET_THERMAL_LIMIT packet becomes a CommandMsg + an ACCEPTED ack in SIL."""
    key = b"sil-test-key-0000000000000000000"
    pkt = build_tc_packet("SET_THERMAL_LIMIT", {"limit_c": 70.0}, "ground", 1, key, apid=1)
    system = build_sil_system(
        PactConfig(),
        ManualClock(),
        build_frames(2),
        plume_detector(),
        inbound_packets=[pkt],
        thermal_readings=[20.0, 20.0],
        power_readings=[10.0, 10.0],
    )
    commands = system.bus.subscribe(CommandMsg)
    acks = system.bus.subscribe(CommandAckMsg)
    SilHarness(system).run_steps(2)
    routed = [c for c in _drain(commands) if c.command_id == "SET_THERMAL_LIMIT"]
    assert len(routed) == 1
    assert routed[0].target == "thermal"
    assert any(a.status is AckStatus.ACCEPTED for a in _drain(acks))


def test_tampered_command_is_rejected_not_routed() -> None:
    """A packet signed with the wrong key yields a REJECTED ack and no CommandMsg."""
    pkt = build_tc_packet("PING", {}, "ground", 1, b"wrong-key-xxxxxxxxxxxxxxxxxxxxxxx", apid=1)
    system = build_sil_system(
        PactConfig(),
        ManualClock(),
        build_frames(2),
        plume_detector(),
        inbound_packets=[pkt],
        thermal_readings=[20.0, 20.0],
        power_readings=[10.0, 10.0],
    )
    commands = system.bus.subscribe(CommandMsg)
    acks = system.bus.subscribe(CommandAckMsg)
    SilHarness(system).run_steps(2)
    assert not [c for c in _drain(commands) if c.source == "ground"]
    rejects = [a for a in _drain(acks) if a.status is AckStatus.REJECTED]
    assert rejects and rejects[0].fault_code is FaultCode.COMMAND_AUTH_FAIL
