"""SIL closed-loop integration: the real flight apps over sim drivers via build_apps."""

import math

from flight.libs.bus import Subscription
from flight.libs.commands import build_tc_packet
from flight.libs.config import PactConfig
from flight.libs.messages import (
    CommandAckMsg,
    CommandMsg,
    FaultEventMsg,
    InferenceResultMsg,
    ModeChangeMsg,
    TelemetryEventMsg,
)
from flight.libs.time import ManualClock
from flight.libs.types import AckStatus, FaultCode, GimbalState, MessageType, Ok, SystemMode
from sim.scene import build_frames, plume_detector
from sim.sil import SilHarness, build_sil_system

_KEY = b"sil-test-key-0000000000000000000"


def _commission(harness: SilHarness, system, *, homing_steps: int = 12) -> None:
    """Leave boot SAFE via ENTER_INIT homing, then ENTER_OPERATE for tracking."""
    now = harness._now

    def advance() -> None:
        nonlocal now
        now += 1.0
        harness.step(now)
        system.clock.advance(1.0)

    system.station.enqueue(
        build_tc_packet("ENTER_INIT", {"phase": "ARM"}, "ground", 1, _KEY, apid=1)
    )
    advance()
    system.station.enqueue(
        build_tc_packet("ENTER_INIT", {"phase": "EXECUTE"}, "ground", 2, _KEY, apid=1)
    )
    advance()
    advance()  # poll ModeChangeMsg(INIT)
    for _ in range(homing_steps):
        advance()
    system.station.enqueue(build_tc_packet("ENTER_OPERATE", {}, "ground", 3, _KEY, apid=1))
    advance()
    advance()  # poll ModeChangeMsg(OPERATE)


def _drain[T](subscription: Subscription[T]) -> list[T]:
    """Drain all pending messages from a subscription into a list."""
    result: list[T] = []
    while not subscription.empty():
        result.append(subscription.get_nowait())
    return result


def test_sil_nominal_closed_loop_tracks_plume() -> None:
    """A plume scene drives payload detection, pointing telemetry, and elevation motion."""
    system = build_sil_system(
        PactConfig(),
        ManualClock(),
        build_frames(30),
        plume_detector(),
        inbound_packets=[],
        thermal_readings=[25.0],
        power_readings=[30.0],
    )
    inf_sub = system.bus.subscribe(InferenceResultMsg)
    telem_sub = system.bus.subscribe(TelemetryEventMsg)
    mode_sub = system.bus.subscribe(ModeChangeMsg)

    harness = SilHarness(system)
    _commission(harness, system)
    _drain(inf_sub)
    harness.run_steps(8, dt=1.0)

    # Payload tracked the plume and moved elevation inside the science window.
    telem = _drain(telem_sub)
    pointing = [m for m in telem if m.subsystem == "payload" and m.event_name == "pointing"]
    assert pointing
    position = system.gimbal.read_position()
    assert isinstance(position, Ok)
    stow_el = PactConfig().gimbal.stow_el_deg
    assert position.value.el_deg > stow_el + 0.5
    assert position.value.el_deg <= 45.0
    transitions = [
        m for m in telem if m.subsystem == "controller" and m.event_name == "state_transition"
    ]
    assert harness.payload_gimbal_state() is not GimbalState.SAFE
    assert harness.payload_gimbal_state() is GimbalState.TRACKING or any(
        m.payload.get("from") == "TRACKING" or m.payload.get("to") == "TRACKING"
        for m in transitions
    )

    # Inference ran once per commissioned tracking frame.
    inference_count = 0
    while not inf_sub.empty():
        inf_sub.get_nowait()
        inference_count += 1
    assert inference_count == 8

    # Commissioning publishes INIT/IDLE/OPERATE transitions before tracking.
    mode_changes = _drain(mode_sub)
    assert any(m.new_mode is SystemMode.INIT for m in mode_changes)
    assert any(m.new_mode is SystemMode.OPERATE for m in mode_changes)
    assert telem


def test_sil_thermal_hot_sample_is_telemetry_only() -> None:
    """A hot thermal sample publishes telemetry and does not drive SAFE."""
    system = build_sil_system(
        PactConfig(),
        ManualClock(),
        build_frames(6),
        plume_detector(),
        inbound_packets=[],
        thermal_readings=[25.0, 25.0, 95.0, 95.0, 95.0, 95.0],
        power_readings=[30.0],
    )
    fault_sub = system.bus.subscribe(FaultEventMsg)
    mode_sub = system.bus.subscribe(ModeChangeMsg)
    telem_sub = system.bus.subscribe(TelemetryEventMsg)

    SilHarness(system).run_steps(6, dt=1.0)

    thermal_samples = [
        m
        for m in _drain(telem_sub)
        if m.subsystem == "thermal" and m.event_name == "thermal_sample"
    ]
    assert any(m.payload["temperature_c"] == 95.0 for m in thermal_samples)
    assert not any(f.fault_code is FaultCode.THERMAL_OVER_LIMIT for f in _drain(fault_sub))
    assert mode_sub.empty()


def test_safe_halts_the_gimbal() -> None:
    """A commanded SAFE mode change halts motion without commanding stow."""
    system = build_sil_system(
        PactConfig(),
        ManualClock(),
        build_frames(15),
        plume_detector(),
        inbound_packets=[],
        thermal_readings=[25.0],
        power_readings=[30.0],
    )
    pos_before = system.gimbal.read_position()
    assert isinstance(pos_before, Ok)
    system.bus.publish(
        ModeChangeMsg(
            msg_type=MessageType.MODE_CHANGE,
            timestamp_utc="2026-06-10T00:00:00.000Z",
            new_mode=SystemMode.SAFE,
            requested_by="test_safe_halt",
        )
    )

    SilHarness(system).run_steps(15, dt=1.0)

    pos_after = system.gimbal.read_position()
    assert isinstance(pos_after, Ok)
    assert abs(pos_after.value.el_deg - pos_before.value.el_deg) < 1.0
    switch = system.gimbal.read_stow_switch()
    assert isinstance(switch, Ok)
    assert switch.value is False


def test_safe_recovery_returns_to_operations() -> None:
    """ENTER_INIT homing then ENTER_OPERATE after SAFE un-latches tracking."""
    system = build_sil_system(
        PactConfig(),
        ManualClock(),
        build_frames(30),
        plume_detector(),
        inbound_packets=[],
        thermal_readings=[25.0],
        power_readings=[30.0],
    )
    harness = SilHarness(system)
    system.bus.publish(
        ModeChangeMsg(
            msg_type=MessageType.MODE_CHANGE,
            timestamp_utc="2026-06-10T00:00:00.000Z",
            new_mode=SystemMode.SAFE,
            requested_by="test_safe_entry",
        )
    )
    harness.run_steps(2, dt=1.0)
    assert harness.payload_gimbal_state() is GimbalState.SAFE

    _commission(harness, system)
    harness.run_steps(4, dt=1.0)

    assert harness.payload_gimbal_state() is not GimbalState.SAFE


def test_tracking_commands_point_toward_the_plume() -> None:
    """Outer rate during TRACKING has the sign of the boresight error and moves that way.

    The plume sits at band-plane (612, 124): on-boresight in x, image-up ->
    +el error, so the gimbal must end inside the science window.
    """
    system = build_sil_system(
        PactConfig(),
        ManualClock(),
        build_frames(30),
        plume_detector(),
        inbound_packets=[],
        thermal_readings=[25.0],
        power_readings=[30.0],
    )
    harness = SilHarness(system)
    _commission(harness, system)
    harness.run_steps(8, dt=1.0)

    pos = system.gimbal.read_position()
    assert isinstance(pos, Ok)
    stow_el = PactConfig().gimbal.stow_el_deg
    assert pos.value.el_deg > stow_el + 0.5
    assert pos.value.el_deg <= 45.0
    assert harness.payload_gimbal_state() is not GimbalState.SAFE
    assert not hasattr(pos.value, "az_deg")


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


def test_sil_non_grid_now_interleaves() -> None:
    """A now that is not a multiple of T_out still runs inner-then-outer slices."""
    system = build_sil_system(
        PactConfig(),
        ManualClock(),
        build_frames(4),
        plume_detector(),
        inbound_packets=[],
        thermal_readings=[25.0],
        power_readings=[30.0],
    )
    harness = SilHarness(system)
    harness.step(1.005)
    system.clock.advance(1.005)
    pos = system.gimbal.read_position()
    assert isinstance(pos, Ok)
    assert math.isfinite(pos.value.el_deg)


def test_encoder_freeze_trips_runaway_and_safe() -> None:
    """A frozen encoder under nonzero r publishes GIMBAL_RUNAWAY and SAFEs."""
    system = build_sil_system(
        PactConfig(),
        ManualClock(),
        build_frames(30),
        plume_detector(),
        inbound_packets=[],
        thermal_readings=[25.0],
        power_readings=[30.0],
    )
    fault_sub = system.bus.subscribe(FaultEventMsg)
    harness = SilHarness(system)
    _commission(harness, system)
    harness.run_steps(6, dt=1.0)
    system.gimbal.freeze_encoder()
    harness.run_steps(4, dt=1.0)
    faults = _drain(fault_sub)
    assert any(f.fault_code is FaultCode.GIMBAL_RUNAWAY for f in faults)
    assert harness.payload_gimbal_state() is GimbalState.SAFE
