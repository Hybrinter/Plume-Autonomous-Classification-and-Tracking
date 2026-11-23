"""SIL closed-loop integration: the real flight apps over sim drivers via build_apps."""

import math
from dataclasses import replace

from flight.libs.bus import Subscription
from flight.libs.commands import build_tc_packet
from flight.libs.config import PactConfig
from flight.libs.messages import (
    CommandAckMsg,
    CommandMsg,
    FaultEventMsg,
    InferenceResultMsg,
    SystemModeActivatedMsg,
    SystemModeRequestMsg,
    TelemetryEventMsg,
)
from flight.libs.time import ManualClock
from flight.libs.types import AckStatus, FaultCode, Ok, SystemMode
from sim.scene import build_frames, plume_detector
from sim.sil import SilHarness, build_sil_system, publish_activation


def _drain[T](subscription: Subscription[T]) -> list[T]:
    """Drain all pending messages from a subscription into a list."""
    result: list[T] = []
    while not subscription.empty():
        result.append(subscription.get_nowait())
    return result


def _config(stow_rate_deg_per_s: float | None = None) -> PactConfig:
    """Default config with zeroed sim encoder noise (keeps the 0-deg bound fresh)."""
    base = PactConfig()
    return replace(
        base,
        gimbal=replace(
            base.gimbal,
            simulation=replace(base.gimbal.simulation, encoder_noise_deg=0.0),
            xeryon=(
                base.gimbal.xeryon
                if stow_rate_deg_per_s is None
                else replace(
                    base.gimbal.xeryon,
                    stow_reference_rate_deg_per_s=stow_rate_deg_per_s,
                )
            ),
        ),
    )


def test_sil_nominal_closed_loop_tracks_plume() -> None:
    """A plume scene drives payload detection, tracking telemetry, and elevation motion."""
    system = build_sil_system(
        _config(),
        ManualClock(),
        build_frames(8),
        plume_detector(),
        inbound_packets=[],
        thermal_readings=[25.0],
        power_readings=[30.0],
    )
    inf_sub = system.bus.subscribe(InferenceResultMsg)
    telem_sub = system.bus.subscribe(TelemetryEventMsg)

    harness = SilHarness(system)
    publish_activation(system, SystemMode.OPERATE, sequence=1)
    harness.run_steps(8, dt=1.0)

    # Payload tracked the plume and moved elevation inside the science window.
    telem = _drain(telem_sub)
    position = system.gimbal.read_position()
    assert isinstance(position, Ok)
    assert 0.0 < position.value.el_deg <= 45.0
    assert harness.payload_system_mode() is SystemMode.OPERATE
    assert harness.payload_graph() == "operate"
    transitions = [m for m in telem if m.event_name == "node_transition"]
    assert harness.payload_node() == "tracking" or any(
        m.payload.get("from") == "tracking" or m.payload.get("to") == "tracking"
        for m in transitions
    )

    # Imaging duty 0.5 captures even opportunities: floor(8 * 0.5) == 4.
    inference_count = 0
    while not inf_sub.empty():
        inf_sub.get_nowait()
        inference_count += 1
    assert inference_count == 4

    # Housekeeping telemetry flowed and the system stayed nominal (no SAFE).
    assert telem
    assert harness.payload_system_mode() is not SystemMode.SAFE


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
    telem_sub = system.bus.subscribe(TelemetryEventMsg)

    SilHarness(system).run_steps(6, dt=1.0)

    thermal_samples = [
        m
        for m in _drain(telem_sub)
        if m.subsystem == "thermal" and m.event_name == "thermal_sample"
    ]
    assert any(m.payload["temperature_c"] == 95.0 for m in thermal_samples)
    assert not any(f.fault_code is FaultCode.THERMAL_OVER_LIMIT for f in _drain(fault_sub))


def test_stow_activation_stows_the_gimbal() -> None:
    """An explicit STOW activation slews the gimbal to the stow pose and trips the switch."""
    system = build_sil_system(
        _config(stow_rate_deg_per_s=8.0),
        ManualClock(),
        build_frames(15),
        plume_detector(),
        inbound_packets=[],
        thermal_readings=[25.0],
        power_readings=[30.0],
    )
    harness = SilHarness(system)
    publish_activation(system, SystemMode.STOW, sequence=1)

    # Stow is +90 deg at 8 deg/s, so the slew from nadir needs longer than the old -45 deg park.
    harness.run_steps(20, dt=1.0)

    switch = system.gimbal.read_stow_switch()
    assert isinstance(switch, Ok)
    assert switch.value is True
    assert harness.payload_system_mode() is SystemMode.STOW


def test_safe_recovery_returns_through_init() -> None:
    """An authority-authorized INIT activation after SAFE releases the containment latch."""
    system = build_sil_system(
        _config(),
        ManualClock(),
        build_frames(8),
        plume_detector(),
        inbound_packets=[],
        thermal_readings=[25.0],
        power_readings=[30.0],
    )
    harness = SilHarness(system)
    publish_activation(system, SystemMode.SAFE, sequence=1, previous_mode=SystemMode.OPERATE)
    harness.run_steps(2, dt=1.0)
    assert harness.payload_system_mode() is SystemMode.SAFE
    assert system.apps.payload.containment.local_latched

    publish_activation(
        system,
        SystemMode.INIT,
        sequence=2,
        previous_mode=SystemMode.SAFE,
        request_id="rec-1",
        recovery_authorized=True,
    )
    harness.run_steps(2, dt=1.0)

    assert harness.payload_system_mode() is SystemMode.INIT
    assert not system.apps.payload.containment.local_latched


def test_tracking_commands_point_toward_the_plume() -> None:
    """Outer rate during TRACKING has the sign of the boresight error and moves that way.

    The plume sits at band-plane (612, 124): on-boresight in x, image-up ->
    +el error, so the gimbal must end inside the science window.
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
    harness = SilHarness(system)
    publish_activation(system, SystemMode.OPERATE, sequence=1)
    harness.run_steps(8, dt=1.0)

    pos = system.gimbal.read_position()
    assert isinstance(pos, Ok)
    assert 0.0 < pos.value.el_deg <= 45.0
    assert harness.payload_system_mode() is SystemMode.OPERATE
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
    publish_activation(system, SystemMode.OPERATE, sequence=1)
    harness.step(1.005)
    pos = system.gimbal.read_position()
    assert isinstance(pos, Ok)
    assert math.isfinite(pos.value.el_deg)


def test_encoder_freeze_trips_runaway_and_latches_containment() -> None:
    """A frozen encoder under nonzero r publishes GIMBAL_RUNAWAY and latches containment.

    The fault-owned evidence inhibits hardware immediately; the real authority
    arbitrates the fault's SAFE request into a SAFE activation.
    """
    system = build_sil_system(
        _config(),
        ManualClock(),
        build_frames(10),
        plume_detector(),
        inbound_packets=[],
        thermal_readings=[25.0],
        power_readings=[30.0],
    )
    fault_sub = system.bus.subscribe(FaultEventMsg)
    request_sub = system.bus.subscribe(SystemModeRequestMsg)
    act_sub = system.bus.subscribe(SystemModeActivatedMsg)
    harness = SilHarness(system)
    publish_activation(system, SystemMode.OPERATE, sequence=1)
    harness.run_steps(4, dt=1.0)
    system.gimbal.freeze_encoder()
    harness.run_steps(3, dt=1.0)
    faults = _drain(fault_sub)
    assert any(f.fault_code is FaultCode.GIMBAL_RUNAWAY for f in faults)
    assert system.apps.payload.containment.local_latched
    assert harness.payload_system_mode() is SystemMode.SAFE
    assert any(
        r.requested_mode is SystemMode.SAFE and r.requested_by == "fault"
        for r in _drain(request_sub)
    )
    activations = _drain(act_sub)
    assert any(
        a.active_mode is SystemMode.SAFE and a.previous_mode is SystemMode.OPERATE
        for a in activations
    )
    health = system.gimbal.read_health()
    assert isinstance(health, Ok)
    assert health.value.inhibit_confirmed


def test_boot_activates_safe_and_captures_nothing() -> None:
    """The authority's boot SAFE inhibits the gimbal and captures no frames."""
    system = build_sil_system(
        PactConfig(),
        ManualClock(),
        build_frames(4),
        plume_detector(),
        inbound_packets=[],
        thermal_readings=[25.0],
        power_readings=[30.0],
    )
    inf_sub = system.bus.subscribe(InferenceResultMsg)
    act_sub = system.bus.subscribe(SystemModeActivatedMsg)
    harness = SilHarness(system)
    harness.run_steps(4, dt=1.0)
    assert inf_sub.empty()
    activations = _drain(act_sub)
    assert activations and activations[0].active_mode is SystemMode.SAFE
    assert harness.payload_system_mode() is SystemMode.SAFE
    assert harness.payload_graph() == "safe"
    health = system.gimbal.read_health()
    assert isinstance(health, Ok)
    assert health.value.inhibit_confirmed
