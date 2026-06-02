"""SIL closed-loop integration: the real flight apps over sim drivers via build_apps."""

from flight.libs.config import PactConfig
from flight.libs.messages import (
    FaultEventMsg,
    GimbalCommandMsg,
    InferenceResultMsg,
    ModeChangeMsg,
    TelemetryEventMsg,
)
from flight.libs.time import ManualClock
from flight.libs.types import Ok, SystemMode
from sim.scene import build_frames, plume_detector
from sim.sil import SilHarness, build_sil_system


def test_sil_nominal_closed_loop_tracks_plume() -> None:
    """A plume scene drives payload detection -> gimbal command + telemetry, no SAFE."""
    system = build_sil_system(
        PactConfig(),
        ManualClock(),
        build_frames(8),
        plume_detector(),
        inbound_commands=[],
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
        inbound_commands=[],
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
