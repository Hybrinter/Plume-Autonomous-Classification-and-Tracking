"""Characterization test for the extracted driver-agnostic step_once."""

import math

from flight.libs.config import PactConfig
from flight.libs.messages import InferenceResultMsg
from flight.libs.time import ManualClock
from flight.libs.types import Ok
from sim.scene import build_frames, plume_detector
from sim.sil import SilHarness, build_sil_system


def test_step_once_processes_one_frame_per_call() -> None:
    """After commission, each harness step publishes one inference."""
    system = build_sil_system(
        PactConfig(),
        ManualClock(),
        build_frames(24),
        plume_detector(),
        inbound_packets=[],
        thermal_readings=[25.0, 25.0, 25.0],
        power_readings=[30.0, 30.0, 30.0],
    )
    inf_sub = system.bus.subscribe(InferenceResultMsg)
    harness = SilHarness(system)
    harness.commission()
    while not inf_sub.empty():
        inf_sub.get_nowait()
    harness.run_steps(3, dt=1.0)

    inference_count = 0
    while not inf_sub.empty():
        inf_sub.get_nowait()
        inference_count += 1
    assert inference_count == 3

    position = system.gimbal.read_position()
    assert isinstance(position, Ok)
    assert math.isfinite(position.value.el_deg)
    assert not hasattr(position.value, "az_deg")
