"""Characterization test for the extracted driver-agnostic step_once."""

from flight.libs.config import PactConfig
from flight.libs.messages import InferenceResultMsg
from flight.libs.time import ManualClock
from flight.libs.types import Ok
from sim.scene import build_frames, plume_detector
from sim.sil import build_sil_system, step_once


def test_step_once_processes_one_frame_per_call() -> None:
    """step_once runs the full per-cycle body: one inference is published per call."""
    system = build_sil_system(
        PactConfig(),
        ManualClock(),
        build_frames(3),
        plume_detector(),
        inbound_packets=[],
        thermal_readings=[25.0, 25.0, 25.0],
        power_readings=[30.0, 30.0, 30.0],
    )
    inf_sub = system.bus.subscribe(InferenceResultMsg)
    payload_state = system.apps.payload.controller.initial_state()
    fault_entries = system.apps.fault.initial_entries()

    now = 0.0
    for _ in range(3):
        now += 1.0
        system.clock.advance(1.0)
        payload_state, fault_entries = step_once(
            system.apps,
            system.sensor,
            system.gimbal,
            system.bus,
            system.clock,
            now,
            payload_state,
            fault_entries,
        )

    inference_count = 0
    while not inf_sub.empty():
        inf_sub.get_nowait()
        inference_count += 1
    assert inference_count == 3

    position = system.gimbal.read_position()
    assert isinstance(position, Ok)
    assert (position.value.az_deg, position.value.el_deg) != (0.0, 0.0)
