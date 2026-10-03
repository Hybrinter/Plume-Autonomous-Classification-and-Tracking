"""Characterization test for the extracted driver-agnostic step_once."""

from dataclasses import replace

import pytest
from flight.libs.config import PactConfig
from flight.libs.messages import InferenceResultMsg
from flight.libs.time import ManualClock
from flight.libs.types import Ok, SystemMode
from sim.scene import build_frames, plume_detector
from sim.sil import SilHarness, build_sil_system, publish_activation, step_once


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


def test_step_once_processes_one_frame_per_call() -> None:
    """step_once runs the full per-cycle body. Duty 0.5 captures even opportunities."""
    system = build_sil_system(
        _config(),
        ManualClock(),
        build_frames(3),
        plume_detector(),
        inbound_packets=[],
        thermal_readings=[25.0, 25.0, 25.0],
        power_readings=[30.0, 30.0, 30.0],
    )
    inf_sub = system.bus.subscribe(InferenceResultMsg)
    publish_activation(system, SystemMode.OPERATE, sequence=1)
    payload_state = system.apps.payload.initial_state()
    fault_entries = system.apps.fault.initial_entries()

    now = 0.0
    for _ in range(3):
        now += 1.0
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
    assert inference_count == 1

    position = system.gimbal.read_position()
    assert isinstance(position, Ok)
    assert position.value.el_deg != 0.0
    assert not hasattr(position.value, "az_deg")


def test_step_once_duty_half_processes_due_scripted_frames() -> None:
    """Duty 0.5 processes scripted frames stamped at even steps, not the skipped ones."""
    frames = build_frames(4)
    assert [frame.timestamp_s for frame in frames] == [1.0, 2.0, 3.0, 4.0]
    system = build_sil_system(
        _config(),
        ManualClock(),
        frames,
        plume_detector(),
        inbound_packets=[],
        thermal_readings=[25.0],
        power_readings=[30.0],
    )
    inf_sub = system.bus.subscribe(InferenceResultMsg)
    publish_activation(system, SystemMode.OPERATE, sequence=1)
    payload_state = system.apps.payload.initial_state()
    fault_entries = system.apps.fault.initial_entries()

    now = 0.0
    for _ in range(4):
        now += 1.0
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

    processed: list[int] = []
    while not inf_sub.empty():
        processed.append(inf_sub.get_nowait().frame_id)
    assert processed == [2, 4]
    assert system.sensor.unread_scripted_count() == 0


def test_inhibited_stepping_reaches_target_with_exact_encoder_stamp() -> None:
    """Unactivated stepping advances the clock to a non-grid target.

    With no activation the actuator stays inhibited, yet the shared
    ManualClock still reaches the step target and the final control-owned
    feedback sample is stamped at the real clock time -- no torque writes or
    forged timestamps are needed to progress simulated device time.
    """
    clock = ManualClock(monotonic_s=100.0)
    system = build_sil_system(
        _config(),
        clock,
        build_frames(2),
        plume_detector(),
        inbound_packets=[],
        thermal_readings=[25.0],
        power_readings=[30.0],
    )
    harness = SilHarness(system)
    harness.step(101.005)
    assert clock.monotonic_s() == pytest.approx(101.005)
    samples = system.apps.payload.encoder_stream.samples
    assert samples
    latest = max(sample.t_s for sample in samples)
    inner_dt = system.apps.payload.servo.cfg.inner.dt_s
    assert latest <= 101.005
    assert 101.005 - latest <= inner_dt + 1.0e-9
