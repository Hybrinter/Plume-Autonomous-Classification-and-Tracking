"""Tests for the passive SIL capture loop (recorder)."""

import math

import pandas as pd
import pytest
from flight.hal.drivers_sim.gimbal import SimGimbal
from flight.hal.interfaces.gimbal import GimbalPosition
from flight.libs.config import PactConfig
from flight.libs.time import ManualClock
from flight.libs.types import FaultCode, Ok, Result
from sim.scene import build_frames, plume_detector
from sim.sil import SilSystem, build_sil_system, step_once
from tools.analysis.datapoints import (
    REGISTRY,
    SampleContext,
    Signal,
    SignalKind,
    accumulable_names,
)
from tools.analysis.recorder import _evaluate, record_run, sample_devices


def _nominal_system(frames: int = 10) -> SilSystem:
    """Build a nominal SIL system with the given frame count."""
    return build_sil_system(
        PactConfig(),
        ManualClock(),
        build_frames(frames),
        plume_detector(),
        thermal_readings=[25.0],
        power_readings=[30.0],
    )


def test_record_run_shapes_and_columns() -> None:
    """A run yields one wide frame per group and the expected emitted-column count."""
    result = record_run(_nominal_system(10), steps=10)
    assert result.n_steps == 10
    assert result.n_signals == len(REGISTRY)
    assert result.n_columns == len(REGISTRY) + len(accumulable_names())
    assert set(result.wide) == {signal.group for signal in REGISTRY}
    for frame in result.wide.values():
        assert "t" in frame.columns
        assert list(frame.index) == list(range(1, 11))
    assert list(result.long.columns) == [
        "step",
        "t",
        "group",
        "signal",
        "unit",
        "kind",
        "value_num",
        "value_str",
    ]
    assert result.long.shape[0] == 10 * result.n_columns


def test_record_run_is_deterministic() -> None:
    """Two identical runs produce identical long frames."""
    a = record_run(_nominal_system(10), steps=10)
    b = record_run(_nominal_system(10), steps=10)
    pd.testing.assert_frame_equal(a.long, b.long)


def test_steps_must_be_positive() -> None:
    """A non-positive step count is rejected."""
    with pytest.raises(ValueError, match="steps must be positive"):
        record_run(_nominal_system(2), steps=0)


def test_failed_extractor_maps_to_sentinel() -> None:
    """An extractor that raises becomes NaN (numeric) or "" (categorical), never an error."""
    system = _nominal_system(2)
    payload_state = system.apps.payload.controller.initial_state()
    fault_entries = system.apps.fault.initial_entries()
    ctx = SampleContext(
        step=1,
        t=1.0,
        system=system,
        payload_state=payload_state,
        fault_entries=fault_entries,
        messages={},
        devices=sample_devices(system),
    )

    def boom(_ctx: SampleContext) -> float:
        raise RuntimeError("extractor failure")

    numeric = Signal("x.num", "system", "num", "count", SignalKind.NUMERIC, boom)
    categorical = Signal("x.cat", "system", "cat", "enum", SignalKind.CATEGORICAL, boom)
    value = _evaluate(numeric, ctx)
    assert isinstance(value, float) and math.isnan(value)
    assert _evaluate(categorical, ctx) == ""


def test_nominal_run_tracks_and_stays_nominal() -> None:
    """The nominal scene ends in TRACKING with no SAFE latch."""
    result = record_run(_nominal_system(12), steps=12)
    payload = result.wide["payload"]
    system = result.wide["system"]
    assert payload["payload.gimbal_state"].iloc[-1] == "TRACKING"
    assert float(system["system.safe_latched"].max()) == 0.0


def test_queue_depth_and_devices_are_sampled() -> None:
    """The bus queue-depth family and the gimbal device reads are present and numeric."""
    result = record_run(_nominal_system(6), steps=6)
    bus = result.wide["bus"]
    assert "bus.depth.total" in bus.columns
    payload = result.wide["payload"]
    assert "payload.gimbal_el_true_deg" in payload.columns
    # Elevation slews while tracking the off-boresight plume.
    assert float(payload["payload.gimbal_el_true_deg"].abs().max()) > 0.0


def _trace_encoder_reads(gimbal: SimGimbal) -> list[float]:
    """Record each elevation ``read_position`` delivers without changing sampling."""
    delivered: list[float] = []
    original = gimbal.read_position

    def traced() -> Result[GimbalPosition, FaultCode]:
        result = original()
        if isinstance(result, Ok):
            delivered.append(result.value.el_deg)
        return result

    setattr(gimbal, "read_position", traced)
    return delivered


def test_sample_devices_does_not_consume_encoder_samples() -> None:
    """Two seeded twins stay aligned when only the first is observed after one encoder read."""
    recorded = _nominal_system(4)
    twin = _nominal_system(4)
    first_recorded = recorded.gimbal.read_position()
    first_twin = twin.gimbal.read_position()
    assert isinstance(first_recorded, Ok)
    assert isinstance(first_twin, Ok)
    assert first_recorded.value.el_deg == first_twin.value.el_deg
    devices = sample_devices(recorded)
    assert devices.gimbal_el_meas_deg == first_recorded.value.el_deg
    sample_devices(recorded)
    recorded.gimbal.snapshot()
    second_recorded = recorded.gimbal.read_position()
    second_twin = twin.gimbal.read_position()
    assert isinstance(second_recorded, Ok)
    assert isinstance(second_twin, Ok)
    assert second_recorded.value.el_deg == second_twin.value.el_deg


def test_record_run_matches_unrecorded_twin_encoder_and_plant() -> None:
    """record_run matches a twin that steps the SIL and never calls sample_devices."""
    steps = 8
    recorded = _nominal_system(steps)
    twin = _nominal_system(steps)
    recorded_reads = _trace_encoder_reads(recorded.gimbal)
    twin_reads = _trace_encoder_reads(twin.gimbal)
    result = record_run(recorded, steps=steps)

    payload_state = twin.apps.payload.controller.initial_state()
    fault_entries = twin.apps.fault.initial_entries()
    now = 0.0
    dt = 1.0
    twin_true: list[float] = []
    twin_tau: list[float] = []
    twin_meas: list[float] = []
    for _step in range(1, steps + 1):
        now += dt
        payload_state, fault_entries = step_once(
            twin.apps,
            twin.sensor,
            twin.gimbal,
            twin.bus,
            twin.clock,
            now,
            payload_state,
            fault_entries,
        )
        twin.clock.advance(dt)
        snap = twin.gimbal.snapshot()
        twin_true.append(snap.true_el_deg)
        twin_tau.append(snap.tau_nm)
        twin_meas.append(snap.last_el_meas_deg)

    assert recorded_reads == twin_reads
    payload = result.wide["payload"]
    assert list(payload["payload.gimbal_el_true_deg"]) == twin_true
    assert list(payload["payload.gimbal_tau_nm"]) == twin_tau
    assert list(payload["payload.gimbal_el_meas_deg"]) == twin_meas


def test_sample_devices_does_not_change_last_feedback_or_next_read() -> None:
    """Observing leaves last-feedback time and the next encoder sample unchanged."""
    observed = _nominal_system(4)
    twin = _nominal_system(4)
    assert observed.gimbal._last_feedback_s is None
    sample_devices(observed)
    assert observed.gimbal._last_feedback_s is None
    first_observed = observed.gimbal.read_position()
    first_twin = twin.gimbal.read_position()
    assert isinstance(first_observed, Ok)
    assert isinstance(first_twin, Ok)
    assert first_observed.value.el_deg == first_twin.value.el_deg
    last_feedback = observed.gimbal._last_feedback_s
    sample_devices(observed)
    assert observed.gimbal._last_feedback_s == last_feedback
    second_observed = observed.gimbal.read_position()
    second_twin = twin.gimbal.read_position()
    assert isinstance(second_observed, Ok)
    assert isinstance(second_twin, Ok)
    assert second_observed.value.el_deg == second_twin.value.el_deg
