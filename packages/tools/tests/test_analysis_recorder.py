"""Tests for the passive SIL capture loop (recorder)."""

import math

import pandas as pd
import pytest
from flight.libs.config import PactConfig
from flight.libs.time import ManualClock
from sim.scene import build_frames, plume_detector
from sim.sil import SilSystem, build_sil_system
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
    assert "payload.gimbal_az_true_deg" in payload.columns
    # The gimbal slews off the origin while tracking the plume.
    assert float(payload["payload.gimbal_az_true_deg"].abs().max()) > 0.0
