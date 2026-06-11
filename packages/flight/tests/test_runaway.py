"""Tests for the encoder-divergence runaway monitor."""

from flight.hal.interfaces import GimbalPosition
from flight.libs.types import FaultCode
from flight.payload.gimbal import RunawayState, check_runaway


def _pos(az: float, el: float, t: float) -> GimbalPosition:
    return GimbalPosition(az_deg=az, el_deg=el, timestamp_s=t)


def test_matching_motion_resets_strikes() -> None:
    """Encoder motion matching the commanded rate keeps the strike count at zero."""
    state = RunawayState(last_pos=_pos(0.0, 0.0, 0.0), strike_count=2)
    new_state, fault = check_runaway(
        state, _pos(2.0, 0.0, 1.0), 2.0, 0.0, True, tolerance_deg_per_s=1.0, strike_limit=3
    )
    assert fault is None
    assert new_state.strike_count == 0


def test_divergence_accumulates_strikes_then_faults() -> None:
    """Sustained commanded-vs-encoder divergence raises GIMBAL_RUNAWAY at the strike limit."""
    state = RunawayState(last_pos=_pos(0.0, 0.0, 0.0), strike_count=0)
    for i in range(1, 3):
        state, fault = check_runaway(
            state,
            _pos(0.0, 0.0, float(i)),  # gimbal not moving
            2.0,  # but commanded 2 deg/s az
            0.0,
            True,
            tolerance_deg_per_s=1.0,
            strike_limit=3,
        )
        assert fault is None
        assert state.strike_count == i
    state, fault = check_runaway(
        state, _pos(0.0, 0.0, 3.0), 2.0, 0.0, True, tolerance_deg_per_s=1.0, strike_limit=3
    )
    assert fault is FaultCode.GIMBAL_RUNAWAY


def test_non_unit_dt_locks_rate_units() -> None:
    """Rate is delta/dt in deg/s: a 4 deg move over 2 s reads as 2 deg/s, not 4."""
    state = RunawayState(last_pos=_pos(0.0, 0.0, 0.0), strike_count=0)
    # 4 deg over dt=2 s -> 2 deg/s; commanding 2 deg/s matches (no strike).
    matched_state, fault = check_runaway(
        state, _pos(4.0, 0.0, 2.0), 2.0, 0.0, True, tolerance_deg_per_s=0.5, strike_limit=3
    )
    assert fault is None
    assert matched_state.strike_count == 0
    # Same encoder motion but commanding the undivided 4 deg/s diverges (would falsely
    # match if the /dt division were dropped).
    diverged_state, fault = check_runaway(
        state, _pos(4.0, 0.0, 2.0), 4.0, 0.0, True, tolerance_deg_per_s=0.5, strike_limit=3
    )
    assert fault is None
    assert diverged_state.strike_count == 1


def test_elevation_axis_divergence_strikes() -> None:
    """Divergence on the elevation axis alone accumulates strikes (hypot el term)."""
    state = RunawayState(last_pos=_pos(0.0, 0.0, 0.0), strike_count=0)
    new_state, fault = check_runaway(
        state,
        _pos(0.0, 0.0, 1.0),  # encoder el static
        0.0,
        2.0,  # but commanded 2 deg/s el
        True,
        tolerance_deg_per_s=1.0,
        strike_limit=3,
    )
    assert fault is None
    assert new_state.strike_count == 1


def test_no_rate_mode_or_missing_data_resets() -> None:
    """Outside RATE mode, or without a prior/current read, the monitor resets quietly."""
    state = RunawayState(last_pos=_pos(0.0, 0.0, 0.0), strike_count=2)
    new_state, fault = check_runaway(
        state, _pos(0.0, 0.0, 1.0), 2.0, 0.0, False, tolerance_deg_per_s=1.0, strike_limit=3
    )
    assert fault is None
    assert new_state.strike_count == 0
    new_state, fault = check_runaway(
        new_state, None, 2.0, 0.0, True, tolerance_deg_per_s=1.0, strike_limit=3
    )
    assert fault is None
    assert new_state.last_pos is None
