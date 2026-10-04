"""Tests for the elevation rate-law primitives."""

import math

from flight.payload.gimbal import RateDecision
from flight.payload.gimbal.outer import (
    boundary_limited,
    clip_rate,
    rate_decision,
    smear_cap_rad_s,
    stopping_cap,
)

_IFOV = 0.002636


def _decision(
    *,
    scene_rate_rad_s: float = 0.0,
    requested_relative_rate_rad_s: float = 0.0,
    requested_rate_rad_s: float | None = None,
    theta_g_rad: float | None = None,
    theta_sci_max_rad: float | None = None,
    omega_hw_rad_s: float | None = None,
    exposure_us: float = 1000.0,
    max_motion_smear_px: float = 1.0,
    ifov_band_deg_per_px: float = _IFOV,
    theta_sci_min_rad: float = 0.0,
    max_decel_rad_s2: float = math.inf,
    rate_loop_bandwidth_s: float = math.inf,
) -> RateDecision:
    """rate_decision with defaults for unused kinematics."""
    theta = math.radians(10.0) if theta_g_rad is None else theta_g_rad
    theta_max = math.radians(45.0) if theta_sci_max_rad is None else theta_sci_max_rad
    omega_hw = math.radians(10.0) if omega_hw_rad_s is None else omega_hw_rad_s
    sharp = smear_cap_rad_s(exposure_us, max_motion_smear_px, ifov_band_deg_per_px)
    requested = (
        scene_rate_rad_s + requested_relative_rate_rad_s
        if requested_rate_rad_s is None
        else requested_rate_rad_s
    )
    return rate_decision(
        scene_rate_rad_s,
        requested_relative_rate_rad_s,
        requested,
        sharp,
        theta,
        theta_sci_min_rad,
        theta_max,
        omega_hw,
        max_decel_rad_s2,
        rate_loop_bandwidth_s,
    )


def test_smear_cap_scales_with_exposure() -> None:
    """Halving exposure doubles the smear-limited relative-rate cap."""
    a = smear_cap_rad_s(1000.0, 1.0, _IFOV)
    b = smear_cap_rad_s(500.0, 1.0, _IFOV)
    assert abs(b - 2.0 * a) < 1e-12


def test_rate_decision_passes_composed_terms() -> None:
    """scene and relative terms pass through to requested and commanded rates."""
    sharp = smear_cap_rad_s(1000.0, 1.0, _IFOV)
    nom = math.radians(1.0)
    rel = clip_rate(8.0 * math.radians(-5.0), sharp)
    decision = _decision(
        scene_rate_rad_s=nom,
        requested_relative_rate_rad_s=rel,
    )
    assert rel == -sharp
    assert abs(decision.commanded_rate_rad_s - (nom - sharp)) < 1e-12
    assert abs(decision.scene_rate_rad_s - nom) < 1e-12
    assert abs(decision.requested_relative_rate_rad_s + sharp) < 1e-12
    assert decision.hardware_limited is False
    assert decision.science_limited is False


def test_scene_term_is_not_smear_clipped() -> None:
    """Only the relative term is clipped; the scene rate passes unclipped."""
    nom = math.radians(1.0)
    decision = _decision(scene_rate_rad_s=nom, exposure_us=2000.0)
    assert abs(decision.commanded_rate_rad_s - nom) < 1e-12


def test_rate_decision_clips_to_hardware_when_relative_exceeds_slew() -> None:
    """A requested rate above omega_hw commands the hardware cap."""
    cap = math.radians(10.0)
    sharp = smear_cap_rad_s(13.0, 1.0, _IFOV)
    rel = clip_rate(8.0 * math.radians(-5.0), sharp)
    decision = _decision(
        requested_relative_rate_rad_s=rel,
        requested_rate_rad_s=rel,
        exposure_us=13.0,
    )
    assert abs(abs(decision.commanded_rate_rad_s) - cap) < 1e-12
    assert decision.hardware_limited is True
    assert decision.science_limited is False


def test_science_window_zeros_outward_r() -> None:
    """Commanded rate that would leave [sci_min, sci_max] is zeroed."""
    sharp = smear_cap_rad_s(1000.0, 1.0, _IFOV)
    rel_min = clip_rate(8.0 * math.radians(-4.0), sharp)
    at_min = _decision(
        scene_rate_rad_s=0.0,
        requested_relative_rate_rad_s=rel_min,
        theta_g_rad=0.0,
    )
    assert at_min.commanded_rate_rad_s == 0.0
    assert at_min.science_limited is True
    rel_max = clip_rate(8.0 * math.radians(4.0), sharp)
    at_max = _decision(
        scene_rate_rad_s=0.0,
        requested_relative_rate_rad_s=rel_max,
        theta_g_rad=math.radians(45.0),
    )
    assert at_max.commanded_rate_rad_s == 0.0
    assert at_max.science_limited is True


def test_finite_stopping_governor_binds_near_sci_max() -> None:
    """Finite detailed-plant limits cut rate that infinite production limits pass."""
    theta_max = math.radians(45.0)
    theta = theta_max - 0.01
    sharp = smear_cap_rad_s(13.0, 1.0, _IFOV)
    rel = clip_rate(8.0 * math.radians(4.0), sharp)
    prod = _decision(
        requested_relative_rate_rad_s=rel,
        requested_rate_rad_s=rel,
        theta_g_rad=theta,
        theta_sci_max_rad=theta_max,
        exposure_us=13.0,
        max_decel_rad_s2=math.inf,
        rate_loop_bandwidth_s=math.inf,
    )
    plant = _decision(
        requested_relative_rate_rad_s=rel,
        requested_rate_rad_s=rel,
        theta_g_rad=theta,
        theta_sci_max_rad=theta_max,
        exposure_us=13.0,
        max_decel_rad_s2=1.0,
        rate_loop_bandwidth_s=10.0,
    )
    assert prod.commanded_rate_rad_s > plant.commanded_rate_rad_s
    assert plant.science_limited is True
    assert prod.science_limited is False
    remaining = theta_max - theta
    assert abs(stopping_cap(remaining, 1.0, 10.0) - plant.commanded_rate_rad_s) < 1e-12


def test_stopping_cap_zero_remaining_is_finite_zero() -> None:
    """Zero or negative remaining is a zero cap before any infinite product."""
    assert stopping_cap(0.0, math.inf, math.inf) == 0.0
    assert stopping_cap(-0.2, math.inf, math.inf) == 0.0
    assert math.isfinite(stopping_cap(0.0, math.inf, math.inf))


def test_stopping_cap_infinite_limits_are_unbounded() -> None:
    """Production infinite limits leave a positive remaining angle uncapped."""
    assert stopping_cap(0.1, math.inf, math.inf) == math.inf


def test_stopping_cap_finite_limits_bind() -> None:
    """Detailed-plant deceleration and bandwidth bind the remaining-angle cap."""
    remaining = 0.1
    max_decel = 2.0
    bandwidth = 10.0
    expected = min(math.sqrt(2.0 * max_decel * remaining), bandwidth * remaining)
    assert stopping_cap(remaining, max_decel, bandwidth) == expected


def test_clip_rate_nonpositive_and_infinite_limits() -> None:
    """A non-positive clip limit yields 0; an infinite limit passes the rate."""
    assert clip_rate(0.5, 0.0) == 0.0
    assert clip_rate(-0.5, 0.0) == 0.0
    assert clip_rate(0.5, math.inf) == 0.5
    assert clip_rate(-0.5, math.inf) == -0.5


def test_boundary_limited_zero_rate_stays_zero() -> None:
    """A zero requested rate is not re-expanded by the boundary limiter."""
    assert boundary_limited(0.0, 0.1, 0.0, 1.0, 1.0, 10.0) == 0.0


def test_zero_decision_has_no_limit_flags() -> None:
    """An all-zero composition reports no hardware or science limiting."""
    sharp = smear_cap_rad_s(1000.0, 1.0, _IFOV)
    decision = _decision(exposure_us=1000.0)
    assert decision.commanded_rate_rad_s == 0.0
    assert decision.requested_relative_rate_rad_s == 0.0
    assert decision.smear_limit_rad_s == sharp
    assert decision.hardware_limited is False
    assert decision.science_limited is False
