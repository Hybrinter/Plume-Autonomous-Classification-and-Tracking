"""Tests for the elevation-relative smear rate law."""

import math

from flight.libs.types import GimbalState
from flight.payload.gimbal import RateDecision
from flight.payload.gimbal.outer import outer_rate, smear_cap_rad_s, stopping_cap

_IFOV = 0.002636


def _tracking(
    *,
    omega_t_nom: float = 0.0,
    omega_t_res: float = 0.0,
    e_hat: float = 0.0,
    k_p: float = 8.0,
    live: bool = True,
    theta_g_rad: float | None = None,
    theta_sci_max_rad: float | None = None,
    omega_hw_rad_s: float | None = None,
    exposure_us: float = 1000.0,
    max_motion_smear_px: float = 1.0,
    ifov_band_deg_per_px: float = _IFOV,
    max_decel_rad_s2: float = math.inf,
    rate_loop_bandwidth_s: float = math.inf,
) -> RateDecision:
    """TRACKING live outer_rate with defaults for unused kinematics."""
    theta = math.radians(10.0) if theta_g_rad is None else theta_g_rad
    theta_max = math.radians(45.0) if theta_sci_max_rad is None else theta_sci_max_rad
    omega_hw = math.radians(10.0) if omega_hw_rad_s is None else omega_hw_rad_s
    return outer_rate(
        omega_t_nom=omega_t_nom,
        omega_t_res=omega_t_res,
        e_hat=e_hat,
        k_p=k_p,
        mode=GimbalState.TRACKING,
        live=live,
        theta_g_rad=theta,
        theta_sci_max_rad=theta_max,
        omega_hw_rad_s=omega_hw,
        exposure_us=exposure_us,
        max_motion_smear_px=max_motion_smear_px,
        ifov_band_deg_per_px=ifov_band_deg_per_px,
        max_decel_rad_s2=max_decel_rad_s2,
        rate_loop_bandwidth_s=rate_loop_bandwidth_s,
    )


def test_smear_cap_scales_with_exposure() -> None:
    """Halving exposure doubles the smear-limited relative-rate cap."""
    a = smear_cap_rad_s(1000.0, 1.0, _IFOV)
    b = smear_cap_rad_s(500.0, 1.0, _IFOV)
    assert abs(b - 2.0 * a) < 1e-12


def test_tracking_caps_only_proportional_term() -> None:
    """Large Kp*e is smear-capped; the scene rate is not."""
    sharp = smear_cap_rad_s(1000.0, 1.0, _IFOV)
    nom = math.radians(1.0)
    decision = _tracking(omega_t_nom=nom, e_hat=math.radians(-5.0))
    assert abs(decision.commanded_rate_rad_s - (nom - sharp)) < 1e-12
    assert abs(decision.scene_rate_rad_s - nom) < 1e-12
    assert abs(decision.requested_relative_rate_rad_s + sharp) < 1e-12
    assert decision.hardware_limited is False
    assert decision.science_limited is False


def test_tracking_does_not_clip_scene_match() -> None:
    """A long exposure does not reduce omega_scene when e_hat is zero."""
    nom = math.radians(1.0)
    decision = _tracking(omega_t_nom=nom, e_hat=0.0, exposure_us=2000.0)
    assert abs(decision.commanded_rate_rad_s - nom) < 1e-12


def test_tracking_adds_residual_to_scene() -> None:
    """TRACKING live scene rate is omega_t_nom + omega_t_res."""
    nom = math.radians(1.0)
    res = math.radians(0.25)
    decision = _tracking(omega_t_nom=nom, omega_t_res=res, e_hat=0.0)
    assert abs(decision.scene_rate_rad_s - (nom + res)) < 1e-12
    assert abs(decision.commanded_rate_rad_s - (nom + res)) < 1e-12


def test_tracking_live_clips_to_hardware_when_smear_is_loose() -> None:
    """A 13 us exposure leaves hardware as the binding clip on Kp*e."""
    decision = _tracking(e_hat=math.radians(-5.0), exposure_us=13.0)
    cap = math.radians(10.0)
    assert abs(abs(decision.commanded_rate_rad_s) - cap) < 1e-12
    assert decision.hardware_limited is True
    assert decision.science_limited is False


def test_cold_tracking_is_zero() -> None:
    """TRACKING before the first vision update holds r = 0."""
    decision = _tracking(omega_t_nom=0.1, e_hat=0.05, live=False, theta_g_rad=0.0)
    assert decision.commanded_rate_rad_s == 0.0
    assert decision.hardware_limited is False
    assert decision.science_limited is False


def test_rewind_sharp_adds_smear_budget_to_scene_rate() -> None:
    """REWIND inside the sharp window hunts at omega_nom + omega_sharp."""
    sharp = smear_cap_rad_s(1000.0, 1.0, _IFOV)
    nom = math.radians(-1.0)
    decision = outer_rate(
        omega_t_nom=nom,
        omega_t_res=math.radians(9.0),
        e_hat=0.0,
        k_p=8.0,
        mode=GimbalState.REWIND,
        live=False,
        theta_g_rad=0.0,
        theta_sci_max_rad=math.radians(45.0),
        omega_hw_rad_s=math.radians(10.0),
        exposure_us=1000.0,
        max_motion_smear_px=1.0,
        ifov_band_deg_per_px=_IFOV,
        rewind_elapsed_s=0.1,
        rewind_sharp_max_s=2.0,
    )
    assert abs(decision.commanded_rate_rad_s - (nom + sharp)) < 1e-12
    assert abs(decision.scene_rate_rad_s - nom) < 1e-12
    assert abs(decision.requested_relative_rate_rad_s - sharp) < 1e-12
    assert decision.requested_relative_rate_rad_s > 0.0


def _rewind(*, omega_t_res: float) -> RateDecision:
    """Sharp REWIND at nadir with a downward scene rate."""
    return outer_rate(
        omega_t_nom=math.radians(-1.0),
        omega_t_res=omega_t_res,
        e_hat=0.0,
        k_p=8.0,
        mode=GimbalState.REWIND,
        live=False,
        theta_g_rad=0.0,
        theta_sci_max_rad=math.radians(45.0),
        omega_hw_rad_s=math.radians(10.0),
        exposure_us=1000.0,
        max_motion_smear_px=1.0,
        ifov_band_deg_per_px=_IFOV,
        rewind_elapsed_s=0.1,
        rewind_sharp_max_s=2.0,
    )


def test_rewind_ignores_residual_rate() -> None:
    """REWIND commanded rate does not change when omega_t_res changes."""
    a = _rewind(omega_t_res=0.0)
    b = _rewind(omega_t_res=math.radians(9.0))
    assert a.commanded_rate_rad_s == b.commanded_rate_rad_s
    assert a.scene_rate_rad_s == b.scene_rate_rad_s


def test_rewind_escape_uses_hardware_rate_toward_limb() -> None:
    """After rewind_sharp_max_s, REWIND drives at the hardware cap."""
    cap = math.radians(10.0)
    decision = outer_rate(
        omega_t_nom=math.radians(-1.0),
        omega_t_res=0.0,
        e_hat=0.0,
        k_p=8.0,
        mode=GimbalState.REWIND,
        live=False,
        theta_g_rad=0.0,
        theta_sci_max_rad=math.radians(45.0),
        omega_hw_rad_s=cap,
        exposure_us=1000.0,
        max_motion_smear_px=1.0,
        ifov_band_deg_per_px=_IFOV,
        rewind_elapsed_s=2.0,
        rewind_sharp_max_s=2.0,
    )
    assert abs(decision.commanded_rate_rad_s - cap) < 1e-12
    assert abs(decision.requested_relative_rate_rad_s - cap) < 1e-12
    assert decision.hardware_limited is False
    assert decision.science_limited is False


def test_science_window_zeros_outward_r() -> None:
    """TRACKING live zeros r that would leave [sci_min, sci_max]."""
    at_min = _tracking(e_hat=math.radians(-4.0), theta_g_rad=0.0)
    assert at_min.commanded_rate_rad_s == 0.0
    assert at_min.science_limited is True
    at_max = _tracking(e_hat=math.radians(4.0), theta_g_rad=math.radians(45.0))
    assert at_max.commanded_rate_rad_s == 0.0
    assert at_max.science_limited is True


def test_rewind_zeros_outward_rate_at_sci_min() -> None:
    """Sharp REWIND at theta_sci_min does not command a negative rate."""
    decision = outer_rate(
        omega_t_nom=-0.02,
        omega_t_res=0.0,
        e_hat=0.0,
        k_p=8.0,
        mode=GimbalState.REWIND,
        live=False,
        theta_g_rad=0.0,
        theta_sci_max_rad=1.0,
        omega_hw_rad_s=0.2,
        exposure_us=1.0e6,
        max_motion_smear_px=1.0,
        ifov_band_deg_per_px=_IFOV,
        theta_sci_min_rad=0.0,
        max_decel_rad_s2=math.inf,
        rate_loop_bandwidth_s=math.inf,
    )
    assert decision.commanded_rate_rad_s == 0.0
    assert math.isfinite(decision.commanded_rate_rad_s)
    assert decision.science_limited is True


def test_rewind_at_sci_max_is_zero() -> None:
    """REWIND at the upper science limb holds commanded rate at 0."""
    decision = outer_rate(
        omega_t_nom=math.radians(-1.0),
        omega_t_res=0.0,
        e_hat=0.0,
        k_p=8.0,
        mode=GimbalState.REWIND,
        live=False,
        theta_g_rad=math.radians(45.0),
        theta_sci_max_rad=math.radians(45.0),
        omega_hw_rad_s=math.radians(10.0),
        exposure_us=1000.0,
        max_motion_smear_px=1.0,
        ifov_band_deg_per_px=_IFOV,
    )
    assert decision.commanded_rate_rad_s == 0.0
    assert decision.science_limited is True
    assert decision.hardware_limited is False


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


def test_finite_stopping_governor_binds_near_sci_max() -> None:
    """Finite detailed-plant limits cut rate that infinite production limits pass."""
    theta_max = math.radians(45.0)
    theta = theta_max - 0.01
    prod = _tracking(
        e_hat=math.radians(4.0),
        theta_g_rad=theta,
        theta_sci_max_rad=theta_max,
        exposure_us=13.0,
        max_decel_rad_s2=math.inf,
        rate_loop_bandwidth_s=math.inf,
    )
    plant = _tracking(
        e_hat=math.radians(4.0),
        theta_g_rad=theta,
        theta_sci_max_rad=theta_max,
        exposure_us=13.0,
        max_decel_rad_s2=1.0,
        rate_loop_bandwidth_s=10.0,
    )
    assert prod.commanded_rate_rad_s > plant.commanded_rate_rad_s
    assert plant.science_limited is True
    assert prod.science_limited is False
    assert stopping_cap(0.01, 1.0, 10.0) == plant.commanded_rate_rad_s
