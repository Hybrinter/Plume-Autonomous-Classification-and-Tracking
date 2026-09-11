"""Tests for the elevation-relative smear rate law."""

import math

from flight.libs.types import GimbalState
from flight.payload.gimbal.outer import outer_rate, smear_cap_rad_s

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
) -> float:
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
    r = _tracking(omega_t_nom=nom, e_hat=math.radians(-5.0))
    assert abs(r - (nom - sharp)) < 1e-12


def test_tracking_does_not_clip_scene_match() -> None:
    """A long exposure does not reduce omega_scene when e_hat is zero."""
    nom = math.radians(1.0)
    r = _tracking(omega_t_nom=nom, e_hat=0.0, exposure_us=2000.0)
    assert abs(r - nom) < 1e-12


def test_tracking_live_clips_to_hardware_when_smear_is_loose() -> None:
    """A 13 us exposure leaves hardware as the binding clip on Kp*e."""
    r = _tracking(e_hat=math.radians(-5.0), exposure_us=13.0)
    cap = math.radians(10.0)
    assert abs(abs(r) - cap) < 1e-12


def test_azimuth_rate_does_not_change_elevation_command() -> None:
    """omega_az is diagnostic and must not shrink the elevation smear cap."""
    a = outer_rate(
        omega_t_nom=0.0,
        omega_t_res=0.0,
        e_hat=math.radians(-5.0),
        k_p=8.0,
        mode=GimbalState.TRACKING,
        live=True,
        theta_g_rad=math.radians(10.0),
        theta_sci_max_rad=math.radians(45.0),
        omega_hw_rad_s=math.radians(10.0),
        exposure_us=1000.0,
        max_motion_smear_px=1.0,
        ifov_band_deg_per_px=_IFOV,
        omega_az=0.0,
    )
    b = outer_rate(
        omega_t_nom=0.0,
        omega_t_res=0.0,
        e_hat=math.radians(-5.0),
        k_p=8.0,
        mode=GimbalState.TRACKING,
        live=True,
        theta_g_rad=math.radians(10.0),
        theta_sci_max_rad=math.radians(45.0),
        omega_hw_rad_s=math.radians(10.0),
        exposure_us=1000.0,
        max_motion_smear_px=1.0,
        ifov_band_deg_per_px=_IFOV,
        omega_az=1.0,
    )
    assert a == b


def test_cold_tracking_is_zero() -> None:
    """TRACKING before the first vision update holds r = 0."""
    r = _tracking(omega_t_nom=0.1, e_hat=0.05, live=False, theta_g_rad=0.0)
    assert r == 0.0


def test_rewind_sharp_adds_smear_budget_to_scene_rate() -> None:
    """REWIND inside the sharp window hunts at omega_nom + sign * omega_sharp."""
    sharp = smear_cap_rad_s(1000.0, 1.0, _IFOV)
    nom = math.radians(-1.0)
    r = outer_rate(
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
    assert abs(r - (nom + sharp)) < 1e-12


def test_rewind_escape_uses_hardware_rate_toward_limb() -> None:
    """After rewind_sharp_max_s, REWIND drives at the hardware cap."""
    r = outer_rate(
        omega_t_nom=math.radians(-1.0),
        omega_t_res=0.0,
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
        rewind_elapsed_s=2.0,
        rewind_sharp_max_s=2.0,
    )
    cap = math.radians(10.0)
    assert abs(r - cap) < 1e-12


def test_science_window_zeros_outward_r() -> None:
    """TRACKING live zeros r that would leave [sci_min, sci_max]."""
    at_min = _tracking(e_hat=math.radians(-4.0), theta_g_rad=0.0)
    assert at_min == 0.0
    at_max = _tracking(e_hat=math.radians(4.0), theta_g_rad=math.radians(45.0))
    assert at_max == 0.0


def test_rewind_zeros_outward_rate_at_sci_min() -> None:
    """Sharp REWIND at theta_sci_min does not command a negative rate."""
    r = outer_rate(
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
    assert r == 0.0
    assert math.isfinite(r)
