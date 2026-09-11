"""Unit tests for analysis.lib.hunt -- rewind/limb Poisson waits and scan rate."""

import math

from analysis.lib.hunt import (
    az_scan_rate_deg_s,
    d_look_ahead_d_eta_km_per_rad,
    ground_central_angle_rad,
    look_ahead_km,
    reacquire,
    two_phase_mean_wait_s,
)
from analysis.lib.optics import band_gsd_along_m, build_optics
from analysis.lib.orbit import build_orbit
from analysis.studies.single_axis_vs_dual_axis_gimbal.assumptions import (
    DESIGN_LAT_DEG,
    GIMBAL_BOX,
    OPTICS_SPEC,
    TLE,
)


def test_two_phase_wait_single_rate() -> None:
    """A zero-length first phase reduces to 1/lambda_b."""
    assert two_phase_mean_wait_s(0.0, 0.0, 0.5) == 2.0


def test_two_phase_wait_finds_during_rewind() -> None:
    """High rewind rate and long rewind -> mean wait ~ 1/lambda_a."""
    wait = two_phase_mean_wait_s(1.0, 100.0, 0.01)
    assert abs(wait - 1.0) < 0.02


def test_two_phase_wait_misses_rewind() -> None:
    """Zero rewind rate -> t_a + 1/lambda_b."""
    wait = two_phase_mean_wait_s(0.0, 10.0, 0.1)
    assert abs(wait - 20.0) < 1e-9


def test_two_phase_wait_both_zero_is_inf() -> None:
    """No search at all is an infinite wait."""
    assert math.isinf(two_phase_mean_wait_s(0.0, 5.0, 0.0))


def test_az_scan_rate_uses_smear_budget() -> None:
    """Leftover smear budget after along-track scene rate is the az cap."""
    rate = az_scan_rate_deg_s(2.5, 1.5)
    assert abs(rate - math.sqrt(2.5**2 - 1.5**2)) < 1e-9


def test_az_scan_rate_zero_when_scene_exceeds_gate() -> None:
    """No science-frame az raster if along-track scene rate uses the gate."""
    assert az_scan_rate_deg_s(1.0, 1.2) == 0.0


def test_nadir_look_ahead_is_zero() -> None:
    """Nadir look-ahead ground range is zero."""
    assert look_ahead_km(0.0, 6800.0, 6371.0) == 0.0


def test_central_angle_increases_with_eta() -> None:
    """Farther off-nadir look points are farther along-track."""
    r_iss, r_earth = 6800.0, 6371.0
    near = ground_central_angle_rad(20.0, r_iss, r_earth)
    far = ground_central_angle_rad(45.0, r_iss, r_earth)
    assert far > near > 0.0


def test_band_gsd_grows_with_incidence() -> None:
    """Along-track GSD at 45 deg incidence is larger than nadir."""
    nadir = band_gsd_along_m(0.00264, 433.0, 0.0)
    off = band_gsd_along_m(0.00264, 635.0, 49.0)
    assert off > nadir
    assert nadir < 25.0
    assert off < 55.0


def test_full_window_dwell_rewinds_at_imaging_rate() -> None:
    """Loss at nadir slews the remaining elevation at omega_img, not t_window."""
    optics = build_optics(OPTICS_SPEC)
    orbit = build_orbit(TLE, use_perigee=False)
    omega_img = 1.725
    window_s = 65.0
    result = reacquire(
        DESIGN_LAT_DEG,
        1.0e-6,
        optics,
        orbit,
        GIMBAL_BOX,
        t_dwell_1_s=window_s,
        t_dwell_2_s=window_s,
        t_window_s=window_s,
        omega_img_deg_s=omega_img,
        omega_rel_deg_s=2.636,
    )
    expected = (GIMBAL_BOX.el_nadir_deg - GIMBAL_BOX.el_limb_deg) / omega_img
    assert abs(result.t_rewind_1_s - expected) < 0.05
    assert abs(result.t_rewind_2_s - expected) < 0.05
    assert result.omega_az_scan_deg_s > 0.0
    assert 0.0 < result.scan_fill <= 1.0
    assert result.swath_box_km > result.swath_fov_km


def test_loss_at_limb_has_zero_rewind() -> None:
    """A target lost at the start of the window is already at the limb stop."""
    optics = build_optics(OPTICS_SPEC)
    orbit = build_orbit(TLE, use_perigee=False)
    result = reacquire(
        DESIGN_LAT_DEG,
        1.0e-6,
        optics,
        orbit,
        GIMBAL_BOX,
        t_dwell_1_s=0.0,
        t_dwell_2_s=0.0,
        t_window_s=65.0,
        omega_img_deg_s=1.725,
        omega_rel_deg_s=2.636,
    )
    assert result.t_rewind_1_s == 0.0
    assert result.t_rewind_2_s == 0.0


def test_rewind_look_point_outpaces_iss_ground_speed() -> None:
    """ds/d(eta) is positive, so slewing toward the limb covers ground faster."""
    orbit = build_orbit(TLE, use_perigee=False)
    r_earth = orbit.earth_radius_km(DESIGN_LAT_DEG)
    mid_eta = 0.5 * (90.0 - GIMBAL_BOX.el_limb_deg)
    ds_deta = d_look_ahead_d_eta_km_per_rad(mid_eta, orbit.radius_km, r_earth)
    v_g = orbit.ground_speed_km_s(DESIGN_LAT_DEG)
    v_rewind = v_g + ds_deta * math.radians(1.725)
    assert look_ahead_km(45.0, orbit.radius_km, r_earth) > 0.0
    assert ds_deta > 0.0
    assert v_rewind > v_g
