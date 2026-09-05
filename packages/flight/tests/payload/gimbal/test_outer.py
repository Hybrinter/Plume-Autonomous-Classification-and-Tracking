"""Tests for the outer rate law and live smear cap."""

import math

from flight.libs.types import GimbalState
from flight.payload.gimbal.outer import outer_rate, smear_cap_rad_s


def test_smear_cap_scales_with_exposure() -> None:
    """Halving exposure doubles the smear-limited |r| cap."""
    ifov = 0.002636
    a = smear_cap_rad_s(1000.0, 1.0, ifov)
    b = smear_cap_rad_s(500.0, 1.0, ifov)
    assert abs(b - 2.0 * a) < 1e-12


def test_tracking_live_clips_to_smear() -> None:
    """TRACKING live |r| respects the live smear cap and hardware slew."""
    ifov = 0.002636
    r = outer_rate(
        omega_t_nom=0.0,
        omega_t_res=0.0,
        e_hat=math.radians(-5.0),
        k_p=8.0,
        mode=GimbalState.TRACKING,
        live=True,
        theta_g_rad=0.0,
        theta_sci_max_rad=math.radians(45.0),
        omega_hw_rad_s=math.radians(10.0),
        exposure_us=1000.0,
        max_motion_smear_px=1.0,
        ifov_band_deg_per_px=ifov,
    )
    cap = smear_cap_rad_s(1000.0, 1.0, ifov)
    assert abs(abs(r) - cap) < 1e-12


def test_cold_tracking_is_zero() -> None:
    """TRACKING before the first vision update holds r = 0."""
    r = outer_rate(
        omega_t_nom=0.1,
        omega_t_res=0.0,
        e_hat=0.05,
        k_p=8.0,
        mode=GimbalState.TRACKING,
        live=False,
        theta_g_rad=0.0,
        theta_sci_max_rad=math.radians(45.0),
        omega_hw_rad_s=math.radians(10.0),
        exposure_us=1000.0,
        max_motion_smear_px=1.0,
        ifov_band_deg_per_px=0.002636,
    )
    assert r == 0.0


def test_rewind_uses_smear_toward_limb() -> None:
    """REWIND drives toward the science limb at the smear/hardware cap."""
    ifov = 0.002636
    r = outer_rate(
        omega_t_nom=0.0,
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
        ifov_band_deg_per_px=ifov,
    )
    cap = min(smear_cap_rad_s(1000.0, 1.0, ifov), math.radians(10.0))
    assert abs(r - cap) < 1e-12
