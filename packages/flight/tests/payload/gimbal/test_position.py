"""Tests for the STOW/HOME/GOTO position loop."""

import math

from flight.payload.gimbal.position import position_rate


def test_position_rate_is_proportional() -> None:
    """Unsaturated output is K_pos times elevation error."""
    r = position_rate(theta_cmd_rad=0.2, theta_g_rad=0.1, k_pos=4.0, r_max_rad_s=1.0)
    assert abs(r - 0.4) < 1e-12


def test_position_rate_saturates() -> None:
    """Large error saturates at r_max with the command sign."""
    r_max = math.radians(8.0)
    r = position_rate(
        theta_cmd_rad=math.radians(-45.0),
        theta_g_rad=0.0,
        k_pos=4.0,
        r_max_rad_s=r_max,
    )
    assert abs(r + r_max) < 1e-12
