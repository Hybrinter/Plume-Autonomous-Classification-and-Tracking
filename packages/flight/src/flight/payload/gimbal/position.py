"""Proportional position loop for STOW / HOME / GOTO (pure, SI).

Writes a rate reference into the same inner PI. The output is saturated at
r_max and is not smear-capped.

Satisfies: REQ-AIML-GIMB-001, REQ-GIMB-HIGH-001.
"""

from __future__ import annotations


def position_rate(
    theta_cmd_rad: float,
    theta_g_rad: float,
    k_pos: float,
    r_max_rad_s: float,
) -> float:
    """Saturated proportional rate toward a pose command.

    Inputs:
        theta_cmd_rad: Target elevation, rad.
        theta_g_rad: Current elevation, rad.
        k_pos: Position-loop gain, 1/s.
        r_max_rad_s: Rate saturation (stow/home cap), rad/s.

    Outputs:
        float: Rate reference r, rad/s.
    """
    r = k_pos * (theta_cmd_rad - theta_g_rad)
    if r > r_max_rad_s:
        return r_max_rad_s
    if r < -r_max_rad_s:
        return -r_max_rad_s
    return r
