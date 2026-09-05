"""Elevation-controller analysis helpers: rigid-body plant matching flight SimGimbal.

The study imports flight pure cores. This module only integrates
J * omega_dot + B * omega = tau in SI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ElevationPlant:
    """Semi-implicit Euler plant used by the elevation-controller block tests."""

    j_kg_m2: float = 0.008
    b_nms_per_rad: float = 0.04
    tau_max_nm: float = 1.0
    omega_max_rad_s: float = math.radians(10.0)
    theta_min_rad: float = math.radians(-45.0)
    theta_max_rad: float = math.radians(90.0)
    theta_rad: float = 0.0
    omega_rad_s: float = 0.0

    def step(self, tau_nm: float, dt_s: float) -> None:
        """Advance one inner period with clipped torque."""
        if dt_s <= 0.0:
            return
        tau = min(max(tau_nm, -self.tau_max_nm), self.tau_max_nm)
        omega_dot = (tau - self.b_nms_per_rad * self.omega_rad_s) / self.j_kg_m2
        omega = self.omega_rad_s + omega_dot * dt_s
        omega = min(max(omega, -self.omega_max_rad_s), self.omega_max_rad_s)
        theta = self.theta_rad + omega * dt_s
        if theta > self.theta_max_rad:
            theta = self.theta_max_rad
            if omega > 0.0:
                omega = 0.0
        elif theta < self.theta_min_rad:
            theta = self.theta_min_rad
            if omega < 0.0:
                omega = 0.0
        self.theta_rad = theta
        self.omega_rad_s = omega
