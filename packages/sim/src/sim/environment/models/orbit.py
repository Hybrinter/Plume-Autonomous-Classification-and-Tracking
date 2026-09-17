"""Truth-orbit models. Circular Keplerian matches SimIssEphemeris (SI meters).

At epoch the satellite is at the ascending node. Do not import this module from
flight HAL; the sim driver keeps its own copy until a shared test vector proves
equality.
"""

from __future__ import annotations

# stdlib
import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# internal
from flight.hal.interfaces.ephemeris import IssState
from flight.libs.config import EphemerisConfig

_TWO_PI = 2.0 * math.pi
_DAY_S = 86400.0


@runtime_checkable
class OrbitModel(Protocol):
    """Truth ISS ECI state as a function of UTC."""

    def state_eci(self, utc_s: float) -> IssState:
        """Return ISS ECI position and velocity at utc_s."""
        ...


@dataclass(frozen=True, slots=True)
class CircularKepler:
    """Circular two-body orbit; same kinematics as SimIssEphemeris."""

    inclination_deg: float
    mean_motion_rev_per_day: float
    mu_m3_s2: float
    epoch_utc_s: float

    @classmethod
    def from_ephemeris_config(cls, cfg: EphemerisConfig) -> CircularKepler:
        """Copy mean elements from flight EphemerisConfig."""
        return cls(
            inclination_deg=cfg.inclination_deg,
            mean_motion_rev_per_day=cfg.mean_motion_rev_per_day,
            mu_m3_s2=cfg.mu_m3_s2,
            epoch_utc_s=cfg.epoch_utc_s,
        )

    def state_eci(self, utc_s: float) -> IssState:
        """Return circular-orbit ISS state at utc_s (ascending node at epoch)."""
        n = self.mean_motion_rev_per_day * _TWO_PI / _DAY_S
        sma = (self.mu_m3_s2 / (n * n)) ** (1.0 / 3.0)
        dt = utc_s - self.epoch_utc_s
        theta = n * dt
        inc = math.radians(self.inclination_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        cos_i, sin_i = math.cos(inc), math.sin(inc)
        r = (
            sma * cos_t,
            sma * sin_t * cos_i,
            sma * sin_t * sin_i,
        )
        v = (
            -n * sma * sin_t,
            n * sma * cos_t * cos_i,
            n * sma * cos_t * sin_i,
        )
        return IssState(r_m=r, v_m_s=v, epoch_utc_s=utc_s, frame="ECI")
