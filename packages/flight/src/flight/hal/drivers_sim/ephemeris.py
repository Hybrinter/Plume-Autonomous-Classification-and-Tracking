"""Circular Keplerian ISS ephemeris in ECI meters (sim).

Propagates a circular orbit from TLE mean elements in EphemerisConfig. No sgp4.

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001.
"""

from __future__ import annotations

# stdlib
import math

from flight.hal.interfaces.ephemeris import IssState
from flight.libs.config import EphemerisConfig
from flight.libs.time import Clock
from flight.libs.types import FaultCode, Ok, Result

_TWO_PI = 2.0 * math.pi
_DAY_S = 86400.0


class SimIssEphemeris:
    """Circular ECI ephemeris stand-in (satisfies IssEphemeris).

    At epoch the satellite is at the ascending node. Inclination and mean motion
    come from config. Position and velocity are SI meters.
    """

    def __init__(self, clock: Clock, cfg: EphemerisConfig | None = None) -> None:
        """Store clock and mean elements.

        Args:
            clock: Injected clock (utc_s used when the caller passes now from it).
            cfg: EphemerisConfig; defaults to EphemerisConfig() if None.
        """
        self._clock = clock
        self._cfg = cfg if cfg is not None else EphemerisConfig()
        n = self._cfg.mean_motion_rev_per_day * _TWO_PI / _DAY_S
        self._n = n
        self._a = (self._cfg.mu_m3_s2 / (n * n)) ** (1.0 / 3.0)
        self._inc = math.radians(self._cfg.inclination_deg)

    def read_state(self, now_utc_s: float) -> Result[IssState, FaultCode]:
        """Return circular-orbit ISS state at now_utc_s.

        Args:
            now_utc_s: UTC seconds.

        Returns:
            Ok(IssState) in ECI meters.
        """
        dt = now_utc_s - self._cfg.epoch_utc_s
        theta = self._n * dt
        inc = self._inc
        a = self._a
        n = self._n
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        cos_i = math.cos(inc)
        sin_i = math.sin(inc)
        r = (
            a * cos_t,
            a * sin_t * cos_i,
            a * sin_t * sin_i,
        )
        v = (
            -n * a * sin_t,
            n * a * cos_t * cos_i,
            n * a * cos_t * sin_i,
        )
        return Ok(IssState(r_m=r, v_m_s=v, epoch_utc_s=now_utc_s, frame="ECI"))
