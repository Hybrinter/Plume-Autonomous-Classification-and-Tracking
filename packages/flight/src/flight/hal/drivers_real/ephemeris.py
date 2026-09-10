"""Real ISS ephemeris stub. The station TLE API is not implemented.

read_state always returns Err(EPHEMERIS_FAULT). The outer loop then uses
omega_t_nom = 0.

Satisfies: REQ-AIML-GIMB-002.
"""

from __future__ import annotations

from flight.hal.interfaces.ephemeris import IssState
from flight.libs.types import Err, FaultCode, Result


class RealIssEphemeris:
    """Stub IssEphemeris. No network, no TLE parse."""

    def read_state(self, now_utc_s: float) -> Result[IssState, FaultCode]:
        """Return Err(EPHEMERIS_FAULT). The station TLE API is out of scope.

        Args:
            now_utc_s: UTC seconds (ignored).

        Returns:
            Err(EPHEMERIS_FAULT).
        """
        del now_utc_s
        return Err(FaultCode.EPHEMERIS_FAULT)
