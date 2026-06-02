"""Real ISS/station data-link driver (stub).

The exact station avionics data interface is a deferred design decision, so this stub
satisfies the StationLink protocol with inert no-ops (no pending command; downlinks
accepted) until the interface is defined. Tests and CI use SimStationLink.
"""

from flight.libs.messages import CommandMsg, DownlinkItemMsg
from flight.libs.types import FaultCode, Ok, Result


class RealStationLink:
    """ISS/station data-link driver (stub). Satisfies StationLink; inert until defined."""

    def receive_command(self) -> Result[CommandMsg | None, FaultCode]:
        """Return Ok(None): no command source wired yet (stub)."""
        return Ok(None)

    def send_downlink(self, item: DownlinkItemMsg) -> Result[None, FaultCode]:
        """Accept and drop the downlink item (stub)."""
        return Ok(None)
