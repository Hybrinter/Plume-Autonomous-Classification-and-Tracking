"""Simulated station link.

Replays a scripted list of inbound commands (one per receive_command() call, then
Ok(None)) and records every downlinked item for inspection. Satisfies StationLink
structurally; used by SIL and tests.
"""

from flight.libs.messages import CommandMsg, DownlinkItemMsg
from flight.libs.types import FaultCode, Ok, Result


class SimStationLink:
    """Station link that replays scripted commands and records downlinks (sim/SIL)."""

    def __init__(self, inbound: list[CommandMsg]) -> None:
        """Initialize with the inbound commands to replay, in order.

        Args:
            inbound: Commands returned one per receive_command() call, in order.
        """
        self._inbound = inbound
        self._index = 0
        self._downlinked: list[DownlinkItemMsg] = []

    def receive_command(self) -> Result[CommandMsg | None, FaultCode]:
        """Return the next scripted command, or Ok(None) once exhausted."""
        if self._index >= len(self._inbound):
            return Ok(None)
        command = self._inbound[self._index]
        self._index += 1
        return Ok(command)

    def send_downlink(self, item: DownlinkItemMsg) -> Result[None, FaultCode]:
        """Record the downlink item and return Ok(None)."""
        self._downlinked.append(item)
        return Ok(None)

    @property
    def downlinked(self) -> tuple[DownlinkItemMsg, ...]:
        """All items passed to send_downlink, in order (test/SIL inspection hook)."""
        return tuple(self._downlinked)
