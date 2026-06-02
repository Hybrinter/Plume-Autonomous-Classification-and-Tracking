"""Station data-link hardware abstraction.

Defines the StationLink protocol: the payload's interface to the ISS/station for
inbound commands and outbound downlink. The station owns the RF/downlink path, so this
abstraction is deliberately thin; the exact ISS data-interface wire protocol is a
deferred decision hidden behind this Protocol (real driver TBD; sim driver drives SIL).
"""

from typing import Protocol, runtime_checkable

from flight.libs.messages import CommandMsg, DownlinkItemMsg
from flight.libs.types import FaultCode, Result


@runtime_checkable
class StationLink(Protocol):
    """Hardware abstraction for the ISS/station command + downlink interface."""

    def receive_command(self) -> Result[CommandMsg | None, FaultCode]:
        """Poll for the next inbound command from the station.

        Returns:
            Result[CommandMsg | None, FaultCode]: Ok(command) when one is pending,
            Ok(None) when the inbound queue is empty, Err(FaultCode.COMM_TIMEOUT) on
            a link error.
        """
        ...

    def send_downlink(self, item: DownlinkItemMsg) -> Result[None, FaultCode]:
        """Hand a downlink item to the station for transmission to the ground."""
        ...
