"""Station data-link hardware abstraction.

Defines the StationLink protocol: the payload's interface to the ISS/station for
CCSDS Space Packet transport (byte-level). Inbound telecommands arrive as raw bytes
(TC packets); outbound telemetry/products leave as raw bytes (TM packets). All framing,
CRC, authentication, and validation live in iss_iface, not in the link driver. The link
reports its acquisition state (AOS/LOS) so the app can gate downlink draining. The
legacy command-level methods (receive_command/send_downlink) are retained transitionally
and are removed once iss_iface migrates to the byte-level API (Phase 6A Task 8).
"""

from typing import Protocol, runtime_checkable

from flight.libs.messages import CommandMsg, DownlinkItemMsg
from flight.libs.types import FaultCode, LinkState, Result


@runtime_checkable
class StationLink(Protocol):
    """Hardware abstraction for the ISS/station command + downlink interface.

    The link is a pure byte transport for CCSDS Space Packets: telecommands inbound,
    telemetry/products outbound. Framing, CRC, authentication, and validation live in
    iss_iface, not here. (The legacy command-level methods are retained transitionally and
    are removed once iss_iface migrates to the byte-level API.)
    """

    def receive_packet(self) -> Result[bytes | None, FaultCode]:
        """Pop the next complete inbound CCSDS packet, or Ok(None) if none is pending."""
        ...

    def send_packet(self, packet: bytes) -> Result[None, FaultCode]:
        """Transmit one complete outbound CCSDS packet (bytes) to the station."""
        ...

    def link_state(self) -> LinkState:
        """Return the current AOS/LOS acquisition state."""
        ...

    def close(self) -> None:
        """Release any sockets/threads. Safe to call multiple times."""
        ...

    def receive_command(self) -> Result[CommandMsg | None, FaultCode]:
        """Deprecated (removed once iss_iface migrates); legacy command-level uplink."""
        ...

    def send_downlink(self, item: DownlinkItemMsg) -> Result[None, FaultCode]:
        """Deprecated (removed once iss_iface migrates); legacy command-level downlink."""
        ...
