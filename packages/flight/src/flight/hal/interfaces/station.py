"""Station data-link hardware abstraction.

Defines the StationLink protocol: the payload's interface to the ISS/station for
CCSDS Space Packet transport (byte-level). Inbound telecommands arrive as raw bytes
(TC packets); outbound telemetry/products leave as raw bytes (TM packets). All framing,
CRC, authentication, and validation live in iss_iface, not in the link driver. The link
reports its acquisition state (AOS/LOS) so the app can gate downlink draining.
"""

from typing import Protocol, runtime_checkable

from flight.libs.types import FaultCode, LinkState, Result


@runtime_checkable
class StationLink(Protocol):
    """Hardware abstraction for the ISS/station command + downlink interface.

    The link is a pure byte transport for CCSDS Space Packets: telecommands inbound,
    telemetry/products outbound. Framing, CRC, authentication, and validation live in
    iss_iface, not here.
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
