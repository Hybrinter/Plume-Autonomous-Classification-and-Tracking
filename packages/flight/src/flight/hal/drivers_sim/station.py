"""Simulated station link (byte-level + legacy command-level during migration).

Replays scripted inbound CCSDS packets (one per receive_packet() call, then Ok(None)) and
records every outbound packet. Link state is scriptable (defaults AOS). Retains the legacy
receive_command/send_downlink during the iss_iface migration. Satisfies StationLink
structurally; used by SIL and tests.
"""

from flight.libs.messages import CommandMsg, DownlinkItemMsg
from flight.libs.types import FaultCode, LinkState, Ok, Result


class SimStationLink:
    """Station link replaying scripted packets and recording outbound packets (sim/SIL)."""

    def __init__(
        self, inbound: list[bytes] | None = None, link_state: LinkState = LinkState.AOS
    ) -> None:
        """Initialize with inbound packets to replay, in order, and a fixed link state.

        Args:
            inbound: CCSDS packets returned one per receive_packet() call, in order.
            link_state: The AOS/LOS state link_state() reports (default AOS).
        """
        self._inbound: list[bytes] = list(inbound) if inbound is not None else []
        self._index = 0
        self._sent: list[bytes] = []
        self._link_state = link_state

    def enqueue(self, packet: bytes) -> None:
        """Append an inbound packet to be returned by a later receive_packet() call."""
        self._inbound.append(packet)

    def set_link_state(self, state: LinkState) -> None:
        """Set the AOS/LOS state reported by link_state() (test/SIL hook)."""
        self._link_state = state

    def receive_packet(self) -> Result[bytes | None, FaultCode]:
        """Return the next scripted inbound packet, or Ok(None) once exhausted."""
        if self._index >= len(self._inbound):
            return Ok(None)
        packet = self._inbound[self._index]
        self._index += 1
        return Ok(packet)

    def send_packet(self, packet: bytes) -> Result[None, FaultCode]:
        """Record the outbound packet and return Ok(None)."""
        self._sent.append(packet)
        return Ok(None)

    def link_state(self) -> LinkState:
        """Return the scripted AOS/LOS state."""
        return self._link_state

    def close(self) -> None:
        """No-op for the sim link."""

    @property
    def sent(self) -> tuple[bytes, ...]:
        """All packets passed to send_packet, in order (test/SIL inspection hook)."""
        return tuple(self._sent)

    # --- legacy command-level API (removed in Task 8) ---
    def receive_command(self) -> Result[CommandMsg | None, FaultCode]:
        """Legacy no-op during migration: always Ok(None)."""
        return Ok(None)

    def send_downlink(self, item: DownlinkItemMsg) -> Result[None, FaultCode]:
        """Legacy no-op during migration: accept and drop."""
        return Ok(None)
