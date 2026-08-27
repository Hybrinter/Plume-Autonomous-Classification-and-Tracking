"""Real ISS/station data-link driver: CCSDS TC over TCP in, TM over UDP out.

Inbound telecommands arrive on a TCP server socket the payload binds (one station client);
a background daemon thread accepts the client, recv-loops, and deframes the byte stream into
complete CCSDS packets (using packet_length on the 6-byte header) which receive_packet() pops
non-blocking. Outbound telemetry/products are sent as UDP datagrams to the station endpoint.
Link state is AOS while a client is connected, LOS otherwise. Sockets are stdlib (no SDK) but
open lazily in __init__; close() tears down the thread and sockets. Library methods return
Result and never raise; only bad startup config raises ValueError.

Satisfies: REQ-COMM-HIGH-001, REQ-COMM-HIGH-002.
"""

from __future__ import annotations

# stdlib
import socket
import threading
from collections import deque

# internal
from flight.libs.ccsds import CCSDS_PRIMARY_HEADER_SIZE, packet_length
from flight.libs.config import LinkConfig
from flight.libs.time import Clock
from flight.libs.types import Err, FaultCode, LinkState, Ok, Result


class RealStationLink:
    """Real CCSDS station link (TCP command-in, UDP telemetry-out). Satisfies StationLink."""

    def __init__(self, cfg: LinkConfig, clock: Clock) -> None:
        """Bind the inbound TCP server and outbound UDP socket; start the accept/recv thread.

        Args:
            cfg: Link transport config (hosts/ports/APIDs/timeout).
            clock: Injected clock (reserved for future timestamping / LOS timeouts).

        Raises:
            ValueError: if host is empty or a port is out of range (startup misconfig).
        """
        if not cfg.command_tcp_host or not cfg.telemetry_udp_host:
            raise ValueError("RealStationLink requires non-empty hosts")
        if not (1 <= cfg.command_tcp_port <= 65535 and 1 <= cfg.telemetry_udp_port <= 65535):
            raise ValueError("RealStationLink requires ports in 1..65535")
        self._cfg = cfg
        self._clock = clock
        self._lock = threading.Lock()
        self._inbound: deque[bytes] = deque()
        self._connected = False
        self._stop = threading.Event()

        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((cfg.command_tcp_host, cfg.command_tcp_port))
        self._server.listen(1)
        self._server.settimeout(cfg.socket_timeout_s)
        self._udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self._thread = threading.Thread(target=self._serve, name="station_link", daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        """Accept one client and recv-loop, deframing packets onto the inbound deque."""
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except TimeoutError, OSError:
                continue
            conn.settimeout(self._cfg.socket_timeout_s)
            with self._lock:
                self._connected = True
            self._recv_loop(conn)
            with self._lock:
                self._connected = False

    def _recv_loop(self, conn: socket.socket) -> None:
        """Read a TCP stream, splitting it into complete CCSDS packets by the length field."""
        buffer = bytearray()
        with conn:
            while not self._stop.is_set():
                try:
                    chunk = conn.recv(4096)
                except TimeoutError, OSError:
                    continue
                if not chunk:
                    return  # peer closed
                buffer.extend(chunk)
                while len(buffer) >= CCSDS_PRIMARY_HEADER_SIZE:
                    length_result = packet_length(bytes(buffer[:CCSDS_PRIMARY_HEADER_SIZE]))
                    if isinstance(length_result, Err):
                        buffer.clear()
                        break
                    total = length_result.value
                    if len(buffer) < total:
                        break
                    with self._lock:
                        self._inbound.append(bytes(buffer[:total]))
                    del buffer[:total]

    def receive_packet(self) -> Result[bytes | None, FaultCode]:
        """Pop the next deframed inbound packet, or Ok(None) if none is pending."""
        with self._lock:
            if self._inbound:
                return Ok(self._inbound.popleft())
        return Ok(None)

    def send_packet(self, packet: bytes) -> Result[None, FaultCode]:
        """Send one packet as a UDP datagram to the station telemetry endpoint."""
        try:
            self._udp.sendto(packet, (self._cfg.telemetry_udp_host, self._cfg.telemetry_udp_port))
        except OSError:
            return Err(FaultCode.COMM_TIMEOUT)
        return Ok(None)

    def link_state(self) -> LinkState:
        """Return AOS while a station client is connected, LOS otherwise."""
        with self._lock:
            return LinkState.AOS if self._connected else LinkState.LOS

    def close(self) -> None:
        """Stop the accept/recv thread and close both sockets (idempotent)."""
        self._stop.set()
        try:
            self._server.close()
        except OSError:
            pass
        try:
            self._udp.close()
        except OSError:
            pass
        self._thread.join(timeout=2.0)
