"""Ground-station emulator: TCP command client + UDP telemetry receiver for the flight link.

StationEmulator is the test-side counterpart to the flight RealStationLink. RealStationLink
binds a TCP SERVER for inbound telecommands and SENDS UDP telemetry; so the emulator connects
a TCP CLIENT to that server (to push signed TC packets) and binds a UDP RECEIVER on the
telemetry endpoint (to drain downlinked TM). Commands are framed + signed by the flight
build_tc_packet so the flight ingress pipeline authenticates them identically to the real
ground segment. This is GSE test tooling (not flight library code): methods raise on misuse
rather than returning Result.

Contains:
  - StationEmulator: connect / send_command / poll_downlink / close.

Satisfies: REQ-VAL-GSE-001.
"""

from __future__ import annotations

# stdlib
import socket

# internal
from flight.libs.commands import build_tc_packet


class StationEmulator:
    """Emulated ISS ground station: signs + sends telecommands, receives telemetry datagrams."""

    def __init__(
        self,
        tcp_host: str,
        tcp_port: int,
        udp_host: str,
        udp_port: int,
        key: bytes,
        tc_apid: int,
    ) -> None:
        """Record the flight link endpoints and the shared HMAC key (no sockets opened yet).

        Args:
            tcp_host: Host of the flight link's inbound TC server (the emulator connects here).
            tcp_port: TCP port of the flight link's inbound TC server.
            udp_host: Host the emulator binds to receive outbound telemetry datagrams.
            udp_port: UDP port the emulator binds to receive outbound telemetry datagrams.
            key: The shared HMAC-SHA256 secret used to sign telecommands.
            tc_apid: The CCSDS APID stamped into outbound telecommand packets.

        Notes:
            Sockets are opened in connect(), not here, so an unconnected emulator is inert.
        """
        self._tcp_host = tcp_host
        self._tcp_port = tcp_port
        self._udp_host = udp_host
        self._udp_port = udp_port
        self._key = key
        self._tc_apid = tc_apid
        self._tcp: socket.socket | None = None
        self._udp: socket.socket | None = None

    def connect(self) -> None:
        """Open the TCP client to the flight TC server and bind the UDP telemetry receiver.

        Raises:
            OSError: if the TCP connect or the UDP bind fails (test-setup error).
            RuntimeError: if called when already connected.
        """
        if self._tcp is not None or self._udp is not None:
            raise RuntimeError("StationEmulator is already connected")
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        udp.bind((self._udp_host, self._udp_port))
        udp.setblocking(False)
        self._udp = udp
        tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp.connect((self._tcp_host, self._tcp_port))
        self._tcp = tcp

    def send_command(
        self,
        command_id: str,
        params: dict[str, str | int | float | bool],
        source: str,
        seq: int,
    ) -> None:
        """Frame, sign, and transmit one telecommand over the TCP connection to the flight link.

        Args:
            command_id: The command opcode string (e.g. "SET_THERMAL_LIMIT", "PING").
            params: The command parameter dict.
            source: The command origin identifier (must be on the flight allow-list to accept).
            seq: The per-source monotonic sequence number.

        Returns:
            None.

        Raises:
            RuntimeError: if called before connect().
            ValueError: if build_tc_packet rejects a field (propagated from the flight builder).
        """
        if self._tcp is None:
            raise RuntimeError("StationEmulator.connect() must be called before send_command()")
        packet = build_tc_packet(command_id, params, source, seq, self._key, self._tc_apid)
        self._tcp.sendall(packet)

    def poll_downlink(self, timeout_s: float = 0.5) -> list[bytes]:
        """Drain all telemetry datagrams currently waiting on the UDP receiver.

        Args:
            timeout_s: Total wall-clock budget to wait for at least one datagram (seconds).

        Returns:
            A list of received datagram payloads (may be empty if none arrived in the budget).

        Raises:
            RuntimeError: if called before connect().

        Notes:
            Blocks up to timeout_s for the first datagram, then drains any others non-blocking,
            so a TM sent just before the call is reliably captured without busy-spinning.
        """
        if self._udp is None:
            raise RuntimeError("StationEmulator.connect() must be called before poll_downlink()")
        self._udp.settimeout(timeout_s)
        received: list[bytes] = []
        try:
            received.append(self._udp.recv(65535))
        except TimeoutError, BlockingIOError, OSError:
            return received
        self._udp.setblocking(False)
        while True:
            try:
                received.append(self._udp.recv(65535))
            except BlockingIOError, OSError:
                break
        return received

    def close(self) -> None:
        """Close both sockets (idempotent; safe to call without a prior connect())."""
        for sock in (self._tcp, self._udp):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        self._tcp = None
        self._udp = None
