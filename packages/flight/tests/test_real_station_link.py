"""RealStationLink loopback-socket integration tests (real TCP/UDP, localhost)."""

import socket
import time

from flight.hal.drivers_real import RealStationLink
from flight.libs.ccsds import CcsdsHeader, encode_packet
from flight.libs.config import LinkConfig
from flight.libs.time import ManualClock
from flight.libs.types import LinkState, Ok


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_receive_packet_deframes_tcp_stream() -> None:
    """RealStationLink deframes two back-to-back CCSDS packets from a TCP stream."""
    cfg = LinkConfig(
        command_tcp_host="127.0.0.1",
        command_tcp_port=_free_port(),
        telemetry_udp_host="127.0.0.1",
        telemetry_udp_port=_free_port(),
        socket_timeout_s=0.5,
    )
    link = RealStationLink(cfg=cfg, clock=ManualClock())
    try:
        # Before any client connects: LOS, no command.
        assert link.link_state() is LinkState.LOS
        assert isinstance(link.receive_command(), Ok)  # legacy method still present, Ok(None)
        client = socket.create_connection((cfg.command_tcp_host, cfg.command_tcp_port), timeout=2.0)
        header = CcsdsHeader(packet_type=1, apid=cfg.tc_apid, sequence_count=0)
        encoded = encode_packet(header, b"hi")
        assert isinstance(encoded, Ok)
        pkt = encoded.value
        client.sendall(pkt + pkt)  # two packets back-to-back to prove deframing
        # poll until both packets surface (daemon thread processes data asynchronously)
        seen = []
        for _ in range(100):
            result = link.receive_packet()
            assert isinstance(result, Ok)
            if result.value is not None:
                seen.append(result.value)
            if len(seen) == 2:
                break
            time.sleep(0.02)
        assert seen == [pkt, pkt]
        assert link.link_state() is LinkState.AOS
        client.close()
    finally:
        link.close()


def test_send_packet_emits_udp() -> None:
    """RealStationLink sends bytes as a UDP datagram to the telemetry endpoint."""
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(2.0)
    udp_port = int(rx.getsockname()[1])
    cfg = LinkConfig(
        command_tcp_host="127.0.0.1",
        command_tcp_port=_free_port(),
        telemetry_udp_host="127.0.0.1",
        telemetry_udp_port=udp_port,
        socket_timeout_s=0.5,
    )
    link = RealStationLink(cfg=cfg, clock=ManualClock())
    try:
        payload = b"telemetry-bytes"
        assert isinstance(link.send_packet(payload), Ok)
        data, _ = rx.recvfrom(4096)
        assert data == payload
    finally:
        link.close()
        rx.close()
