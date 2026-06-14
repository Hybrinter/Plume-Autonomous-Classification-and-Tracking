"""StationEmulator round-trip against a real flight RealStationLink over loopback sockets."""

import dataclasses
import socket
import time

from flight.hal.drivers_real.station import RealStationLink
from flight.libs.config import LinkConfig
from flight.libs.time import RealClock
from flight.libs.types import Ok
from gse.station import StationEmulator

_KEY = b"sil-test-key-0000000000000000000"


def _free_port() -> int:
    """Return an OS-assigned free TCP port (released immediately; race window is tiny)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port: int = s.getsockname()[1]
    s.close()
    return port


def test_station_emulator_round_trip() -> None:
    """A signed command reaches the flight link; a TM datagram reaches the emulator."""
    tc_port = _free_port()
    udp_port = _free_port()
    cfg = dataclasses.replace(
        LinkConfig(),
        command_tcp_host="127.0.0.1",
        command_tcp_port=tc_port,
        telemetry_udp_host="127.0.0.1",
        telemetry_udp_port=udp_port,
    )
    link = RealStationLink(cfg, RealClock())
    emulator = StationEmulator(
        tcp_host="127.0.0.1",
        tcp_port=tc_port,
        udp_host="127.0.0.1",
        udp_port=udp_port,
        key=_KEY,
        tc_apid=cfg.tc_apid,
    )
    try:
        emulator.connect()
        emulator.send_command("SET_THERMAL_LIMIT", {"limit_c": 70.0}, "ground", 1)

        received: bytes | None = None
        for _ in range(50):  # allow the daemon recv/deframe thread to land the packet
            popped = link.receive_packet()
            assert isinstance(popped, Ok)
            if popped.value is not None:
                received = popped.value
                break
            time.sleep(0.02)
        assert received is not None
        assert len(received) > 0

        sent = link.send_packet(b"<tm-downlink>")
        assert isinstance(sent, Ok)
        drained = emulator.poll_downlink(timeout_s=0.5)
        assert b"<tm-downlink>" in drained
    finally:
        emulator.close()
        link.close()
