"""Conformance + behavior tests for the StationLink HAL and its drivers."""

import socket

from flight.hal.drivers_real import RealStationLink
from flight.hal.drivers_sim import SimStationLink
from flight.hal.interfaces import StationLink
from flight.libs.ccsds import CcsdsHeader, encode_packet
from flight.libs.config import LinkConfig
from flight.libs.time import ManualClock
from flight.libs.types import LinkState, Ok


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_sim_station_link_satisfies_protocol() -> None:
    """SimStationLink conforms to StationLink (typed + runtime)."""
    link: StationLink = SimStationLink()
    assert isinstance(link, StationLink)


def test_real_station_link_satisfies_protocol() -> None:
    """RealStationLink conforms to StationLink and constructs with cfg + clock."""
    cfg = LinkConfig(
        command_tcp_host="127.0.0.1",
        command_tcp_port=_free_port(),
        telemetry_udp_host="127.0.0.1",
        telemetry_udp_port=_free_port(),
        socket_timeout_s=0.5,
    )
    link: StationLink = RealStationLink(cfg=cfg, clock=ManualClock())
    try:
        assert isinstance(link, StationLink)
    finally:
        link.close()


def test_sim_receives_scripted_packets_in_order_then_none() -> None:
    """receive_packet yields each scripted packet once, then Ok(None)."""
    r1 = encode_packet(CcsdsHeader(packet_type=1, apid=1, sequence_count=1), b"cmd1")
    r2 = encode_packet(CcsdsHeader(packet_type=1, apid=1, sequence_count=2), b"cmd2")
    assert isinstance(r1, Ok) and isinstance(r2, Ok)
    pkt1, pkt2 = r1.value, r2.value
    link = SimStationLink([pkt1, pkt2])
    first = link.receive_packet()
    second = link.receive_packet()
    third = link.receive_packet()
    assert isinstance(first, Ok) and first.value == pkt1
    assert isinstance(second, Ok) and second.value == pkt2
    assert isinstance(third, Ok) and third.value is None


def test_sim_records_sent_packets() -> None:
    """send_packet records each packet; sent exposes them in order."""
    link = SimStationLink()
    payload = b"telemetry"
    result = link.send_packet(payload)
    assert isinstance(result, Ok)
    assert len(link.sent) == 1
    assert link.sent[0] == payload


def test_sim_link_state_scriptable() -> None:
    """link_state returns the state set at construction or via set_link_state."""
    link_aos = SimStationLink(link_state=LinkState.AOS)
    assert link_aos.link_state() is LinkState.AOS
    link_los = SimStationLink(link_state=LinkState.LOS)
    assert link_los.link_state() is LinkState.LOS
    link_aos.set_link_state(LinkState.LOS)
    assert link_aos.link_state() is LinkState.LOS
