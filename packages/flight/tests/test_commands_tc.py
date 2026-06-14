"""Locality + round-trip test for the relocated build_tc_packet helper."""

from flight.iss_iface.ingress.pipeline import process_inbound
from flight.libs.commands import build_tc_packet as build_tc_packet_pkg
from flight.libs.commands.tc import build_tc_packet
from flight.libs.types import AckStatus

_KEY = b"unit-test-key-0000000000000000000"


def test_build_tc_packet_is_exported_from_commands_package() -> None:
    """The package re-export and the submodule resolve to the same function object."""
    assert build_tc_packet_pkg is build_tc_packet


def test_build_tc_packet_roundtrips_through_process_inbound() -> None:
    """A packet built by the relocated helper is accepted by the ingress pipeline."""
    pkt = build_tc_packet("SET_THERMAL_LIMIT", {"limit_c": 70.0}, "ground", 1, _KEY, apid=1)
    outcome, _ = process_inbound(
        pkt, key=_KEY, require_auth=True, accepted_sources=("ground",), last_seq={}
    )
    assert outcome.status is AckStatus.ACCEPTED
    assert outcome.command is not None
    assert outcome.command_id == "SET_THERMAL_LIMIT"
