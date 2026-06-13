"""Pure command-ingress pipeline tests: decode -> CRC -> auth -> parse -> validate -> dedup."""

from flight.iss_iface.ingress import build_tc_packet, process_inbound
from flight.libs.types import AckStatus, FaultCode

_KEY = b"unit-test-key-0000000000000000000"


def _packet(
    command_id: str,
    params: dict[str, str | int | float | bool],
    source: str,
    seq: int,
    key: bytes = _KEY,
) -> bytes:
    """Build a signed TC packet for the given command fields using build_tc_packet."""
    return build_tc_packet(command_id, params, source, seq, key, apid=1)


def test_accepts_valid_signed_command() -> None:
    """A correctly signed SET_THERMAL_LIMIT command is accepted and CommandMsg is returned."""
    outcome, last_seq = process_inbound(
        _packet("SET_THERMAL_LIMIT", {"limit_c": 70.0}, "ground", 1),
        key=_KEY,
        require_auth=True,
        accepted_sources=("ground",),
        last_seq={},
    )
    assert outcome.status is AckStatus.ACCEPTED
    assert outcome.command is not None
    assert outcome.command.target == "thermal"  # stamped from the dictionary
    assert last_seq["ground"] == 1


def test_rejects_crc_corruption() -> None:
    """A packet with a corrupted byte is rejected with COMMAND_CRC_FAIL."""
    pkt = bytearray(_packet("PING", {}, "ground", 1))
    pkt[7] ^= 0xFF
    outcome, _ = process_inbound(
        bytes(pkt), key=_KEY, require_auth=True, accepted_sources=("ground",), last_seq={}
    )
    assert outcome.status is AckStatus.REJECTED
    assert outcome.fault_code is FaultCode.COMMAND_CRC_FAIL


def test_rejects_bad_hmac() -> None:
    """A packet signed with the wrong key is rejected with COMMAND_AUTH_FAIL."""
    outcome, _ = process_inbound(
        _packet("PING", {}, "ground", 1, key=b"wrong-key"),
        key=_KEY,
        require_auth=True,
        accepted_sources=("ground",),
        last_seq={},
    )
    assert outcome.status is AckStatus.REJECTED
    assert outcome.fault_code is FaultCode.COMMAND_AUTH_FAIL


def test_rejects_unknown_command() -> None:
    """A packet with an unknown command_id is rejected with COMMAND_INVALID."""
    outcome, _ = process_inbound(
        _packet("LAUNCH_NUKE", {}, "ground", 1),
        key=_KEY,
        require_auth=True,
        accepted_sources=("ground",),
        last_seq={},
    )
    assert outcome.status is AckStatus.REJECTED
    assert outcome.fault_code is FaultCode.COMMAND_INVALID


def test_rejects_replay() -> None:
    """A command with a seq <= the last seen seq for that source is rejected as a replay."""
    outcome, _ = process_inbound(
        _packet("PING", {}, "ground", 5),
        key=_KEY,
        require_auth=True,
        accepted_sources=("ground",),
        last_seq={"ground": 5},
    )
    assert outcome.status is AckStatus.REJECTED
    assert outcome.fault_code is FaultCode.COMMAND_SEQ_ERROR


def test_rejects_unaccepted_source() -> None:
    """A command from a source not in accepted_sources is rejected with COMMAND_AUTH_FAIL."""
    outcome, _ = process_inbound(
        _packet("PING", {}, "intruder", 1),
        key=_KEY,
        require_auth=True,
        accepted_sources=("ground",),
        last_seq={},
    )
    assert outcome.status is AckStatus.REJECTED
    assert outcome.fault_code is FaultCode.COMMAND_AUTH_FAIL
