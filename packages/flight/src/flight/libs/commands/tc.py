"""Signed CCSDS telecommand packet builder (for GSE / sim / tests; not used in flight).

Contains:
  - build_tc_packet: construct an HMAC-signed, CRC-framed TC packet from command fields.

This helper lives in flight.libs.commands so that out-of-tree command tooling (the GSE station
emulator, the SIL harness, and tests) can build authenticated telecommands while importing only
flight.libs -- never flight.iss_iface. It is the only command-side function permitted to raise
(at build/test time, on an encode failure); the runtime ingress path stays Result/Outcome-typed.

Satisfies: REQ-COMM-HIGH-003, REQ-COMM-HIGH-004.
"""

from __future__ import annotations

# stdlib
import hashlib
import hmac
import json

# internal
from flight.libs.ccsds import CcsdsHeader, encode_packet
from flight.libs.types import Err


def build_tc_packet(
    command_id: str,
    params: dict[str, str | int | float | bool],
    source: str,
    seq: int,
    key: bytes,
    apid: int,
) -> bytes:
    """Construct a signed CCSDS telecommand packet (for GSE / sim / tests; not used in flight).

    Args:
        command_id: The command opcode string.
        params: The command parameters dict.
        source: The command origin identifier string.
        seq: The per-source monotonic sequence number.
        key: The shared HMAC-SHA256 secret.
        apid: The telecommand APID.

    Returns:
        The framed TC packet bytes (header + body + HMAC tag + CRC trailer).

    Notes:
        params is JSON-serialized with sorted keys so the signed bytes are deterministic.
        Raises ValueError if encode_packet rejects a field (test/build-time error only).
    """
    body = json.dumps(
        {"command_id": command_id, "params": params, "source": source, "seq": seq},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    tag = hmac.new(key, body, hashlib.sha256).digest()
    encoded = encode_packet(
        CcsdsHeader(packet_type=1, apid=apid, sequence_count=seq & 0x3FFF), body + tag
    )
    if isinstance(encoded, Err):
        raise ValueError(f"could not encode TC packet: {encoded.error}")  # test helper only
    return encoded.value
