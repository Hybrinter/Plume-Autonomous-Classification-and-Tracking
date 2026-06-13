"""Pure command-ingress pipeline for iss_iface: bytes -> validated CommandMsg or rejection.

Stages, in order: CCSDS decode + CRC (flight.libs.ccsds) -> JSON parse -> HMAC-SHA256
authentication over the command body -> source allow-list -> typed dictionary validation
(flight.libs.commands) -> monotonic per-source sequence dedup (replay guard). The functions are
pure: no bus, no clock, no I/O, no logging. The app shell owns the HMAC key, the per-source
last-seq map, the clock, and the bus, and turns an IngressOutcome into a CommandMsg + an
always-emitted CommandAckMsg.

Wire format (TC): [CCSDS header type=1] [body = JSON {command_id, params, source, seq}]
[HMAC-SHA256 tag, 32 bytes] [CRC-32 trailer]. The dictionary stamps the canonical target;
the ground frame does not carry target.

Contains:
  - IngressOutcome: the per-packet result (command-or-None + ack status + reason + echo).
  - build_tc_packet: construct a signed TC packet (used by GSE/sim/tests, not flight).
  - process_inbound: run the full pipeline for one raw packet (Result-free; outcome-typed).

Satisfies: REQ-COMM-HIGH-003, REQ-COMM-HIGH-004.
"""

from __future__ import annotations

# stdlib
import hashlib
import hmac
import json
from dataclasses import dataclass

# internal
from flight.libs.ccsds import CcsdsHeader, decode_packet, encode_packet
from flight.libs.commands import lookup_command, validate_command
from flight.libs.messages import CommandMsg
from flight.libs.types import AckStatus, Err, FaultCode, MessageType

_HMAC_TAG_SIZE = 32  # SHA-256 digest length


@dataclass(slots=True)
class IngressOutcome:
    """Result of running one inbound packet through the ingress pipeline.

    Fields:
        command: The validated CommandMsg to publish, or None if rejected.
        status: ACCEPTED or REJECTED.
        fault_code: NONE on accept; the reject reason otherwise.
        command_id: Echoed opcode string ("" if the body was unparseable).
        source: Echoed origin ("" if unparseable).
        seq: Echoed sequence number (-1 if unparseable).
        detail: Human-readable context for the ack/fault.
    """

    command: CommandMsg | None
    status: AckStatus
    fault_code: FaultCode
    command_id: str
    source: str
    seq: int
    detail: str


def _reject(code: FaultCode, detail: str, command_id: str, source: str, seq: int) -> IngressOutcome:
    """Construct a REJECTED IngressOutcome with the given fault code and echo fields."""
    return IngressOutcome(None, AckStatus.REJECTED, code, command_id, source, seq, detail)


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


def process_inbound(
    raw: bytes,
    key: bytes,
    require_auth: bool,
    accepted_sources: tuple[str, ...],
    last_seq: dict[str, int],
) -> tuple[IngressOutcome, dict[str, int]]:
    """Run one raw inbound packet through the full ingress pipeline.

    Args:
        raw: The complete framed CCSDS TC packet.
        key: The shared HMAC-SHA256 secret.
        require_auth: If False, skip the HMAC check (bench/test only).
        accepted_sources: The allow-list of command origins.
        last_seq: Per-source last-accepted sequence map (threaded state, not mutated here).

    Returns:
        (outcome, new_last_seq). On ACCEPTED, new_last_seq[source] is updated to the command's
        seq; on REJECTED, last_seq is returned unchanged. No exceptions: malformed input maps
        to a REJECTED outcome with the appropriate FaultCode.
    """
    decoded = decode_packet(raw)
    if isinstance(decoded, Err):
        return _reject(decoded.error, "ccsds decode/crc failed", "", "", -1), last_seq
    _header, data = decoded.value

    if len(data) < _HMAC_TAG_SIZE:
        return _reject(FaultCode.COMMAND_AUTH_FAIL, "missing hmac tag", "", "", -1), last_seq
    body, tag = data[:-_HMAC_TAG_SIZE], data[-_HMAC_TAG_SIZE:]

    try:
        fields = json.loads(body.decode("utf-8"))
        command_id = str(fields["command_id"])
        params = dict(fields["params"])
        source = str(fields["source"])
        seq = int(fields["seq"])
    except ValueError, KeyError, TypeError:
        return _reject(FaultCode.COMMAND_INVALID, "malformed command body", "", "", -1), last_seq

    if require_auth:
        expected = hmac.new(key, body, hashlib.sha256).digest()
        if not hmac.compare_digest(expected, tag):
            return (
                _reject(FaultCode.COMMAND_AUTH_FAIL, "hmac mismatch", command_id, source, seq),
                last_seq,
            )
    if source not in accepted_sources:
        return (
            _reject(FaultCode.COMMAND_AUTH_FAIL, "source not accepted", command_id, source, seq),
            last_seq,
        )

    spec_result = lookup_command(command_id)
    if isinstance(spec_result, Err):
        return _reject(spec_result.error, "unknown command", command_id, source, seq), last_seq
    spec = spec_result.value
    valid = validate_command(spec, params)
    if isinstance(valid, Err):
        return _reject(valid.error, "param validation failed", command_id, source, seq), last_seq

    if seq <= last_seq.get(source, -1):
        return (
            _reject(FaultCode.COMMAND_SEQ_ERROR, "replay/duplicate seq", command_id, source, seq),
            last_seq,
        )

    command = CommandMsg(
        msg_type=MessageType.COMMAND,
        timestamp_utc="",  # the shell stamps this with the clock
        target=spec.target,
        command_id=command_id,
        params=params,
        source=source,
        seq=seq,
    )
    new_last_seq = dict(last_seq)
    new_last_seq[source] = seq
    outcome = IngressOutcome(
        command, AckStatus.ACCEPTED, FaultCode.NONE, command_id, source, seq, ""
    )
    return outcome, new_last_seq
