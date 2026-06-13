"""CCSDS 133.0-B-2 Space Packet codec with a CRC-32 integrity trailer (pure, stdlib).

Encodes/decodes the 6-byte primary header (big-endian, three 16-bit words) and appends a
4-byte CRC-32 trailer over the whole packet so a decoded packet is self-validating. The
codec is transport-agnostic: it never touches sockets. iss_iface uses it to frame/deframe
command (TC, packet_type=1) and telemetry (TM, packet_type=0) packets; the real station
driver uses packet_length() to deframe a TCP byte stream into discrete packets.

Bit layout (mirrors the standard): word1 = version(3, 0) | type(1) | sec_hdr(1, 0) |
apid(11); word2 = seq_flags(2, 0b11 standalone) | seq_count(14); word3 = data_length(16)
= (len(body) + CRC_TRAILER_SIZE) - 1. CRC-32 = binascii.crc32 & 0xFFFFFFFF (ISO-3309/zlib).

Contains:
  - CcsdsHeader: decoded primary-header fields used by callers.
  - compute_crc32 / verify_crc32: the integrity primitive.
  - encode_packet: header + body -> framed bytes with CRC trailer (Result, never raises).
  - decode_packet: framed bytes -> (header, body), CRC-verified (Result).
  - packet_length: total packet size from the first 6 header bytes (for stream deframing).

Satisfies: REQ-COMM-HIGH-002.
"""

from __future__ import annotations

# stdlib
import binascii
import struct
from dataclasses import dataclass

# internal
from flight.libs.types import Err, FaultCode, Ok, Result

CCSDS_PRIMARY_HEADER_SIZE = 6
CRC_TRAILER_SIZE = 4
APID_MAX = 0x7FF
SEQ_COUNT_MAX = 0x3FFF
_SEQ_FLAGS_STANDALONE = 0b11


@dataclass(slots=True)
class CcsdsHeader:
    """Decoded CCSDS primary-header fields the caller cares about.

    Args/fields:
        packet_type: 0 = telemetry (TM), 1 = telecommand (TC).
        apid: 11-bit application process identifier (0..0x7FF).
        sequence_count: 14-bit per-APID packet sequence count (0..0x3FFF).
    """

    packet_type: int
    apid: int
    sequence_count: int


def compute_crc32(data: bytes) -> int:
    """Return the unsigned CRC-32 (ISO-3309 / zlib) of data.

    Args:
        data: The bytes to checksum.

    Returns:
        Unsigned 32-bit integer CRC value.
    """
    return binascii.crc32(data) & 0xFFFFFFFF


def verify_crc32(data: bytes, expected: int) -> bool:
    """Return True iff compute_crc32(data) equals expected (masked to 32 bits).

    Args:
        data: The bytes to checksum.
        expected: The expected CRC-32 value (will be masked to 32 bits).

    Returns:
        True if the computed CRC matches expected, False otherwise.
    """
    return compute_crc32(data) == (expected & 0xFFFFFFFF)


def encode_packet(header: CcsdsHeader, body: bytes) -> Result[bytes, FaultCode]:
    """Frame body into a CCSDS Space Packet with a CRC-32 trailer.

    Args:
        header: The primary-header fields (type / apid / sequence_count).
        body: The packet data field (already includes any HMAC tag for TC).

    Returns:
        Ok(framed bytes) = primary header + body + CRC-32(header+body), or
        Err(FaultCode.COMMAND_INVALID) if a header field is out of range or body is empty.
    """
    if not (0 <= header.apid <= APID_MAX):
        return Err(FaultCode.COMMAND_INVALID)
    if not (0 <= header.sequence_count <= SEQ_COUNT_MAX):
        return Err(FaultCode.COMMAND_INVALID)
    if header.packet_type not in (0, 1) or len(body) == 0:
        return Err(FaultCode.COMMAND_INVALID)

    data_length = len(body) + CRC_TRAILER_SIZE - 1
    if data_length > 0xFFFF:
        return Err(FaultCode.COMMAND_INVALID)

    word1 = (0 << 13) | ((header.packet_type & 0x01) << 12) | (0 << 11) | (header.apid & APID_MAX)
    word2 = (_SEQ_FLAGS_STANDALONE << 14) | (header.sequence_count & SEQ_COUNT_MAX)
    word3 = data_length & 0xFFFF
    primary = struct.pack(">HHH", word1, word2, word3)
    frame = primary + body
    return Ok(frame + struct.pack(">I", compute_crc32(frame)))


def decode_packet(raw: bytes) -> Result[tuple[CcsdsHeader, bytes], FaultCode]:
    """Decode and CRC-verify a framed CCSDS Space Packet.

    Args:
        raw: The complete framed packet (primary header + body + CRC trailer).

    Returns:
        Ok((header, body)) on success, Err(FaultCode.COMMAND_CRC_FAIL) on a length or CRC
        violation (truncated, inconsistent data_length, or CRC mismatch).
    """
    if len(raw) < CCSDS_PRIMARY_HEADER_SIZE + CRC_TRAILER_SIZE:
        return Err(FaultCode.COMMAND_CRC_FAIL)
    frame, crc_bytes = raw[:-CRC_TRAILER_SIZE], raw[-CRC_TRAILER_SIZE:]
    (expected_crc,) = struct.unpack(">I", crc_bytes)
    if not verify_crc32(frame, expected_crc):
        return Err(FaultCode.COMMAND_CRC_FAIL)

    word1, word2, word3 = struct.unpack(">HHH", frame[:CCSDS_PRIMARY_HEADER_SIZE])
    packet_type = (word1 >> 12) & 0x01
    apid = word1 & APID_MAX
    sequence_count = word2 & SEQ_COUNT_MAX
    body = frame[CCSDS_PRIMARY_HEADER_SIZE:]
    if (word3 & 0xFFFF) != len(body) + CRC_TRAILER_SIZE - 1:
        return Err(FaultCode.COMMAND_CRC_FAIL)
    hdr = CcsdsHeader(packet_type=packet_type, apid=apid, sequence_count=sequence_count)
    return Ok((hdr, body))


def packet_length(primary_header: bytes) -> Result[int, FaultCode]:
    """Return total framed-packet size from the first 6 header bytes (for stream deframing).

    Args:
        primary_header: At least the 6-byte CCSDS primary header.

    Returns:
        Ok(total bytes = 6 + (data_length + 1)) where the trailing CRC is included in
        data_length, or Err(FaultCode.COMMAND_CRC_FAIL) if fewer than 6 bytes are given.
    """
    if len(primary_header) < CCSDS_PRIMARY_HEADER_SIZE:
        return Err(FaultCode.COMMAND_CRC_FAIL)
    (_, _, word3) = struct.unpack(">HHH", primary_header[:CCSDS_PRIMARY_HEADER_SIZE])
    return Ok(CCSDS_PRIMARY_HEADER_SIZE + (word3 & 0xFFFF) + 1)
