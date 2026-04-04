"""
CCSDS Space Packet Protocol implementation for PACT.

Implements the CCSDS Space Packet primary header (CCSDS 133.0-B-2, §4.1).
Only the 6-byte primary header is implemented; no secondary header in Phase I.

Primary header layout (big-endian, 48 bits total):
    Bits  0-2   : Version number (3 bits, always 0b000)
    Bit   3     : Packet type (1 bit: 0=telemetry, 1=telecommand)
    Bit   4     : Secondary header flag (1 bit)
    Bits  5-15  : Application Process ID / APID (11 bits)
    Bits 16-17  : Sequence flags (2 bits: 11=standalone, 01=first, 00=continuation, 10=last)
    Bits 18-31  : Packet sequence count (14 bits)
    Bits 32-47  : Packet data length (16 bits, value = data field length in bytes - 1)

Satisfies: REQ-COMM-HIGH-003
"""

from __future__ import annotations

import binascii
import struct
from dataclasses import dataclass
from typing import Final

from pact.types.enums import Err, Ok, Result

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CCSDS_PRIMARY_HEADER_SIZE: Final[int] = 6  # bytes


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CcsdsPacket:
    """CCSDS Space Packet with primary header only (no secondary header in this implementation).

    All integer fields are validated against their bit-width by encode_packet().
    Do not construct directly if encoding is required — use encode_packet() which validates.

    Fields
    ------
    version:
        3-bit version number. Must be 0b000 (= 0) per CCSDS 133.0-B-2.
    packet_type:
        1-bit field: 0 = telemetry packet, 1 = telecommand packet.
    sec_hdr_flag:
        1-bit secondary header presence flag: 0 = absent (Phase I implementation).
    apid:
        11-bit Application Process ID. Range [0x000, 0x7FF]. Idle packets use 0x7FF.
    sequence_flags:
        2-bit sequence grouping flags.
        0b11 = standalone packet (most common in Phase I).
        0b01 = first segment, 0b00 = continuation, 0b10 = last segment.
    sequence_count:
        14-bit monotonically increasing packet counter (wraps at 16383 → 0).
    data_length:
        16-bit field = (length of data field in bytes) - 1. Per CCSDS spec.
    data:
        Raw data field bytes. len(data) must equal data_length + 1.
    """

    version: int        # 3 bits, always 0
    packet_type: int    # 1 bit
    sec_hdr_flag: int   # 1 bit
    apid: int           # 11 bits
    sequence_flags: int # 2 bits
    sequence_count: int # 14 bits
    data_length: int    # 16 bits (len(data) - 1)
    data: bytes


# ---------------------------------------------------------------------------
# Encoding / Decoding
# ---------------------------------------------------------------------------

def encode_packet(packet: CcsdsPacket) -> bytes:
    """Encode a CcsdsPacket into bytes using the CCSDS primary header format.

    Header word 1 (16 bits, big-endian):
        [version(3)] [packet_type(1)] [sec_hdr_flag(1)] [apid(11)]

    Header word 2 (16 bits, big-endian):
        [sequence_flags(2)] [sequence_count(14)]

    Header word 3 (16 bits, big-endian):
        [data_length(16)]

    Parameters
    ----------
    packet:
        Fully populated CcsdsPacket. data_length must equal len(packet.data) - 1.

    Returns
    -------
    bytes
        6-byte primary header concatenated with packet.data.

    Raises
    ------
    ValueError
        If any field is out of its valid bit-range, or data_length is inconsistent.
    """
    if packet.version not in (0,):
        raise ValueError(f"CCSDS version must be 0, got {packet.version}")
    if not (0 <= packet.packet_type <= 1):
        raise ValueError(f"packet_type must be 0 or 1, got {packet.packet_type}")
    if not (0 <= packet.sec_hdr_flag <= 1):
        raise ValueError(f"sec_hdr_flag must be 0 or 1, got {packet.sec_hdr_flag}")
    if not (0 <= packet.apid <= 0x7FF):
        raise ValueError(f"apid out of range [0, 0x7FF]: {packet.apid:#05x}")
    if not (0 <= packet.sequence_flags <= 3):
        raise ValueError(f"sequence_flags must be in [0,3], got {packet.sequence_flags}")
    if not (0 <= packet.sequence_count <= 0x3FFF):
        raise ValueError(f"sequence_count out of range [0, 16383]: {packet.sequence_count}")
    expected_data_length = len(packet.data) - 1
    if packet.data_length != expected_data_length:
        raise ValueError(
            f"data_length ({packet.data_length}) does not match len(data)-1 "
            f"({expected_data_length})"
        )

    # Word 1: version(3) | packet_type(1) | sec_hdr_flag(1) | apid(11)
    word1 = (
        ((packet.version & 0x07) << 13)
        | ((packet.packet_type & 0x01) << 12)
        | ((packet.sec_hdr_flag & 0x01) << 11)
        | (packet.apid & 0x07FF)
    )

    # Word 2: sequence_flags(2) | sequence_count(14)
    word2 = ((packet.sequence_flags & 0x03) << 14) | (packet.sequence_count & 0x3FFF)

    # Word 3: data_length(16)
    word3 = packet.data_length & 0xFFFF

    header = struct.pack(">HHH", word1, word2, word3)
    return header + packet.data


def decode_packet(raw: bytes) -> "Result[CcsdsPacket, str]":
    """Decode a raw byte buffer into a CcsdsPacket.

    Parameters
    ----------
    raw:
        Raw bytes containing at minimum the 6-byte primary header followed by the data field.

    Returns
    -------
    Result[CcsdsPacket, str]
        Ok(CcsdsPacket) on success.
        Err(str) with a human-readable error message if the buffer is malformed.
    """
    if len(raw) < CCSDS_PRIMARY_HEADER_SIZE:
        return Err(
            f"Buffer too short: need at least {CCSDS_PRIMARY_HEADER_SIZE} bytes, "
            f"got {len(raw)}"
        )

    word1, word2, word3 = struct.unpack(">HHH", raw[:CCSDS_PRIMARY_HEADER_SIZE])

    version      = (word1 >> 13) & 0x07
    packet_type  = (word1 >> 12) & 0x01
    sec_hdr_flag = (word1 >> 11) & 0x01
    apid         = word1 & 0x07FF

    sequence_flags = (word2 >> 14) & 0x03
    sequence_count = word2 & 0x3FFF

    data_length = word3  # value = len(data) - 1

    expected_data_bytes = data_length + 1
    actual_data_bytes = len(raw) - CCSDS_PRIMARY_HEADER_SIZE

    if actual_data_bytes < expected_data_bytes:
        return Err(
            f"Data field truncated: header declares {expected_data_bytes} bytes, "
            f"buffer contains {actual_data_bytes}"
        )

    data = raw[CCSDS_PRIMARY_HEADER_SIZE : CCSDS_PRIMARY_HEADER_SIZE + expected_data_bytes]

    return Ok(
        CcsdsPacket(
            version=version,
            packet_type=packet_type,
            sec_hdr_flag=sec_hdr_flag,
            apid=apid,
            sequence_flags=sequence_flags,
            sequence_count=sequence_count,
            data_length=data_length,
            data=data,
        )
    )


# ---------------------------------------------------------------------------
# CRC-32 utilities
# ---------------------------------------------------------------------------

def compute_crc32(data: bytes) -> int:
    """Compute the CRC-32 checksum of `data`.

    Uses the standard ISO 3309 / ITU-T V.42 CRC-32 polynomial (same as zlib).
    Returns an unsigned 32-bit integer.

    Parameters
    ----------
    data:
        Arbitrary byte string to checksum (e.g., packet.data or the full encoded packet).

    Returns
    -------
    int
        Unsigned 32-bit CRC-32 value in [0, 2^32 - 1].
    """
    return binascii.crc32(data) & 0xFFFF_FFFF


def verify_crc32(data: bytes, expected: int) -> bool:
    """Verify that the CRC-32 of `data` matches `expected`.

    Parameters
    ----------
    data:
        Data to verify.
    expected:
        Expected CRC-32 value (as returned by compute_crc32() at the time of encoding).

    Returns
    -------
    bool
        True if the computed CRC-32 matches `expected`; False if data is corrupt.
    """
    return compute_crc32(data) == (expected & 0xFFFF_FFFF)
