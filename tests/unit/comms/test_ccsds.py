"""Unit tests for pact.comms.ccsds — encode/decode roundtrip and CRC-32 verification.

Satisfies: §6.2 of PACT_SW_ARCH.md — Comms subsystem unit tests.
REQ-COMM-HIGH-001, REQ-COMM-HIGH-002
"""

from __future__ import annotations

# third-party
import pytest

# module under test
from pact.comms.ccsds import (
    CcsdsPacket,
    compute_crc32,
    decode_packet,
    encode_packet,
    verify_crc32,
)

# pact types
from pact.types.enums import Err, Ok


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_packet(
    data: bytes = b"hello world",
    apid: int = 0x001,
    sequence_count: int = 0,
    packet_type: int = 0,
) -> CcsdsPacket:
    """Construct a minimal valid CcsdsPacket for testing."""
    return CcsdsPacket(
        version=0,
        packet_type=packet_type,
        sec_hdr_flag=0,
        apid=apid,
        sequence_flags=0b11,          # unsegmented packet
        sequence_count=sequence_count,
        data_length=len(data) - 1,    # CCSDS: data_length = len(data field) - 1
        data=data,
    )


# ---------------------------------------------------------------------------
# encode / decode roundtrip test
# ---------------------------------------------------------------------------


def test_encode_decode_roundtrip() -> None:
    """encode_packet then decode_packet must recover the original CcsdsPacket.

    All fields that are encoded into the primary header must survive the roundtrip.
    """
    original = make_packet(data=b"PACT test payload", sequence_count=42)
    encoded: bytes = encode_packet(original)

    result = decode_packet(encoded)
    assert isinstance(result, Ok), (
        f"decode_packet returned Err: {result.error if hasattr(result, 'error') else result}"
    )
    decoded: CcsdsPacket = result.value

    assert decoded.version == original.version
    assert decoded.packet_type == original.packet_type
    assert decoded.apid == original.apid
    assert decoded.sequence_flags == original.sequence_flags
    assert decoded.sequence_count == original.sequence_count
    assert decoded.data_length == original.data_length
    assert decoded.data == original.data


def test_encode_produces_bytes() -> None:
    """encode_packet must return a bytes object."""
    packet = make_packet()
    encoded = encode_packet(packet)
    assert isinstance(encoded, bytes), f"Expected bytes, got {type(encoded)}"


def test_encoded_length_correct() -> None:
    """Encoded packet length must be CCSDS_PRIMARY_HEADER_SIZE (6) + len(data)."""
    from pact.comms.ccsds import CCSDS_PRIMARY_HEADER_SIZE
    data = b"test data"
    packet = make_packet(data=data)
    encoded = encode_packet(packet)
    expected_len = CCSDS_PRIMARY_HEADER_SIZE + len(data)
    assert len(encoded) == expected_len, (
        f"Expected {expected_len} bytes, got {len(encoded)}"
    )


def test_decode_packet_too_short_returns_err() -> None:
    """decode_packet on a truncated buffer (< 6 bytes) must return Err."""
    result = decode_packet(b"\x00\x01")
    assert isinstance(result, Err), (
        "Expected Err for too-short packet, but got Ok"
    )


# ---------------------------------------------------------------------------
# CRC-32 tests
# ---------------------------------------------------------------------------


def test_crc32_verify_correct() -> None:
    """verify_crc32 must return True for data whose CRC was computed by compute_crc32."""
    data = b"PACT health telemetry payload"
    crc = compute_crc32(data)
    assert verify_crc32(data, crc) is True, (
        "verify_crc32 returned False for correct CRC — possible implementation bug"
    )


def test_crc32_verify_tampered() -> None:
    """verify_crc32 must return False when the data has been bit-flipped."""
    data = b"PACT health telemetry payload"
    crc = compute_crc32(data)

    # Flip one bit in the data
    tampered = bytearray(data)
    tampered[0] ^= 0xFF
    assert verify_crc32(bytes(tampered), crc) is False, (
        "verify_crc32 returned True for tampered data — CRC is not detecting corruption"
    )


def test_crc32_deterministic() -> None:
    """compute_crc32 must return the same value for the same input."""
    data = b"determinism check"
    assert compute_crc32(data) == compute_crc32(data)


def test_crc32_different_data_different_crc() -> None:
    """Different data must (almost certainly) produce different CRC values."""
    crc_a = compute_crc32(b"data A")
    crc_b = compute_crc32(b"data B")
    assert crc_a != crc_b, (
        "compute_crc32 returned the same value for different inputs — suspicious"
    )


def test_crc32_returns_int() -> None:
    """compute_crc32 must return an int."""
    crc = compute_crc32(b"test")
    assert isinstance(crc, int), f"Expected int, got {type(crc)}"
