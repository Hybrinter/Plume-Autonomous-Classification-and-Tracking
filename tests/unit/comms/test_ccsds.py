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


# ---------------------------------------------------------------------------
# APID boundary tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("apid,should_raise", [
    (0x000, False),   # valid minimum APID
    (0x001, False),   # typical PACT APID
    (0x7FE, False),   # one below idle
    (0x7FF, False),   # CCSDS idle APID — valid maximum
    (0x800, True),    # one over 11-bit max — must raise ValueError
    (0xFFF, True),    # well over limit
])
def test_apid_boundaries(apid: int, should_raise: bool) -> None:
    """encode_packet must accept valid 11-bit APIDs and raise ValueError for out-of-range values."""
    packet = CcsdsPacket(
        version=0,
        packet_type=0,
        sec_hdr_flag=0,
        apid=apid,
        sequence_flags=0b11,
        sequence_count=0,
        data_length=0,   # len(b"\x00") - 1 = 0
        data=b"\x00",
    )
    if should_raise:
        with pytest.raises(ValueError, match="apid"):
            encode_packet(packet)
    else:
        encoded = encode_packet(packet)
        result = decode_packet(encoded)
        assert isinstance(result, Ok)
        assert result.value.apid == apid


def test_encode_decode_idle_packet() -> None:
    """An idle APID (0x7FF) must encode and decode correctly."""
    packet = CcsdsPacket(
        version=0,
        packet_type=0,
        sec_hdr_flag=0,
        apid=0x7FF,
        sequence_flags=0b11,
        sequence_count=0,
        data_length=0,
        data=b"\x00",
    )
    encoded = encode_packet(packet)
    result = decode_packet(encoded)
    assert isinstance(result, Ok)
    assert result.value.apid == 0x7FF


# ---------------------------------------------------------------------------
# sequence_count boundary
# ---------------------------------------------------------------------------


def test_sequence_count_max() -> None:
    """sequence_count at maximum value (16383 = 0x3FFF) must encode and decode correctly."""
    packet = make_packet(sequence_count=0x3FFF)
    encoded = encode_packet(packet)
    result = decode_packet(encoded)
    assert isinstance(result, Ok)
    assert result.value.sequence_count == 0x3FFF, (
        f"Expected 0x3FFF, got {result.value.sequence_count:#x}"
    )


# ---------------------------------------------------------------------------
# sequence_flags — all four values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_flags", [0b00, 0b01, 0b10, 0b11])
def test_sequence_flags_all_values(seq_flags: int) -> None:
    """All four sequence_flags values must survive an encode/decode roundtrip."""
    packet = CcsdsPacket(
        version=0,
        packet_type=0,
        sec_hdr_flag=0,
        apid=0x001,
        sequence_flags=seq_flags,
        sequence_count=0,
        data_length=0,
        data=b"\x00",
    )
    encoded = encode_packet(packet)
    result = decode_packet(encoded)
    assert isinstance(result, Ok)
    assert result.value.sequence_flags == seq_flags, (
        f"sequence_flags {seq_flags:#04b} not preserved through encode/decode"
    )


# ---------------------------------------------------------------------------
# data_length consistency
# ---------------------------------------------------------------------------


def test_data_length_inconsistency_raises() -> None:
    """encode_packet must raise ValueError when data_length does not equal len(data) - 1."""
    packet = CcsdsPacket(
        version=0,
        packet_type=0,
        sec_hdr_flag=0,
        apid=0x001,
        sequence_flags=0b11,
        sequence_count=0,
        data_length=99,   # wrong: data has 1 byte so data_length should be 0
        data=b"\x00",
    )
    with pytest.raises(ValueError, match="data_length"):
        encode_packet(packet)


# ---------------------------------------------------------------------------
# Truncated data field in decode
# ---------------------------------------------------------------------------


def test_decode_data_field_truncated_returns_err() -> None:
    """decode_packet must return Err when the buffer is shorter than the declared data field.

    Constructs a header declaring data_length=10 (11 data bytes) but provides only
    4 data bytes in the buffer.
    """
    import struct

    apid = 0x001
    word1 = (0 << 13) | (0 << 12) | (0 << 11) | apid
    word2 = (0b11 << 14) | 0
    word3 = 10   # data_length=10 means the data field is 11 bytes
    header = struct.pack(">HHH", word1, word2, word3)
    truncated = header + b"\x00" * 4   # only 4 data bytes, but 11 declared

    result = decode_packet(truncated)

    assert isinstance(result, Err), (
        "Expected Err for truncated data field, got Ok"
    )


# ---------------------------------------------------------------------------
# Telecommand packet_type
# ---------------------------------------------------------------------------


def test_encode_telecommand_packet_type() -> None:
    """packet_type=1 (telecommand) must survive an encode/decode roundtrip."""
    packet = make_packet(packet_type=1, data=b"telecommand payload")
    encoded = encode_packet(packet)
    result = decode_packet(encoded)
    assert isinstance(result, Ok)
    assert result.value.packet_type == 1, (
        f"Expected packet_type=1 (telecommand), got {result.value.packet_type}"
    )
