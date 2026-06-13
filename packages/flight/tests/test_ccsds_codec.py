"""CCSDS Space Packet codec round-trip and integrity tests."""

from flight.libs.ccsds import (
    CcsdsHeader,
    compute_crc32,
    decode_packet,
    encode_packet,
    packet_length,
)
from flight.libs.types import Err, FaultCode, Ok


def test_encode_decode_round_trip() -> None:
    """Encode then decode a TC packet and verify all header fields and body match."""
    header = CcsdsHeader(packet_type=1, apid=0x01, sequence_count=5)
    body = b"hello-command"
    encoded = encode_packet(header, body)
    assert isinstance(encoded, Ok)
    decoded = decode_packet(encoded.value)
    assert isinstance(decoded, Ok)
    out_header, out_body = decoded.value
    assert out_header.packet_type == 1
    assert out_header.apid == 0x01
    assert out_header.sequence_count == 5
    assert out_body == body


def test_decode_rejects_crc_corruption() -> None:
    """A single flipped bit in the body produces a CRC failure."""
    header = CcsdsHeader(packet_type=0, apid=0x02, sequence_count=1)
    enc_result = encode_packet(header, b"science")
    assert isinstance(enc_result, Ok)
    corrupted = bytearray(enc_result.value)
    corrupted[8] ^= 0xFF  # flip a body byte; CRC trailer no longer matches
    result = decode_packet(bytes(corrupted))
    assert isinstance(result, Err)
    assert result.error is FaultCode.COMMAND_CRC_FAIL


def test_decode_rejects_truncated() -> None:
    """A truncated packet (fewer than header+CRC bytes) is rejected."""
    assert isinstance(decode_packet(b"\x00\x00\x00"), Err)


def test_encode_rejects_out_of_range_apid() -> None:
    """An APID beyond 11 bits (>0x7FF) is rejected."""
    result = encode_packet(CcsdsHeader(packet_type=1, apid=0x800, sequence_count=0), b"x")
    assert isinstance(result, Err)
    assert result.error is FaultCode.COMMAND_INVALID


def test_packet_length_reads_total_size_from_header() -> None:
    """packet_length returns the total byte count matching the full encoded packet."""
    enc_result = encode_packet(CcsdsHeader(packet_type=1, apid=1, sequence_count=0), b"abcd")
    assert isinstance(enc_result, Ok)
    length = packet_length(enc_result.value[:6])
    assert isinstance(length, Ok)
    assert length.value == len(enc_result.value)


def test_compute_crc32_matches_binascii() -> None:
    """compute_crc32 is consistent with binascii.crc32 masked to unsigned 32 bits."""
    import binascii

    assert compute_crc32(b"abc") == (binascii.crc32(b"abc") & 0xFFFFFFFF)
