"""CCSDS Space Packet codec (see flight.libs.ccsds.codec)."""

from flight.libs.ccsds.codec import (
    CCSDS_PRIMARY_HEADER_SIZE,
    CRC_TRAILER_SIZE,
    CcsdsHeader,
    compute_crc32,
    decode_packet,
    encode_packet,
    packet_length,
    verify_crc32,
)

__all__ = [
    "CCSDS_PRIMARY_HEADER_SIZE",
    "CRC_TRAILER_SIZE",
    "CcsdsHeader",
    "compute_crc32",
    "decode_packet",
    "encode_packet",
    "packet_length",
    "verify_crc32",
]
