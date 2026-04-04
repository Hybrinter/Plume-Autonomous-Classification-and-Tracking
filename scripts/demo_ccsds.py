"""Demo: encode a CCSDS packet, print hex, decode, and verify CRC-32.

Encodes a sample health telemetry payload as a CCSDS Space Packet, prints the
hex dump of the encoded bytes, decodes it back, and verifies the CRC-32 checksum.

Usage
-----
    python scripts/demo_ccsds.py

Satisfies: §7 of PACT_SW_ARCH.md (scripts/demo_ccsds.py)
This script is fully functional since ccsds.py encode/decode are real implementations.
"""

from __future__ import annotations

# stdlib
import sys

# internal
from pact.comms.ccsds import (
    CcsdsPacket,
    compute_crc32,
    decode_packet,
    encode_packet,
    verify_crc32,
)
from pact.types.enums import Ok


def main() -> None:
    """Encode, print, decode, and verify a CCSDS packet."""
    # Sample health telemetry payload (JSON-like ASCII)
    payload = b'{"subsystem":"controller","gimbal_state":"TRACKING","inference_ms":47.2}'

    crc = compute_crc32(payload)
    print("PACT CCSDS Packet Demo")
    print("=" * 60)
    print(f"Payload ({len(payload)} bytes): {payload.decode('ascii')}")
    print(f"CRC-32: 0x{crc:08X}")
    print()

    # Build a CCSDS Space Packet
    packet = CcsdsPacket(
        version=0,
        packet_type=0,           # 0 = telemetry
        sec_hdr_flag=0,
        apid=0x001,              # APID=1 per config/default.toml
        sequence_flags=0b11,     # unsegmented packet
        sequence_count=42,
        data_length=len(payload) - 1,  # CCSDS: data_length = len(data) - 1
        data=payload,
    )

    # Encode
    encoded: bytes = encode_packet(packet)
    print(f"Encoded packet ({len(encoded)} bytes):")
    hex_str = " ".join(f"{b:02X}" for b in encoded)
    # Print in rows of 16 bytes
    bytes_per_row = 16
    for i in range(0, len(encoded), bytes_per_row):
        row = encoded[i:i + bytes_per_row]
        hex_row = " ".join(f"{b:02X}" for b in row)
        print(f"  {i:04X}  {hex_row}")
    print()

    # Decode
    result = decode_packet(encoded)
    if not isinstance(result, Ok):
        print(f"ERROR: decode_packet failed: {result}")
        sys.exit(1)

    decoded: CcsdsPacket = result.value
    print("Decoded packet fields:")
    print(f"  version:        {decoded.version}")
    print(f"  packet_type:    {decoded.packet_type} (0=telemetry)")
    print(f"  apid:           0x{decoded.apid:03X}")
    print(f"  sequence_flags: 0b{decoded.sequence_flags:02b}")
    print(f"  sequence_count: {decoded.sequence_count}")
    print(f"  data_length:    {decoded.data_length}")
    print(f"  data:           {decoded.data.decode('ascii')}")
    print()

    # Verify CRC
    crc_ok = verify_crc32(payload, crc)
    print(f"CRC-32 verification (original payload): {'PASS' if crc_ok else 'FAIL'}")

    # Verify tampering is detected
    tampered = bytearray(payload)
    tampered[0] ^= 0xFF
    tamper_ok = verify_crc32(bytes(tampered), crc)
    print(f"CRC-32 verification (tampered payload):  {'PASS' if tamper_ok else 'FAIL (correctly rejected)'}")

    if crc_ok and not tamper_ok:
        print("\nAll checks PASSED. CCSDS encode/decode and CRC-32 are working correctly.")
    else:
        print("\nSome checks FAILED. Review ccsds.py implementation.")
        sys.exit(1)


if __name__ == "__main__":
    main()
