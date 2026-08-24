# flight.libs.ccsds.codec

**Source:** `packages/flight/src/flight/libs/ccsds/codec.py`
**Kind:** pure module

## Purpose

This module frames and deframes CCSDS Space Packets. It builds a 6-byte big-endian primary
header, appends a body, and adds a 4-byte CRC-32 trailer over header plus body.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `CCSDS_PRIMARY_HEADER_SIZE` | constant | Primary header size in bytes (`6`) |
| `CRC_TRAILER_SIZE` | constant | CRC trailer size in bytes (`4`) |
| `APID_MAX` | constant | Maximum 11-bit APID (`0x7FF`) |
| `SEQ_COUNT_MAX` | constant | Maximum 14-bit sequence count (`0x3FFF`) |
| `CcsdsHeader` | dataclass | `packet_type`, `apid`, `sequence_count` |
| `compute_crc32` | function | Unsigned CRC-32 (ISO-3309 / zlib) of byte data |
| `verify_crc32` | function | Compare computed CRC-32 to an expected value |
| `encode_packet` | function | Build framed packet bytes from header and body |
| `decode_packet` | function | Parse and CRC-verify a framed packet |
| `packet_length` | function | Total packet size from the first 6 header bytes |

## Inputs and outputs

- `compute_crc32(data: bytes) -> int` returns a 32-bit unsigned CRC.
- `verify_crc32(data, expected) -> bool` masks `expected` to 32 bits and compares.
- `encode_packet(header, body) -> Result[bytes, FaultCode]` returns framed bytes or an error.
- `decode_packet(raw) -> Result[tuple[CcsdsHeader, bytes], FaultCode]` returns header and
  body or an error.
- `packet_length(primary_header) -> Result[int, FaultCode]` returns total framed length or an
  error.

## Behavior

1. `encode_packet` validates `apid`, `sequence_count`, `packet_type` (0 or 1), and non-empty
   body. It rejects `data_length` above `0xFFFF`.
2. Word 1 packs version 0, packet type, secondary-header flag 0, and APID.
3. Word 2 packs standalone sequence flags `0b11` and the 14-bit sequence count.
4. Word 3 sets `data_length = len(body) + CRC_TRAILER_SIZE - 1`.
5. The frame is primary header plus body. The CRC-32 trailer covers the frame bytes before
   the trailer.
6. `decode_packet` requires at least 10 bytes. It verifies CRC, unpacks the header fields,
   and checks `data_length` against the body size.
7. `packet_length` reads word 3 from at least 6 bytes and returns
   `6 + (data_length + 1)`.

## Errors and faults

| Result | Trigger |
| --- | --- |
| `Err(FaultCode.COMMAND_INVALID)` | Out-of-range APID or sequence count; invalid packet type; empty body; `data_length` overflow on encode |
| `Err(FaultCode.COMMAND_CRC_FAIL)` | Truncated packet; CRC mismatch; inconsistent `data_length`; fewer than 6 bytes for `packet_length` |

## Messages

None.

## Configuration

None.

## Constraints

- `packet_type` 0 is telemetry (TM). `packet_type` 1 is telecommand (TC).
- CRC uses `binascii.crc32(data) & 0xFFFFFFFF`.
- The module uses only the Python standard library.

## Related documents

- [`flight.libs.ccsds`](flight/libs/ccsds.md)
- [`flight.libs.types`](flight/libs/types.md)
