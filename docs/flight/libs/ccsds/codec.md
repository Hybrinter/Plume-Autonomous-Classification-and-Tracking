# flight.libs.ccsds.codec

**Source:** `packages/flight/src/flight/libs/ccsds/codec.py`
**Kind:** pure module

## Purpose

The module encodes and decodes CCSDS Space Packets with a CRC-32 trailer. It supports stream
deframing via header length extraction.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `CCSDS_PRIMARY_HEADER_SIZE` | constant | Primary header size in bytes (`6`) |
| `CRC_TRAILER_SIZE` | constant | CRC trailer size in bytes (`4`) |
| `CcsdsHeader` | class | Decoded primary-header fields |
| `compute_crc32` | function | Unsigned CRC-32 of byte data |
| `verify_crc32` | function | Compare computed CRC to expected value |
| `encode_packet` | function | Frame header and body with CRC trailer |
| `decode_packet` | function | Decode and CRC-verify a framed packet |
| `packet_length` | function | Total packet size from primary header bytes |

### CcsdsHeader fields

| Field | Type | Description |
| --- | --- | --- |
| `packet_type` | `int` | `0` = TM, `1` = TC |
| `apid` | `int` | 11-bit application process identifier |
| `sequence_count` | `int` | 14-bit per-APID sequence count |

## Inputs and outputs

| Entry point | Inputs | Outputs |
| --- | --- | --- |
| `encode_packet(header, body)` | `CcsdsHeader`, body `bytes` | `Result[bytes, FaultCode]` |
| `decode_packet(raw)` | Complete framed packet `bytes` | `Result[(CcsdsHeader, bytes), FaultCode]` |
| `packet_length(primary_header)` | At least 6 header bytes | `Result[int, FaultCode]` |
| `compute_crc32(data)` | Byte sequence | Unsigned 32-bit CRC |
| `verify_crc32(data, expected)` | Byte sequence and expected CRC | `bool` |

## Behavior

1. `encode_packet` packs three 16-bit big-endian header words, appends the body, then appends
   a 4-byte big-endian CRC-32 over header plus body.
2. Word 1 carries version, packet type, secondary-header flag, and APID.
3. Word 2 carries standalone sequence flags and the 14-bit sequence count.
4. Word 3 carries `data_length = len(body) + CRC_TRAILER_SIZE - 1`.
5. `decode_packet` verifies CRC, unpacks the header, and returns the body without the trailer.
6. `packet_length` reads word 3 from the first six bytes and returns total framed size.
7. APID must be in `0..0x7FF`. Sequence count must be in `0..0x3FFF`. Body must be non-empty.

## Errors and faults

| Result | FaultCode | Trigger |
| --- | --- | --- |
| `Err` | `COMMAND_INVALID` | APID, sequence count, packet type, or body out of range |
| `Err` | `COMMAND_CRC_FAIL` | Truncated packet, length mismatch, or CRC failure |
| `Err` | `COMMAND_CRC_FAIL` | Fewer than six bytes passed to `packet_length` |

## Messages

None.

## Configuration

None.

## Constraints

- The module never raises. All framing operations return `Result`.
- The codec does not open sockets or manage transport.
- CRC uses `binascii.crc32 & 0xFFFFFFFF`.
- TC bodies may include an HMAC tag before framing. The codec treats body bytes as opaque.

## Related documents

- [`flight.libs.ccsds`](../ccsds.md)
- [`flight.libs.commands.tc`](../commands/tc.md)
