# flight.libs.ccsds

**Source:** `packages/flight/src/flight/libs/ccsds/`
**Kind:** package

## Purpose

The ccsds package implements a CCSDS 133.0-B-2 Space Packet codec with a CRC-32 integrity
trailer. It frames and deframes command and telemetry packets. It does not open sockets.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`codec`](ccsds/codec.md) | module | Encode, decode, CRC, and stream length helpers |

## Package interface

`flight.libs.ccsds` re-exports:

| Name | Kind |
| --- | --- |
| `CCSDS_PRIMARY_HEADER_SIZE`, `CRC_TRAILER_SIZE` | constant |
| `CcsdsHeader` | class |
| `compute_crc32`, `verify_crc32` | function |
| `encode_packet`, `decode_packet`, `packet_length` | function |

## Interactions

`iss_iface` and station HAL drivers use the codec to frame TC and TM packets. The real station
driver calls `packet_length()` to deframe a TCP byte stream.

## Constraints

- The codec is transport-agnostic. It never touches sockets.
- All entry points return `Result`. The codec never raises.
- CRC-32 uses ISO-3309 / zlib (`binascii.crc32 & 0xFFFFFFFF`).
- The CRC covers primary header plus body. A 4-byte big-endian CRC trailer follows the body.

## Related documents

- [`flight.libs`](../libs.md)
- [`flight.libs.ccsds.codec`](ccsds/codec.md)
- [`flight.libs.commands`](../commands.md)
