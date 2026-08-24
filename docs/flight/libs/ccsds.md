# flight.libs.ccsds

**Source:** `packages/flight/src/flight/libs/ccsds`
**Kind:** package

## Purpose

This package implements CCSDS 133.0-B-2 Space Packet framing with a CRC-32 trailer. It
encodes and decodes telecommand and telemetry packets. It does not open sockets.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`codec`](flight/libs/ccsds/codec.md) | module | Header types, encode, decode, CRC helpers |

## Package interface

Re-exports from `flight.libs.ccsds.codec`:

- `CCSDS_PRIMARY_HEADER_SIZE`
- `CRC_TRAILER_SIZE`
- `CcsdsHeader`
- `compute_crc32`
- `verify_crc32`
- `encode_packet`
- `decode_packet`
- `packet_length`

## Interactions

`iss_iface` and station drivers use the codec to frame and deframe CCSDS packets over the
station link. `flight.libs.commands.tc` calls `encode_packet` when building test telecommands.

## Constraints

- All entry points return `Result`; the codec does not raise.
- The codec is transport-agnostic. Callers supply and consume byte buffers.

## Related documents

- [`flight.libs`](flight/libs.md)
- [`flight.libs.ccsds.codec`](flight/libs/ccsds/codec.md)
- [`flight.libs.commands`](flight/libs/commands.md)
