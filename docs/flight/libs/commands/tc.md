# flight.libs.commands.tc

**Source:** `packages/flight/src/flight/libs/commands/tc.py`
**Kind:** module

## Purpose

This module builds HMAC-signed CCSDS telecommand packets. GSE, SIL, and tests use it. The
flight runtime ingress path does not call it.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `build_tc_packet` | function | Construct signed, CRC-framed TC packet bytes |

## Inputs and outputs

`build_tc_packet(command_id, params, source, seq, key, apid) -> bytes`

| Argument | Type | Role |
| --- | --- | --- |
| `command_id` | `str` | Command opcode string |
| `params` | `dict[str, str \| int \| float \| bool]` | Command parameters |
| `source` | `str` | Command origin identifier |
| `seq` | `int` | Per-source sequence number |
| `key` | `bytes` | Shared HMAC-SHA256 secret |
| `apid` | `int` | Telecommand APID |

Returns complete framed TC bytes: primary header, JSON body, HMAC tag, and CRC trailer.

## Behavior

1. The function JSON-encodes `command_id`, `params`, `source`, and `seq` with sorted keys and
   compact separators.
2. It computes an HMAC-SHA256 tag over the UTF-8 JSON body.
3. It calls `encode_packet` with `packet_type=1`, the given `apid`, and
   `sequence_count=seq & 0x3FFF`. The body is JSON bytes plus the tag.
4. On `Ok`, it returns the framed packet bytes.

## Errors and faults

`build_tc_packet` raises `ValueError` when `encode_packet` returns `Err`. This applies at
build and test time only.

## Messages

None.

## Configuration

None.

## Constraints

- JSON serialization uses sorted keys for deterministic signed bytes.
- The function imports `flight.libs.ccsds` only, not `flight.iss_iface`.

## Related documents

- [`flight.libs.commands`](flight/libs/commands.md)
- [`flight.libs.ccsds`](flight/libs/ccsds.md)
