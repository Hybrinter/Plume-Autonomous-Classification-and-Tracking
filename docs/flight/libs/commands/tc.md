# flight.libs.commands.tc

**Source:** `packages/flight/src/flight/libs/commands/tc.py`
**Kind:** module

## Purpose

The module builds HMAC-signed CCSDS telecommand packets for GSE, sim, and tests. Flight runtime
ingress does not call this helper.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `build_tc_packet` | function | Construct a signed, CRC-framed TC packet |

## Inputs and outputs

| Entry point | Inputs | Outputs |
| --- | --- | --- |
| `build_tc_packet(...)` | `command_id`, `params`, `source`, `seq`, `key`, `apid` | Framed TC packet `bytes` |

Arguments:

| Name | Type | Description |
| --- | --- | --- |
| `command_id` | `str` | Command opcode string |
| `params` | `dict[str, str \| int \| float \| bool]` | Command parameters |
| `source` | `str` | Command origin identifier |
| `seq` | `int` | Per-source monotonic sequence number |
| `key` | `bytes` | Shared HMAC-SHA256 secret |
| `apid` | `int` | Telecommand APID |

## Behavior

1. Serialize `command_id`, `params`, `source`, and `seq` to JSON with sorted keys and compact
   separators.
2. Compute an HMAC-SHA256 tag over the JSON body bytes.
3. Call `encode_packet` with packet type TC (`1`), the given APID, and sequence count
   `seq & 0x3FFF`.
4. Pass `body + tag` as the packet data field.
5. Return the framed bytes on success.

## Errors and faults

Raises `ValueError` when `encode_packet` returns `Err`. This is a build-time helper only.

Runtime ingress uses `Result`-typed decode and validation paths.

## Messages

Output bytes match the TC format consumed by `iss_iface` ingress. The JSON body fields align
with `CommandMsg` envelope fields after decode.

## Configuration

Callers supply APID and HMAC key bytes. Production values come from `LinkConfig` and
`CommandIngressConfig`.

## Constraints

- Lives in `flight.libs.commands` so GSE and tests import only `flight.libs`.
- Flight runtime ingress does not call `build_tc_packet`.
- JSON serialization uses sorted keys for deterministic signed bytes.
- This is the only command-side function permitted to raise.

## Related documents

- [`flight.libs.commands`](../commands.md)
- [`flight.libs.commands.dictionary`](dictionary.md)
- [`flight.libs.ccsds.codec`](../ccsds/codec.md)
