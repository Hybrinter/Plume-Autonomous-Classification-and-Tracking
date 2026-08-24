# flight.libs.commands

**Source:** `packages/flight/src/flight/libs/commands`
**Kind:** package

## Purpose

This package holds the typed ground-command dictionary and a signed telecommand packet
builder for tests and ground tools.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`dictionary`](flight/libs/commands/dictionary.md) | module | Command specs, lookup, validation |
| [`tc`](flight/libs/commands/tc.md) | module | HMAC-signed TC packet builder |

## Package interface

Re-exports:

- `COMMAND_DICTIONARY`
- `CommandSpec`
- `ParamSpec`
- `lookup_command`
- `validate_command`
- `routable_targets`
- `hazardous_command_ids`
- `build_tc_packet`

## Interactions

`iss_iface` and `flight.core.command_router` use the dictionary to validate and route
inbound commands. GSE, SIL, and tests call `build_tc_packet` to construct authenticated
telecommand bytes. The builder uses `flight.libs.ccsds.encode_packet`.

## Constraints

- Validation is data-driven over declared `ParamSpec` entries. No callable dispatch tables.
- `CommandSpec.target` names the canonical subsystem for each opcode.
- `build_tc_packet` is not used in the flight runtime ingress path.

## Related documents

- [`flight.libs`](flight/libs.md)
- [`flight.libs.commands.dictionary`](flight/libs/commands/dictionary.md)
- [`flight.libs.commands.tc`](flight/libs/commands/tc.md)
- [`flight.libs.ccsds`](flight/libs/ccsds.md)
