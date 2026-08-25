# flight.libs.commands

**Source:** `packages/flight/src/flight/libs/commands/`
**Kind:** package

## Purpose

The commands package holds the typed command dictionary and a signed telecommand packet builder.
It is the validation authority for inbound ground commands.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`dictionary`](commands/dictionary.md) | module | `CommandSpec`, lookup, and parameter validation |
| [`tc`](commands/tc.md) | module | `build_tc_packet` for GSE, sim, and tests |

## Package interface

`flight.libs.commands` re-exports:

| Name | Kind |
| --- | --- |
| `COMMAND_DICTIONARY` | constant |
| `CommandSpec`, `ParamSpec` | class |
| `build_tc_packet` | function |
| `hazardous_command_ids`, `lookup_command`, `routable_targets` | function |
| `validate_command` | function |

## Interactions

`iss_iface` validates inbound commands against the dictionary. The command router derives
routable targets and hazardous opcodes from the dictionary. GSE and test tooling call
`build_tc_packet` to construct authenticated TC packets.

## Constraints

- Validation is data-driven iteration over declared params. There are no callable dispatch
  tables.
- `CommandSpec.target` stamps `CommandMsg.target`. The ground frame does not carry a target.
- Hazardous commands require an ARM/EXECUTE two-step enforced by the command router.
- `build_tc_packet` may raise at build time. Runtime ingress stays `Result`-typed.

## Related documents

- [`flight.libs`](../libs.md)
- [`flight.libs.commands.dictionary`](commands/dictionary.md)
- [`flight.libs.types.enums`](../types/enums.md)
