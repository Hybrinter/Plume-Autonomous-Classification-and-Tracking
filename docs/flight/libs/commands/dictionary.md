# flight.libs.commands.dictionary

**Source:** `packages/flight/src/flight/libs/commands/dictionary.py`
**Kind:** pure module

## Purpose

The module maps each `CommandId` to a frozen `CommandSpec` and validates inbound command
parameters against that schema.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ParamSpec` | class | One required parameter name and `ParamKind` |
| `CommandSpec` | class | Target subsystem, params, and hazard flag |
| `COMMAND_DICTIONARY` | constant | `dict[CommandId, CommandSpec]` registry |
| `lookup_command` | function | Resolve a wire command ID string |
| `validate_command` | function | Check a params dict against a spec |
| `routable_targets` | function | Set of canonical target subsystem names |
| `hazardous_command_ids` | function | Set of hazardous opcode strings |

### CommandSpec fields

| Field | Type | Description |
| --- | --- | --- |
| `command_id` | `CommandId` | Opcode this spec describes |
| `target` | `str` | Canonical destination subsystem name |
| `params` | `tuple[ParamSpec, ...]` | Required parameters in declaration order |
| `hazardous` | `bool` | True when ARM/EXECUTE gating applies |

### Registered commands

| CommandId | Target | Params | Hazardous |
| --- | --- | --- | --- |
| `PING` | `core` | none | no |
| `NOOP` | `core` | none | no |
| `SET_THERMAL_LIMIT` | `thermal` | `limit_c: float` | no |
| `EXIT_SAFE` | `fault` | `phase: str` | yes |
| `RELEASE_LAUNCH_LOCK` | `mechanical` | `phase: str` | yes |
| `UPLOAD_MODEL_CHUNK` | `iss_iface` | chunk fields | no |
| `ACTIVATE_MODEL` | `model_deploy` | `version: str` | no |

## Inputs and outputs

| Entry point | Inputs | Outputs |
| --- | --- | --- |
| `lookup_command(command_id)` | Wire opcode string | `Result[CommandSpec, FaultCode]` |
| `validate_command(spec, params)` | `CommandSpec`, params dict | `Result[None, FaultCode]` |
| `routable_targets()` | None | `frozenset[str]` |
| `hazardous_command_ids()` | None | `frozenset[str]` |

## Behavior

1. `lookup_command` converts the wire string to `CommandId` and fetches the spec.
2. Unknown IDs return `Err(COMMAND_INVALID)`.
3. `validate_command` requires an exact key match between `params` and `spec.params`.
4. Each value must match the declared `ParamKind`.
5. `ParamKind.FLOAT` accepts `int` or `float`.
6. `bool` is rejected where `INT` or `FLOAT` is required. `int` and `float` are rejected
   where `BOOL` is required.
7. `routable_targets()` collects distinct `spec.target` values from the dictionary.
8. `hazardous_command_ids()` collects opcodes where `spec.hazardous` is True.

## Errors and faults

| Result | FaultCode | Trigger |
| --- | --- | --- |
| `Err` | `COMMAND_INVALID` | Unknown command ID |
| `Err` | `COMMAND_INVALID` | Missing, extra, or wrongly typed parameter |

## Messages

The dictionary defines the schema for `CommandMsg` and `RoutedCommandMsg` fields. It does not
publish messages.

## Configuration

None.

## Constraints

- Validation iterates declared `ParamSpec` entries. There are no callable dispatch tables.
- `CommandSpec.target` is stamped onto outbound `CommandMsg.target` by ingress.
- Adding a command updates `routable_targets()` and `hazardous_command_ids()` automatically.

## Related documents

- [`flight.libs.commands`](../commands.md)
- [`flight.libs.commands.tc`](tc.md)
- [`flight.libs.types.enums`](../types/enums.md)
