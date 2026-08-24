# flight.libs.commands.dictionary

**Source:** `packages/flight/src/flight/libs/commands/dictionary.py`
**Kind:** pure module

## Purpose

This module is the validation authority for inbound ground commands. It maps each
`CommandId` to a `CommandSpec` with target subsystem, parameter schema, and hazard flag.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ParamSpec` | dataclass | Parameter name and `ParamKind` |
| `CommandSpec` | dataclass | Opcode, target, params tuple, `hazardous` flag |
| `COMMAND_DICTIONARY` | dict | `CommandId` to `CommandSpec` registry |
| `lookup_command` | function | Resolve a wire opcode string to a spec |
| `validate_command` | function | Check a params dict against a spec |
| `routable_targets` | function | Frozenset of all dictionary target names |
| `hazardous_command_ids` | function | Frozenset of hazardous opcode strings |

### Registered commands

| CommandId | Target | Params | Hazardous |
| --- | --- | --- | --- |
| `PING` | `core` | none | no |
| `NOOP` | `core` | none | no |
| `SET_THERMAL_LIMIT` | `thermal` | `limit_c: float` | no |
| `EXIT_SAFE` | `fault` | `phase: str` | yes |
| `RELEASE_LAUNCH_LOCK` | `mechanical` | `phase: str` | yes |
| `UPLOAD_MODEL_CHUNK` | `iss_iface` | `chunk_index`, `total_chunks`, `data_b64`, `crc32` | no |
| `ACTIVATE_MODEL` | `model_deploy` | `version: str` | no |

## Inputs and outputs

- `lookup_command(command_id: str) -> Result[CommandSpec, FaultCode]`
- `validate_command(spec, params) -> Result[None, FaultCode]`
- `routable_targets() -> frozenset[str]`
- `hazardous_command_ids() -> frozenset[str]`

## Behavior

1. `lookup_command` parses `command_id` as a `CommandId` enum member. It returns the
   matching `CommandSpec` from `COMMAND_DICTIONARY`.
2. `validate_command` requires the params dict keys to match the spec's param names exactly.
3. For each `ParamSpec`, `validate_command` checks the value kind:
   - `BOOL` accepts only `bool`.
   - `INT` accepts only `int` (not `bool`).
   - `FLOAT` accepts `int` or `float` (not `bool`).
   - `STR` accepts `str`.
4. `routable_targets` collects every `CommandSpec.target` value.
5. `hazardous_command_ids` collects `command_id.value` for specs with `hazardous=True`.

## Errors and faults

| Result | Trigger |
| --- | --- |
| `Err(FaultCode.COMMAND_INVALID)` | Unknown opcode string; missing dictionary entry; param key mismatch; wrong param kind |

## Messages

None.

## Configuration

None.

## Constraints

- Parameter validation iterates declared `ParamSpec` entries only.
- `CommandSpec.target` stamps the routed destination; the wire frame does not carry target.
- Hazardous commands require an ARM/EXECUTE two-step at the command router.

## Related documents

- [`flight.libs.commands`](flight/libs/commands.md)
- [`flight.libs.types`](flight/libs/types.md)
