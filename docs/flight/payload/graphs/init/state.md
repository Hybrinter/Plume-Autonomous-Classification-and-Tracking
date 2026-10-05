# flight.payload.graphs.init.state

**Source:** `packages/flight/src/flight/payload/graphs/init/state.py`
**Kind:** pure module

## Purpose

The INIT state record and node vocabulary tracking which effect intents were
issued, whether failure latched, and whether IDLE was already requested.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `InitNode` | enum | `SELFTEST`, `MODEL_LOAD`, `HOME`, `READY` |
| `State` | dataclass | Key, node, `issued`, `failed`, `requested_idle`, `home_target_rad` |

## Inputs and outputs

Data only: `InitNode` enumerates `SELFTEST`/`MODEL_LOAD`/`HOME`/`READY`
and `State` carries `activation_key`, `node`, `issued`, `failed`,
`requested_idle`, and `home_target_rad`.

## Behavior

Pure records only; the graph module advances nodes and latches `failed`/`requested_idle` via `dataclasses.replace`.

## Errors and faults

None directly.

## Messages

None.

## Configuration

None.

## Constraints

Pure data only; frozen slots.

## Related documents

- [`flight.payload.graphs.init`](../init.md)
