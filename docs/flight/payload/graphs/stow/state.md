# flight.payload.graphs.stow.state

**Source:** `packages/flight/src/flight/payload/graphs/stow/state.py`
**Kind:** pure module

## Purpose

The STOW state record and node vocabulary. `MOVING` drives the bounded stow
reference; `HELD` inhibits after confirmed arrival.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `StowNode` | enum | `MOVING`, `HELD` |
| `State` | dataclass | Activation key, node, `entered_s`, `timeout_latched` |

## Inputs and outputs

Data only: `StowNode` enumerates `MOVING`/`HELD` and `State` carries
`activation_key`, `node`, `entered_s`, and `timeout_latched`.

## Behavior

Pure records only; graph transitions mutate via `dataclasses.replace` in the graph module.

## Errors and faults

None directly.

## Messages

None.

## Configuration

None.

## Constraints

Pure data only; frozen slots.

## Related documents

- [`flight.payload.graphs.stow`](../stow.md)
