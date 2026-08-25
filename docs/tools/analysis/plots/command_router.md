# tools.analysis.plots.command_router

**Source:** `packages/tools/src/tools/analysis/plots/command_router.py`
**Kind:** module

## Purpose

The command_router plot builder renders armed hazardous commands, SAFE mirror state, and
routing throughput.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `build` | function | Build command_router figures from a wide DataFrame |

## Inputs and outputs

None.

## Behavior

1. Armed hazardous command count and SAFE mirror line panel.
2. Stacked commands seen, routed, acked, and unroutable faults per step.
3. Cumulative routed commands and acks.

## Errors and faults

None.

## Messages

None.

## Configuration

None.

## Constraints

None.

## Related documents

- [`tools.analysis.plots`](plots.md)
- [`tools.analysis.plots.common`](common.md)
