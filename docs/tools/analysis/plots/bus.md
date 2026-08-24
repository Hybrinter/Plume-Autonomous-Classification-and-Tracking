# tools.analysis.plots.bus

**Source:** `packages/tools/src/tools/analysis/plots/bus.py`
**Kind:** module

## Purpose

The bus plot builder renders message-bus figures: publish throughput, queue depth, drops and
overflow, and active type count.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `build` | function | Build bus figures from a wide DataFrame |

## Behavior

1. Total published messages per step.
2. Stacked publish mix for key types (InferenceResult, TelemetryEvent, FaultEvent,
   GimbalCommand, Heartbeat, SafetyState, LinkState, DownlinkItem, Command, CommandAck).
3. Total and per-type queue depth.
4. Cumulative drops and soft-bound overflow totals.
5. Distinct message types active per step.
6. Cumulative published total.

## Errors and faults

None.

## Messages

None.

## Configuration

None.

## Constraints

Key message types are a fixed representative subset for readability.

## Related documents

- [`tools.analysis.plots`](plots.md)
- [`tools.analysis.plots.common`](common.md)
