# tools.analysis.plots.downlink

**Source:** `packages/tools/src/tools/analysis/plots/downlink.py`
**Kind:** module

## Purpose

The downlink plot builder renders queue depth, backlog versus budget, AOS gate, priority mix,
and emission throughput.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `build` | function | Build downlink figures from a wide DataFrame |

## Behavior

1. Pending item count and cumulative enqueued order.
2. Pending bytes and backlog fraction versus per-pass budget.
3. AOS gate line panel.
4. Stacked queued items by priority (FAULT_EVENT, COMMAND_ACK, HK_TELEMETRY,
   SCIENCE_PRODUCT).
5. Downlink items emitted per step.
6. Cumulative downlink item emission.

## Errors and faults

None.

## Messages

None.

## Configuration

None.

## Constraints

Priority names match `DownlinkPriority` enum members.

## Related documents

- [`tools.analysis.plots`](plots.md)
- [`tools.analysis.plots.common`](common.md)
