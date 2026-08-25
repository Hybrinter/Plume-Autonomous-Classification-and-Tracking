# tools.analysis.plots.storage

**Source:** `packages/tools/src/tools/analysis/plots/storage.py`
**Kind:** module

## Purpose

The storage plot builder renders live byte usage, entry counts, quota fullness, and write
activity.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `build` | function | Build storage figures from a wide DataFrame |

## Inputs and outputs

None.

## Behavior

1. Live stored bytes and quota headroom.
2. Live entries, cumulative stored count, and cumulative evicted count.
3. Quota fraction used.
4. Stacked writes, telemetry persisted, and storage-full faults per step.
5. Cumulative storage writes.

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
