# tools.analysis.plots.fault

**Source:** `packages/tools/src/tools/analysis/plots/fault.py`
**Kind:** module

## Purpose

The fault plot builder renders FDIR figures: SAFE latch and reason, mode changes, per-subsystem
watchdog metrics, and per-code fault activity.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `build` | function | Build fault figures from a wide DataFrame |

## Inputs and outputs

None.

## Behavior

1. Latched SAFE reason categorical timeline.
2. SAFE latch, fault event count, mode-change count, and active fault count line panel.
3. Per-subsystem watchdog consecutive miss counts.
4. Per-subsystem heartbeat age in seconds.
5. Stacked per-step fault events by code (thermal, power, gimbal runaway, watchdog, model
   corrupt, process died, storage full, command unroutable).
6. Cumulative fault events and mode changes.

## Errors and faults

None.

## Messages

None.

## Configuration

None.

## Constraints

Monitored subsystem list mirrors `MONITORED` in datapoints.

## Related documents

- [`tools.analysis.plots`](plots.md)
- [`tools.analysis.plots.common`](common.md)
