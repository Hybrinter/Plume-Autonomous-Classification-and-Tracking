# tools.analysis.plots.system

**Source:** `packages/tools/src/tools/analysis/plots/system.py`
**Kind:** module

## Purpose

The system plot builder renders rollup figures from the `system` wide frame: mode timeline,
SAFE latch, and gross message and fault throughput.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `build` | function | Build system figures from a wide DataFrame |

## Inputs and outputs

**`build(wide) -> list[LabeledFigure]`**

- Output: non-`None` figures from candidate panels (may be empty).

## Behavior

1. System mode categorical timeline from `system.mode`.
2. SAFE latch line panel from `system.safe_latched`.
3. Total messages per step from `system.total_messages`.
4. Stacked fault, command, and ack mix from system event count columns.
5. Cumulative messages, faults, and acks.

## Errors and faults

None.

## Messages

None.

## Configuration

None.

## Constraints

- Drops panels with no finite data via `common` helpers.

## Related documents

- [`tools.analysis.plots`](plots.md)
- [`tools.analysis.plots.common`](common.md)
