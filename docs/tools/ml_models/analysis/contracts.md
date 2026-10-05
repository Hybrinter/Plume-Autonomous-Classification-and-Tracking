# tools.ml_models.analysis.contracts

**Source:** `packages/tools/src/tools/ml_models/analysis/contracts.py`
**Kind:** module
**Status:** stub

## Purpose

This module declares the frozen typed records that join evaluation, capture,
and rendering: the sample alignment key, one named measurement, and one
split of evidence.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SampleKey` | dataclass | Dataset, task, split, shard, row, tile, and element identity |
| `MetricValue` | dataclass | One named value or an explicit `UNAVAILABLE` status with reason |
| `SplitEvidence` | dataclass | Task, split, and dataset identity plus metric records |

## Inputs and outputs

Constructor arguments only. `SampleKey` takes `dataset_hash`, `task`,
`split`, `spatial_shard`, `row_index`, `tile_id`, and `element`.
`MetricValue` takes `name`, `value`, `status`, and an optional `reason`.
`SplitEvidence` takes `task`, `split`, `dataset_hash`, and an optional
`metrics` tuple.

## Behavior

The records are frozen slots dataclasses. This scaffold declares the field
shapes only; validation lands in a later phase.

## Errors and faults

None at this layer.

## Messages

None.

## Configuration

None.

## Constraints

- Unavailable measurements carry `status` `UNAVAILABLE` and a `reason`;
  no fabricated numeric value is stored.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.analysis.evaluate`](evaluate.md)
