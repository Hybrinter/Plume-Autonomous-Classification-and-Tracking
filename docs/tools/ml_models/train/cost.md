# tools.ml_models.train.cost

**Source:** `packages/tools/src/tools/ml_models/train/cost.py`
**Kind:** module

## Purpose

This module counts parameters and FLOPs for a classifier or segmentor graph.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `count_params` | function | Sum of parameter elements |
| `count_flops` | function | FLOPs for one forward pass |

## Inputs and outputs

`count_params(model) -> int`.

`count_flops(model, input_shape) -> int`. `input_shape` is an NCHW tuple.

## Behavior

1. `count_params` sums `numel()` over every parameter.
2. `count_flops` runs one eval-mode forward pass on a zero tensor of
   `input_shape` and reads `FlopCounterMode.get_total_flops()`.

## Errors and faults

None.

## Messages

None.

## Configuration

None.

## Constraints

Torch is a required tools dependency. The FLOP count is the torch counter
total for that dummy pass.

## Related documents

- [`tools.ml_models.train`](../train.md)
- [`tools.ml_models.train.loop`](loop.md)
