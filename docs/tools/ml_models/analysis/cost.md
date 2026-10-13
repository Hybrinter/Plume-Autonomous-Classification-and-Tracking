# tools.ml_models.analysis.cost

**Source:** `packages/tools/src/tools/ml_models/analysis/cost.py`
**Kind:** module

## Purpose

This module counts model parameters. The one-input FLOP executor is
removed; the FLOP boundary is unavailable until conditioned resource
measurement lands.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `count_params` | function | Total parameters of a torch module |
| `count_flops` | function | FLOP boundary; returns `Err` while unimplemented |

## Inputs and outputs

`count_params(model) -> int`. `count_flops(model, input_shape) ->
Result[int, str]` expects a full `(N, C, H, W)` tuple.

## Behavior

`count_params` sums `numel` over parameters. `count_flops` returns an
explicit unavailable error and never executes the model.

## Errors and faults

`count_flops` always returns `Err`.

## Messages

None.

## Configuration

None.

## Constraints

- No dummy-input forward pass runs at the FLOP boundary.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.arch.registry`](../arch/registry.md)
