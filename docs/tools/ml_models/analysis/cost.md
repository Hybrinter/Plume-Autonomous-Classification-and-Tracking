# tools.ml_models.analysis.cost

**Source:** `packages/tools/src/tools/ml_models/analysis/cost.py`
**Kind:** module

## Purpose

This module counts model parameters and forward-pass FLOPs for
architecture records in run summaries and Pareto tables.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `count_params` | function | Total trainable parameters of a torch module |
| `count_flops` | function | Forward FLOPs for one input shape via `FlopCounterMode` |

## Inputs and outputs

`count_params(model) -> int`. `count_flops(model, input_shape) -> int`
expects a full `(N, C, H, W)` tuple.

## Behavior

`count_params` sums `numel` over parameters. `count_flops` runs one forward
pass under `torch.utils.flop_counter.FlopCounterMode` on a zero input.

## Errors and faults

`count_flops` raises whatever the forward pass or flop counter raises.

## Messages

None.

## Configuration

None.

## Constraints

- Models run on CPU with a zero tensor.
- Only single-input module calls are counted; conditioned models are
  profiled through wrapper modules.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.arch.registry`](../arch/registry.md)
