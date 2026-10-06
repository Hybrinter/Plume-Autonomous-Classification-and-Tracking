# tools.ml_models.analysis.cost

**Source:** `packages/tools/src/tools/ml_models/analysis/cost.py`
**Kind:** module
**Status:** implemented

## Purpose

This module measures parameters and a dense Conv2d/Linear operation
lower bound for the two-input conditioned models. Counts are explicitly
PARTIAL: they are not total graph FLOPs and not a latency estimate.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `count_params` | function | Total parameters of a torch module |
| `measure_resources` | function | Partial per-shape resource evidence |
| `count_flops` | function | Accessor for the same partial count |
| `ResourceEvidence` | dataclass | Parameters, shapes, counts, coverage, limitations |
| `OperationCost` | dataclass | One executed supported module call |

## Inputs and outputs

`count_params(model) -> int` sums `numel` over parameters.

`measure_resources(model, input_shape) -> Result[ResourceEvidence, str]`
expects a positive integer `(batch, channels, height, width)` and a
single-device, single-floating-dtype two-input module. It calls
`model(image, encoded_gsd)` on zeros at the reference GSD under
`inference_mode`, counting forward hooks on leaf modules.

`count_flops(model, input_shape) -> Result[int, str]` returns the same
`counted_flops` field.

## Behavior

- Each executed `Conv2d` or `Linear` leaf call contributes
  `2 * output.numel() * fan_in` (kernel product over in-channels per
  group for convolutions).
- Bias, activation, normalization, pooling, interpolation, FiLM
  arithmetic, and functional operations are excluded.
- Non-finite or batch-mismatched outputs return `Err`.
- Module training modes, RNG state, device, and dtype are preserved.

## Errors and faults

`Err` on malformed input shape, mixed device or dtype, unsupported
model behavior, or forward errors.

## Messages

None.

## Configuration

None.

## Constraints

- Coverage is `PARTIAL`; `limitations` enumerate the excluded work.
- No hardware counter, memory, optimizer, backward, or latency claim.

## Related documents

- [`tools.ml_models.analysis`](../analysis.md)
- [`tools.ml_models.arch.registry`](../arch/registry.md)
- [`tools.ml_models.train.loop`](../train/loop.md)
