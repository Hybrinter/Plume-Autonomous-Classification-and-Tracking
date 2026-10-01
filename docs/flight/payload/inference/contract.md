# flight.payload.inference.contract

**Source:** `packages/flight/src/flight/payload/inference/contract.py`
**Kind:** pure module

## Purpose

This module is the single implementation of the shape formulas for the
GSD-conditioned two-input flight graph: `image` `(N, C, H, W)` plus `gsd`
`(N, 2)` in, conditioned logits out. Both tools-side export/session
validation and flight consumers call this verifier; no other module restates
the shape rules.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `Shape` | type alias | `tuple[int | None, ...]`; `None` is a dynamic dim |
| `verify_conditioned_shapes` | function | Checks image/gsd/output shapes for a model kind |

## Inputs and outputs

`verify_conditioned_shapes(image, gsd, output, channels, tile_hw, kind)`
returns `Result[None, FaultCode]`.

## Behavior

1. `image` must be rank 4 with a dynamic batch (`None`), exactly `channels`
   channels, and spatial dims that are either `None` or the `tile_hw` size.
2. `gsd` must equal `(None, 2)`.
3. A `classifier` output must equal `(None, 1)`; a `segmentor` output must be
   rank 4 with head dims `(None, 1)` and spatial dims dynamic or `tile_hw`.
4. Any other `kind` or shape returns `Err(FaultCode.MODEL_CORRUPT)`.

## Errors and faults

- `Err(FaultCode.MODEL_CORRUPT)` on any contract violation.
- `Ok(None)` on conformance.

## Constraints

- Pure function: no I/O, no clock, no SDK imports.
- The batch dim must be dynamic on every I/O tensor; fixed-batch graphs fail.
- `tools.ml_models.export` re-exports this verifier; the formulas are not duplicated.

## Messages

None.

## Configuration

None.

## Related documents

- [flight.payload.inference](inference.md)
- [flight.payload.inference.verify](inference/verify.md)
- [tools.ml_models.export.contract](../../tools/ml_models/export/contract.md)
