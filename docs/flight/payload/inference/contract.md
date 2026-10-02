# flight.payload.inference.contract

**Source:** `packages/flight/src/flight/payload/inference/contract.py`
**Kind:** pure module

## Purpose

This module defines the shape formulas for flight's GSD-conditioned ONNX models.
The classifier and segmentor each consume an image tile plus its two-axis GSD
encoding.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `Shape` | type alias | Tensor dimensions with `None` for dynamic dimensions |
| `verify_conditioned_shapes` | function | Checks image, GSD, and output shapes |

## Inputs and outputs

`verify_conditioned_shapes(image, gsd, output, channels, tile_hw, kind)` returns
`Result[None, FaultCode]`.

## Behavior

- Image shape is `(None, channels, H, W)` and GSD shape is `(None, 2)`.
- A classifier output is `(None, 1)`; a segmentor output is
  `(None, 1, H, W)`.
- Batch is always dynamic. When `tile_hw` is supplied, image and segmentor
  spatial dimensions must equal it exactly. Without a tile size, spatial
  dimensions may be dynamic but concrete image and output dimensions agree.
- Invalid kinds, ranks, or dimensions return `Err(MODEL_CORRUPT)`.

## Errors and faults

`Err(FaultCode.MODEL_CORRUPT)` indicates a shape contract violation.

## Messages

None.

## Configuration

None. Callers supply channel count, tile size, and model kind.

## Constraints

The function is pure and imports neither onnxruntime nor filesystem APIs.

## Related documents

- [`flight.payload.inference.onnx_session`](onnx_session.md)
- [`flight.payload.inference.verify`](verify.md)
