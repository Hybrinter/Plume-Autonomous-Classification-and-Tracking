# tools.ml_models.dataset.preprocess

**Source:** `packages/tools/src/tools/ml_models/dataset/preprocess.py`
**Kind:** module

## Purpose

This module maps raw tile pixels to the unit interval, packs them on the
uint16 storage grid, and encodes GSD for the model.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `IMAGE_SCALE` | constant | uint16 storage full scale, 65535 |
| `to_unit` | function | DN or unit image to float32 unit interval |
| `quantize_unit` | function | Unit image to uint16 storage grid |
| `dequantize_unit` | function | Stored uint16 back to float32 unit |
| `to_model_gsd` | function | `ln(gsd / gsd_reference_m)` per component |

## Inputs and outputs

`to_unit(image, domain, bit_depth) -> np.ndarray[float32, (C, H, W)]`.
Domain `dn` passes the image and bit depth through
`normalize_dn`. Domain `unit` clips to `[0, 1]`.

`quantize_unit(image) -> np.ndarray[uint16, (C, H, W)]` computes
`round(clip(image) * 65535)`.

`dequantize_unit(image) -> np.ndarray[float32, (C, H, W)]` divides by
65535.

`to_model_gsd(gsd_m, reference_m) -> np.ndarray[float32, (..., 2)]` delegates
to the flight GSD encoder, takes lateral then along-track metres along a
trailing length-2 axis, and returns the log ratio to the reference.

## Behavior

1. The uint16 round trip is exact on the 65535 grid.
2. Storage is a lossy quantization: the absolute error on a stored unit
   value is at most 0.5 / 65535. Normalized float values are not exactly
   preserved. A 12-bit integer DN can be recovered exactly by rounding the
   stored value back onto the DN grid.
3. DN normalization uses flight `normalize_dn`; GSD encoding uses the flight
   footprint contract. Their `Result` errors become `ValueError` for the
   dataset tools API.

## Errors and faults

`ValueError` on an unknown domain, a `bit_depth` below 1, or a GSD input
that the flight GSD encoder rejects, including an invalid reference or a GSD
component that is not finite and greater than 0.

## Messages

None.

## Configuration

`IMAGE_SCALE` is fixed at 65535. The GSD reference comes from
`BuildSpec.gsd_reference_m`. There is no TOML file in this module.

## Constraints

This module calls `flight.payload.preprocess.normalize.normalize_dn` for
the `dn` domain. It does not import torch.

## Related documents

- [`tools.ml_models.dataset`](../dataset.md)
- [`tools.ml_models.dataset.build`](build.md)
- [`tools.ml_models.dataset.loader`](loader.md)
- [`flight.payload.preprocess.normalize`](../../../../flight/payload/preprocess/normalize.md)
