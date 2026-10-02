# tools.ml_models.dataset.gsd

**Source:** `packages/tools/src/tools/ml_models/dataset/gsd.py`
**Kind:** module

## Purpose

This module encodes stored GSD metres for the model's second input. It
carries no pixel conversion: finished datasets already store float32 unit
pixels.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `to_model_gsd` | function | `ln(gsd / gsd_reference_m)` per component |

## Inputs and outputs

`to_model_gsd(gsd_m, reference_m) -> np.ndarray[float32, (..., 2)]` delegates
to the flight GSD encoder, takes lateral then along-track metres along a
trailing length-2 axis, and returns the log ratio to the reference.

## Behavior

1. GSD encoding uses the flight footprint contract. Its `Result` errors
   become `ValueError` for the dataset tools API.
2. A GSD equal to the reference encodes to the zero vector.

## Errors and faults

`ValueError` when the flight GSD encoder rejects the input, including an
invalid reference or a GSD component that is not finite and greater than 0.

## Messages

None.

## Configuration

The GSD reference comes from `BuildSpec.gsd_reference_m`. There is no TOML
file in this module.

## Constraints

This module does not import torch. Stored pixels are float32 unit values;
no normalization or quantization happens in this module.

## Related documents

- [`tools.ml_models.dataset`](../dataset.md)
- [`tools.ml_models.dataset.build`](build.md)
- [`tools.ml_models.dataset.loader`](loader.md)
- [`flight.payload.gimbal.footprint`](../../../../flight/payload/gimbal/footprint.md)
