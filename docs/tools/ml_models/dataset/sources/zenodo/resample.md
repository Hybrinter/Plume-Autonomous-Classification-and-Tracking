# tools.ml_models.dataset.sources.zenodo.resample

**Source:** `packages/tools/src/tools/ml_models/dataset/sources/zenodo/resample.py`
**Kind:** module

## Purpose

This module downsamples channel-major image planes by area overlap, so a
coarser GSD bin averages the native pixels it covers.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `resample_area` | function | `(C, H, W)` planes to a `(h, w)` output grid |

## Inputs and outputs

`resample_area(planes, tile_hw)` takes a finite `(C, H, W)` array and a
positive output size, and returns float32 `(C, h, w)`. Each output cell is
the area-weighted mean of the source pixels it overlaps.

## Behavior

1. Per-axis edges split `[0, source)` into `target` uniform intervals; the
   weight of a source pixel is its overlap length with the target cell.
2. Both axes apply through one `einsum`, so output energy is the
   area-weighted source mean: a constant input stays constant.
3. Upsampling is rejected. A target equal to the source size reduces to a
   normalized identity resample.

## Errors and faults

`ValueError` on a non-3-D input, an empty or non-finite array, a
non-positive output size, or an output side larger than the source side.

## Messages

None.

## Configuration

None.

## Constraints

This module does not import torch. No detail is synthesized: the output
grid may only retain or coarsen native resolution.

## Related documents

- [`tools.ml_models.dataset.sources.zenodo`](../zenodo.md)
- [`tools.ml_models.dataset.sources.zenodo.bins`](bins.md)
- [`tools.ml_models.dataset.sources.zenodo.adapt`](adapt.md)
