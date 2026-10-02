# tools.ml_models.dataset.sources.zenodo.annotations

**Source:** `packages/tools/src/tools/ml_models/dataset/sources/zenodo/annotations.py`
**Kind:** module

## Purpose

This module parses Label Studio smoke polygons in percent coordinates and
rasterizes them onto a target tile grid.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `parse_polygons` | function | Smoke polygons from one annotation payload |
| `rasterize_percent_mask` | function | `(V, 2)` percent polygons to a `(1, h, w)` mask |

## Inputs and outputs

`parse_polygons(payload)` takes a decoded Label Studio JSON object and
returns a tuple of float64 `(V, 2)` percent-coordinate polygons. A caller
uses `None` for a missing annotation file and `()` for an annotation with
no smoke polygon. `rasterize_percent_mask(polygons, tile_hw, rule="half")`
returns a uint8 `(1, h, w)` mask.

## Behavior

1. `completions` is a list of objects; each `result` item of type
   `polygonlabels` whose labels include `smoke` contributes its `points`.
2. Polygon vertices must be finite percent coordinates in `[0, 100]` with
   at least three points.
3. Rasterization supersamples each output cell on a 4-by-4 grid, unions all
   polygons, then thresholds coverage: `half` requires at least half the
   subsamples, `touch` requires any subsample.

## Errors and faults

`ValueError` on a non-object payload, a non-list `completions`, a
non-object completion, a malformed smoke polygon, a non-positive output
size, an unknown coverage rule, or an invalid mask polygon.

## Messages

None.

## Configuration

None. The coverage rule defaults to `half`; `matplotlib.path` performs the
point-in-polygon tests.

## Constraints

This module does not import torch.

## Related documents

- [`tools.ml_models.dataset.sources.zenodo`](../zenodo.md)
- [`tools.ml_models.dataset.sources.zenodo.archive`](archive.md)
- [`tools.ml_models.dataset.sources.zenodo.adapt`](adapt.md)
