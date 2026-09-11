# tools.inference.annotations

**Source:** `packages/tools/src/tools/inference/annotations.py`
**Kind:** module

## Purpose

This module reads polygon segmentation labels for the Zenodo 4250706 smoke-plume
corpus. The corpus stores label-studio annotation exports, not raster masks.
Each JSON file holds zero or more `smoke` polygons with vertices as percentages
of the image extent.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SMOKE_LABEL` | constant | Polygon class name (`smoke`) |
| `LABEL_SUFFIX` | constant | Filename suffix (`_features`) |
| `annotation_stem` | function | Image stem for a label filename |
| `parse_polygons` | function | Percentage-space polygons from decoded JSON |
| `load_polygons` | function | Percentage-space polygons from one file |
| `rasterize_polygons` | function | Polygons to a binary mask |
| `load_annotation_mask` | function | One file to a `(1, H, W)` mask |

## Inputs and outputs

`annotation_stem(name) -> str`.

`parse_polygons(data, label=SMOKE_LABEL) -> tuple[np.ndarray, ...]`. Each array
is `(V, 2)` float32 with vertices in `[0, 100]`.

`load_polygons(path, label=SMOKE_LABEL) -> tuple[np.ndarray, ...]`.

`rasterize_polygons(polygons, height, width) -> np.ndarray` with shape
`(height, width)` and values in `{0, 1}`.

`load_annotation_mask(path, height, width, label=SMOKE_LABEL) -> np.ndarray`
with shape `(1, height, width)`.

## Behavior

1. `annotation_stem` strips a `.json` suffix and the trailing `_features`
   marker from a label filename.
2. `parse_polygons` walks `completions` and `result` entries. It keeps
   `polygonlabels` results whose class list contains `label`. Polygons with
   fewer than three vertices are dropped. Non-dict input returns an empty tuple.
3. `load_polygons` reads and decodes one annotation file, then calls
   `parse_polygons`.
4. `rasterize_polygons` scales percentage vertices to pixel coordinates,
   tests pixel centres at the half-pixel offset, and unions overlapping
   polygons. No polygons yields an all-zero mask.
5. `load_annotation_mask` loads polygons and rasterizes them with a leading
   channel dimension.

## Errors and faults

`OSError` and `json.JSONDecodeError` on a missing or malformed annotation file.

`ValueError` when `height` or `width` is below one.

## Messages

None.

## Configuration

Default polygon class is `smoke`. Default label filename suffix is `_features`.

## Constraints

Depends on numpy and matplotlib for point-in-polygon tests. Percentage vertices
are rasterized at the requested resolution without resampling from a fixed source
size.

## Related documents

- [`tools.inference.fetch`](fetch.md)
- [`tools.inference.data`](data.md)
- [`tools.inference`](../inference.md)
