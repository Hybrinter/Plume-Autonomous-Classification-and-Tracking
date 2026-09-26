"""Polygon segmentation labels for the Zenodo 4250706 corpus.

The corpus ships segmentation labels as label-studio annotation exports, not as
raster masks: one JSON file per annotated image, holding zero or more ``smoke``
polygons. Each polygon stores its vertices as percentages of the image extent,
so a vertex is in ``[0, 100]`` regardless of pixel size. The dataset README
states the same fact as a scale factor of 1.2 for its 120 px tiles, which is
what a percentage becomes at that size.

Ten of the 1,437 annotated images carry no polygon. Those are genuine
negatives, and rasterise to an empty mask rather than being dropped.

Contains:
  - SMOKE_LABEL: the only polygon class in the corpus.
  - LABEL_SUFFIX: filename suffix that separates a label from its image stem.
  - annotation_stem: image stem for a label filename.
  - parse_polygons: percentage-space polygons from a decoded annotation.
  - load_polygons: percentage-space polygons from one annotation file.
  - rasterize_polygons: polygons to a binary mask.
  - load_annotation_mask: one annotation file to a (1, H, W) mask.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SMOKE_LABEL = "smoke"
LABEL_SUFFIX = "_features"

_POLYGON_TYPE = "polygonlabels"
_PERCENT_FULL_SCALE = 100.0
_MIN_POLYGON_VERTICES = 3


def annotation_stem(name: str) -> str:
    """Return the image stem that an annotation filename refers to.

    Args:
        name: Annotation filename or stem, such as
            ``10003_2019-01-21T10:56:41.330Z_0_features``.

    Returns:
        str: Stem with any ``.json`` suffix and the trailing ``_features``
        marker removed, which is what the matching GeoTIFF is called.
    """
    stem = Path(name).stem if name.endswith(".json") else name
    if stem.endswith(LABEL_SUFFIX):
        return stem[: -len(LABEL_SUFFIX)]
    return stem


def parse_polygons(data: object, label: str = SMOKE_LABEL) -> tuple[np.ndarray, ...]:
    """Return the polygons of a decoded annotation in percentage space.

    Args:
        data: Decoded annotation JSON.
        label: Polygon class to keep.

    Returns:
        tuple[np.ndarray, ...]: One ``(V, 2)`` float32 array per polygon, each
        row an ``(x, y)`` vertex in ``[0, 100]``. Degenerate polygons with
        fewer than three vertices are dropped. An image annotated as having no
        plume yields an empty tuple.
    """
    if not isinstance(data, dict):
        return ()
    polygons: list[np.ndarray] = []
    for completion in data.get("completions", []):
        for result in completion.get("result", []):
            if result.get("type") != _POLYGON_TYPE:
                continue
            value = result.get("value", {})
            if label not in value.get("polygonlabels", []):
                continue
            points = np.asarray(value.get("points", []), dtype=np.float32)
            if points.ndim == 2 and points.shape[0] >= _MIN_POLYGON_VERTICES:
                polygons.append(points)
    return tuple(polygons)


def load_polygons(path: str | Path, label: str = SMOKE_LABEL) -> tuple[np.ndarray, ...]:
    """Return the polygons of one annotation file in percentage space.

    Args:
        path: Annotation JSON file.
        label: Polygon class to keep.

    Returns:
        tuple[np.ndarray, ...]: Polygons as described by :func:`parse_polygons`.

    Raises:
        OSError / json.JSONDecodeError: on a missing or malformed file.
    """
    return parse_polygons(json.loads(Path(path).read_text(encoding="utf-8")), label=label)


def rasterize_polygons(
    polygons: tuple[np.ndarray, ...],
    height: int,
    width: int,
) -> np.ndarray:
    """Fill percentage-space polygons into a binary mask.

    Args:
        polygons: Percentage-space ``(V, 2)`` vertex arrays as returned by
            :func:`load_polygons`.
        height: Output mask height in pixels.
        width: Output mask width in pixels.

    Returns:
        np.ndarray[float32, (height, width)] in {0, 1}. Overlapping polygons
        union together. No polygons gives an all-zero mask.

    Raises:
        ValueError: If ``height`` or ``width`` is below one.

    Notes:
        Percentages are scaled to the requested pixel grid directly, so the
        mask is produced at the target resolution instead of being rasterised
        at the source size and resampled. Pixel centres, at the half-pixel
        offset, decide membership.
    """
    if height < 1 or width < 1:
        raise ValueError(f"mask size must be positive; got {height}x{width}")
    mask = np.zeros((height, width), dtype=bool)
    if not polygons:
        return mask.astype(np.float32)
    from matplotlib.path import Path as MplPath

    ys, xs = np.mgrid[0:height, 0:width]
    centres = np.column_stack((xs.ravel() + 0.5, ys.ravel() + 0.5)).astype(np.float64)
    for polygon in polygons:
        scaled = polygon.astype(np.float64) / _PERCENT_FULL_SCALE
        scaled[:, 0] *= float(width)
        scaled[:, 1] *= float(height)
        inside = MplPath(scaled).contains_points(centres)
        mask |= inside.reshape(height, width)
    return mask.astype(np.float32)


def load_annotation_mask(
    path: str | Path,
    height: int,
    width: int,
    label: str = SMOKE_LABEL,
) -> np.ndarray:
    """Return the (1, H, W) mask for one annotation file.

    Args:
        path: Annotation JSON file.
        height: Output mask height.
        width: Output mask width.
        label: Polygon class to keep.

    Returns:
        np.ndarray[float32, (1, height, width)] in {0, 1}.

    Raises:
        OSError / json.JSONDecodeError: on a missing or malformed file.
        ValueError: If the requested size is not positive.
    """
    polygons = load_polygons(path, label=label)
    mask = rasterize_polygons(polygons, height, width)
    return mask[np.newaxis, ...]
