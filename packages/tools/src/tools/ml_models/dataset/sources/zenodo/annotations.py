"""Label Studio smoke polygons and rectangular mask rasterization.

Contains:
  - parse_polygons: smoke polygons from one annotation file; ``None`` when
    every smoke entry is degenerate.
  - rasterize_percent_mask: binary mask from percent-coordinate polygons.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def parse_polygons(payload: object) -> tuple[np.ndarray, ...] | None:
    """Read smoke polygons in percent coordinates, preserving annotated negatives.

    Args:
        payload: Decoded Label Studio JSON for one annotation file.

    Returns:
        tuple[np.ndarray, ...] | None: Each kept polygon is float64 ``(V, 2)``
        with ``V >= 3``. ``()`` when the file has no smoke ``polygonlabels``
        entry (an annotated negative). ``None`` when smoke entries exist but
        none is kept, so the file gives no usable geometry.

    Raises:
        ValueError: On a malformed payload, or on smoke vertices that are not
            (x, y) pairs, are non-finite, or lie outside 0-100 percent.

    Notes:
        An empty ``points`` list or a polygon with fewer than three vertices
        has no area and is dropped; the corpus contains stray single-click
        polygons. Such an entry still counts as a smoke entry, so a file whose
        smoke entries are all dropped returns ``None``, not ``()``. Callers
        also use ``None`` for a stem with no annotation file.
    """
    if not isinstance(payload, dict):
        raise ValueError("annotation must be a JSON object")
    polygons: list[np.ndarray] = []
    smoke_seen = False
    completions = payload.get("completions", [])
    if not isinstance(completions, list):
        raise ValueError("annotation completions must be a list")
    for completion in completions:
        if not isinstance(completion, dict):
            raise ValueError("annotation completion must be an object")
        for result in completion.get("result", []):
            if not isinstance(result, dict) or result.get("type") != "polygonlabels":
                continue
            value = result.get("value", {})
            if not isinstance(value, dict) or "smoke" not in value.get("polygonlabels", []):
                continue
            smoke_seen = True
            points = np.asarray(value.get("points", []), dtype=np.float64)
            if points.size == 0:
                continue
            if points.ndim != 2 or points.shape[1] != 2:
                raise ValueError("smoke polygon vertices must be (x, y) pairs")
            if not np.all(np.isfinite(points)) or np.any(points < 0) or np.any(points > 100):
                raise ValueError("smoke polygon must have finite percent-coordinate vertices")
            if len(points) < 3:
                continue
            polygons.append(points)
    if smoke_seen and not polygons:
        return None
    return tuple(polygons)


def rasterize_percent_mask(
    polygons: Sequence[np.ndarray],
    tile_hw: tuple[int, int],
    *,
    rule: str = "half",
    source_hw: tuple[int, int] | None = None,
    fitted_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    """Supersample each cell on a 4-by-4 grid and union polygons before thresholding."""
    from matplotlib.path import Path

    height, width = tile_hw
    if min(tile_hw) < 1 or rule not in ("half", "touch"):
        raise ValueError("invalid output size or mask coverage rule")
    offsets = (np.arange(4, dtype=np.float64) + 0.5) / 4
    xs = (np.arange(width)[:, None] + offsets).reshape(-1) * (100 / width)
    ys = (np.arange(height)[:, None] + offsets).reshape(-1) * (100 / height)
    x, y = np.meshgrid(xs, ys)
    samples = np.column_stack((x.ravel(), y.ravel()))
    if (source_hw is None) != (fitted_hw is None):
        raise ValueError("source and fitted mask geometry must be specified together")
    if source_hw is not None and fitted_hw is not None:
        if min(*source_hw, *fitted_hw) < 1:
            raise ValueError("mask fitting dimensions must be positive")
        # Undo the image's origin crop / edge pad before testing percentage polygons.
        for axis, source, fitted in (
            (0, source_hw[1], fitted_hw[1]),
            (1, source_hw[0], fitted_hw[0]),
        ):
            pixels = samples[:, axis] * fitted / 100
            pixels = np.where(pixels >= source, source - 0.5, pixels)
            samples[:, axis] = pixels * 100 / source
    inside = np.zeros(len(samples), dtype=bool)
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or len(points) < 3
            or not np.all(np.isfinite(points))
        ):
            raise ValueError("invalid mask polygon")
        if not np.array_equal(points[0], points[-1]):
            points = np.vstack((points, points[0]))
        inside |= Path(points, closed=True).contains_points(samples, radius=1e-6)
    fractions = inside.reshape(height, 4, width, 4).mean(axis=(1, 3))
    mask = fractions >= 0.5 if rule == "half" else fractions > 0
    return np.asarray(mask.astype(np.uint8)[None])
