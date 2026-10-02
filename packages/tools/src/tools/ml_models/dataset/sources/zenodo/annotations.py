"""Label Studio smoke polygons and rectangular mask rasterization."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def parse_polygons(payload: object) -> tuple[np.ndarray, ...]:
    """Read smoke polygons in percent coordinates, preserving annotated negatives.

    A caller uses None for missing annotation files, and () for empty files.
    """
    if not isinstance(payload, dict):
        raise ValueError("annotation must be a JSON object")
    polygons: list[np.ndarray] = []
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
            points = np.asarray(value.get("points", []), dtype=np.float64)
            if (
                points.ndim != 2
                or points.shape[1] != 2
                or len(points) < 3
                or not np.all(np.isfinite(points))
                or np.any(points < 0)
                or np.any(points > 100)
            ):
                raise ValueError("smoke polygon must have finite percent-coordinate vertices")
            polygons.append(points)
    return tuple(polygons)


def rasterize_percent_mask(
    polygons: Sequence[np.ndarray],
    tile_hw: tuple[int, int],
    *,
    rule: str = "half",
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
