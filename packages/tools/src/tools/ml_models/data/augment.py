"""Geometric transforms shared by a chip and its mask.

Contains:
  - dihedral: one of eight flips and rotations, applied to image and mask.
  - feather_paste: in-place border blend of a chip onto a canvas.
"""

from __future__ import annotations

import numpy as np


def _require_pair(image: np.ndarray, mask: np.ndarray) -> tuple[int, int]:
    """Return ``(H, W)`` when image and mask share a spatial grid.

    Args:
        image: Array ``(C, H, W)``.
        mask: Array ``(1, H, W)``.

    Returns:
        tuple[int, int]: Height and width.

    Raises:
        ValueError: If the ranks or spatial sizes disagree.
    """
    if image.ndim != 3 or image.shape[0] < 1 or image.shape[1] < 1 or image.shape[2] < 1:
        raise ValueError(f"image must have shape (C, H, W); got {image.shape}")
    height = int(image.shape[1])
    width = int(image.shape[2])
    if mask.shape != (1, height, width):
        raise ValueError(f"mask must have shape {(1, height, width)}; got {mask.shape}")
    return height, width


def dihedral(image: np.ndarray, mask: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Apply one dihedral transform to an image and a mask.

    Args:
        image: Float array ``(C, H, W)``.
        mask: Float array ``(1, H, W)``.
        k: Index in ``0..7``. ``k % 4`` is the number of counterclockwise
            quarter turns. ``k >= 4`` flips the width axis before those turns.

    Returns:
        tuple[np.ndarray, np.ndarray]: Contiguous float32 image and mask.
        Both arrays receive the same flip and rotation.

    Raises:
        ValueError: If ``k`` is outside ``0..7`` or the shapes disagree.
    """
    _require_pair(image, mask)
    if isinstance(k, bool) or not isinstance(k, int) or k < 0 or k > 7:
        raise ValueError(f"k must be an int in 0..7; got {k!r}")
    out_image = np.asarray(image, dtype=np.float32)
    out_mask = np.asarray(mask, dtype=np.float32)
    if k >= 4:
        out_image = np.flip(out_image, axis=2)
        out_mask = np.flip(out_mask, axis=2)
    turns = k % 4
    if turns:
        out_image = np.rot90(out_image, turns, axes=(1, 2))
        out_mask = np.rot90(out_mask, turns, axes=(1, 2))
    return (
        np.ascontiguousarray(out_image, dtype=np.float32),
        np.ascontiguousarray(out_mask, dtype=np.float32),
    )


def overlap_window(
    canvas_hw: tuple[int, int],
    chip_hw: tuple[int, int],
    top: int,
    left: int,
) -> tuple[int, int, int, int, int, int] | None:
    """Return the overlapping source and destination rectangles.

    Args:
        canvas_hw: Canvas ``(height, width)``.
        chip_hw: Chip ``(height, width)``.
        top: Destination row of the chip's first row. Negative values clip
            the chip.
        left: Destination column of the chip's first column.

    Returns:
        tuple[int, int, int, int, int, int] | None: ``src_y``, ``src_x``,
        ``dst_y``, ``dst_x``, ``height``, ``width``. ``None`` when the chip
        misses the canvas.
    """
    canvas_h, canvas_w = canvas_hw
    chip_h, chip_w = chip_hw
    src_y = max(0, -top)
    src_x = max(0, -left)
    dst_y = max(0, top)
    dst_x = max(0, left)
    height = min(chip_h - src_y, canvas_h - dst_y)
    width = min(chip_w - src_x, canvas_w - dst_x)
    if height <= 0 or width <= 0:
        return None
    return src_y, src_x, dst_y, dst_x, height, width


def _feather_alpha(chip_h: int, chip_w: int, feather_px: int) -> np.ndarray:
    """Return per-pixel blend weights, shape ``(chip_h, chip_w)``.

    Args:
        chip_h: Chip height.
        chip_w: Chip width.
        feather_px: Border thickness in pixels. ``0`` is a hard replace.

    Returns:
        np.ndarray[float32, (chip_h, chip_w)]: ``1`` on the interior. On the
        outer ``feather_px`` pixels, the weight is ``(inset + 1) / (feather_px + 1)``.
        ``inset`` is the distance in pixels to the nearest chip edge.
    """
    if feather_px <= 0:
        return np.ones((chip_h, chip_w), dtype=np.float32)
    rows = np.arange(chip_h, dtype=np.float32)
    cols = np.arange(chip_w, dtype=np.float32)
    dist_y = np.minimum(rows, np.float32(chip_h - 1) - rows)
    dist_x = np.minimum(cols, np.float32(chip_w - 1) - cols)
    dist = np.minimum(dist_y[:, None], dist_x[None, :])  # np.ndarray[float32, (H, W)]
    alpha = np.ones((chip_h, chip_w), dtype=np.float32)
    border = dist < float(feather_px)
    alpha[border] = (dist[border] + 1.0) / float(feather_px + 1)
    return alpha


def feather_paste(
    canvas: np.ndarray,
    chip: np.ndarray,
    top: int,
    left: int,
    feather_px: int,
) -> None:
    """Blend ``chip`` onto ``canvas`` in place.

    Args:
        canvas: Array ``(C, H, W)`` updated in place.
        chip: Array ``(C, h, w)``. The interior replaces the canvas. The outer
            ``feather_px`` pixels blend ``alpha * chip + (1 - alpha) * canvas``.
        top: Destination row of the chip origin. May be negative.
        left: Destination column of the chip origin. May be negative.
        feather_px: Border thickness. ``0`` replaces every overlapping pixel.

    Returns:
        None.

    Raises:
        ValueError: If the arrays are not ``(C, H, W)``, the channel counts
            differ, or ``feather_px`` is negative.

    Notes:
        This function does not read or write a mask.
    """
    if isinstance(feather_px, bool) or not isinstance(feather_px, int) or feather_px < 0:
        raise ValueError(f"feather_px must be an int >= 0; got {feather_px!r}")
    if canvas.ndim != 3 or chip.ndim != 3:
        raise ValueError(
            f"canvas and chip must have shape (C, H, W); got {canvas.shape}, {chip.shape}"
        )
    if int(canvas.shape[0]) != int(chip.shape[0]) or int(canvas.shape[0]) < 1:
        raise ValueError(f"channel count {canvas.shape[0]} != {chip.shape[0]}")
    chip_h = int(chip.shape[1])
    chip_w = int(chip.shape[2])
    if chip_h < 1 or chip_w < 1 or int(canvas.shape[1]) < 1 or int(canvas.shape[2]) < 1:
        raise ValueError("canvas and chip spatial axes must be positive")
    window = overlap_window(
        (int(canvas.shape[1]), int(canvas.shape[2])), (chip_h, chip_w), top, left
    )
    if window is None:
        return
    src_y, src_x, dst_y, dst_x, height, width = window
    alpha = _feather_alpha(chip_h, chip_w, feather_px)
    weights = alpha[src_y : src_y + height, src_x : src_x + width]  # np.ndarray[float32, (h, w)]
    dst = canvas[:, dst_y : dst_y + height, dst_x : dst_x + width]
    src = np.asarray(chip[:, src_y : src_y + height, src_x : src_x + width], dtype=np.float32)
    blended = weights[None, :, :] * src + (1.0 - weights[None, :, :]) * dst
    dst[...] = blended
