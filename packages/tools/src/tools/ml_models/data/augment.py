"""Geometric transforms shared by a chip and its mask.

Contains:
  - dihedral: one of eight flips and rotations, applied to image and mask.
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
