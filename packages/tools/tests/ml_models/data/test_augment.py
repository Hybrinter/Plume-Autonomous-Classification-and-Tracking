"""Dihedral transforms and feathered paste."""

from __future__ import annotations

import numpy as np
import pytest
from tools.ml_models.data.augment import dihedral, feather_paste


def test_dihedral_keeps_image_and_mask_together() -> None:
    """The same flip and rotation land on the image and the mask."""
    image = np.zeros((2, 2, 3), dtype=np.float32)
    image[0, 0, 0] = 4.0
    image[1, 1, 2] = 7.0
    mask = np.zeros((1, 2, 3), dtype=np.float32)
    mask[0, 0, 0] = 1.0
    identity, identity_mask = dihedral(image, mask, 0)
    assert np.array_equal(identity, image)
    assert np.array_equal(identity_mask, mask)
    flipped, flipped_mask = dihedral(image, mask, 4)
    assert flipped[0, 0, -1] == 4.0
    assert flipped_mask[0, 0, -1] == 1.0
    turned, turned_mask = dihedral(image, mask, 1)
    assert turned.shape == (2, 3, 2)
    assert turned_mask.shape == (1, 3, 2)
    assert np.array_equal(turned[0] > 0.0, turned_mask[0] > 0.0)
    with pytest.raises(ValueError, match="0..7"):
        dihedral(image, mask, 8)


def test_feather_replaces_the_interior_and_blends_the_border() -> None:
    """Interior pixels copy the chip. Border pixels blend with the canvas."""
    canvas = np.zeros((1, 6, 6), dtype=np.float32)
    chip = np.ones((1, 4, 4), dtype=np.float32)
    feather_paste(canvas, chip, 1, 1, 1)
    assert canvas[0, 1, 1] == pytest.approx(0.5)
    assert canvas[0, 2, 2] == pytest.approx(1.0)
    assert canvas[0, 0, 0] == 0.0
    assert canvas[0, 4, 4] == pytest.approx(0.5)
