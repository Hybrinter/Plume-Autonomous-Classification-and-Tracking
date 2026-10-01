"""Tests for dihedral elements."""

import numpy as np
import pytest
from tools.ml_models.dataset.augment import (
    ELEMENT_NAMES,
    SHAPE_PRESERVING,
    apply_dihedral,
    legal_elements,
)


def test_square_tile_keeps_eight_distinct_elements() -> None:
    """A square tile has eight elements and they do not repeat."""
    image = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
    assert legal_elements(4, 4) == ELEMENT_NAMES
    seen: set[bytes] = set()
    for name in ELEMENT_NAMES:
        out = apply_dihedral(image, name)
        assert out.shape == image.shape
        seen.add(out.tobytes())
    assert len(seen) == 8


def test_nonsquare_tile_keeps_shape_preserving_elements() -> None:
    """A non-square tile drops 90 degree and transpose elements."""
    assert legal_elements(4, 6) == SHAPE_PRESERVING
    image = np.arange(3 * 4 * 6, dtype=np.float32).reshape(3, 4, 6)
    for name in SHAPE_PRESERVING:
        assert apply_dihedral(image, name).shape == (3, 4, 6)


def test_rot180_reverses_both_axes() -> None:
    """rot180 places the first pixel at the opposite corner."""
    image = np.zeros((1, 2, 3), dtype=np.float32)
    image[0, 0, 0] = 1.0
    out = apply_dihedral(image, "rot180")
    assert out[0, -1, -1] == 1.0
    assert out[0, 0, 0] == 0.0


def test_unknown_element_rejected() -> None:
    """An unknown element name raises ValueError."""
    image = np.zeros((1, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="unknown augment"):
        apply_dihedral(image, "spin")
