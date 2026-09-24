"""Near-native tiles are stored as 120-pixel stacks."""

from __future__ import annotations

import numpy as np
import pytest
from tools.original_dataset_analysis.cache import to_native_stack


def test_short_width_is_edge_padded() -> None:
    """A 120 by 119 tile gains one repeated column."""
    stack = np.ones((2, 120, 119), dtype=np.float32)
    stack[:, :, -1] = 4.0
    fitted = to_native_stack(stack)
    assert fitted.shape == (2, 120, 120)
    assert fitted[0, 0, -1] == 4.0


def test_far_side_is_refused() -> None:
    """A tile more than 2 pixels off 120 is refused."""
    with pytest.raises(ValueError, match="120"):
        to_native_stack(np.zeros((1, 120, 100), dtype=np.float32))
