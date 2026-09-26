"""Area resample and percent masks, including side 76."""

from __future__ import annotations

import numpy as np
import pytest
from tools.ml_models.data.grid import (
    LEGAL_SIDES,
    coarsen,
    gsd_m,
    rasterize_mask,
    rasterize_percent_mask,
    resample_area,
)

_CENTER = (np.array([[25.0, 25.0], [75.0, 25.0], [75.0, 75.0], [25.0, 75.0]], dtype=np.float32),)


def test_seventy_six_is_not_a_legal_side() -> None:
    """Side 76 stays out of the study set and gsd_m refuses it."""
    assert 76 not in LEGAL_SIDES
    with pytest.raises(ValueError, match="side_px"):
        gsd_m(76)
    with pytest.raises(ValueError, match="side_px"):
        coarsen(np.zeros((1, 120, 120), dtype=np.float32), 76)


def test_resample_area_preserves_a_constant_at_76() -> None:
    """A constant native field stays constant after an area resample to 76."""
    planes = np.full((2, 120, 120), 0.25, dtype=np.float32)
    out = resample_area(planes, 76)
    assert out.shape == (2, 76, 76)
    assert out.dtype == np.float32
    assert np.allclose(out, 0.25)


def test_rasterize_percent_mask_matches_legal_side() -> None:
    """The any-side rasterizer agrees with rasterize_mask on a legal side."""
    percent = rasterize_percent_mask(_CENTER, 40, rule="half")
    legal = rasterize_mask(_CENTER, 40, rule="half")
    assert percent.shape == (1, 40, 40)
    assert np.array_equal(percent, legal)


def test_rasterize_percent_mask_accepts_76() -> None:
    """A center polygon covers the center of a 76 px mask."""
    mask = rasterize_percent_mask(_CENTER, 76)[0]
    assert mask.shape == (76, 76)
    start = 76 // 4
    stop = (3 * 76) // 4
    assert np.all(mask[start:stop, start:stop] == 1.0)
    assert mask[0, 0] == 0.0
