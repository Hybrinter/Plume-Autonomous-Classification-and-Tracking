"""Per-band moments ignore the held-out stack."""

from __future__ import annotations

import numpy as np
from tools.original_dataset_analysis.normalize import apply_band_stats, fit_band_stats


def test_held_out_stack_does_not_change_moments() -> None:
    """A second stack is scaled by moments fit on the first stack alone."""
    train = np.ones((2, 4, 4), dtype=np.float32)
    train[0] = 2.0
    train[1] = 4.0
    held_out = np.full((2, 4, 4), 100.0, dtype=np.float32)
    stats = fit_band_stats([train])
    scaled = apply_band_stats(held_out, stats)
    assert stats.mean.tolist() == [2.0, 4.0]
    assert np.allclose(scaled[0], (100.0 - 2.0) / stats.std[0])
    refit = fit_band_stats([train, held_out])
    assert not np.allclose(refit.mean, stats.mean)
