"""Per-band moments ignore the held-out stack."""

from __future__ import annotations

import numpy as np
import pytest
from tools.ml_models.data.moments import (
    BandStats,
    MomentAccumulator,
    apply_band_stats,
    fit_band_stats,
)


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


def test_accumulator_matches_fit() -> None:
    """Streaming updates match a single fit over the same stacks."""
    first = np.ones((2, 3, 3), dtype=np.float32)
    second = np.full((2, 3, 3), 3.0, dtype=np.float32)
    accumulator = MomentAccumulator(2)
    accumulator.update(first)
    accumulator.update(second)
    assert np.allclose(accumulator.finish().mean, fit_band_stats([first, second]).mean)


def test_apply_rejects_moments_that_are_not_vectors() -> None:
    """A shared standard deviation or a column vector is refused."""
    stack = np.zeros((3, 2, 2), dtype=np.float32)
    shared = BandStats(mean=np.zeros(3, dtype=np.float32), std=np.ones(1, dtype=np.float32))
    column = BandStats(mean=np.zeros((3, 1), dtype=np.float32), std=np.ones(3, dtype=np.float32))
    with pytest.raises(ValueError, match="must both be"):
        apply_band_stats(stack, shared)
    with pytest.raises(ValueError, match="must both be"):
        apply_band_stats(stack, column)
