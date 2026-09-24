"""Train-split per-band mean and standard deviation.

Contains:
  - BandStats: one mean and one standard deviation per channel.
  - fit_band_stats: moments from train stacks.
  - apply_band_stats: zero-mean unit-variance scaling.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

_EPS = 1e-6


@dataclass(frozen=True, slots=True)
class BandStats:
    """Per-channel moments computed on the train split only.

    Attributes:
        mean: Shape ``(C,)``.
        std: Shape ``(C,)``, floored at ``1e-6``.
    """

    mean: np.ndarray
    std: np.ndarray


def fit_band_stats(stacks: Sequence[np.ndarray]) -> BandStats:
    """Fit per-channel moments.

    Args:
        stacks: Arrays of shape ``(C, H, W)`` sharing ``C``.

    Returns:
        BandStats: Mean and standard deviation across pixels and stacks.

    Raises:
        ValueError: If ``stacks`` is empty or the channel counts differ.
    """
    if not stacks:
        raise ValueError("need at least one stack to fit band stats")
    channels = int(stacks[0].shape[0])
    total = np.zeros(channels, dtype=np.float64)
    total_sq = np.zeros(channels, dtype=np.float64)
    count = 0
    for stack in stacks:
        array = np.asarray(stack, dtype=np.float64)
        if array.ndim != 3 or array.shape[0] != channels:
            raise ValueError(f"expected (C={channels}, H, W); got {array.shape}")
        flat = array.reshape(channels, -1)
        total += flat.sum(axis=1)
        total_sq += np.square(flat).sum(axis=1)
        count += flat.shape[1]
    mean = total / float(count)
    variance = total_sq / float(count) - np.square(mean)
    std = np.sqrt(np.maximum(variance, 0.0))
    std = np.maximum(std, _EPS)
    return BandStats(mean=mean.astype(np.float32), std=std.astype(np.float32))


def apply_band_stats(stack: np.ndarray, stats: BandStats) -> np.ndarray:
    """Scale one stack with frozen moments.

    Args:
        stack: Array ``(C, H, W)``.
        stats: Moments with the same channel count.

    Returns:
        np.ndarray: Float32 ``(stack - mean) / std``.

    Raises:
        ValueError: If the channel counts differ.
    """
    array = np.asarray(stack, dtype=np.float32)
    if array.shape[0] != stats.mean.shape[0]:
        raise ValueError(f"stack has {array.shape[0]} channels; stats have {stats.mean.shape[0]}")
    mean = stats.mean.astype(np.float32)[:, None, None]
    std = stats.std.astype(np.float32)[:, None, None]
    return (array - mean) / std
