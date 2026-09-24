"""Train-split per-band mean and standard deviation.

Contains:
  - BandStats: one mean and one standard deviation per channel.
  - fit_band_stats: moments from train stacks.
  - check_band_stats: both moment vectors have shape ``(C,)``.
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


def check_band_stats(stats: BandStats, channels: int) -> None:
    """Require both moment vectors to have shape ``(channels,)``.

    Args:
        stats: Frozen moments.
        channels: Channel count of the stack being scaled.

    Raises:
        ValueError: If ``mean`` or ``std`` is not exactly that shape.
    """
    mean = np.asarray(stats.mean)
    std = np.asarray(stats.std)
    expected = (channels,)
    if mean.shape != expected or std.shape != expected:
        raise ValueError(
            f"mean shape {mean.shape} and std shape {std.shape} must both be {expected}"
        )


def apply_band_stats(stack: np.ndarray, stats: BandStats) -> np.ndarray:
    """Scale one stack with frozen moments.

    Args:
        stack: Array ``(C, H, W)``.
        stats: Moments with the same channel count. Both vectors are shape ``(C,)``.

    Returns:
        np.ndarray: Float32 ``(stack - mean) / std``.

    Raises:
        ValueError: If either moment vector is not exactly shape ``(C,)``.
    """
    array = np.asarray(stack, dtype=np.float32)
    check_band_stats(stats, int(array.shape[0]))
    mean = np.asarray(stats.mean, dtype=np.float32)[:, None, None]
    std = np.asarray(stats.std, dtype=np.float32)[:, None, None]
    return (array - mean) / std
