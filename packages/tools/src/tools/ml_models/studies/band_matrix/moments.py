"""Train-split per-band mean and standard deviation.

Contains:
  - BandStats: one mean and one standard deviation per channel.
  - MomentAccumulator: running train-split moments.
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


class MomentAccumulator:
    """Running sum of pixels for one channel count.

    Call :meth:`update` once per train stack, then :meth:`finish`.
    """

    def __init__(self, channels: int) -> None:
        """Allocate the running sums.

        Args:
            channels: Channel count of every later stack.

        Raises:
            ValueError: If ``channels`` is below 1.
        """
        if channels < 1:
            raise ValueError(f"channels must be at least 1; got {channels}")
        self._channels = channels
        self._total = np.zeros(channels, dtype=np.float64)
        self._total_sq = np.zeros(channels, dtype=np.float64)
        self._count = 0

    def update(self, stack: np.ndarray) -> None:
        """Add one ``(C, H, W)`` stack.

        Args:
            stack: Train pixels. ``C`` matches the constructor.

        Raises:
            ValueError: If the channel count differs.
        """
        array = np.asarray(stack, dtype=np.float64)
        if array.ndim != 3 or array.shape[0] != self._channels:
            raise ValueError(f"expected (C={self._channels}, H, W); got {array.shape}")
        flat = array.reshape(self._channels, -1)
        self._total += flat.sum(axis=1)
        self._total_sq += np.square(flat).sum(axis=1)
        self._count += int(flat.shape[1])

    def finish(self) -> BandStats:
        """Return the mean and standard deviation.

        Returns:
            BandStats: Moments across every pixel passed to :meth:`update`.

        Raises:
            ValueError: If :meth:`update` was never called.
        """
        if self._count == 0:
            raise ValueError("need at least one stack to fit band stats")
        mean = self._total / float(self._count)
        variance = self._total_sq / float(self._count) - np.square(mean)
        std = np.sqrt(np.maximum(variance, 0.0))
        std = np.maximum(std, _EPS)
        return BandStats(mean=mean.astype(np.float32), std=std.astype(np.float32))


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
    channels = int(np.asarray(stacks[0]).shape[0])
    accumulator = MomentAccumulator(channels)
    for stack in stacks:
        accumulator.update(stack)
    return accumulator.finish()


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
