"""Normalization recipes for processed-pack planes.

Contains:
  - apply_normalize_dn: ADC full-scale scaling through flight ``normalize_dn``.
  - apply_unit: clip to the unit interval.
  - BandStats / fit_band_stats / apply_band_z: per-band population z-score.
  - apply_norm: dispatch on ``normalize_dn``, ``unit``, and ``band_z``.

``fit_band_stats`` accepts ``(C, H, W)`` or ``(N, C, H, W)``. Population standard
deviation uses divisor P (pixel count). Values below ``1e-6`` are raised to
``1e-6``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from flight.payload.preprocess.normalize import normalize_dn

from tools.ml_models.data.meta import NormName

_STD_FLOOR = 1e-6


@dataclass(frozen=True, slots=True)
class BandStats:
    """Per-channel mean and population standard deviation.

    Attributes:
        mean: One mean per channel.
        std: One standard deviation per channel, each at least ``1e-6``.
    """

    mean: tuple[float, ...]
    std: tuple[float, ...]


def apply_normalize_dn(planes: np.ndarray, bit_depth: int) -> np.ndarray:
    """Scale calibrated DN by ADC full scale and clip to ``[0, 1]``.

    Args:
        planes: np.ndarray[float, (C, H, W)] or ``(N, C, H, W)`` DN values.
        bit_depth: ADC bit depth. Full scale is ``2**bit_depth - 1``.

    Returns:
        np.ndarray[float32]: Same shape as ``planes``, values in ``[0, 1]``.

    Raises:
        ValueError: If ``bit_depth`` is not an int >= 1.
    """
    _require_bit_depth(bit_depth)
    return normalize_dn(planes, bit_depth)


def apply_unit(planes: np.ndarray) -> np.ndarray:
    """Clip planes to the unit interval.

    Args:
        planes: np.ndarray of any rank.

    Returns:
        np.ndarray[float32]: ``clip(planes, 0, 1)``.
    """
    array = np.asarray(planes, dtype=np.float32)  # np.ndarray[float32]
    return np.clip(array, 0.0, 1.0)


def fit_band_stats(stack: np.ndarray) -> BandStats:
    """Fit per-channel population moments.

    Args:
        stack: np.ndarray[(C, H, W)] or ``(N, C, H, W)``.

    Returns:
        BandStats: Mean and population standard deviation across pixels
        (and samples, for rank 4). Each std is at least ``1e-6``.

    Raises:
        ValueError: If the rank is not 3 or 4, or an axis is empty.

    Notes:
        Population standard deviation is ``sqrt(mean((x - mean)^2))`` with
        divisor P, the number of pixels in the channel.
    """
    matrix = _channel_matrix(stack)  # np.ndarray[float64, (C, P)]
    means = tuple(float(value) for value in np.mean(matrix, axis=1))
    stds = tuple(max(float(value), _STD_FLOOR) for value in np.std(matrix, axis=1, ddof=0))
    return BandStats(mean=means, std=stds)


def apply_band_z(planes: np.ndarray, stats: BandStats) -> np.ndarray:
    """Scale planes to zero mean and unit variance per channel.

    Args:
        planes: np.ndarray[(C, H, W)] or ``(N, C, H, W)``.
        stats: Moments whose length equals the channel count.

    Returns:
        np.ndarray[float32]: ``(planes - mean) / std`` with the same rank.

    Raises:
        ValueError: If the rank is not 3 or 4, or the moment length disagrees
            with the channel count.
    """
    array = np.asarray(planes, dtype=np.float32)
    if array.ndim == 3:
        channels = int(array.shape[0])
    elif array.ndim == 4:
        channels = int(array.shape[1])
    else:
        raise ValueError(f"planes must have shape (C, H, W) or (N, C, H, W); got {array.shape}")
    _require_stats_length(stats, channels)
    flat_mean = np.asarray(stats.mean, dtype=np.float32)
    flat_std = np.asarray(stats.std, dtype=np.float32)
    if array.ndim == 3:
        mean3 = flat_mean.reshape(channels, 1, 1)  # np.ndarray[float32, (C, 1, 1)]
        std3 = flat_std.reshape(channels, 1, 1)  # np.ndarray[float32, (C, 1, 1)]
        return ((array - mean3) / std3).astype(np.float32)
    mean4 = flat_mean.reshape(1, channels, 1, 1)  # np.ndarray[float32, (1, C, 1, 1)]
    std4 = flat_std.reshape(1, channels, 1, 1)  # np.ndarray[float32, (1, C, 1, 1)]
    return ((array - mean4) / std4).astype(np.float32)


def apply_norm(
    planes: np.ndarray,
    norm: NormName,
    *,
    bit_depth: int = 12,
    stats: BandStats | None = None,
) -> np.ndarray:
    """Dispatch on ``normalize_dn``, ``unit``, or ``band_z``.

    Args:
        planes: np.ndarray[(C, H, W)] or ``(N, C, H, W)``.
        norm: Recipe name.
        bit_depth: ADC bit depth for ``normalize_dn``. Defaults to 12.
        stats: Moments for ``band_z``.

    Returns:
        np.ndarray[float32]: Normalized planes.

    Raises:
        ValueError: If ``band_z`` is missing ``stats``, ``bit_depth`` is below 1,
            or ``norm`` is not one of the three recipe names.
    """
    if norm == "normalize_dn":
        return apply_normalize_dn(planes, bit_depth)
    if norm == "unit":
        return apply_unit(planes)
    if norm == "band_z":
        if stats is None:
            raise ValueError("band_z requires stats")
        return apply_band_z(planes, stats)
    raise ValueError(f"unknown norm {norm!r}")


def _require_bit_depth(bit_depth: int) -> None:
    """Raise when bit depth is not an int >= 1.

    Args:
        bit_depth: ADC bit depth.

    Returns:
        None.

    Raises:
        ValueError: If ``bit_depth`` is a bool or is below 1.
    """
    if isinstance(bit_depth, bool) or not isinstance(bit_depth, int) or bit_depth < 1:
        raise ValueError(f"bit_depth must be an int >= 1; got {bit_depth!r}")


def _require_stats_length(stats: BandStats, channels: int) -> None:
    """Raise when moment tuples do not have length ``channels``.

    Args:
        stats: Frozen moments.
        channels: Channel count of the array being scaled.

    Returns:
        None.

    Raises:
        ValueError: If ``mean`` and ``std`` differ, or either length is not ``channels``.
    """
    if len(stats.mean) != len(stats.std) or len(stats.mean) != channels:
        raise ValueError(
            f"mean length {len(stats.mean)} and std length {len(stats.std)} "
            f"must both equal channel count {channels}"
        )
    if any(value <= 0.0 for value in stats.std):
        raise ValueError("std values must be > 0")


def _channel_matrix(stack: np.ndarray) -> np.ndarray:
    """Return float64 pixels shaped ``(C, P)``.

    Args:
        stack: Array ``(C, H, W)`` or ``(N, C, H, W)``.

    Returns:
        np.ndarray[float64, (C, P)]: One row per channel.

    Raises:
        ValueError: If the rank is not 3 or 4, or an axis is empty.
    """
    array = np.asarray(stack, dtype=np.float64)
    if array.ndim == 3:
        channels = int(array.shape[0])
        if channels < 1 or int(array.shape[1]) < 1 or int(array.shape[2]) < 1:
            raise ValueError(f"stack shape {tuple(int(v) for v in array.shape)} has an empty axis")
        return array.reshape(channels, -1)
    if array.ndim == 4:
        samples = int(array.shape[0])
        channels = int(array.shape[1])
        height = int(array.shape[2])
        width = int(array.shape[3])
        if samples < 1 or channels < 1 or height < 1 or width < 1:
            raise ValueError(f"stack shape {tuple(int(v) for v in array.shape)} has an empty axis")
        moved = np.transpose(array, (1, 0, 2, 3))  # np.ndarray[float64, (C, N, H, W)]
        return moved.reshape(channels, -1)
    raise ValueError(f"stack must have shape (C, H, W) or (N, C, H, W); got {array.shape}")
