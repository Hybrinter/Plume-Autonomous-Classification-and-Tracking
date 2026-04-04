"""
EMA centroid filter for PACT controller subsystem.

Applies an Exponential Moving Average to blob centroid coordinates, reducing
jitter from per-frame inference noise before centroid coordinates are used to
compute gimbal displacement commands.

Satisfies: REQ-AIML-DATA-007
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmaFilterState:
    """Immutable EMA filter state for a single tracked blob centroid.

    Fields
    ------
    centroid:
        Most recently smoothed (x, y) centroid in pixel-space (float, float).
    initialized:
        False until the first observation has been processed. When False, the
        first call to ema_update() returns the raw centroid with no smoothing.
    """

    centroid: tuple[float, float]
    initialized: bool


def ema_update(
    state: EmaFilterState,
    new_centroid: tuple[float, float],
    alpha: float,
) -> EmaFilterState:
    """Exponential moving average centroid filter. REQ-AIML-DATA-007.

    Formula
    -------
    On the **first** detection (state.initialized is False):
        smoothed = new_centroid  (no blending; return raw value directly)

    On subsequent detections:
        smoothed_x = alpha * new_centroid[0] + (1 - alpha) * state.centroid[0]
        smoothed_y = alpha * new_centroid[1] + (1 - alpha) * state.centroid[1]

    Parameters
    ----------
    state:
        Current EMA filter state. Pass EmaFilterState(centroid=(0.0, 0.0), initialized=False)
        at the start of a new track.
    new_centroid:
        Raw centroid from the current inference frame (x, y) in pixel space.
    alpha:
        Smoothing factor in (0.0, 1.0]. Higher alpha means less smoothing (more responsive).
        Sourced from ControllerConfig.ema_alpha (default 0.4).

    Returns
    -------
    EmaFilterState
        New immutable filter state with updated centroid and initialized=True.
    """
    if not state.initialized:
        return EmaFilterState(centroid=new_centroid, initialized=True)

    prev_x, prev_y = state.centroid
    new_x, new_y = new_centroid

    smoothed_x = alpha * new_x + (1.0 - alpha) * prev_x
    smoothed_y = alpha * new_y + (1.0 - alpha) * prev_y

    return EmaFilterState(centroid=(smoothed_x, smoothed_y), initialized=True)
