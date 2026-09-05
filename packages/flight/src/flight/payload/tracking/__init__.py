"""Payload tracking: residual Kalman filter and blob association (pure functions).

residual -- two-state elevation-error / residual-rate filter;
tracker -- IoU blob matching and persistence counting.
"""

from flight.payload.tracking.residual import (
    ResidualFilter,
    ResidualSnapshot,
    ResidualState,
    predict,
    push_snapshot,
    rewind_update,
    update,
)
from flight.payload.tracking.tracker import compute_iou, match_blobs

__all__ = [
    "ResidualFilter",
    "ResidualSnapshot",
    "ResidualState",
    "compute_iou",
    "match_blobs",
    "predict",
    "push_snapshot",
    "rewind_update",
    "update",
]
