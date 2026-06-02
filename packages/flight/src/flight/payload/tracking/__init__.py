"""Payload tracking: target-state estimation and blob association (pure functions).

filter -- EMA centroid smoothing; kalman -- constant-velocity pointing estimator;
tracker -- IoU blob matching and persistence counting.
"""

from flight.payload.tracking.filter import EmaFilterState, ema_update
from flight.payload.tracking.kalman import KalmanFilter, KalmanState, predict, update
from flight.payload.tracking.tracker import compute_iou, match_blobs

__all__ = [
    "EmaFilterState",
    "KalmanFilter",
    "KalmanState",
    "compute_iou",
    "ema_update",
    "match_blobs",
    "predict",
    "update",
]
