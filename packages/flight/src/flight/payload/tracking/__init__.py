"""Payload tracking: target-state estimation and blob association (pure functions).

filter -- EMA centroid smoothing; kalman -- per-axis 4-state pointing estimator;
rewind -- vision-latency snapshot ring; tracker -- IoU blob matching.
"""

from flight.payload.tracking.filter import EmaFilterState, ema_update
from flight.payload.tracking.kalman import (
    AxisKalmanState,
    DualKalmanState,
    KalmanFilter,
    predict,
    update_enc,
    update_vis,
)
from flight.payload.tracking.rewind import (
    EstimatorRing,
    EstimatorSnapshot,
    apply_vision,
    empty_ring,
    push_snapshot,
)
from flight.payload.tracking.tracker import compute_iou, match_blobs

__all__ = [
    "AxisKalmanState",
    "DualKalmanState",
    "EmaFilterState",
    "EstimatorRing",
    "EstimatorSnapshot",
    "KalmanFilter",
    "apply_vision",
    "compute_iou",
    "ema_update",
    "empty_ring",
    "match_blobs",
    "predict",
    "push_snapshot",
    "update_enc",
    "update_vis",
]
