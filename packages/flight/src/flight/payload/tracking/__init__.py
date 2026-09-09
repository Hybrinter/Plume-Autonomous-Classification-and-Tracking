"""Payload tracking: residual Kalman filter and blob association (pure functions).

residual -- two-state elevation-error / residual-rate filter;
tracker -- IoU blob matching and persistence counting.
"""

from flight.payload.tracking.residual import (
    EncoderSample,
    EstimateResult,
    EventDisposition,
    NominalRateSample,
    ObservationDisposition,
    PredictorReferenceChange,
    ResidualCheckpoint,
    ResidualFilter,
    ResidualHistory,
    ResidualSnapshot,
    ResidualState,
    VisionObservation,
    estimate_at,
    predict,
    propagate_displacement,
    push_snapshot,
    rewind_update,
    submit_event,
    update,
)
from flight.payload.tracking.tracker import compute_iou, match_blobs

__all__ = [
    "ResidualFilter",
    "ResidualCheckpoint",
    "ResidualHistory",
    "ResidualSnapshot",
    "ResidualState",
    "EncoderSample",
    "NominalRateSample",
    "PredictorReferenceChange",
    "VisionObservation",
    "ObservationDisposition",
    "EventDisposition",
    "EstimateResult",
    "compute_iou",
    "match_blobs",
    "estimate_at",
    "predict",
    "propagate_displacement",
    "push_snapshot",
    "rewind_update",
    "submit_event",
    "update",
]
