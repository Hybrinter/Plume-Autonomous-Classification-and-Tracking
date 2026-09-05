"""
Safety gates for PACT controller subsystem.

Confidence and minimum-area gates run before blob matching. They are pure
functions with no side effects.

Satisfies: REQ-AIML-DATA-008, REQ-AIML-DATA-009,
           REQ-AIML-GIMB-006, REQ-AIML-GIMB-007
"""

from __future__ import annotations

from flight.libs.messages import BlobMeta


def apply_confidence_gate(
    blobs: tuple[BlobMeta, ...],
    threshold: float,
) -> tuple[BlobMeta, ...]:
    """Reject blobs whose mean_confidence is strictly below threshold. REQ-AIML-DATA-008.

    Parameters
    ----------
    blobs:
        Candidate blobs from the current inference result.
    threshold:
        Minimum acceptable mean_confidence (exclusive lower bound).
        Sourced from ControllerConfig.confidence_gate (default 0.55).

    Returns
    -------
    tuple[BlobMeta, ...]
        Filtered blobs; may be empty.
    """
    return tuple(b for b in blobs if b.mean_confidence >= threshold)


def apply_min_area_gate(
    blobs: tuple[BlobMeta, ...],
    min_px: int,
) -> tuple[BlobMeta, ...]:
    """Reject blobs whose pixel_area is strictly below min_px. REQ-AIML-DATA-009.

    Parameters
    ----------
    blobs:
        Candidate blobs (typically already confidence-gated).
    min_px:
        Minimum acceptable pixel area (exclusive lower bound).
        Sourced from ControllerConfig.min_blob_area_px (default 15).

    Returns
    -------
    tuple[BlobMeta, ...]
        Filtered blobs; may be empty.
    """
    return tuple(b for b in blobs if b.pixel_area >= min_px)
