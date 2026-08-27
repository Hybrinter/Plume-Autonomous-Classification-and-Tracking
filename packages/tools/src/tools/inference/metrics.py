"""Classifier and segmentation scoring helpers (pure NumPy).

Mask IoU lives in tools.inference.accept.compute_iou. This module holds the
image-level binary accuracy used for the presence classifier.

Contains:
  - binary_accuracy: 1.0 on a matching boolean pair, else 0.0.
  - mean_binary_accuracy: mean over paired predictions and labels.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations


def binary_accuracy(pred_positive: bool, label_positive: bool) -> float:
    """Return 1.0 when the predicted class matches the label, else 0.0.

    Args:
        pred_positive: Predicted plume-present flag.
        label_positive: Ground-truth plume-present flag.

    Returns:
        float: 1.0 on a match, 0.0 on a mismatch.
    """
    return 1.0 if pred_positive is label_positive else 0.0


def mean_binary_accuracy(
    pred_positive: tuple[bool, ...],
    label_positive: tuple[bool, ...],
) -> float:
    """Return mean binary accuracy over aligned prediction/label pairs.

    Args:
        pred_positive: Predicted presence flags.
        label_positive: Ground-truth presence flags. Must match length.

    Returns:
        float: Mean accuracy in [0, 1]. Empty inputs return 0.0.
    """
    if len(pred_positive) != len(label_positive):
        raise ValueError("pred_positive and label_positive must have equal length")
    if not pred_positive:
        return 0.0
    total = 0.0
    for pred, label in zip(pred_positive, label_positive, strict=True):
        total += binary_accuracy(pred, label)
    return total / float(len(pred_positive))
