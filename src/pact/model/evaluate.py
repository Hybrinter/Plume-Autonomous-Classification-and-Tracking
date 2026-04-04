"""
pact.model.evaluate — Segmentation evaluation metrics.

Satisfies: REQ-AIML-HIGH-001, REQ-AIML-HIGH-002

Implements standard binary segmentation metrics. All functions accept either
``np.ndarray`` or ``torch.Tensor`` inputs and return Python ``float`` values.
Inputs are expected to be probability maps (after sigmoid) in [0, 1]; a threshold
is applied internally to convert to binary predictions.
"""

from __future__ import annotations

# stdlib
from typing import Union

# third-party
import numpy as np
import torch


# Type alias for acceptable array inputs.
ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_numpy_binary(arr: ArrayLike, threshold: float) -> np.ndarray:
    """Convert an ArrayLike probability map to a binary numpy array.

    Args:
        arr:       Probability map, any shape, values in [0, 1].
        threshold: Pixels strictly above this value are set to 1, others to 0.

    Returns:
        Boolean numpy array of the same shape.
    """
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    # arr is np.ndarray here; dtype may be float32, float64, etc.
    return (arr > threshold).astype(np.uint8)  # np.ndarray[uint8, ...]


def iou_score(
    pred_mask: ArrayLike,
    true_mask: ArrayLike,
    threshold: float = 0.5,
) -> float:
    """Intersection over Union (Jaccard index) for binary segmentation.

    IoU = |pred ∩ true| / |pred ∪ true|

    Returns 1.0 when both masks are all-zero (correct empty prediction).
    Returns 0.0 when one mask is non-zero and the other is zero (full mismatch).

    Args:
        pred_mask: Predicted probability map, shape (H, W) or (1, H, W) or (B, 1, H, W).
        true_mask: Ground-truth binary mask, same shape as pred_mask.
        threshold: Binarisation threshold for pred_mask. Default 0.5.

    Returns:
        IoU score as a Python float in [0.0, 1.0].
    """
    pred_bin = _to_numpy_binary(pred_mask, threshold).ravel()  # np.ndarray[uint8, (N,)]
    true_bin = _to_numpy_binary(true_mask, 0.5).ravel()        # np.ndarray[uint8, (N,)]

    intersection: int = int(np.logical_and(pred_bin, true_bin).sum())
    union: int = int(np.logical_or(pred_bin, true_bin).sum())

    if union == 0:
        # Both masks are empty — correct prediction.
        return 1.0
    return float(intersection) / float(union)


def dice_score(
    pred_mask: ArrayLike,
    true_mask: ArrayLike,
    threshold: float = 0.5,
) -> float:
    """Dice coefficient (F1 score) for binary segmentation.

    Dice = 2 * |pred ∩ true| / (|pred| + |true|)

    Returns 1.0 when both masks are all-zero. Returns 0.0 on full mismatch.

    Args:
        pred_mask: Predicted probability map.
        true_mask: Ground-truth binary mask.
        threshold: Binarisation threshold. Default 0.5.

    Returns:
        Dice score as a Python float in [0.0, 1.0].
    """
    pred_bin = _to_numpy_binary(pred_mask, threshold).ravel()  # np.ndarray[uint8, (N,)]
    true_bin = _to_numpy_binary(true_mask, 0.5).ravel()        # np.ndarray[uint8, (N,)]

    intersection: int = int(np.logical_and(pred_bin, true_bin).sum())
    denominator: int = int(pred_bin.sum()) + int(true_bin.sum())

    if denominator == 0:
        # Both masks are empty — correct prediction.
        return 1.0
    return float(2 * intersection) / float(denominator)


def precision_recall(
    pred_mask: ArrayLike,
    true_mask: ArrayLike,
    threshold: float = 0.5,
) -> tuple[float, float]:
    """Binary segmentation precision and recall.

    precision = TP / (TP + FP)    — "of all predicted positives, how many are correct?"
    recall    = TP / (TP + FN)    — "of all true positives, how many did we find?"

    Edge cases: precision = 1.0 when no positives are predicted (no false positives).
                recall    = 1.0 when there are no true positives to miss.

    Args:
        pred_mask: Predicted probability map.
        true_mask: Ground-truth binary mask.
        threshold: Binarisation threshold. Default 0.5.

    Returns:
        ``(precision, recall)`` as a tuple of Python floats in [0.0, 1.0].
    """
    pred_bin = _to_numpy_binary(pred_mask, threshold).ravel()  # np.ndarray[uint8, (N,)]
    true_bin = _to_numpy_binary(true_mask, 0.5).ravel()        # np.ndarray[uint8, (N,)]

    tp: int = int(np.logical_and(pred_bin, true_bin).sum())
    fp: int = int(np.logical_and(pred_bin, np.logical_not(true_bin)).sum())
    fn: int = int(np.logical_and(np.logical_not(pred_bin), true_bin).sum())

    precision: float = float(tp) / float(tp + fp) if (tp + fp) > 0 else 1.0
    recall: float = float(tp) / float(tp + fn) if (tp + fn) > 0 else 1.0

    return (precision, recall)
