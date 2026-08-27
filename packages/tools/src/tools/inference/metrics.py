"""Classifier and segmentation scoring helpers (pure NumPy).

This module holds design metrics for both heads. Acceptance and training import
these helpers. Graphs emit logits; ROC, PR, and Brier use sigmoid probabilities.
Empty inputs return 0.0 with n=0.

Contains:
  - binary_accuracy / mean_binary_accuracy: image-level presence scores.
  - compute_iou / compute_dice: per-mask overlap at a probability threshold.
  - confusion_counts / precision_recall_f1: thresholded binary scores.
  - roc_auc / average_precision / brier_score / reliability_bins.
  - classifier_metrics / segmentor_metrics: split-level summaries.
  - binary_cross_entropy_with_logits: numpy BCE for reports.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BLOB_GATE = 0.55
MASK_THRESHOLD = 0.5
LOGIT_THRESHOLD = 0.0


@dataclass(frozen=True, slots=True)
class ConfusionCounts:
    """Binary confusion counts.

    Attributes:
        tp: True positives.
        fp: False positives.
        tn: True negatives.
        fn: False negatives.
    """

    tp: int
    fp: int
    tn: int
    fn: int


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    """One calibration bin.

    Attributes:
        confidence: Mean predicted probability in the bin.
        accuracy: Mean label in the bin.
        count: Sample count in the bin.
    """

    confidence: float
    accuracy: float
    count: int


@dataclass(frozen=True, slots=True)
class ClassifierMetrics:
    """Split-level classifier scores.

    Attributes:
        n: Sample count.
        accuracy: Thresholded accuracy.
        precision: Positive-class precision.
        recall: Positive-class recall.
        f1: Harmonic mean of precision and recall.
        roc_auc: ROC area. 0.0 when a class is missing.
        pr_auc: Average precision. 0.0 when n is 0.
        brier: Mean squared error of probabilities.
        bce: Mean binary cross-entropy with logits.
    """

    n: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    brier: float
    bce: float


@dataclass(frozen=True, slots=True)
class SegmentorMetrics:
    """Split-level segmentor overlap scores.

    Attributes:
        n: Sample count.
        mean_iou: Mean IoU at ``MASK_THRESHOLD`` (0.5).
        mean_dice: Mean Dice at 0.5.
        mean_iou_blob_gate: Mean IoU at onboard blob gate 0.55.
        bce: Mean binary cross-entropy with logits over pixels.
    """

    n: int
    mean_iou: float
    mean_dice: float
    mean_iou_blob_gate: float
    bce: float


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Return elementwise logistic sigmoid of logits.

    Args:
        logits: Unbounded scores.

    Returns:
        np.ndarray[float64]: Probabilities in (0, 1) with the same shape.
    """
    clipped = np.clip(np.asarray(logits, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


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


def compute_iou(pred_mask: np.ndarray, gold_mask: np.ndarray, threshold: float = 0.5) -> float:
    """Compute the binary intersection-over-union of a predicted vs golden mask.

    Args:
        pred_mask: Predicted probability mask (H, W).
        gold_mask: Golden probability mask (H, W).
        threshold: Probability threshold for binarization.

    Returns:
        float: IoU in [0, 1]. Two empty masks score 1.0.
    """
    pred = np.asarray(pred_mask) >= threshold
    gold = np.asarray(gold_mask) >= threshold
    intersection = float(np.logical_and(pred, gold).sum())
    union = float(np.logical_or(pred, gold).sum())
    if union == 0.0:
        return 1.0
    return intersection / union


def compute_dice(pred_mask: np.ndarray, gold_mask: np.ndarray, threshold: float = 0.5) -> float:
    """Compute the binary Dice coefficient of a predicted vs golden mask.

    Args:
        pred_mask: Predicted probability mask (H, W).
        gold_mask: Golden probability mask (H, W).
        threshold: Probability threshold for binarization.

    Returns:
        float: Dice in [0, 1]. Two empty masks score 1.0.
    """
    pred = np.asarray(pred_mask) >= threshold
    gold = np.asarray(gold_mask) >= threshold
    intersection = float(np.logical_and(pred, gold).sum())
    total = float(pred.sum() + gold.sum())
    if total == 0.0:
        return 1.0
    return (2.0 * intersection) / total


def confusion_counts(pred_positive: np.ndarray, label_positive: np.ndarray) -> ConfusionCounts:
    """Return confusion counts for aligned boolean arrays.

    Args:
        pred_positive: Predicted flags.
        label_positive: Ground-truth flags. Must match length.

    Returns:
        ConfusionCounts: tp, fp, tn, fn.

    Raises:
        ValueError: If lengths differ.
    """
    pred = np.asarray(pred_positive, dtype=bool).ravel()
    label = np.asarray(label_positive, dtype=bool).ravel()
    if pred.size != label.size:
        raise ValueError("pred_positive and label_positive must have equal length")
    tp = int(np.logical_and(pred, label).sum())
    fp = int(np.logical_and(pred, np.logical_not(label)).sum())
    tn = int(np.logical_and(np.logical_not(pred), np.logical_not(label)).sum())
    fn = int(np.logical_and(np.logical_not(pred), label).sum())
    return ConfusionCounts(tp=tp, fp=fp, tn=tn, fn=fn)


def _ratio(numerator: int, denominator: int) -> float:
    """Return numerator/denominator, or 0.0 when the denominator is 0."""
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def precision_recall_f1(counts: ConfusionCounts) -> tuple[float, float, float]:
    """Return precision, recall, and F1 from confusion counts.

    Args:
        counts: Binary confusion counts.

    Returns:
        tuple[float, float, float]: precision, recall, f1. Zero denominators
        yield 0.0.
    """
    precision = _ratio(counts.tp, counts.tp + counts.fp)
    recall = _ratio(counts.tp, counts.tp + counts.fn)
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Return ROC-AUC from scores and binary labels.

    Args:
        scores: Higher means more positive. Logits or probabilities.
        labels: Binary labels {0, 1}.

    Returns:
        float: Area under the ROC curve. 0.0 when n is 0 or a class is missing.
    """
    y = np.asarray(labels, dtype=np.float64).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    if y.size == 0 or y.size != s.size:
        if y.size != s.size:
            raise ValueError("scores and labels must have equal length")
        return 0.0
    n_pos = float(y.sum())
    n_neg = float(y.size) - n_pos
    if n_pos == 0.0 or n_neg == 0.0:
        return 0.0
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1.0 - y_sorted)
    tpr = np.concatenate(([0.0], tps / n_pos))
    fpr = np.concatenate(([0.0], fps / n_neg))
    return float(np.trapezoid(tpr, fpr))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Return average precision (area under the precision-recall curve).

    Args:
        scores: Higher means more positive.
        labels: Binary labels {0, 1}.

    Returns:
        float: Average precision in [0, 1]. 0.0 when n is 0 or there is no
        positive label.
    """
    y = np.asarray(labels, dtype=np.float64).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    if y.size == 0:
        return 0.0
    if y.size != s.size:
        raise ValueError("scores and labels must have equal length")
    n_pos = float(y.sum())
    if n_pos == 0.0:
        return 0.0
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1.0 - y_sorted)
    precision = tps / np.maximum(tps + fps, 1e-12)
    recall = tps / n_pos
    recall_ext = np.concatenate(([0.0], recall))
    precision_ext = np.concatenate(([1.0], precision))
    return float(np.sum(np.diff(recall_ext) * precision_ext[1:]))


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Return mean squared error between probabilities and labels.

    Args:
        probs: Predicted probabilities in [0, 1].
        labels: Binary labels {0, 1}.

    Returns:
        float: Brier score. 0.0 when n is 0.
    """
    p = np.asarray(probs, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    if p.size == 0:
        return 0.0
    if p.size != y.size:
        raise ValueError("probs and labels must have equal length")
    return float(np.mean((p - y) ** 2))


def reliability_bins(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> tuple[ReliabilityBin, ...]:
    """Return calibration bins of mean confidence versus mean label.

    Args:
        probs: Predicted probabilities in [0, 1].
        labels: Binary labels {0, 1}.
        n_bins: Equal-width bins on [0, 1].

    Returns:
        tuple[ReliabilityBin, ...]: Occupied bins only.

    Raises:
        ValueError: If n_bins < 1 or lengths differ.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    p = np.asarray(probs, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    if p.size != y.size:
        raise ValueError("probs and labels must have equal length")
    if p.size == 0:
        return ()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[ReliabilityBin] = []
    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        bins.append(
            ReliabilityBin(
                confidence=float(p[mask].mean()),
                accuracy=float(y[mask].mean()),
                count=count,
            )
        )
    return tuple(bins)


def binary_cross_entropy_with_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    """Return mean BCE with logits (numerically stable).

    Args:
        logits: Unbounded scores.
        targets: Labels or masks in {0, 1} with the same shape.

    Returns:
        float: Mean BCE. 0.0 when n is 0.
    """
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if z.size == 0:
        return 0.0
    if z.shape != y.shape:
        raise ValueError("logits and targets must have the same shape")
    # log(1 + exp(z)) - y z, stable via max(z, 0) + log(1 + exp(-|z|)) - y z
    loss = np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))
    return float(np.mean(loss))


def classifier_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    logit_threshold: float = LOGIT_THRESHOLD,
) -> ClassifierMetrics:
    """Score a classifier split from logits and labels.

    Args:
        logits: np.ndarray[float, (N,) or (N, 1)].
        labels: np.ndarray[float, (N,) or (N, 1)] in {0, 1}.
        logit_threshold: Positive when logit >= this value. Default 0.0.

    Returns:
        ClassifierMetrics: Thresholded and ranking scores. n=0 yields zeros.
    """
    z = np.asarray(logits, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    if z.size != y.size:
        raise ValueError("logits and labels must have equal length")
    n = int(z.size)
    if n == 0:
        return ClassifierMetrics(
            n=0,
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            roc_auc=0.0,
            pr_auc=0.0,
            brier=0.0,
            bce=0.0,
        )
    pred = z >= logit_threshold
    label_bool = y >= 0.5
    counts = confusion_counts(pred, label_bool)
    precision, recall, f1 = precision_recall_f1(counts)
    accuracy = float((pred == label_bool).mean())
    probs = sigmoid(z)
    return ClassifierMetrics(
        n=n,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc(z, y),
        pr_auc=average_precision(z, y),
        brier=brier_score(probs, y),
        bce=binary_cross_entropy_with_logits(z, y),
    )


def _iter_masks(stack: np.ndarray) -> list[np.ndarray]:
    """Yield 2-D masks from (N, H, W) or (N, 1, H, W)."""
    arr = np.asarray(stack)
    if arr.ndim == 2:
        return [arr]
    if arr.ndim == 3:
        return [arr[i] for i in range(arr.shape[0])]
    if arr.ndim == 4:
        return [arr[i, 0] for i in range(arr.shape[0])]
    raise ValueError(f"expected (H, W), (N, H, W), or (N, 1, H, W); got {arr.shape}")


def segmentor_metrics(
    logits: np.ndarray,
    masks: np.ndarray,
    mask_threshold: float = MASK_THRESHOLD,
    blob_gate: float = BLOB_GATE,
) -> SegmentorMetrics:
    """Score a segmentor split from logits and gold masks.

    Args:
        logits: np.ndarray[float, (N, 1, H, W) or (N, H, W)].
        masks: Gold masks with the same rank, values in {0, 1}.
        mask_threshold: Probability threshold for IoU and Dice. Default 0.5.
        blob_gate: Onboard blob-gate threshold. Default 0.55.

    Returns:
        SegmentorMetrics: Mean overlap and pixel BCE. n=0 yields zeros.
    """
    pred_logits = _iter_masks(logits)
    gold = _iter_masks(masks)
    if len(pred_logits) != len(gold):
        raise ValueError("logits and masks must have equal N")
    n = len(pred_logits)
    if n == 0:
        return SegmentorMetrics(
            n=0,
            mean_iou=0.0,
            mean_dice=0.0,
            mean_iou_blob_gate=0.0,
            bce=0.0,
        )
    ious: list[float] = []
    dices: list[float] = []
    blob_ious: list[float] = []
    for pred_z, gold_mask in zip(pred_logits, gold, strict=True):
        probs = sigmoid(pred_z)
        ious.append(compute_iou(probs, gold_mask, mask_threshold))
        dices.append(compute_dice(probs, gold_mask, mask_threshold))
        blob_ious.append(compute_iou(probs, gold_mask, blob_gate))
    bce = binary_cross_entropy_with_logits(
        np.asarray(logits, dtype=np.float64),
        np.asarray(masks, dtype=np.float64),
    )
    return SegmentorMetrics(
        n=n,
        mean_iou=float(np.mean(ious)),
        mean_dice=float(np.mean(dices)),
        mean_iou_blob_gate=float(np.mean(blob_ious)),
        bce=bce,
    )
