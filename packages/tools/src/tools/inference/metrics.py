"""Classifier and segmentation scoring helpers (torch).

This module holds design metrics for both heads. Acceptance and training import
these helpers. Graphs emit logits; ROC, PR, and Brier use sigmoid probabilities.
Empty inputs return 0.0 with n=0.

Contains:
  - binary_accuracy / mean_binary_accuracy: image-level presence scores.
  - compute_iou / compute_dice: per-mask overlap at a probability threshold.
  - confusion_counts / precision_recall_f1: thresholded binary scores.
  - roc_auc / average_precision / brier_score / reliability_bins.
  - classifier_metrics / segmentor_metrics: split-level summaries.
  - binary_cross_entropy_with_logits: BCE for reports.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as functional

BLOB_GATE = 0.55
MASK_THRESHOLD = 0.5
LOGIT_THRESHOLD = 0.0
_LOGIT_CLAMP = 60.0
_EPS = 1e-12

ArrayLike = torch.Tensor | np.ndarray


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


def _as_tensor(value: ArrayLike, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Return a CPU tensor view of ``value``."""
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _as_bool(value: ArrayLike) -> torch.Tensor:
    """Ravel ``value`` to a 1-D bool tensor (nonzero is True)."""
    tensor = _as_tensor(value).reshape(-1)
    if tensor.dtype == torch.bool:
        return tensor
    return tensor != 0


def sigmoid(logits: ArrayLike) -> torch.Tensor:
    """Return elementwise logistic sigmoid of logits.

    Args:
        logits: Unbounded scores.

    Returns:
        torch.Tensor: Probabilities in (0, 1) with the same shape.
    """
    tensor = _as_tensor(logits, dtype=torch.float64)
    return torch.sigmoid(tensor.clamp(-_LOGIT_CLAMP, _LOGIT_CLAMP))


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


def compute_iou(pred_mask: ArrayLike, gold_mask: ArrayLike, threshold: float = 0.5) -> float:
    """Compute the binary intersection-over-union of a predicted vs golden mask.

    Args:
        pred_mask: Predicted probability mask (H, W).
        gold_mask: Golden probability mask (H, W).
        threshold: Probability threshold for binarization.

    Returns:
        float: IoU in [0, 1]. Two empty masks score 1.0.
    """
    pred = _as_tensor(pred_mask) >= threshold
    gold = _as_tensor(gold_mask) >= threshold
    intersection = float(torch.logical_and(pred, gold).sum().item())
    union = float(torch.logical_or(pred, gold).sum().item())
    if union == 0.0:
        return 1.0
    return intersection / union


def compute_dice(pred_mask: ArrayLike, gold_mask: ArrayLike, threshold: float = 0.5) -> float:
    """Compute the binary Dice coefficient of a predicted vs golden mask.

    Args:
        pred_mask: Predicted probability mask (H, W).
        gold_mask: Golden probability mask (H, W).
        threshold: Probability threshold for binarization.

    Returns:
        float: Dice in [0, 1]. Two empty masks score 1.0.
    """
    pred = _as_tensor(pred_mask) >= threshold
    gold = _as_tensor(gold_mask) >= threshold
    intersection = float(torch.logical_and(pred, gold).sum().item())
    total = float(pred.sum().item() + gold.sum().item())
    if total == 0.0:
        return 1.0
    return (2.0 * intersection) / total


def confusion_counts(pred_positive: ArrayLike, label_positive: ArrayLike) -> ConfusionCounts:
    """Return confusion counts for aligned boolean arrays.

    Args:
        pred_positive: Predicted flags.
        label_positive: Ground-truth flags. Must match length.

    Returns:
        ConfusionCounts: tp, fp, tn, fn.

    Raises:
        ValueError: If lengths differ.
    """
    pred = _as_bool(pred_positive)
    label = _as_bool(label_positive)
    if pred.numel() != label.numel():
        raise ValueError("pred_positive and label_positive must have equal length")
    tp = int(torch.logical_and(pred, label).sum().item())
    fp = int(torch.logical_and(pred, torch.logical_not(label)).sum().item())
    tn = int(torch.logical_and(torch.logical_not(pred), torch.logical_not(label)).sum().item())
    fn = int(torch.logical_and(torch.logical_not(pred), label).sum().item())
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


def _flatten_pair(scores: ArrayLike, labels: ArrayLike) -> tuple[torch.Tensor, torch.Tensor]:
    """Return 1-D float64 score and label tensors of equal length."""
    score_t = _as_tensor(scores, dtype=torch.float64).reshape(-1)
    label_t = _as_tensor(labels, dtype=torch.float64).reshape(-1)
    if score_t.numel() != label_t.numel():
        raise ValueError("scores and labels must have equal length")
    return score_t, label_t


def roc_auc(scores: ArrayLike, labels: ArrayLike) -> float:
    """Return ROC-AUC from scores and binary labels.

    Args:
        scores: Higher means more positive. Logits or probabilities.
        labels: Binary labels {0, 1}.

    Returns:
        float: Area under the ROC curve. 0.0 when n is 0 or a class is missing.
    """
    score_t, label_t = _flatten_pair(scores, labels)
    if score_t.numel() == 0:
        return 0.0
    n_pos = float(label_t.sum().item())
    n_neg = float(label_t.numel()) - n_pos
    if n_pos == 0.0 or n_neg == 0.0:
        return 0.0
    order = torch.argsort(score_t, descending=True, stable=True)
    y_sorted = label_t[order]
    tps = torch.cumsum(y_sorted, dim=0)
    fps = torch.cumsum(1.0 - y_sorted, dim=0)
    tpr = torch.cat((torch.zeros(1, dtype=torch.float64), tps / n_pos))
    fpr = torch.cat((torch.zeros(1, dtype=torch.float64), fps / n_neg))
    return float(torch.trapezoid(tpr, fpr).item())


def average_precision(scores: ArrayLike, labels: ArrayLike) -> float:
    """Return average precision (area under the precision-recall curve).

    Args:
        scores: Higher means more positive.
        labels: Binary labels {0, 1}.

    Returns:
        float: Average precision in [0, 1]. 0.0 when n is 0 or there is no
        positive label.
    """
    score_t, label_t = _flatten_pair(scores, labels)
    if score_t.numel() == 0:
        return 0.0
    n_pos = float(label_t.sum().item())
    if n_pos == 0.0:
        return 0.0
    order = torch.argsort(score_t, descending=True, stable=True)
    y_sorted = label_t[order]
    tps = torch.cumsum(y_sorted, dim=0)
    fps = torch.cumsum(1.0 - y_sorted, dim=0)
    precision = tps / torch.clamp(tps + fps, min=_EPS)
    recall = tps / n_pos
    recall_ext = torch.cat((torch.zeros(1, dtype=torch.float64), recall))
    precision_ext = torch.cat((torch.ones(1, dtype=torch.float64), precision))
    return float(torch.sum((recall_ext[1:] - recall_ext[:-1]) * precision_ext[1:]).item())


def brier_score(probs: ArrayLike, labels: ArrayLike) -> float:
    """Return mean squared error between probabilities and labels.

    Args:
        probs: Predicted probabilities in [0, 1].
        labels: Binary labels {0, 1}.

    Returns:
        float: Brier score. 0.0 when n is 0.
    """
    prob_t, label_t = _flatten_pair(probs, labels)
    if prob_t.numel() == 0:
        return 0.0
    return float(torch.mean((prob_t - label_t) ** 2).item())


def reliability_bins(
    probs: ArrayLike,
    labels: ArrayLike,
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
    prob_t, label_t = _flatten_pair(probs, labels)
    if prob_t.numel() == 0:
        return ()
    edges = torch.linspace(0.0, 1.0, n_bins + 1, dtype=torch.float64)
    bins: list[ReliabilityBin] = []
    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == n_bins - 1:
            in_bin = (prob_t >= lo) & (prob_t <= hi)
        else:
            in_bin = (prob_t >= lo) & (prob_t < hi)
        count = int(in_bin.sum().item())
        if count == 0:
            continue
        bins.append(
            ReliabilityBin(
                confidence=float(prob_t[in_bin].mean().item()),
                accuracy=float(label_t[in_bin].mean().item()),
                count=count,
            )
        )
    return tuple(bins)


def binary_cross_entropy_with_logits(logits: ArrayLike, targets: ArrayLike) -> float:
    """Return mean BCE with logits (numerically stable).

    Args:
        logits: Unbounded scores.
        targets: Labels or masks in {0, 1} with the same shape.

    Returns:
        float: Mean BCE. 0.0 when n is 0.
    """
    logit_t = _as_tensor(logits, dtype=torch.float32)
    target_t = _as_tensor(targets, dtype=torch.float32)
    if logit_t.numel() == 0:
        return 0.0
    if logit_t.shape != target_t.shape:
        raise ValueError("logits and targets must have the same shape")
    return float(functional.binary_cross_entropy_with_logits(logit_t, target_t).item())


def classifier_metrics(
    logits: ArrayLike,
    labels: ArrayLike,
    logit_threshold: float = LOGIT_THRESHOLD,
) -> ClassifierMetrics:
    """Score a classifier split from logits and labels.

    Args:
        logits: Tensor[float, (N,) or (N, 1)].
        labels: Tensor[float, (N,) or (N, 1)] in {0, 1}.
        logit_threshold: Positive when logit >= this value. Default 0.0.

    Returns:
        ClassifierMetrics: Thresholded and ranking scores. n=0 yields zeros.
    """
    logit_t = _as_tensor(logits, dtype=torch.float64).reshape(-1)
    label_t = _as_tensor(labels, dtype=torch.float64).reshape(-1)
    if logit_t.numel() != label_t.numel():
        raise ValueError("logits and labels must have equal length")
    n = int(logit_t.numel())
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
    pred = logit_t >= logit_threshold
    label_bool = label_t >= 0.5
    counts = confusion_counts(pred, label_bool)
    precision, recall, f1 = precision_recall_f1(counts)
    accuracy = float((pred == label_bool).to(dtype=torch.float64).mean().item())
    probs = sigmoid(logit_t)
    return ClassifierMetrics(
        n=n,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc(logit_t, label_t),
        pr_auc=average_precision(logit_t, label_t),
        brier=brier_score(probs, label_t),
        bce=binary_cross_entropy_with_logits(logit_t, label_t),
    )


def _iter_masks(stack: ArrayLike) -> list[torch.Tensor]:
    """Yield 2-D masks from (N, H, W) or (N, 1, H, W)."""
    arr = _as_tensor(stack)
    if arr.ndim == 2:
        return [arr]
    if arr.ndim == 3:
        return [arr[i] for i in range(arr.shape[0])]
    if arr.ndim == 4:
        return [arr[i, 0] for i in range(arr.shape[0])]
    raise ValueError(f"expected (H, W), (N, H, W), or (N, 1, H, W); got {tuple(arr.shape)}")


def segmentor_metrics(
    logits: ArrayLike,
    masks: ArrayLike,
    mask_threshold: float = MASK_THRESHOLD,
    blob_gate: float = BLOB_GATE,
) -> SegmentorMetrics:
    """Score a segmentor split from logits and gold masks.

    Args:
        logits: Tensor[float, (N, 1, H, W) or (N, H, W)].
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
    bce = binary_cross_entropy_with_logits(logits, masks)
    return SegmentorMetrics(
        n=n,
        mean_iou=float(torch.tensor(ious, dtype=torch.float64).mean().item()),
        mean_dice=float(torch.tensor(dices, dtype=torch.float64).mean().item()),
        mean_iou_blob_gate=float(torch.tensor(blob_ious, dtype=torch.float64).mean().item()),
        bce=bce,
    )
