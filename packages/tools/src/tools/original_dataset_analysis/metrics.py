"""Test-split scores for the classifier and the segmentor.

Contains:
  - ClassificationScores: precision, recall, F1, PR-AUC, and ROC-AUC.
  - SegmentationScores: mean IoU, Dice, and accuracy.
  - score_classifier: scores from logits and binary labels.
  - score_segmentor: scores from logit planes and binary masks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class ClassificationScores:
    """Threshold-0.5 rates plus ranking curves.

    Attributes:
        precision: Positive predictive value.
        recall: True positive rate.
        f1: Harmonic mean of precision and recall.
        pr_auc: Area under the precision-recall curve.
        roc_auc: Area under the receiver-operating curve.
    """

    precision: float
    recall: float
    f1: float
    pr_auc: float
    roc_auc: float


@dataclass(frozen=True, slots=True)
class SegmentationScores:
    """Mask overlap at threshold 0.5.

    Attributes:
        mean_iou: Mean of the positive-class and background IoU.
        dice: Dice of the positive class.
        accuracy: Fraction of pixels that match the mask.
    """

    mean_iou: float
    dice: float
    accuracy: float


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Return the trapezoid integral of ``y`` along ``x``."""
    area = np.trapezoid(y, x)
    return float(area)


def _ranking_areas(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Return PR-AUC and ROC-AUC from ranking scores.

    Equal scores form one threshold. The curve steps once for that group.
    """
    order = np.argsort(-scores, kind="mergesort")
    ranked_scores = scores[order]
    ranked_labels = labels[order]
    positives = float(ranked_labels.sum())
    negatives = float(len(ranked_labels) - positives)
    if positives == 0.0 or negatives == 0.0:
        return 0.0, 0.0
    group_end = np.empty(len(ranked_scores), dtype=bool)
    group_end[:-1] = ranked_scores[:-1] != ranked_scores[1:]
    group_end[-1] = True
    tp = np.cumsum(ranked_labels)[group_end]
    fp = np.cumsum(1.0 - ranked_labels)[group_end]
    recall = tp / positives
    precision = tp / np.maximum(tp + fp, 1.0)
    tpr = recall
    fpr = fp / negatives
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    fpr = np.concatenate([[0.0], fpr])
    tpr = np.concatenate([[0.0], tpr])
    return _trapz(precision, recall), _trapz(tpr, fpr)


def score_classifier(logits: torch.Tensor, labels: torch.Tensor) -> ClassificationScores:
    """Score binary logits.

    Args:
        logits: Shape ``(N,)`` or ``(N, 1)``.
        labels: Matching shape, values in {0, 1}.

    Returns:
        ClassificationScores: Threshold-0.5 rates and areas from the logits.

    Raises:
        ValueError: If the tensors do not share a length.
    """
    raw = logits.detach().double().reshape(-1).cpu().numpy()
    truth = labels.detach().double().reshape(-1).cpu().numpy()
    if raw.shape != truth.shape:
        raise ValueError(f"logits length {raw.shape} != labels length {truth.shape}")
    predicted = raw >= 0.0
    truth_bool = truth >= 0.5
    tp = float(np.logical_and(predicted, truth_bool).sum())
    fp = float(np.logical_and(predicted, np.logical_not(truth_bool)).sum())
    fn = float(np.logical_and(np.logical_not(predicted), truth_bool).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    pr_auc, roc_auc = _ranking_areas(raw, truth_bool.astype(np.float64))
    return ClassificationScores(precision, recall, f1, pr_auc, roc_auc)


def score_segmentor(logits: torch.Tensor, masks: torch.Tensor) -> SegmentationScores:
    """Score a logit plane against a binary mask.

    Args:
        logits: Shape ``(N, 1, H, W)``.
        masks: Matching shape, values in {0, 1}.

    Returns:
        SegmentationScores: Mean IoU, Dice, and pixel accuracy.

    Raises:
        ValueError: If the shapes differ.
    """
    if logits.shape != masks.shape:
        raise ValueError(f"logits shape {tuple(logits.shape)} != masks {tuple(masks.shape)}")
    predicted = torch.sigmoid(logits.detach().float()) >= 0.5
    truth = masks.detach().float() >= 0.5
    intersection = torch.logical_and(predicted, truth).sum().item()
    union = torch.logical_or(predicted, truth).sum().item()
    pred_sum = predicted.sum().item()
    truth_sum = truth.sum().item()
    iou_pos = intersection / union if union else 1.0
    bg_intersection = torch.logical_and(~predicted, ~truth).sum().item()
    bg_union = torch.logical_or(~predicted, ~truth).sum().item()
    iou_bg = bg_intersection / bg_union if bg_union else 1.0
    dice = 2.0 * intersection / (pred_sum + truth_sum) if pred_sum + truth_sum else 1.0
    accuracy = float(torch.eq(predicted, truth).float().mean().item())
    return SegmentationScores(0.5 * (iou_pos + iou_bg), float(dice), accuracy)
