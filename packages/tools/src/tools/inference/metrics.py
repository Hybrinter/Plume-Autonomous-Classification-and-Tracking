"""Classifier and segmentation scoring helpers (torch).

Public names are defined in ``tools.ml_models.train.metrics``.

Contains:
  - binary_accuracy / mean_binary_accuracy: image-level presence scores.
  - compute_iou / compute_dice: per-mask overlap at a probability threshold.
  - confusion_counts / precision_recall_f1: thresholded binary scores.
  - roc_auc / average_precision / brier_score / reliability_bins.
  - classifier_metrics / segmentor_metrics: split-level summaries.
  - binary_cross_entropy_with_logits: BCE for reports.
"""

from __future__ import annotations

from tools.ml_models.train.metrics import (
    BLOB_GATE,
    LOGIT_THRESHOLD,
    MASK_THRESHOLD,
    ArrayLike,
    ClassifierMetrics,
    ConfusionCounts,
    ReliabilityBin,
    SegmentorMetrics,
    average_precision,
    binary_accuracy,
    binary_cross_entropy_with_logits,
    brier_score,
    classifier_metrics,
    compute_dice,
    compute_iou,
    confusion_counts,
    mean_binary_accuracy,
    precision_recall_f1,
    reliability_bins,
    roc_auc,
    segmentor_metrics,
    sigmoid,
)

__all__ = [
    "BLOB_GATE",
    "LOGIT_THRESHOLD",
    "MASK_THRESHOLD",
    "ArrayLike",
    "ClassifierMetrics",
    "ConfusionCounts",
    "ReliabilityBin",
    "SegmentorMetrics",
    "average_precision",
    "binary_accuracy",
    "binary_cross_entropy_with_logits",
    "brier_score",
    "classifier_metrics",
    "compute_dice",
    "compute_iou",
    "confusion_counts",
    "mean_binary_accuracy",
    "precision_recall_f1",
    "reliability_bins",
    "roc_auc",
    "segmentor_metrics",
    "sigmoid",
]
