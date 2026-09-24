"""Ranking scores and overlap scores."""

from __future__ import annotations

import torch
from tools.original_dataset_analysis.metrics import score_classifier, score_segmentor


def test_perfect_classifier_scores() -> None:
    """Separated logits score precision, recall, and F1 of one."""
    logits = torch.tensor([10.0, -10.0, 10.0, -10.0])
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    scores = score_classifier(logits, labels)
    assert scores.precision == 1.0
    assert scores.recall == 1.0
    assert scores.f1 == 1.0
    assert scores.roc_auc == 1.0


def test_perfect_segmentor_scores() -> None:
    """A logit plane that matches its mask scores Dice and accuracy of one."""
    mask = torch.zeros(1, 1, 4, 4)
    mask[:, :, :2, :] = 1.0
    logits = torch.where(mask > 0.5, torch.full_like(mask, 10.0), torch.full_like(mask, -10.0))
    scores = score_segmentor(logits, mask)
    assert scores.dice == 1.0
    assert scores.accuracy == 1.0
    assert scores.mean_iou == 1.0
