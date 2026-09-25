"""Ranking scores and overlap scores."""

from __future__ import annotations

import torch
from tools.original_dataset_analysis.metrics import (
    score_classifier,
    score_on_native_grid,
    score_segmentor,
)


def test_perfect_classifier_scores() -> None:
    """Separated logits score precision, recall, and F1 of one."""
    logits = torch.tensor([10.0, -10.0, 10.0, -10.0])
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    scores = score_classifier(logits, labels)
    assert scores.precision == 1.0
    assert scores.recall == 1.0
    assert scores.f1 == 1.0
    assert scores.roc_auc == 1.0


def test_tied_classifier_scores_are_order_independent() -> None:
    """Equal scores report ROC-AUC 0.5 for either label order."""
    forward = score_classifier(torch.tensor([1.0, 1.0]), torch.tensor([1.0, 0.0]))
    reverse = score_classifier(torch.tensor([1.0, 1.0]), torch.tensor([0.0, 1.0]))
    assert forward.roc_auc == 0.5
    assert reverse.roc_auc == 0.5
    assert forward.pr_auc == reverse.pr_auc


def test_large_logits_keep_their_rank() -> None:
    """Logits past float32 sigmoid saturation still rank by their value."""
    positive_higher = score_classifier(torch.tensor([20.0, 30.0]), torch.tensor([0.0, 1.0]))
    negative_higher = score_classifier(torch.tensor([30.0, 20.0]), torch.tensor([0.0, 1.0]))
    assert positive_higher.roc_auc == 1.0
    assert negative_higher.roc_auc == 0.0


def test_perfect_segmentor_scores() -> None:
    """A logit plane that matches its mask scores Dice and accuracy of one."""
    mask = torch.zeros(1, 1, 4, 4)
    mask[:, :, :2, :] = 1.0
    logits = torch.where(mask > 0.5, torch.full_like(mask, 10.0), torch.full_like(mask, -10.0))
    scores = score_segmentor(logits, mask)
    assert scores.dice == 1.0
    assert scores.accuracy == 1.0
    assert scores.mean_iou == 1.0


def test_native_grid_expands_a_coarse_hit() -> None:
    """A positive coarse cell scores as that block on the 120 px mask."""
    logits = torch.full((1, 1, 1, 1), 10.0)
    native = torch.ones(1, 120, 120)
    scores = score_on_native_grid(logits, native)
    assert scores.dice == 1.0
    assert scores.positive_iou == 1.0
