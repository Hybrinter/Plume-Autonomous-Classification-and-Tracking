"""Tests for tools.inference metrics."""

import numpy as np
import pytest
from tools.inference.metrics import (
    BLOB_GATE,
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


def test_binary_accuracy_match_and_mismatch() -> None:
    """binary_accuracy is 1.0 on a match and 0.0 on a mismatch."""
    assert binary_accuracy(True, True) == 1.0
    assert binary_accuracy(False, True) == 0.0


def test_mean_binary_accuracy() -> None:
    """mean_binary_accuracy averages pairwise scores."""
    assert mean_binary_accuracy((True, False), (True, False)) == 1.0
    assert mean_binary_accuracy((True, True), (True, False)) == 0.5
    assert mean_binary_accuracy((), ()) == 0.0


def test_mean_binary_accuracy_rejects_length_mismatch() -> None:
    """Unequal prediction/label lengths raise ValueError."""
    with pytest.raises(ValueError):
        mean_binary_accuracy((True,), (True, False))


def test_compute_iou() -> None:
    """compute_iou is 1.0 for identical masks and < 1 for partial overlap."""
    a = np.zeros((4, 4), dtype=np.float32)
    a[0:2, 0:2] = 1.0
    assert compute_iou(a, a) == 1.0
    b = np.zeros((4, 4), dtype=np.float32)
    b[1:3, 1:3] = 1.0
    assert 0.0 < compute_iou(a, b) < 1.0
    empty = np.zeros((4, 4), dtype=np.float32)
    assert compute_iou(empty, empty) == 1.0


def test_compute_dice() -> None:
    """compute_dice is 1.0 for identical masks."""
    a = np.zeros((4, 4), dtype=np.float32)
    a[0:2, 0:2] = 1.0
    assert compute_dice(a, a) == 1.0
    empty = np.zeros((4, 4), dtype=np.float32)
    assert compute_dice(empty, empty) == 1.0


def test_precision_recall_f1() -> None:
    """Perfect predictions yield precision, recall, and F1 of 1.0."""
    pred = np.array([True, False, True, False])
    label = np.array([True, False, True, False])
    precision, recall, f1 = precision_recall_f1(confusion_counts(pred, label))
    assert precision == 1.0
    assert recall == 1.0
    assert f1 == 1.0


def test_precision_recall_zero_positive_predictions() -> None:
    """No predicted positives yields precision 0.0."""
    pred = np.array([False, False])
    label = np.array([True, False])
    precision, recall, f1 = precision_recall_f1(confusion_counts(pred, label))
    assert precision == 0.0
    assert recall == 0.0
    assert f1 == 0.0


def test_roc_auc_perfect_and_empty() -> None:
    """Perfect ranking is 1.0. Empty or single-class inputs are 0.0."""
    scores = np.array([3.0, 2.0, -1.0, -2.0])
    labels = np.array([1.0, 1.0, 0.0, 0.0])
    assert roc_auc(scores, labels) == pytest.approx(1.0)
    assert roc_auc(np.array([]), np.array([])) == 0.0
    assert roc_auc(np.array([1.0, 2.0]), np.array([1.0, 1.0])) == 0.0


def test_average_precision_and_brier() -> None:
    """Average precision is 1.0 for a perfect ranking. Brier is 0 for 0/1 probs."""
    scores = np.array([3.0, 2.0, -1.0])
    labels = np.array([1.0, 1.0, 0.0])
    assert average_precision(scores, labels) == pytest.approx(1.0)
    probs = np.array([1.0, 0.0, 1.0])
    y = np.array([1.0, 0.0, 1.0])
    assert brier_score(probs, y) == pytest.approx(0.0)
    assert average_precision(np.array([]), np.array([])) == 0.0


def test_reliability_bins_skips_empty() -> None:
    """reliability_bins returns occupied bins only."""
    probs = np.array([0.1, 0.1, 0.9])
    labels = np.array([0.0, 0.0, 1.0])
    bins = reliability_bins(probs, labels, n_bins=2)
    assert len(bins) == 2
    assert bins[0].count == 2
    assert bins[1].accuracy == pytest.approx(1.0)


def test_classifier_metrics_empty_and_perfect() -> None:
    """Empty n is 0. Perfect logits yield accuracy and F1 of 1.0."""
    empty = classifier_metrics(np.zeros((0, 1)), np.zeros((0, 1)))
    assert empty.n == 0
    assert empty.accuracy == 0.0
    logits = np.array([[4.0], [-4.0], [3.0], [-3.0]])
    labels = np.array([[1.0], [0.0], [1.0], [0.0]])
    report = classifier_metrics(logits, labels)
    assert report.n == 4
    assert report.accuracy == pytest.approx(1.0)
    assert report.f1 == pytest.approx(1.0)
    assert report.roc_auc == pytest.approx(1.0)


def test_segmentor_metrics_reports_blob_gate() -> None:
    """segmentor_metrics reports IoU at 0.5 and at the 0.55 blob gate."""
    logits = np.full((2, 1, 4, 4), 6.0, dtype=np.float32)
    masks = np.ones((2, 1, 4, 4), dtype=np.float32)
    report = segmentor_metrics(logits, masks)
    assert report.n == 2
    assert report.mean_iou == pytest.approx(1.0)
    assert report.mean_dice == pytest.approx(1.0)
    assert report.mean_iou_blob_gate == pytest.approx(1.0)
    assert BLOB_GATE == pytest.approx(0.55)
    empty = segmentor_metrics(np.zeros((0, 1, 4, 4)), np.zeros((0, 1, 4, 4)))
    assert empty.n == 0
    assert empty.mean_iou == 0.0


def test_binary_cross_entropy_with_logits_zero_on_empty() -> None:
    """Empty tensors yield BCE 0.0. Confident correct logits are small."""
    assert binary_cross_entropy_with_logits(np.zeros((0,)), np.zeros((0,))) == 0.0
    loss = binary_cross_entropy_with_logits(np.array([8.0]), np.array([1.0]))
    assert loss < 0.01


def test_sigmoid_maps_zero_to_half() -> None:
    """sigmoid(0) is 0.5."""
    assert float(sigmoid(np.array([0.0]))[0]) == pytest.approx(0.5)
