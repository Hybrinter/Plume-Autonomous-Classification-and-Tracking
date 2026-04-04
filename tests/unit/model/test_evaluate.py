"""Unit tests for pact.model.evaluate — iou_score, dice_score, precision_recall.

Satisfies: §6.2 of PACT_SW_ARCH.md — Model subsystem unit tests.
REQ-AIML-HIGH-001, REQ-AIML-HIGH-002
"""

from __future__ import annotations

# third-party
import numpy as np
import pytest

# module under test
from pact.model.evaluate import dice_score, iou_score, precision_recall


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ones_mask(shape: tuple[int, int] = (4, 4)) -> np.ndarray:
    """Return a float32 array of ones."""
    return np.ones(shape, dtype=np.float32)


def _zeros_mask(shape: tuple[int, int] = (4, 4)) -> np.ndarray:
    """Return a float32 array of zeros."""
    return np.zeros(shape, dtype=np.float32)


def _checkerboard_mask(shape: tuple[int, int] = (4, 4)) -> np.ndarray:
    """Return a checkerboard binary float32 mask — alternating 0 and 1."""
    arr = np.zeros(shape, dtype=np.float32)
    arr[::2, ::2] = 1.0
    arr[1::2, 1::2] = 1.0
    return arr


# ---------------------------------------------------------------------------
# iou_score tests
# ---------------------------------------------------------------------------


def test_iou_perfect_overlap() -> None:
    """Identical non-empty masks yield IoU = 1.0."""
    mask = _ones_mask()
    assert iou_score(mask, mask) == pytest.approx(1.0)


def test_iou_no_overlap() -> None:
    """Non-overlapping masks yield IoU = 0.0."""
    pred = np.zeros((4, 4), dtype=np.float32)
    pred[:2, :] = 1.0   # top half
    true = np.zeros((4, 4), dtype=np.float32)
    true[2:, :] = 1.0   # bottom half
    assert iou_score(pred, true) == pytest.approx(0.0)


def test_iou_both_empty_returns_one() -> None:
    """Both masks all-zero (empty) is a correct empty prediction — IoU = 1.0."""
    pred = _zeros_mask()
    true = _zeros_mask()
    assert iou_score(pred, true) == pytest.approx(1.0)


def test_iou_partial_overlap() -> None:
    """Partial overlap yields IoU strictly between 0 and 1."""
    pred = np.zeros((4, 4), dtype=np.float32)
    pred[:3, :3] = 1.0
    true = np.zeros((4, 4), dtype=np.float32)
    true[1:, 1:] = 1.0
    score = iou_score(pred, true)
    assert 0.0 < score < 1.0


@pytest.mark.parametrize("threshold,expected", [
    (0.0, 1.0),    # threshold=0 → all pixels predicted → checkerboard ∩ checkerboard = 1.0
    (0.49, 1.0),   # just below 0.5 → same as above for 0/1 masks
    (0.5, 1.0),    # at 0.5 → pixels > 0.5 ↦ only the 1.0 pixels → identical → 1.0
    (0.99, 1.0),   # above 0.5 → identical logic, only 1.0 pixels selected
])
def test_iou_threshold_boundary(threshold: float, expected: float) -> None:
    """iou_score threshold boundary: identical masks always yield 1.0 regardless of threshold."""
    mask = _checkerboard_mask()
    assert iou_score(mask, mask, threshold=threshold) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# dice_score tests
# ---------------------------------------------------------------------------


def test_dice_perfect_overlap() -> None:
    """Identical non-empty masks yield Dice = 1.0."""
    mask = _ones_mask()
    assert dice_score(mask, mask) == pytest.approx(1.0)


def test_dice_no_overlap() -> None:
    """Non-overlapping masks yield Dice = 0.0."""
    pred = np.zeros((4, 4), dtype=np.float32)
    pred[:2, :] = 1.0
    true = np.zeros((4, 4), dtype=np.float32)
    true[2:, :] = 1.0
    assert dice_score(pred, true) == pytest.approx(0.0)


def test_dice_both_empty_returns_one() -> None:
    """Both empty masks yield Dice = 1.0 (correct empty prediction)."""
    assert dice_score(_zeros_mask(), _zeros_mask()) == pytest.approx(1.0)


@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7, 0.9])
def test_dice_threshold_parametrize(threshold: float) -> None:
    """dice_score on identical binary masks returns 1.0 for any threshold < 1.0."""
    mask = _ones_mask()
    assert dice_score(mask, mask, threshold=threshold) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# precision_recall tests
# ---------------------------------------------------------------------------


def test_precision_recall_all_correct() -> None:
    """Identical non-empty masks yield (precision=1.0, recall=1.0)."""
    mask = _ones_mask()
    precision, recall = precision_recall(mask, mask)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(1.0)


def test_precision_recall_no_tp_fp_pred_empty() -> None:
    """When pred is empty: no FP, no TP. precision=1.0 (no false positives), recall=0.0."""
    pred = _zeros_mask()
    true = _ones_mask()
    precision, recall = precision_recall(pred, true)
    # precision: no predicted positives → no FP → 1.0 (by convention in evaluate.py)
    assert precision == pytest.approx(1.0)
    # recall: all true positives missed → 0.0
    assert recall == pytest.approx(0.0)


def test_precision_recall_all_fp_true_empty() -> None:
    """When true is empty and pred is full: all predictions are FP → precision=0.0."""
    pred = _ones_mask()
    true = _zeros_mask()
    precision, recall = precision_recall(pred, true)
    assert precision == pytest.approx(0.0)
    # recall: no true positives → 1.0 (by convention — nothing to recall)
    assert recall == pytest.approx(1.0)


@pytest.mark.parametrize("threshold,expected_prec,expected_rec", [
    (0.3, 1.0, 1.0),   # well below 0.5 → pred=ones → all TP
    (0.49, 1.0, 1.0),  # just below 0.5 → same
    (0.5, 1.0, 1.0),   # at boundary → pixels strictly > 0.5 → none for 0.5-valued pixels
    (0.9, 1.0, 1.0),   # above blob value → matches ones mask
])
def test_precision_recall_threshold_boundary(
    threshold: float,
    expected_prec: float,
    expected_rec: float,
) -> None:
    """precision_recall threshold boundary on identical all-ones masks."""
    mask = _ones_mask()
    precision, recall = precision_recall(mask, mask, threshold=threshold)
    assert precision == pytest.approx(expected_prec)
    assert recall == pytest.approx(expected_rec)
