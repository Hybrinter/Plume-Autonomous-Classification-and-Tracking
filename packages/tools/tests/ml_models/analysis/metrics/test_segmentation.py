"""Independent overlap, population, area and histogram oracles."""

import math

import numpy as np
import pytest
from flight.libs.types import Err, Ok
from tools.ml_models.analysis.config import ScoreConfig
from tools.ml_models.analysis.metrics.segmentation import (
    SegmentationEvidence,
    aggregate_segmentation,
    score_segmentation_image,
)


def _scores(evidence: SegmentationEvidence) -> dict[str, float | None]:
    return {metric.name: metric.value for metric in evidence.metrics}


def test_partial_overlap_and_anisotropic_area() -> None:
    """One intersection, three union pixels give IoU 1/3 and Dice 1/2."""
    result = score_segmentation_image(
        [[2.0, -2.0], [2.0, -2.0]],
        [[1, 1], [0, 0]],
        label=1.0,
        verified_empty=False,
        gsd=(2.0, 3.0),
    )
    assert isinstance(result, Ok)
    row = result.value
    assert (row.tp, row.fp, row.tn, row.fn) == (1, 1, 1, 1)
    assert row.iou == pytest.approx(1 / 3)
    assert row.dice == pytest.approx(0.5)
    assert row.target_area_px == row.predicted_area_px == 2
    assert row.target_area_m2 == row.predicted_area_m2 == 12
    assert row.bce == pytest.approx(1 + math.log1p(math.exp(-2)))
    aggregate = aggregate_segmentation((row,))
    assert isinstance(aggregate, Ok)
    scores = _scores(aggregate.value)
    assert scores["foreground_iou_mean_positive_images"] == pytest.approx(1 / 3)
    assert scores["foreground_precision_global"] == 0.5
    assert scores["foreground_recall_global"] == 0.5
    assert scores["area_signed_error_mean_m2"] == 0


def test_empty_masks_do_not_inflate_positive_headline() -> None:
    """Perfect verified negatives cannot hide a fully missed positive plume."""
    positive = score_segmentation_image([[-2.0]], [[1]], label=1.0, verified_empty=False)
    negative = score_segmentation_image([[-2.0]], [[0]], label=0.0, verified_empty=True)
    unresolved = score_segmentation_image([[-2.0]], [[0]], label=1.0, verified_empty=False)
    assert isinstance(positive, Ok) and isinstance(negative, Ok) and isinstance(unresolved, Ok)
    result = aggregate_segmentation((positive.value, negative.value, unresolved.value))
    assert isinstance(result, Ok)
    scores = _scores(result.value)
    assert scores["foreground_iou_mean_positive_images"] == 0
    assert scores["foreground_dice_mean_positive_images"] == 0
    assert scores["foreground_iou_mean_all_annotated_images"] == pytest.approx(2 / 3)
    assert scores["verified_negative_any_foreground_rate"] == 0
    counts = {count.name: count.value for count in result.value.support.counts}
    assert counts["n_positive_truth_images"] == 1
    assert counts["n_verified_empty_images"] == 1
    assert counts["n_positive_label_empty_masks"] == 1
    negative_only = aggregate_segmentation((negative.value,))
    assert isinstance(negative_only, Ok)
    assert _scores(negative_only.value)["foreground_iou_mean_positive_images"] is None


def test_sample_weighted_and_pixel_weighted_losses_differ() -> None:
    """A one-pixel and three-pixel image have distinct explicit weighting."""
    small = score_segmentation_image([[2.0]], [[1]], label=1.0, verified_empty=False)
    large = score_segmentation_image([[2.0, 2.0, 2.0]], [[0, 0, 0]], label=0.0, verified_empty=True)
    assert isinstance(small, Ok) and isinstance(large, Ok)
    result = aggregate_segmentation((small.value, large.value))
    reverse = aggregate_segmentation((large.value, small.value))
    assert isinstance(result, Ok) and isinstance(reverse, Ok)
    assert result.value.metrics == reverse.value.metrics
    scores = _scores(result.value)
    base = math.log1p(math.exp(-2))
    assert scores["binary_cross_entropy_mean_images"] == pytest.approx(base + 1)
    assert scores["binary_cross_entropy_mean_pixels"] == pytest.approx(base + 1.5)
    assert scores["foreground_iou_global"] == pytest.approx(0.25)
    assert scores["foreground_iou_mean_positive_images"] == 1
    assert scores["area_signed_error_mean_px"] == 1.5


def test_target_validation_does_not_use_prediction_threshold() -> None:
    """A low probability threshold does not turn zero-valued targets positive."""
    cfg = ScoreConfig(mask_probability_threshold=0.0, pixel_histogram_bins=4)
    result = score_segmentation_image(
        [[-1000.0, -1000.0]],
        [[1, 0]],
        label=1.0,
        verified_empty=False,
        cfg=cfg,
    )
    assert isinstance(result, Ok)
    assert result.value.tp == 1 and result.value.fp == 1
    assert result.value.target_area_px == 1
    assert result.value.histogram.negative_counts[0] == 1
    assert result.value.histogram.positive_counts[0] == 1


def test_verified_negative_blob_gate_is_distinct_from_raw_mask() -> None:
    """An isolated false pixel is foreground but not a retained two-pixel blob."""
    cfg = ScoreConfig(min_blob_area_px=2, pixel_histogram_bins=4)
    result = score_segmentation_image(
        [[2.0, -2.0], [-2.0, -2.0]],
        [[0, 0], [0, 0]],
        label=0.0,
        verified_empty=True,
        cfg=cfg,
    )
    assert isinstance(result, Ok)
    assert result.value.predicted_area_px == 1
    assert result.value.n_predicted_blobs == 0
    aggregate = aggregate_segmentation((result.value,))
    assert isinstance(aggregate, Ok)
    scores = _scores(aggregate.value)
    assert scores["verified_negative_any_foreground_rate"] == 1
    assert scores["verified_negative_any_blob_rate"] == 0
    assert scores["verified_negative_false_blobs_mean"] == 0


def test_histogram_exact_counts_and_approximation_labels() -> None:
    """Pixel histogram diagnostics preserve counts and declare approximate ranking."""
    cfg = ScoreConfig(pixel_histogram_bins=4, n_calibration_bins=2)
    logits = np.array([[-1000.0, math.log(1 / 3), 0.0, 1000.0]])
    scored = score_segmentation_image(
        logits,
        [[0, 0, 1, 1]],
        label=1.0,
        verified_empty=False,
        cfg=cfg,
    )
    assert isinstance(scored, Ok)
    hist = scored.value.histogram
    assert hist.positive_counts == (0, 0, 1, 1)
    assert hist.negative_counts == (1, 1, 0, 0)
    assert sum(hist.positive_counts) + sum(hist.negative_counts) == 4
    assert len(hist.positive_counts) == len(hist.probability_sums) == 4
    aggregate = aggregate_segmentation((scored.value,))
    assert isinstance(aggregate, Ok)
    scores = _scores(aggregate.value)
    assert scores["pixel_average_precision_histogram"] == 1
    assert scores["pixel_roc_auc_histogram"] == 1
    curves = {curve.name: curve for curve in aggregate.value.curves}
    assert curves["pixel_roc_histogram"].method == "HISTOGRAM"
    assert curves["pixel_roc_histogram"].n_bins == 4
    assert curves["pixel_roc_histogram"].notes
    tied = score_segmentation_image(
        [[0.0, 0.0]],
        [[1, 0]],
        label=1.0,
        verified_empty=False,
        cfg=cfg,
    )
    assert isinstance(tied, Ok)
    tied_aggregate = aggregate_segmentation((tied.value,))
    assert isinstance(tied_aggregate, Ok)
    assert _scores(tied_aggregate.value)["pixel_roc_auc_histogram"] == 0.5
    assert _scores(tied_aggregate.value)["pixel_average_precision_histogram"] == 0.5


def test_invalid_masks_and_population_contradictions() -> None:
    """Invalid annotation/domain information fails instead of becoming a score."""
    for logits, mask, label, verified in (
        ([[0.0]], [[0.5]], 1.0, False),
        ([[0.0]], [[1]], 1.0, True),
        ([[0.0]], [[0]], 1.0, True),
        ([[float("nan")]], [[0]], 0.0, True),
        ([[0.0]], [[0, 1]], 1.0, False),
        ([[True]], [[0]], 0.0, True),
        ([[0.0]], [[0]], 0.5, False),
    ):
        assert isinstance(
            score_segmentation_image(
                logits,
                mask,
                label=label,
                verified_empty=verified,
            ),
            Err,
        )
    assert isinstance(
        score_segmentation_image([[0.0]], [[0]], label=0.0, verified_empty=True, gsd=(0.0, 1.0)),
        Err,
    )
    assert isinstance(aggregate_segmentation(()), Err)


def test_exact_pixel_bin_edges_and_mixed_settings_refusal() -> None:
    """Zero/one probabilities enter the endpoint bins and unlike settings cannot merge."""
    first = score_segmentation_image(
        [[-1000.0, 1000.0]],
        [[0, 1]],
        label=1.0,
        verified_empty=False,
        cfg=ScoreConfig(pixel_histogram_bins=4, n_calibration_bins=2),
    )
    second = score_segmentation_image(
        [[-1000.0, 1000.0]],
        [[0, 1]],
        label=1.0,
        verified_empty=False,
        cfg=ScoreConfig(pixel_histogram_bins=8, n_calibration_bins=2),
    )
    assert isinstance(first, Ok) and isinstance(second, Ok)
    assert first.value.histogram.calibration_counts == (1, 1)
    assert first.value.histogram.calibration_probability_sums == (0.0, 1.0)
    assert isinstance(aggregate_segmentation((first.value, second.value)), Err)


def test_one_pixel_translation_and_truth_class_residuals() -> None:
    """A pure displaced prediction is zero overlap despite identical area."""
    row = score_segmentation_image(
        [[-2.0, 2.0]],
        [[1, 0]],
        label=1.0,
        verified_empty=False,
    )
    assert isinstance(row, Ok)
    assert row.value.iou == row.value.dice == 0
    result = aggregate_segmentation((row.value,))
    assert isinstance(result, Ok)
    scores = _scores(result.value)
    probability = 1 / (1 + math.exp(-2))
    assert scores["foreground_brier_score"] == pytest.approx(probability**2)
    assert scores["background_brier_score"] == pytest.approx(probability**2)
    assert scores["foreground_probability_residual_mean"] == pytest.approx(-probability)
    assert scores["background_probability_residual_mean"] == pytest.approx(probability)
