"""Independent explicit-mask population and miss-inclusive coordinate references."""

import math
from dataclasses import replace

import numpy as np
from flight.libs.types import Err, Ok
from tools.ml_models.analysis.capture import CaptureRow
from tools.ml_models.analysis.contracts import (
    CurveEvidence,
    MetricSupport,
    MetricValue,
    SampleKey,
    SplitEvidence,
)
from tools.ml_models.analysis.metrics.boundary import BoundaryRow
from tools.ml_models.analysis.metrics.localization import (
    ComponentMatch,
    LocalizationRow,
    MaskComponent,
)
from tools.ml_models.analysis.metrics.spatial import SpatialRow
from tools.ml_models.analysis.prediction_display import segmentation_display_data
from tools.ml_models.analysis.segmentation_figures import segmentation_figure_data

_HASH = "a" * 64


def _row(index: int, label: float, counts: tuple[int, int, int, int]) -> CaptureRow:
    tp, fp, tn, fn = counts
    truth, prediction = tp + fn, tp + fp
    values = (
        ("true_positive_pixels", tp),
        ("false_positive_pixels", fp),
        ("true_negative_pixels", tn),
        ("false_negative_pixels", fn),
        ("target_area_px", truth),
        ("predicted_area_px", prediction),
        ("target_area_m2", truth * 6),
        ("predicted_area_m2", prediction * 6),
        ("foreground_iou", tp / (tp + fp + fn) if tp + fp + fn else 1),
        ("foreground_dice", 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1),
        ("binary_cross_entropy", 0.5),
        ("brier_score", 0.25),
    )
    return CaptureRow(
        key=SampleKey(
            dataset_hash=_HASH,
            task="segmentor",
            split="val",
            spatial_shard=(2, 3),
            row_index=index,
            tile_id=str(index),
            element="I",
        ),
        group_id=str(index),
        bin_id="bin",
        label=label,
        gsd_m=(2.0, 3.0),
        metrics=tuple(
            MetricValue(
                name=name,
                value=float(value),
                status="AVAILABLE",
                support=MetricSupport(unit="IMAGE", n=1),
            )
            for name, value in values
        ),
        failure_score=0.5,
    )


def _cohort() -> tuple[SplitEvidence, tuple[CaptureRow, ...]]:
    truth = (
        MaskComponent(0, 1, 6.0, 0.0, 0.0, (0, 0, 0, 0)),
        MaskComponent(1, 1, 6.0, 2.0, 1.0, (2, 1, 2, 1)),
    )
    local = LocalizationRow(
        truth,
        (MaskComponent(0, 2, 12.0, 0.5, 0.0, (0, 0, 1, 0)),),
        (ComponentMatch(0, 0, 0.5, 0.5, 0.0, 0.5, 1.0),),
        (1,),
        (),
        0,
        0,
        (2.0, 3.0),
        0.55,
        1,
        0.5,
    )
    boundary = BoundaryRow(
        2,
        2,
        1,
        2,
        1.0,
        "pixel",
        (math.sqrt(2) + 1) / 4,
        1 + 0.85 * (math.sqrt(2) - 1),
        (math.sqrt(13) + 2) / 4,
        2 + 0.85 * (math.sqrt(13) - 2),
        None,
        None,
    )
    rows = (
        replace(_row(0, 1.0, (1, 1, 3, 1)), spatial=SpatialRow(local, boundary)),
        _row(1, 0.0, (0, 1, 5, 0)),
        _row(2, 1.0, (0, 0, 6, 0)),
    )
    support = MetricSupport(unit="IMAGE", n=3)
    curves = (
        CurveEvidence(
            name="localization_success_px",
            x_name="tolerance",
            y_name="success",
            x=(0.0, 0.5),
            y=(0.0, 0.5),
            x_unit="pixel",
            y_unit="fraction",
            support=MetricSupport(unit="COMPONENT", n=2),
            notes=("Includes unmatched truth",),
        ),
        CurveEvidence(
            name="pixel_precision_recall_histogram",
            x_name="recall",
            y_name="precision",
            x=(0.0, 1.0),
            y=(1.0, 0.2),
            x_unit="fraction",
            y_unit="fraction",
            support=MetricSupport(unit="PIXEL", n=18),
            method="HISTOGRAM",
            n_bins=8,
            notes=("Approximate fixed probability bin ranking; correlated pixels",),
        ),
    )
    return SplitEvidence(
        task="segmentor", split="val", dataset_hash=_HASH, support=support, curves=curves
    ), rows


def test_overlap_and_verified_negative_populations_are_separate() -> None:
    evidence, rows = _cohort()
    result = segmentation_figure_data(evidence, rows)
    assert isinstance(result, Ok)
    figures = {figure.identifier: figure for figure in result.value}
    overlap = figures["foreground_iou_distribution"]
    assert len(overlap.series) == 1
    assert overlap.series[0].x == (1 / 3,)
    assert overlap.series[0].support.n == 1
    prediction = figures["predicted_area_px_distribution"]
    assert [series.x for series in prediction.series] == [(2.0,), (1.0,), (0.0,)]
    precision = figures["image_foreground_precision"]
    assert [series.x for series in precision.series] == [(0.5,), (0.0,), ()]
    assert figures["image_foreground_recall"].series[1].support.n == 0


def test_conditional_centroids_do_not_replace_miss_inclusive_success_curve() -> None:
    evidence, rows = _cohort()
    result = segmentation_figure_data(evidence, rows)
    assert isinstance(result, Ok)
    figures = {figure.identifier: figure for figure in result.value}
    conditional = figures["matched_centroid_error_px"].series[0]
    assert conditional.x == (0.5,) and conditional.y == (1.0,) and conditional.support.n == 1
    assert figures["matched_centroid_error_m"].series[0].x == (1.0,)
    success = figures["localization_success_px"]
    assert success.series[0].y == (0.0, 0.5) and success.series[0].support.n == 2
    assert success.series[0].style == "POST"


def test_area_errors_boundary_units_and_histogram_method_are_preserved() -> None:
    evidence, rows = _cohort()
    result = segmentation_figure_data(evidence, rows)
    assert isinstance(result, Ok)
    figures = {figure.identifier: figure for figure in result.value}
    assert figures["area_signed_error_m2"].series[1].x == (6.0,)
    assert figures["boundary_asd_m_distribution"].series[0].x == ((math.sqrt(13) + 2) / 4,)
    approximate = figures["pixel_precision_recall_histogram"]
    assert "Approximate histogram" in approximate.title
    assert approximate.series[0].style == "PRE"
    assert approximate.series[0].y == (1.0, 0.2)
    assert "correlated pixels" in approximate.notes[0]
    assert figures["pixel_roc_histogram"].reason


def test_missing_counts_wrong_task_or_mixed_identity_fail_closed() -> None:
    evidence, rows = _cohort()
    missing = replace(
        rows[0],
        metrics=tuple(metric for metric in rows[0].metrics if metric.name != "target_area_px"),
    )
    assert isinstance(segmentation_figure_data(evidence, (missing, *rows[1:])), Err)
    assert isinstance(segmentation_figure_data(replace(evidence, task="classifier"), rows), Err)
    changed = replace(
        rows[0],
        metrics=tuple(
            replace(metric, value=3.0) if metric.name == "target_area_px" else metric
            for metric in rows[0].metrics
        ),
    )
    assert isinstance(segmentation_figure_data(evidence, (changed, *rows[1:])), Err)


def test_no_spatial_capture_is_explicitly_indexed_not_fabricated() -> None:
    evidence, rows = _cohort()
    result = segmentation_figure_data(
        replace(evidence, curves=()), tuple(replace(row, spatial=None) for row in rows)
    )
    assert isinstance(result, Ok)
    figures = {figure.identifier: figure for figure in result.value}
    assert figures["spatial_capture"].reason
    assert figures["matched_centroid_error_px"].reason
    assert figures["localization_success_px"].reason


def test_overlap_scalars_cannot_disagree_with_captured_pixel_counts() -> None:
    evidence, rows = _cohort()
    changed = replace(
        rows[0],
        metrics=tuple(
            replace(metric, value=1.0) if metric.name == "foreground_iou" else metric
            for metric in rows[0].metrics
        ),
    )
    assert isinstance(segmentation_figure_data(evidence, (changed, *rows[1:])), Err)


def test_prediction_display_uses_cached_logits_and_exact_frozen_mask_threshold() -> None:
    _, rows = _cohort()
    target = np.array([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]], dtype=np.float32)
    logits = np.array([[[1.0, 1.0, -1.0], [-1.0, -1.0, -1.0]]], dtype=np.float32)
    result = segmentation_display_data(rows[0], target, logits)
    assert isinstance(result, Ok)
    assert result.value.truth.tolist() == [[True, False, False], [False, False, True]]
    assert result.value.predicted.tolist() == [[True, True, False], [False, False, False]]
    assert result.value.false_positive.tolist() == [[False, True, False], [False, False, False]]
    assert result.value.false_negative.tolist() == [[False, False, False], [False, False, True]]
    assert result.value.probability[0, 0] == 1 / (1 + np.exp(-1))
    assert result.value.threshold == 0.5
    assert not result.value.probability.flags.writeable
    assert not result.value.predicted.flags.writeable
    assert isinstance(segmentation_display_data(rows[0], target, -logits), Err)
    assert isinstance(
        segmentation_display_data(replace(rows[0], spatial=None), target, logits), Err
    )


def test_prediction_display_threshold_endpoints_do_not_use_rounded_probabilities() -> None:
    _, rows = _cohort()
    spatial = rows[0].spatial
    assert spatial is not None
    target = np.array([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]], dtype=np.float32)
    logits = np.full((1, 2, 3), 1000.0, dtype=np.float32)
    for threshold, counts, expected in ((0.0, (2, 4, 0, 0), True), (1.0, (0, 0, 4, 2), False)):
        row = replace(
            _row(0, 1.0, counts),
            spatial=replace(
                spatial, boundary=replace(spatial.boundary, mask_probability_threshold=threshold)
            ),
        )
        result = segmentation_display_data(row, target, logits)
        assert isinstance(result, Ok)
        assert np.all(result.value.probability == 1.0)
        assert np.all(result.value.predicted == expected)
