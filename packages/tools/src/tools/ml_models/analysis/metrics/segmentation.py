"""Exact per-image foreground evidence and bounded pixel diagnostics.

Positive-truth overlap excludes empty masks. Explicit verified empty masks
form their own false-mask population. Pixel ranking uses a labeled histogram
approximation; overlap, areas, counts, probability losses and bin sums are exact.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt
from flight.libs.types import Err, Ok, Result
from scipy import ndimage

from tools.ml_models.analysis.config import ScoreConfig
from tools.ml_models.analysis.contracts import (
    AvailabilityRecord,
    CurveEvidence,
    MetricSupport,
    MetricValue,
    NamedCount,
)
from tools.ml_models.analysis.metrics.inputs import binary_vectors, finite_mean, sigmoid


@dataclass(frozen=True, slots=True)
class PixelHistogram:
    """Fixed-memory class counts and probability sums on explicit bin grids."""

    positive_counts: tuple[int, ...]
    negative_counts: tuple[int, ...]
    probability_sums: tuple[float, ...]
    calibration_counts: tuple[int, ...]
    calibration_positive_counts: tuple[int, ...]
    calibration_probability_sums: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ThresholdCounts:
    """Exact foreground confusion values at one probability threshold."""

    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int


@dataclass(frozen=True, slots=True)
class SegmentationRow:
    """Scalar sufficient statistics for one explicitly annotated image."""

    label: float
    verified_empty: bool
    n_pixels: int
    target_area_px: int
    predicted_area_px: int
    target_area_m2: float | None
    predicted_area_m2: float | None
    tp: int
    fp: int
    tn: int
    fn: int
    iou: float
    dice: float
    bce: float
    brier: float
    foreground_brier: float | None
    background_brier: float | None
    foreground_probability_residual: float | None
    background_probability_residual: float | None
    n_predicted_blobs: int
    histogram: PixelHistogram
    thresholds: tuple[ThresholdCounts, ...]
    score_config: ScoreConfig


@dataclass(frozen=True, slots=True)
class SegmentationEvidence:
    """Aggregate exact scores and persisted histogram/curve diagnostics."""

    metrics: tuple[MetricValue, ...]
    curves: tuple[CurveEvidence, ...]
    support: MetricSupport
    histogram: PixelHistogram
    outputs: tuple[AvailabilityRecord, ...]


def _threshold(score: npt.NDArray[np.float64], value: float) -> npt.NDArray[np.bool_]:
    if value == 0:
        return np.ones(score.shape, dtype=np.bool_)
    if value == 1:
        return np.zeros(score.shape, dtype=np.bool_)
    return score >= math.log(value) - math.log1p(-value)


def _confusion(
    predicted: npt.NDArray[np.bool_],
    target: npt.NDArray[np.bool_],
    threshold: float,
) -> ThresholdCounts:
    return ThresholdCounts(
        threshold=threshold,
        tp=int(np.count_nonzero(predicted & target)),
        fp=int(np.count_nonzero(predicted & ~target)),
        tn=int(np.count_nonzero(~predicted & ~target)),
        fn=int(np.count_nonzero(~predicted & target)),
    )


def _histogram(
    probability: npt.NDArray[np.float64],
    target: npt.NDArray[np.bool_],
    cfg: ScoreConfig,
) -> PixelHistogram:
    rank_edges = (
        np.arange(cfg.pixel_histogram_bins + 1, dtype=np.float64) / cfg.pixel_histogram_bins
    )
    rank = np.minimum(
        np.searchsorted(rank_edges, probability, side="right") - 1,
        cfg.pixel_histogram_bins - 1,
    )
    calibration_edges = (
        np.arange(cfg.n_calibration_bins + 1, dtype=np.float64) / cfg.n_calibration_bins
    )
    calibration = np.minimum(
        np.searchsorted(calibration_edges, probability, side="right") - 1,
        cfg.n_calibration_bins - 1,
    )
    return PixelHistogram(
        positive_counts=tuple(
            int(value) for value in np.bincount(rank[target], minlength=cfg.pixel_histogram_bins)
        ),
        negative_counts=tuple(
            int(value) for value in np.bincount(rank[~target], minlength=cfg.pixel_histogram_bins)
        ),
        probability_sums=tuple(
            float(value)
            for value in np.bincount(rank, weights=probability, minlength=cfg.pixel_histogram_bins)
        ),
        calibration_counts=tuple(
            int(value) for value in np.bincount(calibration, minlength=cfg.n_calibration_bins)
        ),
        calibration_positive_counts=tuple(
            int(value)
            for value in np.bincount(calibration[target], minlength=cfg.n_calibration_bins)
        ),
        calibration_probability_sums=tuple(
            float(value)
            for value in np.bincount(
                calibration, weights=probability, minlength=cfg.n_calibration_bins
            )
        ),
    )


def score_segmentation_image(
    logits: npt.ArrayLike,
    mask: npt.ArrayLike,
    *,
    label: float,
    verified_empty: bool,
    gsd: tuple[float, float] | None = None,
    cfg: ScoreConfig | None = None,
) -> Result[SegmentationRow, str]:
    """Measure one aligned H-by-W or singleton-channel annotated mask."""
    resolved = cfg if cfg is not None else ScoreConfig()
    if isinstance(label, bool) or label not in (0.0, 1.0) or not isinstance(verified_empty, bool):
        return Err("segmentation requires a binary image label and explicit verification flag")
    try:
        raw_score, raw_target = np.asarray(logits), np.asarray(mask)
        if raw_score.shape != raw_target.shape:
            return Err("segmentation logits and mask shapes disagree")
        if raw_score.ndim == 3 and raw_score.shape[0] == 1:
            raw_score, raw_target = raw_score[0], raw_target[0]
        if raw_score.ndim != 2 or min(raw_score.shape) < 1:
            return Err("segmentation expects one nonempty H-by-W mask")
        shape = raw_score.shape
        validated = binary_vectors(raw_score.reshape(-1), raw_target.reshape(-1))
    except (TypeError, ValueError) as exc:
        return Err(f"invalid mask arrays: {exc}")
    if isinstance(validated, Err):
        return Err(validated.error)
    score, target = validated.value
    positive_area = int(target.sum())
    if verified_empty and (label != 0.0 or positive_area):
        return Err("verified-empty annotations must be negative-labeled and have zero truth area")
    if label == 0.0 and positive_area:
        return Err("negative image label conflicts with positive mask pixels")
    pixel_area: float | None = None
    if gsd is not None:
        if len(gsd) != 2 or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in gsd
        ):
            return Err("GSD requires finite positive lateral and along-track metres")
        pixel_area = float(gsd[0] * gsd[1])
        if not math.isfinite(pixel_area) or not math.isfinite(pixel_area * len(score)):
            return Err("GSD-scaled image area is not finite")
    predicted = _threshold(score, resolved.mask_probability_threshold)
    counts = _confusion(predicted, target, resolved.mask_probability_threshold)
    predicted_area = int(predicted.sum())
    union = counts.tp + counts.fp + counts.fn
    total_area = positive_area + predicted_area
    probability = sigmoid(score)
    losses = np.logaddexp(0.0, -np.abs(score)) + np.where(
        target,
        np.maximum(-score, 0.0),
        np.maximum(score, 0.0),
    )
    residual = probability - target
    squared = residual**2
    blob_map, n_components = ndimage.label(
        _threshold(score, resolved.blob_probability_threshold).reshape(shape),
    )
    areas = np.bincount(blob_map.reshape(-1), minlength=n_components + 1)[1:]
    n_blobs = int(np.count_nonzero(areas >= resolved.min_blob_area_px))
    return Ok(
        SegmentationRow(
            label=float(label),
            verified_empty=verified_empty,
            n_pixels=len(score),
            target_area_px=positive_area,
            predicted_area_px=predicted_area,
            target_area_m2=positive_area * pixel_area if pixel_area is not None else None,
            predicted_area_m2=predicted_area * pixel_area if pixel_area is not None else None,
            tp=counts.tp,
            fp=counts.fp,
            tn=counts.tn,
            fn=counts.fn,
            iou=counts.tp / union if union else 1.0,
            dice=2 * counts.tp / total_area if total_area else 1.0,
            bce=finite_mean(losses),
            brier=finite_mean(squared),
            foreground_brier=finite_mean(squared[target]) if positive_area else None,
            background_brier=finite_mean(squared[~target]) if positive_area < len(score) else None,
            foreground_probability_residual=finite_mean(residual[target])
            if positive_area
            else None,
            background_probability_residual=finite_mean(residual[~target])
            if positive_area < len(score)
            else None,
            n_predicted_blobs=n_blobs,
            histogram=_histogram(probability, target, resolved),
            thresholds=tuple(
                _confusion(_threshold(score, threshold), target, threshold)
                for threshold in resolved.probability_thresholds
            ),
            score_config=resolved,
        )
    )


def _value(
    name: str,
    value: float | None,
    support: MetricSupport,
    *,
    aggregation: str,
    unit: str = "dimensionless",
    threshold: float | None = None,
) -> MetricValue:
    return MetricValue(
        name=name,
        value=value,
        status="AVAILABLE" if value is not None else "UNAVAILABLE",
        reason=None if value is not None else "no eligible observations or denominator",
        support=support,
        aggregation=aggregation,
        unit=unit,
        threshold=threshold,
    )


def _mean(values: tuple[float, ...]) -> float | None:
    return math.fsum(value / len(values) for value in values) if values else None


def _merge_histograms(rows: tuple[SegmentationRow, ...]) -> PixelHistogram:
    histograms = tuple(row.histogram for row in rows)
    first = histograms[0]
    return PixelHistogram(
        positive_counts=tuple(
            sum(hist.positive_counts[i] for hist in histograms)
            for i in range(len(first.positive_counts))
        ),
        negative_counts=tuple(
            sum(hist.negative_counts[i] for hist in histograms)
            for i in range(len(first.negative_counts))
        ),
        probability_sums=tuple(
            math.fsum(hist.probability_sums[i] for hist in histograms)
            for i in range(len(first.probability_sums))
        ),
        calibration_counts=tuple(
            sum(hist.calibration_counts[i] for hist in histograms)
            for i in range(len(first.calibration_counts))
        ),
        calibration_positive_counts=tuple(
            sum(hist.calibration_positive_counts[i] for hist in histograms)
            for i in range(len(first.calibration_positive_counts))
        ),
        calibration_probability_sums=tuple(
            math.fsum(hist.calibration_probability_sums[i] for hist in histograms)
            for i in range(len(first.calibration_probability_sums))
        ),
    )


def _pixel_curves(
    histogram: PixelHistogram,
    support: MetricSupport,
) -> tuple[tuple[MetricValue, ...], tuple[CurveEvidence, ...], tuple[AvailabilityRecord, ...]]:
    positive, negative = sum(histogram.positive_counts), sum(histogram.negative_counts)
    n_bins = len(histogram.positive_counts)
    selected_bins = tuple(
        index
        for index in reversed(range(n_bins))
        if histogram.positive_counts[index] + histogram.negative_counts[index]
    )
    tp, fp = 0, 0
    tprs, fprs, precision = [0.0], [0.0], [1.0]
    thresholds: list[float | None] = [None]
    for index in selected_bins:
        tp += histogram.positive_counts[index]
        fp += histogram.negative_counts[index]
        tprs.append(tp / positive if positive else 0.0)
        fprs.append(fp / negative if negative else 0.0)
        precision.append(tp / (tp + fp))
        thresholds.append(index / n_bins)
    roc = (
        math.fsum(
            (tprs[i] + tprs[i - 1]) / 2 * (fprs[i] - fprs[i - 1]) for i in range(1, len(tprs))
        )
        if positive and negative
        else None
    )
    ap = (
        math.fsum((tprs[i] - tprs[i - 1]) * precision[i] for i in range(1, len(tprs)))
        if positive
        else None
    )
    curves: list[CurveEvidence] = []
    outputs: list[AvailabilityRecord] = []
    for name, xs, ys, available in (
        ("pixel_roc_histogram", fprs, tprs, positive > 0 and negative > 0),
        ("pixel_precision_recall_histogram", tprs, precision, positive > 0),
    ):
        if available:
            curves.append(
                CurveEvidence(
                    name=name,
                    x_name="false_positive_rate" if name == "pixel_roc_histogram" else "recall",
                    y_name="true_positive_rate" if name == "pixel_roc_histogram" else "precision",
                    x=tuple(xs),
                    y=tuple(ys),
                    x_unit="fraction",
                    y_unit="fraction",
                    support=support,
                    method="HISTOGRAM",
                    n_bins=n_bins,
                    thresholds=tuple(thresholds),
                    notes=(
                        "Approximate probability ranking: all pixels inside a "
                        "fixed-width bin are tied.",
                    ),
                )
            )
            outputs.append(AvailabilityRecord(name=name, status="AVAILABLE"))
        else:
            outputs.append(
                AvailabilityRecord(
                    name=name, status="UNAVAILABLE", reason="required truth classes are absent"
                )
            )
    calibration_total = sum(histogram.calibration_counts)
    ece_terms: list[float] = []
    means: list[float] = []
    fractions: list[float | None] = []
    calibration_bins = len(histogram.calibration_counts)
    for index, count in enumerate(histogram.calibration_counts):
        if count:
            mean = histogram.calibration_probability_sums[index] / count
            fraction = histogram.calibration_positive_counts[index] / count
            ece_terms.append(count / calibration_total * abs(mean - fraction))
            means.append(mean)
            fractions.append(fraction)
        else:
            means.append((index + 0.5) / calibration_bins)
            fractions.append(None)
    curves.extend(
        (
            CurveEvidence(
                name="pixel_calibration_reliability",
                x_name="mean_probability",
                y_name="positive_fraction",
                x=tuple(means),
                y=tuple(fractions),
                x_unit="probability",
                y_unit="fraction",
                support=support,
                notes=(
                    "Exact bin aggregates; correlated pixels are not independent observations.",
                ),
            ),
            CurveEvidence(
                name="pixel_calibration_support",
                x_name="probability_bin_midpoint",
                y_name="n_pixels",
                x=tuple((index + 0.5) / calibration_bins for index in range(calibration_bins)),
                y=tuple(float(count) for count in histogram.calibration_counts),
                x_unit="probability",
                y_unit="count",
                support=support,
            ),
        )
    )
    metrics = (
        _value(
            "pixel_roc_auc_histogram",
            roc,
            support,
            aggregation="fixed_width_probability_histogram_approximation",
        ),
        _value(
            "pixel_average_precision_histogram",
            ap,
            support,
            aggregation="fixed_width_probability_histogram_approximation",
        ),
        _value(
            "pixel_expected_calibration_error",
            math.fsum(ece_terms),
            support,
            aggregation="exact_support_weighted_equal_width_bins",
        ),
    )
    return metrics, tuple(curves), tuple(outputs)


def aggregate_segmentation(
    rows: tuple[SegmentationRow, ...],
    *,
    histogram: PixelHistogram | None = None,
) -> Result[SegmentationEvidence, str]:
    """Aggregate whole-image records independently of batch grouping or spatial size."""
    if not rows:
        return Err("segmentation aggregation requires annotated images")
    cfg = rows[0].score_config
    if any(row.score_config != cfg for row in rows):
        return Err("segmentation rows disagree on scoring settings")
    n = len(rows)
    positive = tuple(row for row in rows if row.target_area_px > 0)
    empty = tuple(row for row in rows if row.verified_empty)
    support = MetricSupport(
        unit="IMAGE",
        n=n,
        counts=(
            NamedCount(name="n_positive_truth_images", value=len(positive)),
            NamedCount(name="n_verified_empty_images", value=len(empty)),
            NamedCount(
                name="n_positive_label_empty_masks",
                value=sum(row.label == 1 and row.target_area_px == 0 for row in rows),
            ),
            NamedCount(
                name="n_unverified_negative_empty_masks",
                value=sum(
                    row.label == 0 and row.target_area_px == 0 and not row.verified_empty
                    for row in rows
                ),
            ),
        ),
    )
    positive_support = MetricSupport(unit="IMAGE", n=len(positive))
    empty_support = MetricSupport(unit="IMAGE", n=len(empty))
    total_pixels = sum(row.n_pixels for row in rows)
    pixel_support = MetricSupport(
        unit="PIXEL",
        n=total_pixels,
        counts=(
            NamedCount(name="n_foreground_pixels", value=sum(row.target_area_px for row in rows)),
            NamedCount(
                name="n_background_pixels",
                value=sum(row.n_pixels - row.target_area_px for row in rows),
            ),
        ),
    )
    tp, fp, fn = (
        sum(row.tp for row in rows),
        sum(row.fp for row in rows),
        sum(row.fn for row in rows),
    )
    metrics: list[MetricValue] = []
    for name, values, cohort in (
        (
            "foreground_iou_mean_positive_images",
            tuple(row.iou for row in positive),
            positive_support,
        ),
        (
            "foreground_dice_mean_positive_images",
            tuple(row.dice for row in positive),
            positive_support,
        ),
        ("foreground_iou_mean_all_annotated_images", tuple(row.iou for row in rows), support),
        ("foreground_dice_mean_all_annotated_images", tuple(row.dice for row in rows), support),
    ):
        metrics.append(
            _value(
                name,
                _mean(values),
                cohort,
                aggregation="equal_image_mean",
                threshold=cfg.mask_probability_threshold,
            )
        )
    for name, numerator, denominator in (
        ("foreground_iou_global", tp, tp + fp + fn),
        ("foreground_dice_global", 2 * tp, 2 * tp + fp + fn),
        ("foreground_precision_global", tp, tp + fp),
        ("foreground_recall_global", tp, tp + fn),
    ):
        metrics.append(
            _value(
                name,
                numerator / denominator if denominator else None,
                pixel_support,
                aggregation="pooled_pixel_counts",
                threshold=cfg.mask_probability_threshold,
            )
        )
    for name, values, cohort, unit in (
        (
            "binary_cross_entropy_mean_images",
            tuple(row.bce for row in rows),
            support,
            "dimensionless",
        ),
        ("brier_score_mean_images", tuple(row.brier for row in rows), support, "dimensionless"),
        (
            "area_signed_error_mean_px",
            tuple(float(row.predicted_area_px - row.target_area_px) for row in rows),
            support,
            "pixels",
        ),
        (
            "area_absolute_error_mean_px",
            tuple(float(abs(row.predicted_area_px - row.target_area_px)) for row in rows),
            support,
            "pixels",
        ),
        (
            "verified_negative_any_foreground_rate",
            tuple(float(row.predicted_area_px > 0) for row in empty),
            empty_support,
            "fraction",
        ),
        (
            "verified_negative_any_blob_rate",
            tuple(float(row.n_predicted_blobs > 0) for row in empty),
            empty_support,
            "fraction",
        ),
        (
            "verified_negative_false_blobs_mean",
            tuple(float(row.n_predicted_blobs) for row in empty),
            empty_support,
            "blobs_per_image",
        ),
    ):
        metrics.append(
            _value(name, _mean(values), cohort, aggregation="equal_image_mean", unit=unit)
        )
    spatial_rows = tuple(
        row for row in rows if row.target_area_m2 is not None and row.predicted_area_m2 is not None
    )
    for name, absolute in (
        ("area_signed_error_mean_m2", False),
        ("area_absolute_error_mean_m2", True),
    ):
        values = tuple(
            abs(row.predicted_area_m2 - row.target_area_m2)
            if absolute
            else row.predicted_area_m2 - row.target_area_m2
            for row in spatial_rows
            if row.predicted_area_m2 is not None and row.target_area_m2 is not None
        )
        metrics.append(
            _value(
                name,
                _mean(values),
                MetricSupport(unit="IMAGE", n=len(spatial_rows)),
                aggregation="equal_image_mean_local_gsd_approximation",
                unit="square_metres",
            )
        )
    metrics.extend(
        (
            _value(
                "binary_cross_entropy_mean_pixels",
                math.fsum(row.bce * (row.n_pixels / total_pixels) for row in rows),
                pixel_support,
                aggregation="pixel_weighted_image_means",
            ),
            _value(
                "brier_score_mean_pixels",
                math.fsum(row.brier * (row.n_pixels / total_pixels) for row in rows),
                pixel_support,
                aggregation="pixel_weighted_image_means",
            ),
        )
    )
    for name, foreground in (
        ("foreground_brier_score", True),
        ("background_brier_score", False),
        ("foreground_probability_residual_mean", True),
        ("background_probability_residual_mean", False),
    ):
        weighted: list[tuple[float, int]] = []
        for row in rows:
            count = row.target_area_px if foreground else row.n_pixels - row.target_area_px
            value = row.foreground_brier if foreground else row.background_brier
            if "residual" in name:
                value = (
                    row.foreground_probability_residual
                    if foreground
                    else row.background_probability_residual
                )
            if value is not None:
                weighted.append((value, count))
        count = sum(weight for _, weight in weighted)
        mean = math.fsum(value * (weight / count) for value, weight in weighted) if count else None
        metrics.append(
            _value(
                name,
                mean,
                MetricSupport(unit="PIXEL", n=count),
                aggregation="truth_class_pixel_weighted_image_means",
            )
        )
    histogram = histogram if histogram is not None else _merge_histograms(rows)
    histogram_metrics, curves, outputs = _pixel_curves(histogram, pixel_support)
    metrics.extend(histogram_metrics)
    threshold_ious: list[float | None] = []
    threshold_dices: list[float | None] = []
    for index in range(len(cfg.probability_thresholds)):
        ious, dices = [], []
        for row in positive:
            counts = row.thresholds[index]
            ious.append(counts.tp / (counts.tp + counts.fp + counts.fn))
            dices.append(2 * counts.tp / (2 * counts.tp + counts.fp + counts.fn))
        threshold_ious.append(_mean(tuple(ious)))
        threshold_dices.append(_mean(tuple(dices)))
    curves += tuple(
        CurveEvidence(
            name=name,
            x_name="probability_threshold",
            y_name=name,
            x=cfg.probability_thresholds,
            y=tuple(values),
            x_unit="probability",
            y_unit="fraction",
            support=positive_support,
        )
        for name, values in (
            ("threshold_positive_image_iou", threshold_ious),
            ("threshold_positive_image_dice", threshold_dices),
        )
    )
    return Ok(SegmentationEvidence(tuple(metrics), curves, support, histogram, outputs))


class SegmentationAccumulator:
    """Keep one pooled histogram plus scalar image records, never per-image bin arrays."""

    def __init__(self) -> None:
        self._rows: list[SegmentationRow] = []
        self._histogram: PixelHistogram | None = None

    def add(self, row: SegmentationRow) -> Result[None, str]:
        """Accumulate validated image statistics with one common scoring convention."""
        if self._rows and row.score_config != self._rows[0].score_config:
            return Err("segmentation rows disagree on scoring settings")
        if self._histogram is None:
            self._histogram = row.histogram
        else:
            self._histogram = _merge_histograms((replace(row, histogram=self._histogram), row))
        self._rows.append(replace(row, histogram=PixelHistogram((), (), (), (), (), ())))
        return Ok(None)

    def result(self) -> Result[SegmentationEvidence, str]:
        """Aggregate the scalar records against their one pooled histogram."""
        return aggregate_segmentation(tuple(self._rows), histogram=self._histogram)
