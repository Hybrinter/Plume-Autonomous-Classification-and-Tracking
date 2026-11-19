"""Declarative binary metric definitions, directions and limitations.

Definitions are scientific metadata, not a callable dispatch mechanism.
TP/FP/TN/FN refer to the explicitly selected operating point.

Satisfies: REQ-AIML-HIGH-004.
"""

from dataclasses import dataclass
from typing import Literal

from flight.libs.types import Err, Ok, Result


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    """Formula, direction, target population and unavailable-score convention."""

    name: str
    direction: Literal["MINIMIZE", "MAXIMIZE", "DESCRIPTIVE"]
    formula: str
    population: str
    aggregation: str
    undefined_policy: str
    limitations: str
    unit: str = "dimensionless"


_POINT_LIMIT = "Threshold-dependent; report confusion counts, class support and frozen settings."
_NO_DENOMINATOR = "Unavailable when the formula denominator is zero."
_COHORT = "Every eligible image once; no augmentation or training resampling weighting."
_POINT_FORMULAS = (
    ("accuracy", "(TP+TN)/(TP+FP+TN+FN)"),
    ("precision", "TP/(TP+FP)"),
    ("recall", "TP/(TP+FN)"),
    ("specificity", "TN/(TN+FP)"),
    ("negative_predictive_value", "TN/(TN+FN)"),
    ("false_positive_rate", "FP/(FP+TN)"),
    ("false_negative_rate", "FN/(FN+TP)"),
    ("f1", "2*TP/(2*TP+FP+FN)"),
    ("f_beta", "(1+beta^2)*TP/((1+beta^2)*TP+beta^2*FN+FP)"),
    ("balanced_accuracy", "(recall+specificity)/2"),
    ("matthews_correlation", "(TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))"),
    ("predicted_positive_fraction", "(TP+FP)/(TP+FP+TN+FN)"),
)
CLASSIFIER_DEFINITIONS = tuple(
    MetricDefinition(
        name=name,
        direction="MINIMIZE"
        if name in ("false_positive_rate", "false_negative_rate")
        else "DESCRIPTIVE"
        if name == "predicted_positive_fraction"
        else "MAXIMIZE",
        formula=formula,
        population=_COHORT,
        aggregation="confusion_ratio",
        undefined_policy="Unavailable if either truth class is missing."
        if name == "balanced_accuracy"
        else _NO_DENOMINATOR,
        limitations=_POINT_LIMIT,
    )
    for name, formula in _POINT_FORMULAS
) + (
    MetricDefinition(
        name="roc_auc",
        direction="MAXIMIZE",
        formula="Trapezoidal ROC area at complete equal-logit score groups.",
        population=_COHORT,
        aggregation="complete_tie_group_trapezoid",
        undefined_policy="Unavailable if either truth class is missing.",
        limitations="Ranking score, not precision at an operational false-alarm rate.",
    ),
    MetricDefinition(
        name="average_precision",
        direction="MAXIMIZE",
        formula="Sum over complete score groups of (recall_k-recall_previous)*precision_k.",
        population=_COHORT,
        aggregation="complete_tie_group_recall_increments",
        undefined_policy="Unavailable without positive labels; all-positive AP is valid.",
        limitations="Not trapezoidal PR area. Depends on evaluated class prevalence.",
    ),
    MetricDefinition(
        name="binary_cross_entropy",
        direction="MINIMIZE",
        formula="Mean softplus(-abs(z)) + max(z,0) for y=0 or max(-z,0) for y=1.",
        population=_COHORT,
        aggregation="unweighted_image_mean",
        undefined_policy="Infinite probability-baseline loss is unavailable with a reason.",
        limitations="Unweighted probability quality; not a weighted/focal training objective.",
    ),
)
CALIBRATION_DEFINITIONS = (
    MetricDefinition(
        name="brier_score",
        direction="MINIMIZE",
        formula="Mean (p-y)^2.",
        population=_COHORT,
        aggregation="unweighted_probability_mean",
        undefined_policy="Empty cohorts are invalid.",
        limitations="Class prevalence affects the constant-probability baseline.",
    ),
    MetricDefinition(
        name="expected_calibration_error",
        direction="MINIMIZE",
        formula="Sum over occupied bins of (n_bin/N)*abs(mean_probability-positive_fraction).",
        population=_COHORT,
        aggregation="support_weighted_equal_width_bins",
        undefined_policy="Empty bins have null means and zero weight; empty cohorts are invalid.",
        limitations=(
            "Bin-dependent diagnostic; not a proper scoring rule or substitute "
            "for reliability plots."
        ),
    ),
)


_SEGMENTATION_FORMULAS = (
    (
        "foreground_iou_mean_positive_images",
        "Mean TP/(TP+FP+FN) over nonempty truth images.",
        "nonempty_truth_images",
        "equal_image_mean",
        "dimensionless",
    ),
    (
        "foreground_dice_mean_positive_images",
        "Mean 2*TP/(2*TP+FP+FN) over nonempty truth images.",
        "nonempty_truth_images",
        "equal_image_mean",
        "dimensionless",
    ),
    (
        "foreground_iou_mean_all_annotated_images",
        "Mean image foreground IoU; empty/empty is one.",
        "all_annotated_images",
        "equal_image_mean",
        "dimensionless",
    ),
    (
        "foreground_dice_mean_all_annotated_images",
        "Mean image foreground Dice; empty/empty is one.",
        "all_annotated_images",
        "equal_image_mean",
        "dimensionless",
    ),
    (
        "foreground_iou_global",
        "Pooled TP/(TP+FP+FN).",
        "all_annotated_pixels",
        "pooled_pixel_counts",
        "dimensionless",
    ),
    (
        "foreground_dice_global",
        "Pooled 2*TP/(2*TP+FP+FN).",
        "all_annotated_pixels",
        "pooled_pixel_counts",
        "dimensionless",
    ),
    (
        "foreground_precision_global",
        "Pooled TP/(TP+FP).",
        "all_annotated_pixels",
        "pooled_pixel_counts",
        "dimensionless",
    ),
    (
        "foreground_recall_global",
        "Pooled TP/(TP+FN).",
        "all_annotated_pixels",
        "pooled_pixel_counts",
        "dimensionless",
    ),
    (
        "binary_cross_entropy_mean_images",
        "Mean per-image unweighted pixel BCE.",
        "all_annotated_images",
        "equal_image_mean",
        "dimensionless",
    ),
    (
        "binary_cross_entropy_mean_pixels",
        "Sum image_mean_BCE*image_pixels/total_pixels.",
        "all_annotated_pixels",
        "pixel_weighted_image_means",
        "dimensionless",
    ),
    (
        "brier_score_mean_images",
        "Mean per-image mean (p-y)^2.",
        "all_annotated_images",
        "equal_image_mean",
        "dimensionless",
    ),
    (
        "brier_score_mean_pixels",
        "Sum image_mean_Brier*image_pixels/total_pixels.",
        "all_annotated_pixels",
        "pixel_weighted_image_means",
        "dimensionless",
    ),
    (
        "area_signed_error_mean_px",
        "Mean predicted_area_px-truth_area_px.",
        "all_annotated_images",
        "equal_image_mean",
        "pixels",
    ),
    (
        "area_absolute_error_mean_px",
        "Mean abs(predicted_area_px-truth_area_px).",
        "all_annotated_images",
        "equal_image_mean",
        "pixels",
    ),
    (
        "area_signed_error_mean_m2",
        "Mean (predicted_area_px-truth_area_px)*lateral_GSD*along_GSD.",
        "images_with_gsd",
        "equal_image_mean_local_gsd_approximation",
        "square_metres",
    ),
    (
        "area_absolute_error_mean_m2",
        "Mean abs(predicted_area_px-truth_area_px)*lateral_GSD*along_GSD.",
        "images_with_gsd",
        "equal_image_mean_local_gsd_approximation",
        "square_metres",
    ),
    (
        "verified_negative_any_foreground_rate",
        "Fraction with any raw predicted foreground pixel.",
        "verified_negative_empty_images",
        "equal_image_mean",
        "fraction",
    ),
    (
        "verified_negative_any_blob_rate",
        "Fraction with any blob surviving threshold and minimum area.",
        "verified_negative_empty_images",
        "equal_image_mean",
        "fraction",
    ),
    (
        "verified_negative_false_blobs_mean",
        "Mean retained four-connected component count.",
        "verified_negative_empty_images",
        "equal_image_mean",
        "blobs_per_image",
    ),
    (
        "foreground_brier_score",
        "Mean (p-1)^2 over truth foreground pixels.",
        "truth_foreground_pixels",
        "truth_class_pixel_weighted_image_means",
        "dimensionless",
    ),
    (
        "background_brier_score",
        "Mean p^2 over truth background pixels.",
        "truth_background_pixels",
        "truth_class_pixel_weighted_image_means",
        "dimensionless",
    ),
    (
        "foreground_probability_residual_mean",
        "Mean p-1 over truth foreground pixels.",
        "truth_foreground_pixels",
        "truth_class_pixel_weighted_image_means",
        "dimensionless",
    ),
    (
        "background_probability_residual_mean",
        "Mean p over truth background pixels.",
        "truth_background_pixels",
        "truth_class_pixel_weighted_image_means",
        "dimensionless",
    ),
    (
        "pixel_roc_auc_histogram",
        "ROC trapezoidal area with equal-width probability bins treated as ties.",
        "all_annotated_pixels",
        "fixed_width_probability_histogram_approximation",
        "dimensionless",
    ),
    (
        "pixel_average_precision_histogram",
        "Recall-increment AP with equal-width probability bins treated as ties.",
        "all_annotated_pixels",
        "fixed_width_probability_histogram_approximation",
        "dimensionless",
    ),
    (
        "pixel_expected_calibration_error",
        "Sum occupied-bin count/N * abs(mean_probability-positive_fraction).",
        "all_annotated_pixels",
        "exact_support_weighted_equal_width_bins",
        "dimensionless",
    ),
)
SEGMENTATION_DEFINITIONS = tuple(
    MetricDefinition(
        name=name,
        direction="MAXIMIZE"
        if name.startswith("foreground_iou")
        or name.startswith("foreground_dice")
        or name
        in (
            "foreground_precision_global",
            "foreground_recall_global",
            "pixel_roc_auc_histogram",
            "pixel_average_precision_histogram",
        )
        else "DESCRIPTIVE"
        if name.startswith("area_signed") or "probability_residual" in name
        else "MINIMIZE",
        formula=formula,
        population=population,
        aggregation=aggregation,
        undefined_policy=(
            "Unavailable when no eligible observations or a denominator is zero; "
            "ROC needs both classes and AP needs positives."
        ),
        limitations=(
            "Pixels are correlated. GSD area is a local approximation. Empty "
            "images cannot inflate positive-image headlines. Histogram ranking "
            "is approximate; thresholds and aggregation are explicit."
        ),
        unit=unit,
    )
    for name, formula, population, aggregation, unit in _SEGMENTATION_FORMULAS
)


_LOCALIZATION_FORMULAS = (
    (
        "truth_components",
        "Count all unfiltered four-connected explicit truth components.",
        "exact_component_count",
        "component",
    ),
    (
        "predicted_components",
        "Count four-connected prediction components passing the configured blob area gate.",
        "exact_component_count",
        "component",
    ),
    (
        "matched_components",
        "Maximum-cardinality eligible one-to-one IoU matches, then maximum summed IoU.",
        "exact_component_count",
        "component",
    ),
    (
        "unmatched_truth_components",
        "Truth component count minus eligible matches.",
        "exact_component_count",
        "component",
    ),
    (
        "unmatched_prediction_components",
        "Retained prediction component count minus eligible matches.",
        "exact_component_count",
        "component",
    ),
    (
        "component_precision",
        "Eligible matches / retained prediction components.",
        "pooled_retained_prediction_match_fraction",
        "dimensionless",
    ),
    (
        "component_recall",
        "Eligible matches / all unfiltered truth components.",
        "pooled_truth_match_fraction",
        "dimensionless",
    ),
    (
        "component_f1",
        "2 * eligible matches / (truth components + retained prediction components).",
        "pooled_component_f1",
        "dimensionless",
    ),
    (
        "split_truth_components",
        "Truth components intersecting more than one retained prediction component.",
        "truths_overlapping_multiple_retained_predictions",
        "component",
    ),
    (
        "merge_predicted_components",
        "Retained prediction components intersecting more than one truth component.",
        "retained_predictions_overlapping_multiple_truths",
        "component",
    ),
)
LOCALIZATION_DEFINITIONS = tuple(
    MetricDefinition(
        name=name,
        direction="MAXIMIZE"
        if name in ("component_precision", "component_recall", "component_f1")
        else "DESCRIPTIVE",
        formula=formula,
        population=(
            "Explicit-mask four-connected components; truth unfiltered, predictions area-gated."
        ),
        aggregation=aggregation,
        undefined_policy=(
            "Rates are unavailable at zero denominator; F1 is zero with supported errors."
        ),
        limitations=(
            "Components do not prove physical plume independence. Matching IoU and blob "
            "gates are diagnostic settings. Counts include misses and spurious retained "
            "components."
        ),
        unit=unit,
    )
    for name, formula, aggregation, unit in _LOCALIZATION_FORMULAS
) + tuple(
    MetricDefinition(
        name="matched_centroid_error_" + unit + "_" + reduction,
        direction="MINIMIZE",
        formula=(
            "Euclidean distance between matched component pixel-center centroids."
            if unit == "px"
            else "hypot(dx * lateral_GSD, dy * along_GSD) for eligible matched components."
        ),
        population="Eligible matched components with requested distance geometry only.",
        aggregation="matched_component_conditional_" + reduction,
        undefined_policy=(
            "Unavailable with no eligible matches or absent recorded GSD for metre distances."
        ),
        limitations=(
            "Conditional on matching; inspect miss-inclusive localization success curves "
            "separately. Ground geometry is a local-GSD approximation. Quantiles use "
            "linear interpolation."
        ),
        unit="pixel" if unit == "px" else "m",
    )
    for unit in ("px", "m")
    for reduction in ("mean", "median", "p95")
)
_BOUNDARY_FORMULAS = (
    (
        "boundary_precision",
        "Predicted interior-boundary pixels within inclusive configured tolerance of "
        "truth / predicted boundary pixels.",
        "pooled_predicted_boundary_hit_fraction",
        "dimensionless",
    ),
    (
        "boundary_recall",
        "Truth interior-boundary pixels within inclusive configured tolerance of "
        "prediction / truth boundary pixels.",
        "pooled_truth_boundary_hit_fraction",
        "dimensionless",
    ),
    (
        "boundary_f1",
        "Harmonic mean of pooled boundary precision and recall; zero for supported no-hit cases.",
        "harmonic_pooled_boundary_hit_rates",
        "dimensionless",
    ),
    (
        "boundary_missed_images",
        "Images with nonempty truth boundary and empty prediction boundary.",
        "exact_image_count",
        "image",
    ),
    (
        "boundary_spurious_images",
        "Images with empty truth boundary and nonempty prediction boundary.",
        "exact_image_count",
        "image",
    ),
    ("boundary_empty_images", "Images with both boundaries empty.", "exact_image_count", "image"),
)
BOUNDARY_DEFINITIONS = tuple(
    MetricDefinition(
        name=name,
        direction="MAXIMIZE"
        if name in ("boundary_precision", "boundary_recall", "boundary_f1")
        else "DESCRIPTIVE",
        formula=formula,
        population="All explicitly annotated images; one-pixel interior four-neighbour boundaries.",
        aggregation=aggregation,
        undefined_policy=(
            "Zero-denominator rates and both-empty F1 unavailable; supported no-hit rates/F1 zero."
        ),
        limitations=(
            "Outside-image pixels are background for erosion. Inclusive pixel tolerance "
            "is diagnostic; configured physical tolerance requires recorded GSD and "
            "overrides pixel tolerance."
        ),
        unit=unit,
    )
    for name, formula, aggregation, unit in _BOUNDARY_FORMULAS
) + tuple(
    MetricDefinition(
        name="boundary_" + statistic + "_" + unit + "_mean",
        direction="MINIMIZE",
        formula="Per-image mean of pooled bidirectional nearest-boundary distances."
        if statistic == "asd"
        else "Per-image linear 95th percentile of pooled bidirectional nearest-boundary distances.",
        population="Images with both boundaries nonempty and requested distance geometry.",
        aggregation="equal_image_mean_conditional_on_both_nonempty_boundaries",
        undefined_policy=(
            "Unavailable when either boundary is empty, or metre geometry lacks recorded GSD."
        ),
        limitations=(
            "Conditional distances do not include detection misses; missed/spurious/"
            "empty image counts are separate. Metres use anisotropic local GSD: "
            "x lateral, y along."
        ),
        unit="pixel" if unit == "px" else "m",
    )
    for statistic in ("asd", "hd95")
    for unit in ("px", "m")
)


def metric_definition(name: str) -> Result[MetricDefinition, str]:
    """Look up an explicit scientific definition, never infer metric direction."""
    for definition in (
        CLASSIFIER_DEFINITIONS
        + CALIBRATION_DEFINITIONS
        + SEGMENTATION_DEFINITIONS
        + LOCALIZATION_DEFINITIONS
        + BOUNDARY_DEFINITIONS
    ):
        if definition.name == name:
            return Ok(definition)
    return Err(f"unknown metric definition {name!r}")
