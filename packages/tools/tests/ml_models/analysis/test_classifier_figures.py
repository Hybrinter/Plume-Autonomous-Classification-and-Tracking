"""Independent frozen classifier coordinate, normalization and identity references."""

from dataclasses import replace

from flight.libs.types import Err, Ok
from tools.ml_models.analysis.capture import CaptureRow
from tools.ml_models.analysis.classifier_figures import classifier_figure_data
from tools.ml_models.analysis.config import CaptureConfig
from tools.ml_models.analysis.contracts import (
    CurveEvidence,
    MetricSupport,
    MetricValue,
    SampleKey,
    SplitEvidence,
)
from tools.ml_models.analysis.prediction_selections import prediction_gallery_data

_HASH = "a" * 64
_MANIFEST = "b" * 64


def _metric(name: str, value: float | None, n: int = 4) -> MetricValue:
    return MetricValue(
        name=name,
        value=value,
        status="AVAILABLE" if value is not None else "UNAVAILABLE",
        reason=None if value is not None else "No eligible denominator",
        support=MetricSupport(unit="IMAGE", n=n),
        threshold=0.3
        if name in ("precision", "recall", "accuracy", "false_positive_rate")
        else None,
    )


def _rows() -> tuple[CaptureRow, ...]:
    rows: list[CaptureRow] = []
    for index, (label, p, logit, loss, fp, fn) in enumerate(
        (
            (0, 0.1, -2.0, 0.1, False, False),
            (0, 0.4, -0.4, 0.5, True, False),
            (1, 0.2, -1.0, 1.6, False, True),
            (1, 0.8, 1.0, 0.2, False, False),
        )
    ):
        rows.append(
            CaptureRow(
                key=SampleKey(
                    dataset_hash=_HASH,
                    task="classifier",
                    split="val",
                    spatial_shard=(2, 3),
                    row_index=index,
                    tile_id=str(index),
                    element="I",
                ),
                group_id=str(index),
                bin_id="bin",
                label=float(label),
                gsd_m=(2.0, 3.0),
                metrics=tuple(
                    _metric(name, value, 1)
                    for name, value in (
                        ("probability", p),
                        ("logit", logit),
                        ("binary_cross_entropy", loss),
                    )
                ),
                failure_score=loss,
                false_positive=fp,
                false_negative=fn,
                dataset_manifest_hash=_MANIFEST,
            )
        )
    return tuple(rows)


def _evidence() -> SplitEvidence:
    support = MetricSupport(unit="IMAGE", n=4)
    curves = (
        CurveEvidence(
            name="precision_recall",
            x_name="recall",
            y_name="precision",
            x=(0.0, 0.5, 1.0),
            y=(1.0, 0.5, 0.5),
            x_unit="fraction",
            y_unit="fraction",
            support=support,
            thresholds=(None, 1.0, -1.0),
            notes=("captured complete ties",),
        ),
        CurveEvidence(
            name="calibration_reliability",
            x_name="mean_probability",
            y_name="truth",
            x=(0.1, 0.5, 0.8),
            y=(0.0, None, 1.0),
            x_unit="probability",
            y_unit="fraction",
            support=support,
            notes=("Empty bin midpoint; missing y",),
        ),
        CurveEvidence(
            name="threshold_precision",
            x_name="probability",
            y_name="precision",
            x=(0.0, 0.5, 1.0),
            y=(0.5, 1.0, None),
            x_unit="probability",
            y_unit="fraction",
            support=support,
        ),
    )
    return SplitEvidence(
        task="classifier",
        split="val",
        dataset_hash=_HASH,
        dataset_manifest_hash=_MANIFEST,
        checkpoint_hash="c" * 64,
        support=support,
        curves=curves,
        metrics=tuple(
            _metric(name, value)
            for name, value in (
                ("precision", 0.5),
                ("recall", 0.5),
                ("accuracy", 0.5),
                ("false_positive_rate", 0.5),
            )
        ),
    )


def test_ranking_curve_is_verbatim_with_pre_step_and_captured_operating_point() -> None:
    result = classifier_figure_data(_evidence(), _rows())
    assert isinstance(result, Ok)
    pr = next(figure for figure in result.value if figure.identifier == "precision_recall")
    assert pr.series[0].x == (0.0, 0.5, 1.0)
    assert pr.series[0].y == (1.0, 0.5, 0.5)
    assert pr.series[0].style == "PRE"
    assert pr.series[1].y == (0.5, 0.5)
    assert [(point.x, point.y) for point in pr.points] == [(0.5, 0.5)]
    assert pr.identity.checkpoint_hash == "c" * 64
    assert pr.notes == ("captured complete ties",)


def test_confusion_denominators_and_zero_support_classes_are_not_fabricated() -> None:
    rows = tuple(row for row in _rows() if row.label == 0)
    evidence = replace(
        _evidence(),
        support=MetricSupport(unit="IMAGE", n=2),
        metrics=(_metric("precision", 0.0, 2), _metric("recall", None, 2)),
    )
    result = classifier_figure_data(evidence, rows)
    assert isinstance(result, Ok)
    figures = {figure.identifier: figure for figure in result.value}
    assert figures["confusion_counts"].matrix == ((1.0, 1.0), (0.0, 0.0))
    assert figures["confusion_truth_normalized"].matrix == ((0.5, 0.5), (None, None))
    assert figures["confusion_prediction_normalized"].matrix == ((1.0, 1.0), (0.0, 0.0))
    empty_predictions = tuple(replace(row, false_positive=False) for row in rows)
    result = classifier_figure_data(replace(evidence, metrics=()), empty_predictions)
    assert isinstance(result, Ok)
    figures = {figure.identifier: figure for figure in result.value}
    assert figures["confusion_prediction_normalized"].matrix == ((1.0, None), (0.0, None))


def test_distribution_ties_and_empty_class_keep_support() -> None:
    rows = _rows()
    rows = (rows[0], replace(rows[1], metrics=rows[0].metrics, false_positive=False))
    evidence = replace(_evidence(), support=MetricSupport(unit="IMAGE", n=2), metrics=())
    result = classifier_figure_data(evidence, rows)
    assert isinstance(result, Ok)
    figure = next(
        figure for figure in result.value if figure.identifier == "probability_distribution"
    )
    assert figure.series[0].x == (0.1,) and figure.series[0].y == (1.0,)
    assert figure.series[0].support.n == 2
    assert figure.series[1].x == () and figure.series[1].support.n == 0


def test_confidence_uses_actual_predicted_class_at_non_half_threshold() -> None:
    result = classifier_figure_data(_evidence(), _rows())
    assert isinstance(result, Ok)
    figure = next(figure for figure in result.value if figure.identifier == "prediction_confidence")
    assert figure.series[0].x == (0.8, 0.9)
    assert figure.series[1].x == (0.4, 0.8)
    assert figure.series[1].y == (0.5, 1.0)


def test_calibration_null_bins_and_threshold_operating_point_are_not_interpolated() -> None:
    result = classifier_figure_data(_evidence(), _rows())
    assert isinstance(result, Ok)
    figures = {figure.identifier: figure for figure in result.value}
    assert figures["calibration_reliability"].series[0].y == (0.0, None, 1.0)
    assert figures["threshold_precision"].series[0].y == (0.5, 1.0, None)
    assert [(point.x, point.y) for point in figures["threshold_precision"].points] == [(0.3, 0.5)]
    assert figures["roc"].reason
    assert figures["roc"].series == ()


def test_scalar_capture_mismatch_and_duplicate_variants_fail_closed() -> None:
    evidence, rows = _evidence(), _rows()
    assert isinstance(
        classifier_figure_data(replace(evidence, dataset_manifest_hash="d" * 64), rows), Err
    )
    assert isinstance(classifier_figure_data(evidence, rows[:-1]), Err)
    duplicate = replace(rows[1], key=replace(rows[0].key, row_index=99, element="R90"))
    assert isinstance(classifier_figure_data(evidence, (rows[0], duplicate, *rows[2:])), Err)
    assert isinstance(classifier_figure_data(replace(evidence, task="segmentor"), rows), Err)


def test_missing_scalars_or_conflicting_error_flags_fail_closed() -> None:
    evidence, rows = _evidence(), _rows()
    missing = replace(
        rows[0], metrics=tuple(metric for metric in rows[0].metrics if metric.name != "logit")
    )
    assert isinstance(classifier_figure_data(evidence, (missing, *rows[1:])), Err)
    conflict = replace(rows[0], false_negative=True)
    assert isinstance(classifier_figure_data(evidence, (conflict, *rows[1:])), Err)
    assert isinstance(
        classifier_figure_data(
            replace(evidence, metrics=(_metric("accuracy", 1.0),)),
            rows,
        ),
        Err,
    )


def test_error_galleries_select_globally_and_are_independent_of_input_order() -> None:
    cfg = CaptureConfig(max_preview_images=4, examples_per_family=1, seed=11)
    forward = prediction_gallery_data(_evidence(), _rows(), cfg)
    reverse = prediction_gallery_data(_evidence(), tuple(reversed(_rows())), cfg)
    assert isinstance(forward, Ok) and isinstance(reverse, Ok)
    assert forward == reverse
    galleries = {gallery.family: gallery for gallery in forward.value}
    assert galleries["false_positive"].rows[0].key.tile_id == "1"
    assert galleries["false_negative"].rows[0].key.tile_id == "2"
    assert galleries["high_confidence_error"].rows[0].key.tile_id == "2"
    assert galleries["high_confidence_error"].rows[0].failure_score == 1.6
    assert len(galleries["representative"].rows) == 1
    altered = replace(_rows()[2], failure_score=0.0)
    assert isinstance(
        prediction_gallery_data(_evidence(), (*_rows()[:2], altered, _rows()[3]), cfg), Err
    )


def test_disabled_or_empty_error_galleries_keep_explicit_index_states() -> None:
    cfg = CaptureConfig(max_preview_images=0, examples_per_family=0)
    result = prediction_gallery_data(_evidence(), _rows(), cfg)
    assert isinstance(result, Ok)
    assert all(
        gallery.availability.status == "SKIPPED" and gallery.availability.reason
        for gallery in result.value
    )
    cfg = replace(cfg, max_preview_images=4, examples_per_family=1)
    rows = tuple(replace(row, false_positive=False, false_negative=False) for row in _rows())
    result = prediction_gallery_data(_evidence(), rows, cfg)
    assert isinstance(result, Ok)
    assert all(
        gallery.availability.status == "UNAVAILABLE" and gallery.availability.reason
        for gallery in result.value
        if gallery.family != "representative"
    )


def test_captured_curve_with_no_eligible_values_is_explicitly_unavailable() -> None:
    rows = _rows()[:2]
    curve = CurveEvidence(
        name="threshold_recall",
        x_name="probability_threshold",
        y_name="recall",
        x=(0.1, 0.9),
        y=(None, None),
        x_unit="probability",
        y_unit="fraction",
        support=MetricSupport(unit="IMAGE", n=2),
    )
    evidence = replace(
        _evidence(),
        support=MetricSupport(unit="IMAGE", n=2),
        metrics=(),
        curves=(curve,),
    )
    result = classifier_figure_data(evidence, rows)
    assert isinstance(result, Ok)
    frozen = next(figure for figure in result.value if figure.identifier == "threshold_recall")
    assert frozen.reason == "No eligible captured values for this curve"
    assert frozen.series[0].x == (0.1, 0.9)
    assert frozen.series[0].y == (None, None)
