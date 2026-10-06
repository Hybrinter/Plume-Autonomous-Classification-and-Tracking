"""Frozen history-chart coordinate, selection and exposure references."""

from dataclasses import replace

from flight.libs.types import Err, Ok
from tools.ml_models.analysis.contracts import (
    CheckpointIdentity,
    MetricSupport,
    MetricValue,
    Split,
    SplitEvidence,
)
from tools.ml_models.analysis.training import (
    CheckpointRecord,
    EvaluationRecord,
    LossObservation,
    StepRecord,
    TrainingExecution,
    TrainingHistory,
    reduce_epoch,
)
from tools.ml_models.analysis.training_figures import training_figure_data

_HASH = "a" * 64


def _evidence(split: Split, checkpoint: str, loss: float, ap: float) -> SplitEvidence:
    return SplitEvidence(
        task="classifier",
        split=split,
        dataset_hash=_HASH,
        checkpoint_hash=checkpoint,
        metrics=tuple(
            MetricValue(
                name=name,
                value=value,
                status="AVAILABLE",
                aggregation=aggregation,
                support=MetricSupport(unit="IMAGE", n=2),
            )
            for name, value, aggregation in (
                ("objective_loss", loss, "equal_image_mean"),
                ("average_precision", ap, "exact_score_ties"),
            )
        ),
    )


def _history() -> TrainingHistory:
    first = StepRecord(
        epoch=1,
        batch=1,
        optimizer_step=1,
        samples_seen=2,
        n_samples=2,
        loss=LossObservation(total=3.0, bce=1.0, dice=2.0),
        learning_rates=(0.01,),
        duration_s=2.0,
        throughput_images_s=1.0,
        gradient_norm=2.0,
    )
    zero = replace(
        first,
        batch=2,
        optimizer_step=2,
        samples_seen=3,
        n_samples=1,
        loss=LossObservation(total=0.0, bce=0.0, dice=0.0),
        duration_s=1.0,
        throughput_images_s=1.0,
        gradient_norm=0.0,
    )
    last_step = replace(
        first,
        epoch=2,
        batch=3,
        optimizer_step=3,
        samples_seen=5,
        loss=LossObservation(total=1.0, bce=0.5, dice=0.5),
    )
    epoch_a, epoch_b = reduce_epoch((first, zero)), reduce_epoch((last_step,))
    assert isinstance(epoch_a, Ok) and isinstance(epoch_b, Ok)
    evaluations = []
    for epoch, step, seen, digest, loss, ap in (
        (1, 2, 3, "1" * 64, 2.0, 0.8),
        (2, 3, 5, "2" * 64, 2.5, 0.7),
    ):
        validation = _evidence("val", digest, loss, ap)
        checkpoint = CheckpointRecord(
            path="checkpoints/last.pt",
            identity=CheckpointIdentity(
                sha256=digest,
                kind="classifier",
                arch="tiny",
                training_dataset_hash=_HASH,
                epoch=epoch,
                step=step,
            ),
            metric="average_precision",
            direction="MAXIMIZE",
            value=ap,
            validation=validation,
        )
        evaluations.append(
            EvaluationRecord(
                epoch=epoch,
                optimizer_step=step,
                samples_seen=seen,
                train=_evidence("train", digest, loss - 0.5, ap + 0.1),
                validation=validation,
                checkpoint=checkpoint,
                improved=epoch == 1,
                duration_s=3.0,
            )
        )
    best = replace(evaluations[0].checkpoint, path="checkpoints/best.pt")
    execution = TrainingExecution(
        kind="classifier",
        arch="tiny",
        dataset_hash=_HASH,
        conditioning="film-log-gsd-v1",
        config={},
        provenance={},
        metric="average_precision",
        status="COMPLETED",
        epoch=2,
        optimizer_step=3,
        samples_seen=5,
        stop_reason="max_epochs",
        best=best,
        last=evaluations[1].checkpoint,
        selected=best,
    )
    return TrainingHistory(
        execution,
        (first, zero, epoch_a.value, evaluations[0], last_step, epoch_b.value, evaluations[1]),
        (),
    )


def test_optimization_epochs_and_canonical_evaluations_have_distinct_frozen_series() -> None:
    history = _history()
    measured = training_figure_data(history)
    assert isinstance(measured, Ok)
    figures = {figure.identifier: figure for figure in measured.value}
    attempts = figures["optimization_loss_loglog"]
    assert attempts.x_scale == attempts.y_scale == "log"
    assert attempts.series[0].x == (1.0, 2.0, 3.0)
    assert attempts.series[0].y == (3.0, 0.0, 1.0)
    assert attempts.series[0].n_exposures == 5
    assert figures["optimization_loss_semilog"].x_scale == "linear"
    assert figures["optimization_loss_semilog"].y_scale == "log"
    assert figures["optimization_epoch_total"].series[0].y == (2.0, 1.0)
    canonical = figures["evaluation_loss_linear"]
    by_name = {series.name: series for series in canonical.series}
    assert by_name["canonical train"].x == by_name["validation"].x == (2.0, 3.0)
    assert by_name["canonical train"].y == (1.5, 2.0)
    assert by_name["validation"].y == (2.0, 2.5)
    assert {marker.name: marker.x for marker in canonical.markers} == {
        "best checkpoint": 2.0,
        "last evaluated checkpoint": 3.0,
        "stopping state": 3.0,
    }


def test_final_test_is_one_selected_checkpoint_point_and_never_a_history_curve() -> None:
    history = _history()
    assert history.execution.selected is not None
    selected = history.execution.selected.identity
    test = _evidence("test", selected.sha256, 3.0, 0.6)
    result = training_figure_data(history, final_test=test, checkpoint=selected)
    assert isinstance(result, Ok)
    figures = {figure.identifier: figure for figure in result.value}
    series = next(
        s for s in figures["evaluation_loss_linear"].series if s.name == "selected-checkpoint test"
    )
    assert series.x == (2.0,) and series.y == (3.0,) and series.points_only
    assert not any(
        s.name == "selected-checkpoint test" for s in figures["optimization_loss_linear"].series
    )
    assert isinstance(
        training_figure_data(
            history,
            final_test=replace(test, checkpoint_hash="b" * 64),
            checkpoint=selected,
        ),
        Err,
    )


def test_partial_history_and_missing_telemetry_do_not_invent_completed_evaluation() -> None:
    history = _history()
    execution = replace(
        history.execution,
        status="INTERRUPTED",
        stop_reason="user_interrupt",
        epoch=1,
        optimizer_step=1,
        samples_seen=2,
        best=None,
        last=None,
        selected=None,
    )
    partial = TrainingHistory(execution, history.records[:1], ("Incomplete prefix",))
    result = training_figure_data(partial)
    assert isinstance(result, Ok)
    figures = {figure.identifier: figure for figure in result.value}
    assert figures["evaluation_loss_linear"].reason
    assert figures["amp_scale"].reason
    assert figures["optimization_component_focal"].reason
    assert figures["optimization_loss_linear"].run_status == "INTERRUPTED"


def test_metric_gap_coordinates_use_comparable_captured_values_only() -> None:
    result = training_figure_data(_history())
    assert isinstance(result, Ok)
    figures = {figure.identifier: figure for figure in result.value}
    gaps = figures["development_gap_average_precision"]
    assert gaps.series[0].x == (2.0, 3.0)
    assert all(abs(value - 0.1) < 1e-12 for value in gaps.series[0].y if value is not None)
    rates = figures["learning_rates"].series[0]
    assert rates.x == (1.0, 2.0, 3.0) and rates.y == (0.01, 0.01, 0.01)


def test_final_test_cannot_claim_an_unrecorded_checkpoint_or_loss_definition() -> None:
    history = _history()
    assert history.execution.selected is not None
    selected = history.execution.selected.identity
    unknown = replace(selected, sha256="b" * 64, step=999)
    assert isinstance(
        training_figure_data(
            history,
            final_test=_evidence("test", unknown.sha256, 3.0, 0.6),
            checkpoint=unknown,
        ),
        Err,
    )
    test = _evidence("test", selected.sha256, 3.0, 0.6)
    incompatible = replace(
        test,
        metrics=tuple(
            replace(metric, unit="different_loss_unit")
            if metric.name == "objective_loss"
            else metric
            for metric in test.metrics
        ),
    )
    assert isinstance(
        training_figure_data(history, final_test=incompatible, checkpoint=selected), Err
    )


def test_amp_skipped_exposure_keeps_duplicate_or_zero_update_positions() -> None:
    history = _history()
    first = history.records[0]
    second = history.records[1]
    assert isinstance(first, StepRecord) and isinstance(second, StepRecord)
    skipped = replace(
        first,
        optimizer_step=0,
        update_applied=False,
        amp_scale_before=128.0,
        amp_scale_after=64.0,
    )
    applied = replace(
        second,
        optimizer_step=1,
        amp_scale_before=64.0,
        amp_scale_after=64.0,
    )
    execution = replace(
        history.execution,
        status="RUNNING",
        epoch=0,
        optimizer_step=0,
        samples_seen=0,
        stop_reason=None,
        best=None,
        last=None,
        selected=None,
    )
    result = training_figure_data(TrainingHistory(execution, (skipped, applied), ()))
    assert isinstance(result, Ok)
    figures = {figure.identifier: figure for figure in result.value}
    assert figures["optimization_loss_loglog"].series[0].x == (0.0, 1.0)
    assert figures["optimization_loss_loglog"].series[0].y == (3.0, 0.0)
    assert figures["optimization_loss_linear"].series[0].n_exposures == 3
    assert figures["optimizer_update_applied"].series[0].y == (0.0, 1.0)
    assert figures["amp_scale"].series[0].y == (128.0, 64.0)
    assert figures["amp_scale"].series[1].y == (64.0, 64.0)
    assert figures["optimization_loss_linear"].markers[0].name == "last captured running state"
    assert figures["optimization_loss_linear"].markers[0].x == 1.0


def test_batch_telemetry_support_counts_events_not_image_exposures() -> None:
    result = training_figure_data(_history())
    assert isinstance(result, Ok)
    figures = {figure.identifier: figure for figure in result.value}
    for name in (
        "batch_duration",
        "batch_throughput",
        "gradient_norm",
        "optimizer_update_applied",
    ):
        series = figures[name].series[0]
        assert series.n_exposures == 3
        assert series.exposure_unit == "BATCH"
    assert figures["optimization_loss_linear"].series[0].n_exposures == 5
