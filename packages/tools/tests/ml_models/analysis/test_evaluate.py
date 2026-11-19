"""Cohort traversal and canonical input oracles for exhaustive evaluation."""

import hashlib
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from flight.libs.types import Err, Ok
from tools.ml_models.analysis.capture import CaptureRow
from tools.ml_models.analysis.config import CaptureConfig, EvaluationConfig, ScoreConfig
from tools.ml_models.analysis.contracts import ArtifactRef
from tools.ml_models.analysis.evaluate import evaluate_split
from tools.ml_models.analysis.metrics.spatial import SpatialRow, aggregate_spatial
from tools.ml_models.dataset.augment import ELEMENT_NAMES, apply_dihedral
from tools.ml_models.dataset.manifest import (
    ShardCount,
    compute_dataset_hash,
    load_manifest,
    write_manifest,
)
from tools.ml_models.dataset.raw import ObservationMetadata
from tools.ml_models.dataset.store import (
    RowRecord,
    ShardWriter,
    read_images,
    read_labels,
    read_masks,
    read_rows,
)
from tools.ml_models.train.losses import build_loss
from torch import nn


class MarkerModel(nn.Module):
    """Observe every canonical input and emit deterministic aligned logits."""

    def __init__(self, kind: str = "classifier", invalid: bool = False) -> None:
        super().__init__()
        self.kind = kind
        self.invalid = invalid
        self.seen: list[np.ndarray] = []

    def forward(self, image: torch.Tensor, gsd: torch.Tensor) -> torch.Tensor:
        assert not torch.is_grad_enabled()
        self.seen.extend(image.detach().cpu().numpy().copy())
        logits = (image[:, :1] - 0.5) * 4
        if self.invalid:
            logits = logits * float("nan")
        return logits.mean(dim=(2, 3)) if self.kind == "classifier" else logits


def test_each_validation_row_once_and_batch_invariance(
    tmp_path: Path,
    build_synthetic_dataset: Callable[..., Path],
) -> None:
    """Each original validation image contributes exactly once for any batch size."""
    dataset = build_synthetic_dataset(tmp_path / "ds", n=9)
    manifest = load_manifest(dataset / "dataset.json")
    expected = tuple(
        image
        for shard in manifest.shards
        if shard.task == "classifier" and shard.split == "val"
        for image in read_images(dataset / "classifier" / "val" / f"{shard.height}x{shard.width}")
    )
    first, second = MarkerModel(), MarkerModel()
    small = evaluate_split(
        first, dataset, manifest, EvaluationConfig(kind="classifier", split="val", batch_size=1)
    )
    large = evaluate_split(
        second, dataset, manifest, EvaluationConfig(kind="classifier", split="val", batch_size=7)
    )
    assert isinstance(small, Ok) and isinstance(large, Ok)
    assert first.training and second.training
    assert len(first.seen) == len(second.seen) == len(expected) == small.value.support.n
    for observed, original in zip(first.seen, expected, strict=True):
        np.testing.assert_array_equal(observed, original)
    for observed, original in zip(second.seen, expected, strict=True):
        np.testing.assert_array_equal(observed, original)
    for a, b in zip(small.value.metrics, large.value.metrics, strict=True):
        assert a.name == b.name
        assert a.value == pytest.approx(b.value) if b.value is not None else a.value is None
    assert sum(stratum.support.n for stratum in small.value.strata) == len(expected)


def test_canonical_train_does_not_weight_augmentation_copies(
    tmp_path: Path,
    build_synthetic_dataset: Callable[..., Path],
) -> None:
    """Train evaluation chooses one identity view per source tile/GSD variant."""
    dataset = build_synthetic_dataset(tmp_path / "ds", n=9)
    manifest = load_manifest(dataset / "dataset.json")
    shard = next(
        shard for shard in manifest.shards if shard.task == "classifier" and shard.split == "train"
    )
    directory = dataset / "classifier" / "train" / f"{shard.height}x{shard.width}"
    rows, images = read_rows(directory), read_images(directory)
    expected = [image for row, image in zip(rows, images, strict=True) if row.element == "id"]
    model = MarkerModel()
    result = evaluate_split(
        model, dataset, manifest, EvaluationConfig(kind="classifier", split="train")
    )
    assert isinstance(result, Ok)
    assert result.value.support.n == len(expected) < len(rows)
    for observed, original in zip(model.seen, expected, strict=True):
        np.testing.assert_array_equal(observed, original)


def test_segmentation_objective_and_no_fabricated_verified_negatives(
    tmp_path: Path,
    build_synthetic_dataset: Callable[..., Path],
) -> None:
    """Explicit scalar objective evidence is separate from unweighted BCE."""
    dataset = build_synthetic_dataset(tmp_path / "ds", n=9)
    manifest = load_manifest(dataset / "dataset.json")
    result = evaluate_split(
        MarkerModel("segmentor"),
        dataset,
        manifest,
        EvaluationConfig(kind="segmentor", split="val", score=ScoreConfig(pixel_histogram_bins=8)),
        objective=build_loss("bce_dice", pos_weight=2),
    )
    assert isinstance(result, Ok)
    metrics = {metric.name: metric for metric in result.value.metrics}
    assert metrics["objective_loss"].value is not None
    assert metrics["objective_bce"].value is not None
    assert metrics["objective_dice"].value is not None
    assert metrics["objective_focal"].value is None
    explicit_empty = 0
    for shard in manifest.shards:
        if shard.task != "segmentor" or shard.split != "val":
            continue
        directory = dataset / "segmentor" / "val" / f"{shard.height}x{shard.width}"
        masks = read_masks(directory)
        assert masks is not None
        explicit_empty += sum(
            float(label[0]) == 0.0 and not bool(mask.any())
            for label, mask in zip(read_labels(directory), masks, strict=True)
        )
    assert metrics["verified_negative_any_blob_rate"].support.n == explicit_empty


def test_invalid_output_restores_exact_model_modes(
    tmp_path: Path,
    build_synthetic_dataset: Callable[..., Path],
) -> None:
    """Evaluation failures cannot leave the caller's modules in evaluation mode."""
    dataset = build_synthetic_dataset(tmp_path / "ds", n=9)
    manifest = load_manifest(dataset / "dataset.json")
    model = MarkerModel(invalid=True)
    model.add_module("child", nn.Dropout())
    model.train()
    model.get_submodule("child").eval()
    result = evaluate_split(
        model, dataset, manifest, EvaluationConfig(kind="classifier", split="val")
    )
    assert isinstance(result, Err)
    assert model.training and not model.get_submodule("child").training


@pytest.mark.parametrize("element", ELEMENT_NAMES)
def test_inverse_canonical_view(element: str) -> None:
    """Every legal square transform round-trips independently authored inverse choices."""
    from tools.ml_models.analysis.evaluate import canonical_array

    image = np.arange(2 * 5 * 5, dtype=np.float32).reshape(2, 5, 5)
    transformed = apply_dihedral(image, element)
    np.testing.assert_array_equal(canonical_array(transformed, element), image)


def _small_dataset(
    tmp_path: Path,
    builder: Callable[..., Path],
    *,
    include_empty_mask: bool = True,
    split: str = "test",
    elements: tuple[str, ...] = ("id",),
    metadata: ObservationMetadata | None = None,
    gsd_nominal: bool = False,
    schema_version: int | None = None,
) -> Path:
    template = load_manifest(builder(tmp_path / "template", n=3) / "dataset.json")
    root = tmp_path / "small"
    counts: list[ShardCount] = []
    for kind in ("classifier", "segmentor"):
        for height, width, labels in ((2, 3, (1.0, 0.0)), (3, 4, (1.0,))):
            selected = tuple(
                label for label in labels if kind == "classifier" or include_empty_mask or label
            )
            directory = root / kind / split / f"{height}x{width}"
            writer = ShardWriter(
                directory,
                len(selected) * len(elements),
                height,
                width,
                channels=3,
                with_masks=kind == "segmentor",
            )
            for index, label in enumerate(selected):
                mask = np.zeros((1, height, width), dtype=np.uint8)
                mask[0, 0, : 1 if height == 2 else 2] = int(label)
                image = np.repeat(mask.astype(np.float32), 3, axis=0)
                for element in elements:
                    writer.append(
                        apply_dihedral(image, element),
                        np.array([16.0, 32.0], dtype=np.float32),
                        label,
                        apply_dihedral(mask, element) if kind == "segmentor" else None,
                        RowRecord(
                            tile_id=f"{height}-{index}",
                            group_id=f"group-{index}",
                            frame_id=None,
                            grid_rc=None,
                            bin_id="near" if height == 2 else "far",
                            element=element,
                            gsd_nominal=gsd_nominal,
                            metadata=(metadata if metadata is not None else ObservationMetadata()),
                        ),
                    )
            writer.close()
            counts.append(
                ShardCount(
                    task=kind,
                    split=split,
                    height=height,
                    width=width,
                    n=len(selected) * len(elements),
                    n_positive=int(sum(selected)) * len(elements),
                )
            )
    manifest = replace(template, shards=tuple(counts), dataset_hash=compute_dataset_hash(root))
    if schema_version is not None:
        manifest = replace(manifest, schema_version=schema_version)
    write_manifest(root / "dataset.json", manifest)
    return root


def test_train_fallback_inverts_one_deterministic_view(
    tmp_path: Path, build_synthetic_dataset: Callable[..., Path]
) -> None:
    """Missing identity rows are inverted, rather than omitted or augmentation-weighted."""
    dataset = _small_dataset(
        tmp_path, build_synthetic_dataset, split="train", elements=("flip_v", "rot180")
    )
    model = MarkerModel("segmentor")
    sink = _FailingCloseSink()
    result = evaluate_split(
        model,
        dataset,
        load_manifest(dataset / "dataset.json"),
        EvaluationConfig(
            kind="segmentor", split="train", score=ScoreConfig(pixel_histogram_bins=8)
        ),
        capture=sink,
    )
    assert isinstance(result, Err)
    assert "flush failure" in result.error
    assert len(model.seen) == len(sink.rows) == 3
    assert all(row.key.element == "rot180" for row in sink.rows)
    assert [row.key.row_index for row in sink.rows] == [1, 3, 1]
    for image, row in zip(model.seen, sink.rows, strict=True):
        expected = np.zeros(image.shape, dtype=np.float32)
        expected[:, 0, : 1 if image.shape[1] == 2 else 2] = row.label
        np.testing.assert_array_equal(image, expected)


def test_variable_shape_segmentation_loss_oracle(
    tmp_path: Path, build_synthetic_dataset: Callable[..., Path]
) -> None:
    """Equal-image configured loss is not a batch mean or a pooled-pixel mean."""
    dataset = _small_dataset(tmp_path, build_synthetic_dataset)
    manifest = load_manifest(dataset / "dataset.json")
    results = [
        evaluate_split(
            MarkerModel("segmentor"),
            dataset,
            manifest,
            EvaluationConfig(
                kind="segmentor",
                split="test",
                batch_size=batch_size,
                score=ScoreConfig(pixel_histogram_bins=8),
            ),
            objective=build_loss("bce_dice", pos_weight=2.0),
        )
        for batch_size in (1, 5)
    ]
    for result in results:
        assert isinstance(result, Ok), result
        metrics = {metric.name: metric for metric in result.value.metrics}
        assert metrics["foreground_iou_mean_positive_images"].value == 1.0
        assert metrics["foreground_iou_mean_positive_images"].support.n == 2
        assert metrics["verified_negative_any_foreground_rate"].support.n == 1
        assert metrics["verified_negative_any_foreground_rate"].value == 0.0
        assert result.value.support.n == 3
        expected_bce = float(np.logaddexp(0.0, -2.0))
        probability = 1.0 / (1.0 + float(np.exp(-2.0)))
        dice_losses = [
            1.0
            - (2 * area * probability + 1)
            / (area + area * probability + (pixels - area) * (1 - probability) + 1)
            for area, pixels in ((1, 6), (0, 6), (2, 12))
        ]
        objective_bce = expected_bce * (1 + (1 / 6 + 0 + 2 / 12) / 3)
        objective_dice = sum(dice_losses) / 3
        assert metrics["binary_cross_entropy_mean_images"].value == pytest.approx(expected_bce)
        assert metrics["objective_bce"].value == pytest.approx(objective_bce)
        assert metrics["objective_dice"].value == pytest.approx(objective_dice)
        assert metrics["objective_loss"].value == pytest.approx(objective_bce + objective_dice)
        assert sum(stratum.support.n for stratum in result.value.strata) == 3


def test_missing_mask_is_not_a_verified_negative(
    tmp_path: Path, build_synthetic_dataset: Callable[..., Path]
) -> None:
    """The classifier-only negative is excluded, never converted to empty ground truth."""
    dataset = _small_dataset(tmp_path, build_synthetic_dataset, include_empty_mask=False)
    result = evaluate_split(
        MarkerModel("segmentor"),
        dataset,
        load_manifest(dataset / "dataset.json"),
        EvaluationConfig(kind="segmentor", split="test", score=ScoreConfig(pixel_histogram_bins=8)),
    )
    assert isinstance(result, Ok)
    metrics = {metric.name: metric for metric in result.value.metrics}
    assert result.value.support.n == 2
    assert metrics["verified_negative_any_foreground_rate"].value is None
    assert metrics["verified_negative_any_foreground_rate"].support.n == 0


class _FailingCloseSink:
    def __init__(self) -> None:
        self.rows: list[CaptureRow] = []
        self.closed = 0
        self.aborted = False

    def add(
        self, row: CaptureRow, *, image: np.ndarray, target: np.ndarray, logits: np.ndarray
    ) -> Ok[None]:
        self.rows.append(row)
        return Ok(None)

    def abort(self, reason: str) -> Ok[None]:
        self.aborted = True
        return Ok(None)

    def close(self) -> Err[str]:
        self.closed += 1
        return Err("flush failure")

    def references(self) -> tuple[ArtifactRef, ...]:
        return ()


def test_capture_close_errors_and_original_row_identity(
    tmp_path: Path, build_synthetic_dataset: Callable[..., Path]
) -> None:
    """Close failures propagate after preserving shard/row/group/bin alignment."""
    dataset = _small_dataset(tmp_path, build_synthetic_dataset)
    sink = _FailingCloseSink()
    result = evaluate_split(
        MarkerModel(),
        dataset,
        load_manifest(dataset / "dataset.json"),
        EvaluationConfig(kind="classifier", split="test"),
        capture=sink,
    )
    assert isinstance(result, Err)
    assert "flush failure" in result.error
    assert sink.closed == 1 and not sink.aborted
    assert [
        (row.key.spatial_shard, row.key.row_index, row.bin_id, row.group_id) for row in sink.rows
    ] == [
        ((2, 3), 0, "near", "group-0"),
        ((2, 3), 1, "near", "group-1"),
        ((3, 4), 0, "far", "group-0"),
    ]


def test_failed_inference_aborts_capture(
    tmp_path: Path, build_synthetic_dataset: Callable[..., Path]
) -> None:
    """A failed evaluator cannot publish a partial capture as complete."""
    dataset = _small_dataset(tmp_path, build_synthetic_dataset)
    sink = _FailingCloseSink()
    result = evaluate_split(
        MarkerModel(invalid=True),
        dataset,
        load_manifest(dataset / "dataset.json"),
        EvaluationConfig(kind="classifier", split="test"),
        capture=sink,
    )
    assert isinstance(result, Err)
    assert sink.closed == 1 and sink.aborted


def test_stale_supplied_manifest_fails_before_inference(
    tmp_path: Path, build_synthetic_dataset: Callable[..., Path]
) -> None:
    """Even a valid content hash cannot bind evidence to a different manifest."""
    dataset = _small_dataset(tmp_path, build_synthetic_dataset)
    manifest = replace(load_manifest(dataset / "dataset.json"), source="different")
    model = MarkerModel()
    result = evaluate_split(
        model, dataset, manifest, EvaluationConfig(kind="classifier", split="test")
    )
    assert isinstance(result, Err)
    assert "identity" in result.error
    assert not model.seen


def test_compact_capture_integration(
    tmp_path: Path, build_synthetic_dataset: Callable[..., Path]
) -> None:
    """Evaluation publishes compact row artifacts without default full predictions."""
    from tools.ml_models.analysis.capture import BoundedCaptureSink

    dataset = _small_dataset(tmp_path, build_synthetic_dataset)
    sink = BoundedCaptureSink(
        tmp_path / "capture",
        CaptureConfig(max_preview_images=2, examples_per_family=1),
        dataset=dataset,
    )
    result = evaluate_split(
        MarkerModel("segmentor"),
        dataset,
        load_manifest(dataset / "dataset.json"),
        EvaluationConfig(kind="segmentor", split="test", score=ScoreConfig(pixel_histogram_bins=8)),
        capture=sink,
    )
    assert isinstance(result, Ok), result
    assert result.value.artifacts
    assert not (tmp_path / "capture" / ".incomplete").exists()
    assert not (tmp_path / "capture" / "full").exists()


class _CollectSink:
    """Pass-through CaptureSink that keeps every row for inspection."""

    def __init__(self) -> None:
        self.rows: list[CaptureRow] = []
        self.closed = 0
        self.aborted = False

    def add(
        self, row: CaptureRow, *, image: np.ndarray, target: np.ndarray, logits: np.ndarray
    ) -> Ok[None]:
        self.rows.append(row)
        return Ok(None)

    def abort(self, reason: str) -> Ok[None]:
        self.aborted = True
        return Ok(None)

    def close(self) -> Ok[None]:
        self.closed += 1
        return Ok(None)

    def references(self) -> tuple[ArtifactRef, ...]:
        return ()


def _spatial_rows(sink: _CollectSink) -> tuple[SpatialRow, ...]:
    rows = tuple(row.spatial for row in sink.rows)
    assert rows and all(row is not None for row in rows)
    return cast("tuple[SpatialRow, ...]", rows)


def test_segmentor_spatial_metrics_equal_frozen_capture_aggregation(
    tmp_path: Path, build_synthetic_dataset: Callable[..., Path]
) -> None:
    """Split evidence appends the exact spatial aggregate of captured rows."""
    dataset = _small_dataset(tmp_path, build_synthetic_dataset)
    sink = _CollectSink()
    result = evaluate_split(
        MarkerModel("segmentor"),
        dataset,
        load_manifest(dataset / "dataset.json"),
        EvaluationConfig(kind="segmentor", split="test", score=ScoreConfig(pixel_histogram_bins=8)),
        capture=sink,
    )
    assert isinstance(result, Ok), result
    metrics = {metric.name: metric for metric in result.value.metrics}
    for name in (
        "truth_components",
        "component_recall",
        "matched_centroid_error_px_mean",
        "boundary_precision",
        "boundary_f1",
    ):
        assert name in metrics
    assert "localization_success_px" in {curve.name for curve in result.value.curves}
    aggregate = aggregate_spatial(_spatial_rows(sink))
    assert isinstance(aggregate, Ok)
    expected = {metric.name: metric.value for metric in aggregate.value.metrics}
    for name, value in expected.items():
        assert metrics[name].value == value
    expected_curves = {curve.name: curve for curve in aggregate.value.curves}
    for curve in result.value.curves:
        if curve.name in expected_curves:
            assert curve == expected_curves[curve.name]


def test_spatial_metrics_batch_invariance_and_row_identity(
    tmp_path: Path, build_synthetic_dataset: Callable[..., Path]
) -> None:
    """Batch size changes neither spatial metrics nor per-row spatial records."""
    dataset = _small_dataset(tmp_path, build_synthetic_dataset)
    manifest = load_manifest(dataset / "dataset.json")
    metrics, by_key = [], []
    for batch_size in (1, 7):
        sink = _CollectSink()
        result = evaluate_split(
            MarkerModel("segmentor"),
            dataset,
            manifest,
            EvaluationConfig(
                kind="segmentor",
                split="test",
                batch_size=batch_size,
                score=ScoreConfig(pixel_histogram_bins=8),
            ),
            capture=sink,
        )
        assert isinstance(result, Ok), result
        metrics.append({metric.name: metric.value for metric in result.value.metrics})
        by_key.append({row.key: row.spatial for row in sink.rows})
    assert metrics[0] == metrics[1]
    assert by_key[0] == by_key[1]
    assert by_key[0] and all(spatial is not None for spatial in by_key[0].values())


def test_spatial_scoring_failure_aborts_capture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    build_synthetic_dataset: Callable[..., Path],
) -> None:
    """A spatial-scoring Err aborts evaluation and capture; nothing is Ok."""
    import tools.ml_models.analysis.evaluate as evaluate_module

    dataset = _small_dataset(tmp_path, build_synthetic_dataset)
    sink = _CollectSink()

    def broken(*args: object, **kwargs: object) -> Err[str]:
        return Err("spatial scoring rejected the row")

    monkeypatch.setattr(evaluate_module, "score_spatial", broken)
    result = evaluate_split(
        MarkerModel("segmentor"),
        dataset,
        load_manifest(dataset / "dataset.json"),
        EvaluationConfig(kind="segmentor", split="test", score=ScoreConfig(pixel_histogram_bins=8)),
        capture=sink,
    )
    assert isinstance(result, Err)
    assert "spatial scoring rejected the row" in result.error
    assert sink.aborted


def test_classifier_never_calls_spatial_core(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    build_synthetic_dataset: Callable[..., Path],
) -> None:
    """Classifier evaluation skips spatial scoring and captures None rows."""
    import tools.ml_models.analysis.evaluate as evaluate_module

    dataset = _small_dataset(tmp_path, build_synthetic_dataset)
    sink = _CollectSink()

    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("spatial core ran during classifier evaluation")

    monkeypatch.setattr(evaluate_module, "score_spatial", boom)
    result = evaluate_split(
        MarkerModel(),
        dataset,
        load_manifest(dataset / "dataset.json"),
        EvaluationConfig(kind="classifier", split="test"),
        capture=sink,
    )
    assert isinstance(result, Ok), result
    assert sink.rows and all(row.spatial is None for row in sink.rows)


def test_mismatched_spatial_conventions_fail_closed() -> None:
    """Rows scored under different thresholds cannot aggregate to success."""
    from tools.ml_models.analysis.metrics.spatial import score_spatial

    truth = np.zeros((1, 8, 8), dtype=np.float32)
    truth[0, :5, :4] = 1.0
    logits = np.zeros((1, 8, 8), dtype=np.float32)
    logits[0, :5, :4] = 2.0
    first = score_spatial(logits, truth, cfg=ScoreConfig(blob_probability_threshold=0.5))
    second = score_spatial(logits, truth, cfg=ScoreConfig(blob_probability_threshold=0.6))
    assert isinstance(first, Ok) and isinstance(second, Ok)
    assert isinstance(aggregate_spatial((first.value, second.value)), Err)


def test_captured_rows_retain_recorded_metadata_and_nominal_flag(
    tmp_path: Path, build_synthetic_dataset: Callable[..., Path]
) -> None:
    """Recorded observation provenance and a nominal flag reach capture rows unchanged."""
    from tools.ml_models.dataset.raw import ConditionTag

    metadata = ObservationMetadata(
        observation_id="obs-9",
        acquired_at_utc="2026-02-03T04:05:06Z",
        conditions=(ConditionTag(name="sky", value="clear"),),
        annotation_source="survey",
    )
    dataset = _small_dataset(tmp_path, build_synthetic_dataset, metadata=metadata, gsd_nominal=True)
    sink = _CollectSink()
    result = evaluate_split(
        MarkerModel(),
        dataset,
        load_manifest(dataset / "dataset.json"),
        EvaluationConfig(kind="classifier", split="test"),
        capture=sink,
    )
    assert isinstance(result, Ok), result
    assert sink.rows
    assert all(row.metadata == metadata for row in sink.rows)
    assert all(row.gsd_nominal is True for row in sink.rows)


@pytest.mark.parametrize(
    ("schema_version", "nominal", "expected"),
    [
        (2, False, None),
        (2, True, True),
        (3, False, False),
        (3, True, True),
    ],
)
def test_gsd_nominal_propagation_respects_manifest_schema(
    tmp_path: Path,
    build_synthetic_dataset: Callable[..., Path],
    schema_version: int,
    nominal: bool,
    expected: bool | None,
) -> None:
    """A recorded nominal flag stays; a False flag is explicit on schema 3 only."""
    dataset = _small_dataset(
        tmp_path,
        build_synthetic_dataset,
        gsd_nominal=nominal,
        schema_version=schema_version,
    )
    sink = _CollectSink()
    result = evaluate_split(
        MarkerModel(),
        dataset,
        load_manifest(dataset / "dataset.json"),
        EvaluationConfig(kind="classifier", split="test"),
        capture=sink,
    )
    assert isinstance(result, Ok), result
    assert sink.rows
    assert all(row.gsd_nominal is expected for row in sink.rows)


def test_captured_rows_record_the_verified_manifest_hash(
    tmp_path: Path, build_synthetic_dataset: Callable[..., Path]
) -> None:
    """Every captured row carries the manifest digest bound to the evidence."""
    dataset = _small_dataset(tmp_path, build_synthetic_dataset)
    sink = _CollectSink()
    result = evaluate_split(
        MarkerModel(),
        dataset,
        load_manifest(dataset / "dataset.json"),
        EvaluationConfig(kind="classifier", split="test"),
        capture=sink,
    )
    assert isinstance(result, Ok), result
    expected = hashlib.sha256((dataset / "dataset.json").read_bytes()).hexdigest()
    assert result.value.dataset_manifest_hash == expected
    assert sink.rows
    assert {row.dataset_manifest_hash for row in sink.rows} == {expected}
