"""Independent dataset cohort, coverage, and integrity references."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from flight.libs.types import Err, Ok
from tools.ml_models.analysis.dataset import _baselines, measure_dataset
from tools.ml_models.dataset.augment import ELEMENT_NAMES, AugmentRecipe, apply_dihedral
from tools.ml_models.dataset.manifest import (
    DatasetManifest,
    ShardCount,
    compute_dataset_hash,
    write_manifest,
)
from tools.ml_models.dataset.raw import ConditionTag, ObservationMetadata
from tools.ml_models.dataset.split import SplitRecipe
from tools.ml_models.dataset.store import RowRecord, ShardWriter


def _fixture(root: Path, *, identity: bool = True) -> Path:
    root.mkdir()
    a = ObservationMetadata(
        observation_id="observation-a",
        acquired_at_utc="2026-01-02T03:04:05Z",
        conditions=(ConditionTag(name="sky", value="clear"),),
        source_annotation_state="NONEMPTY",
        annotation_source="annotations/a.json",
    )
    c = ObservationMetadata(
        observation_id="observation-c",
        source_annotation_state="EXPLICIT_EMPTY",
    )
    elements = ELEMENT_NAMES if identity else ("rot90", "flip_h")
    low = np.array([[[0.0, 0.25], [0.5, 1.0]]], dtype=np.float32)
    high = np.full((1, 2, 2), 0.75, dtype=np.float32)
    zero = np.zeros((1, 2, 2), dtype=np.float32)
    positive_mask = np.array([[[1, 0], [0, 0]]], dtype=np.uint8)
    empty_mask = np.zeros((1, 2, 2), dtype=np.uint8)
    shards = []
    for task, split, tiles in (
        ("classifier", "train", (("a-low", low, positive_mask), ("a-high", high, empty_mask))),
        ("segmentor", "train", (("a-low", low, positive_mask), ("a-high", high, empty_mask))),
        ("classifier", "val", (("b", zero, None),)),
        ("classifier", "test", (("c", zero, empty_mask),)),
        ("segmentor", "test", (("c", zero, empty_mask),)),
    ):
        transforms = elements if split == "train" else ("id",)
        n = len(tiles) * len(transforms)
        writer = ShardWriter(
            root / task / split / "2x2", n, 2, 2, channels=1, with_masks=task == "segmentor"
        )
        for tile, image, mask in tiles:
            for element in transforms:
                writer.append(
                    apply_dihedral(image, element),
                    np.array([2.0, 3.0], dtype=np.float32),
                    1.0 if split == "train" else 0.0,
                    apply_dihedral(mask, element)
                    if task == "segmentor" and mask is not None
                    else None,
                    RowRecord(
                        tile_id=tile,
                        group_id=split,
                        frame_id=None,
                        grid_rc=None,
                        bin_id="",
                        element=element,
                        metadata=a
                        if split == "train"
                        else c
                        if split == "test"
                        else ObservationMetadata(source_annotation_state="MISSING"),
                        prepared_mask_state="MISSING"
                        if mask is None
                        else "NONEMPTY"
                        if mask.any()
                        else "EMPTY",
                    ),
                )
        writer.close()
        shards.append(
            ShardCount(
                task=task,
                split=split,
                height=2,
                width=2,
                n=n,
                n_positive=n if split == "train" else 0,
            )
        )
    manifest = DatasetManifest(
        source="reference",
        source_ref="",
        weight_table_id="reference",
        band_names=("RED",),
        norm="unit",
        image_dtype="float32",
        gsd_reference_m=1.0,
        split=SplitRecipe(),
        augment=AugmentRecipe(elements=elements),
        bins=(),
        shards=tuple(shards),
        gsd_lateral_min_m=2.0,
        gsd_lateral_max_m=2.0,
        gsd_along_min_m=3.0,
        gsd_along_max_m=3.0,
        dataset_hash=compute_dataset_hash(root),
        schema_version=3,
    )
    write_manifest(root / "dataset.json", manifest)
    return root


def _rehash(root: Path) -> None:
    payload = json.loads((root / "dataset.json").read_text())
    payload["dataset_hash"] = compute_dataset_hash(root)
    (root / "dataset.json").write_text(json.dumps(payload))


def test_counts_remove_task_and_augmentation_weighting(tmp_path: Path) -> None:
    measured = measure_dataset(_fixture(tmp_path / "ds"))
    assert isinstance(measured, Ok)
    result = measured.value
    metrics = {metric.name: metric for metric in result.metrics}
    assert metrics["stored_rows"].value == 35
    assert metrics["canonical_task_variants"].value == 7
    assert metrics["canonical_variants"].value == 4
    assert metrics["augmentation_copies"].value == 28
    assert metrics["split_groups"].value == 3
    assert metrics["recorded_observations"].value == 2
    assert metrics["total_observations"].value is None
    assert metrics["total_observations"].reason
    assert len(result.samples) == 4
    assert sum(sample.stored_rows for sample in result.samples) == 35
    assert len(result.components) == 1
    assert result.components[0].area_px == 1
    assert result.components[0].area_m2 == 6
    samples = {sample.row.tile_id: sample for sample in result.samples}
    assert samples["a-high"].mask is not None and samples["a-high"].mask.area_px == 0
    assert samples["b"].mask is None
    assert samples["c"].mask is not None and samples["c"].mask.area_px == 0
    assert samples["a-low"].key.task == "classifier"
    assert samples["a-low"].mask_key is not None
    assert samples["a-low"].mask_key.task == "segmentor"
    assert samples["a-low"].mask_key.element == "id"
    assert samples["b"].mask_key is None
    assert result.manifest.source == "reference"
    assert result.manifest.augment.elements == ELEMENT_NAMES
    coverage = {
        (row.population, row.field, row.value): row.n
        for row in result.coverage
        if row.split is None
    }
    assert coverage["canonical_variant", "mask_category", "positive_empty"] == 1
    assert coverage["canonical_variant", "mask_category", "verified_negative_empty"] == 1
    assert coverage["canonical_variant", "mask_category", "missing"] == 1
    assert coverage["canonical_variant", "timestamp", "recorded"] == 2
    assert coverage["recorded_observation", "timestamp", "recorded"] == 1
    assert coverage["recorded_observation", "condition:sky", "clear"] == 1


def test_pixels_and_duplicates_use_one_copy_of_each_variant(tmp_path: Path) -> None:
    measured = measure_dataset(_fixture(tmp_path / "ds"))
    assert isinstance(measured, Ok)
    result = measured.value
    pixels = next(cohort.pixels for cohort in result.pixels if cohort.split is None)
    assert pixels.n_images == 4 and pixels.n_pixels == 16
    expected = np.array([0, 0.25, 0.5, 1, 0.75, 0.75, 0.75, 0.75] + [0] * 8, dtype=np.float64)
    assert pixels.bands[0].mean == expected.mean()
    assert pixels.bands[0].std == pytest.approx(expected.std())
    assert pixels.bands[0].at_zero == 9
    assert pixels.bands[0].at_one == 1
    assert len(result.duplicates) == 1
    assert result.duplicates[0].n_variants == 2
    assert result.duplicates[0].splits == ("test", "val")
    baseline = next(b for b in result.baselines if b.task == "classifier" and b.split == "val")
    values = {metric.name: metric for metric in baseline.metrics}
    assert values["baseline_train_prevalence"].value == 1
    assert values["baseline_train_prevalence_brier"].value == 1
    assert values["baseline_train_prevalence_bce"].value is None
    assert values["baseline_always_negative_accuracy"].value == 1
    assert values["baseline_always_positive_accuracy"].value == 0


def test_inverse_view_has_same_frozen_measurements(tmp_path: Path) -> None:
    full = measure_dataset(_fixture(tmp_path / "full"))
    inverted = measure_dataset(_fixture(tmp_path / "inverse", identity=False))
    assert isinstance(full, Ok) and isinstance(inverted, Ok)
    assert full.value.pixels == inverted.value.pixels
    assert full.value.components == inverted.value.components
    assert all(
        sample.key.element != "id"
        for sample in inverted.value.samples
        if sample.key.split == "train"
    )


@pytest.mark.parametrize(
    "fault",
    [
        "group",
        "observation",
        "task_pixels",
        "augmentation",
        "state",
        "counts",
        "hash",
        "extra_shard",
    ],
)
def test_invalid_required_evidence_fails_closed(tmp_path: Path, fault: str) -> None:
    root = _fixture(tmp_path / "ds")
    path = root / "classifier" / "val" / "2x2" / "rows.jsonl"
    payload = json.loads(path.read_text())
    if fault == "group":
        payload["group_id"] = "train"
    elif fault == "observation":
        payload["metadata"]["observation_id"] = "observation-a"
    elif fault == "state":
        payload["prepared_mask_state"] = "NONEMPTY"
    if fault in ("group", "observation", "state"):
        path.write_text(json.dumps(payload) + "\n")
    elif fault in ("task_pixels", "augmentation"):
        path = root / "segmentor" / "train" / "2x2" / "images.npy"
        array = np.load(path, allow_pickle=False)
        array[0 if fault == "task_pixels" else 1, 0, 0, 0] = np.float32(0.125)
        np.save(path, array, allow_pickle=False)
    elif fault == "counts":
        path = root / "dataset.json"
        manifest = json.loads(path.read_text())
        manifest["shards"][0]["n"] += 1
        path.write_text(json.dumps(manifest))
    elif fault == "hash":
        (root / "arbitrary.txt").write_text("new bytes")
    elif fault == "extra_shard":
        path = root / "classifier" / "val" / "1x1"
        path.mkdir()
        np.save(path / "images.npy", np.zeros((1, 1, 1, 1), dtype=np.float32), allow_pickle=False)
    if fault != "hash":
        _rehash(root)
    measured = measure_dataset(root)
    assert isinstance(measured, Err), fault


def test_complete_recorded_ids_enable_total_not_independence(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "ds")
    path = root / "classifier" / "val" / "2x2" / "rows.jsonl"
    payload = json.loads(path.read_text())
    payload["metadata"]["observation_id"] = "observation-b"
    path.write_text(json.dumps(payload) + "\n")
    _rehash(root)
    measured = measure_dataset(root)
    assert isinstance(measured, Ok)
    metric = next(m for m in measured.value.metrics if m.name == "total_observations")
    assert metric.value == 3
    assert any("independ" in warning for warning in measured.value.warnings)


def test_schema_two_keeps_original_counts_and_mask_missingness_unknown(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "ds")
    for path in root.rglob("rows.jsonl"):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            del row["metadata"]
            del row["prepared_mask_state"]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    path = root / "dataset.json"
    payload = json.loads(path.read_text())
    payload["schema"] = 2
    path.write_text(json.dumps(payload))
    _rehash(root)
    before = {
        str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }
    measured = measure_dataset(root)
    assert isinstance(measured, Ok)
    assert measured.value.identity.schema_version == 2
    metrics = {m.name: m for m in measured.value.metrics}
    assert metrics["canonical_variants"].value == 4
    assert metrics["recorded_observations"].value == 0
    assert metrics["total_observations"].value is None
    assert all(s.row.metadata.observation_id is None for s in measured.value.samples)
    assert sum(s.mask is not None for s in measured.value.samples) == 3
    assert before == {
        str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }


def test_literal_missing_condition_is_not_absent_metadata(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "ds")
    for task in ("classifier", "segmentor"):
        path = root / task / "test" / "2x2" / "rows.jsonl"
        payload = json.loads(path.read_text())
        payload["metadata"]["conditions"] = [{"name": "sky", "value": "missing"}]
        path.write_text(json.dumps(payload) + "\n")
    _rehash(root)
    measured = measure_dataset(root)
    assert isinstance(measured, Ok)
    coverage = {
        row.value: row.n
        for row in measured.value.coverage
        if row.split is None
        and row.population == "canonical_variant"
        and row.field == "condition:sky"
    }
    assert coverage == {"clear": 2, "missing": 1, None: 1}


def test_baseline_probability_is_fitted_on_train_only(tmp_path: Path) -> None:
    measured = measure_dataset(_fixture(tmp_path / "ds"))
    assert isinstance(measured, Ok)
    samples = tuple(
        replace(s, label=0 if s.row.tile_id == "a-low" else 1) for s in measured.value.samples
    )
    baseline = next(b for b in _baselines(samples) if b.task == "classifier" and b.split == "val")
    values = {metric.name: metric.value for metric in baseline.metrics}
    assert values["baseline_train_prevalence"] == 0.5
    assert values["baseline_train_prevalence_brier"] == 0.25
    assert values["baseline_train_prevalence_bce"] == pytest.approx(np.log(2))
    assert values["baseline_train_majority_accuracy"] == 1
    assert next(m for m in baseline.metrics if m.name == "baseline_train_prevalence").support.n == 2
    segmentor = next(b for b in _baselines(samples) if b.task == "segmentor" and b.split == "train")
    false_negatives = next(
        m for m in segmentor.metrics if m.name == "baseline_background_false_negative_pixels"
    )
    assert false_negatives.support.unit == "PIXEL"
    assert false_negatives.support.n == 8
    no_train = _baselines(tuple(s for s in samples if s.key.split != "train"))
    assert all(
        m.value is None
        for b in no_train
        if b.task == "classifier"
        for m in b.metrics
        if m.name.startswith("baseline_train")
    )


@pytest.mark.parametrize("fault", ["frame_leakage", "extra_mask"])
def test_cross_source_integrity_checks_are_exhaustive(tmp_path: Path, fault: str) -> None:
    root = _fixture(tmp_path / "ds")
    if fault == "frame_leakage":
        for path in root.rglob("rows.jsonl"):
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            for row in rows:
                row["frame_id"] = "shared-frame"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    else:
        path = root / "segmentor" / "val" / "1x1"
        path.mkdir(parents=True)
        np.save(path / "masks.npy", np.zeros((1, 1, 1, 1), dtype=np.uint8), allow_pickle=False)
    _rehash(root)
    assert isinstance(measure_dataset(root), Err)


def test_geometry_support_does_not_mix_empty_masks_into_positive_area(tmp_path: Path) -> None:
    measured = measure_dataset(_fixture(tmp_path / "ds"))
    assert isinstance(measured, Ok)
    metrics = {m.name: m for m in measured.value.metrics}
    assert metrics["positive_mask_area_px_mean"].value == 1
    assert metrics["positive_mask_area_px_mean"].support.n == 1
    assert metrics["positive_mask_area_m2_mean"].value == 6
    assert metrics["explicit_mask_components_mean"].value == pytest.approx(1 / 3)
    assert metrics["explicit_mask_components_mean"].support.n == 3
    assert metrics["explicit_mask_border_touching_mean"].value == pytest.approx(1 / 3)
    assert metrics["gsd_anisotropy_mean"].value == 1.5
    assert metrics["tile_area_m2_mean"].value == 24


def test_task_copy_gsd_conflict_is_not_counted_as_an_extra_variant(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "ds")
    path = root / "segmentor" / "train" / "2x2" / "gsd.npy"
    gsd = np.load(path, allow_pickle=False)
    gsd[:, 0] = np.float32(4.0)
    np.save(path, gsd, allow_pickle=False)
    path = root / "dataset.json"
    payload = json.loads(path.read_text())
    payload["gsd_lateral_max_m"] = 4.0
    path.write_text(json.dumps(payload))
    _rehash(root)
    measured = measure_dataset(root)
    assert isinstance(measured, Err)
    assert "tile identity" in measured.error


def test_missing_source_annotation_cannot_claim_an_empty_stored_mask(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "ds")
    for task in ("classifier", "segmentor"):
        path = root / task / "test" / "2x2" / "rows.jsonl"
        payload = json.loads(path.read_text())
        payload["metadata"]["source_annotation_state"] = "MISSING"
        payload["prepared_mask_state"] = "UNKNOWN"
        path.write_text(json.dumps(payload) + "\n")
    _rehash(root)
    result = measure_dataset(root)
    assert isinstance(result, Err)
    assert "source annotation" in result.error
