"""Deterministic bounded dataset-gallery and exact-array capture references."""

import io
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
from flight.libs.types import Err, Ok
from tools.ml_models.analysis.config import CaptureConfig
from tools.ml_models.analysis.dataset import measure_dataset
from tools.ml_models.analysis.dataset_previews import capture_dataset_previews, display_channels
from tools.ml_models.analysis.visuals.selection import select_dataset_galleries
from tools.ml_models.dataset.augment import AugmentRecipe
from tools.ml_models.dataset.build import build_dataset
from tools.ml_models.dataset.raw import BinSpec, GsdPair, ObservationMetadata, RawTile, RawTileRef
from tools.ml_models.dataset.spec import BuildSpec


def test_semantic_rgb_never_uses_storage_order_or_invents_missing_bands() -> None:
    assert display_channels(("BLUE", "GREEN", "RED")) == ((2, 1, 0), "RGB (RED, GREEN, BLUE)")
    assert display_channels(("RED", "GREEN", "BLUE")) == ((0, 1, 2), "RGB (RED, GREEN, BLUE)")
    indices, label = display_channels(("NIR", "RED", "THERMAL"))
    assert indices == (0, 1, 2)
    assert "not RGB" in label
    assert display_channels(("RED", "NIR")) == ((0,), "RED band (unit grayscale)")
    indices, label = display_channels(("RED", "red", "BLUE", "GREEN"))
    assert indices == (0, 1, 2) and "not RGB" in label


def test_selection_is_order_invariant_and_bounded(
    tmp_path: Path,
    build_synthetic_dataset: Callable[..., Path],
) -> None:
    measured = measure_dataset(build_synthetic_dataset(tmp_path / "ds"))
    assert isinstance(measured, Ok)
    frozen = measured.value
    cfg = CaptureConfig(max_preview_images=2, examples_per_family=2)
    first = select_dataset_galleries(frozen, cfg)
    second = select_dataset_galleries(replace(frozen, samples=tuple(reversed(frozen.samples))), cfg)
    assert first == second
    assert len(first.variant_ids) <= 2
    assert first.reserved_bytes <= cfg.max_capture_bytes
    assert all(set(gallery.variant_ids) <= set(first.variant_ids) for gallery in first.galleries)
    unavailable = next(o for o in first.outputs if o.name == "gallery:same_observation_gsd")
    assert unavailable.status == "UNAVAILABLE" and unavailable.reason


def test_disabled_and_tiny_budgets_do_not_create_fake_previews(
    tmp_path: Path,
    build_synthetic_dataset: Callable[..., Path],
) -> None:
    measured = measure_dataset(build_synthetic_dataset(tmp_path / "ds"))
    assert isinstance(measured, Ok)
    for cfg in (
        CaptureConfig(max_preview_images=0, examples_per_family=0),
        CaptureConfig(max_capture_bytes=1),
    ):
        plan = select_dataset_galleries(measured.value, cfg)
        assert plan.variant_ids == () and plan.galleries == ()
        assert all(output.status != "AVAILABLE" and output.reason for output in plan.outputs)


def test_same_observation_pairs_require_authoritative_identity_and_different_gsd(
    tmp_path: Path,
    build_synthetic_dataset: Callable[..., Path],
) -> None:
    measured = measure_dataset(build_synthetic_dataset(tmp_path / "ds"))
    assert isinstance(measured, Ok)
    frozen = measured.value
    samples = tuple(
        replace(s, row=replace(s.row, metadata=ObservationMetadata(observation_id="recorded")))
        for s in frozen.samples
    )
    plan = select_dataset_galleries(replace(frozen, samples=samples), CaptureConfig())
    pairs = [gallery for gallery in plan.galleries if gallery.family == "same_observation_gsd"]
    assert len(pairs) == 1 and len(pairs[0].variant_ids) == 2
    by_id = {sample.variant_id: sample for sample in samples}
    selected = [by_id[identity] for identity in pairs[0].variant_ids]
    assert selected[0].gsd_m != selected[1].gsd_m
    assert all(sample.row.metadata.observation_id == "recorded" for sample in selected)


def test_capture_uses_exact_canonical_arrays_and_explicit_mask_keys(
    tmp_path: Path,
    build_synthetic_dataset: Callable[..., Path],
) -> None:
    dataset = build_synthetic_dataset(tmp_path / "ds")
    measured = measure_dataset(dataset)
    assert isinstance(measured, Ok)
    cfg = CaptureConfig(max_preview_images=3, examples_per_family=3)
    captured = capture_dataset_previews(dataset, measured.value, cfg)
    assert isinstance(captured, Ok)
    capture = captured.value
    assert len(capture.previews) <= 3
    assert sum(len(file.data) for file in capture.files) <= cfg.max_capture_bytes
    files = {file.path: file.data for file in capture.files}
    for preview in capture.previews:
        with np.load(io.BytesIO(files[preview.path]), allow_pickle=False) as arrays:
            image = arrays["image"]
            key = preview.sample.key
            source = np.load(
                dataset
                / key.task
                / key.split
                / f"{key.spatial_shard[0]}x{key.spatial_shard[1]}"
                / "images.npy",
                mmap_mode="r",
                allow_pickle=False,
            )[key.row_index]
            assert key.element == "id"
            np.testing.assert_array_equal(image, source)
            assert image.dtype == np.float32
            assert arrays["mask"].dtype == np.uint8
            assert preview.sample.mask_key is not None
            mask_key = preview.sample.mask_key
            mask = np.load(
                dataset
                / mask_key.task
                / mask_key.split
                / f"{mask_key.spatial_shard[0]}x{mask_key.spatial_shard[1]}"
                / "masks.npy",
                mmap_mode="r",
                allow_pickle=False,
            )[mask_key.row_index]
            np.testing.assert_array_equal(arrays["mask"], mask)
    (dataset / "tampered.txt").write_text("changed source")
    assert isinstance(capture_dataset_previews(dataset, measured.value, cfg), Err)


class _RectangularSource:
    name = "preview-reference"
    source_ref = ""
    domain = "unit"
    band_names: tuple[str, ...] = ("BLUE", "GREEN", "RED")
    bins: tuple[BinSpec, ...] = ()

    def __init__(self) -> None:
        image = np.arange(45, dtype=np.float32).reshape(3, 3, 5) / np.float32(44)
        mask = np.zeros((1, 3, 5), dtype=np.uint8)
        mask[0, 0, 0] = 1
        self.tiles = tuple(
            RawTile(
                ref=RawTileRef(
                    tile_id=tile_id,
                    group_id=group,
                    frame_id=group,
                    grid_rc=None,
                    bin_id="",
                    label=float(group == "a"),
                    has_mask=group == "a",
                    gsd=GsdPair(gsd, gsd * 1.5),
                    height=3,
                    width=5,
                    metadata=ObservationMetadata(
                        observation_id=group,
                        source_annotation_state="NONEMPTY" if group == "a" else "MISSING",
                    ),
                ),
                image=image.copy(),
                mask=mask.copy() if group == "a" else None,
            )
            for tile_id, group, gsd in (
                ("a-low", "a", 2.0),
                ("a-high", "a", 4.0),
                ("b", "b", 2.0),
                ("c", "c", 2.0),
            )
        )

    def index(self) -> tuple[RawTileRef, ...]:
        return tuple(tile.ref for tile in self.tiles)

    def iter_tiles(self) -> Iterator[RawTile]:
        yield from self.tiles


def test_rectangular_inverse_capture_preserves_missing_masks_and_gsd_pairs(tmp_path: Path) -> None:
    source = _RectangularSource()
    root = tmp_path / "ds"
    build_dataset(source, root, BuildSpec(augment=AugmentRecipe(elements=("rot180", "flip_h"))))
    measured = measure_dataset(root)
    assert isinstance(measured, Ok)
    captured = capture_dataset_previews(root, measured.value, CaptureConfig())
    assert isinstance(captured, Ok)
    capture = captured.value
    files = {file.path: file.data for file in capture.files}
    originals = {tile.ref.tile_id: tile for tile in source.tiles}
    for preview in capture.previews:
        original = originals[preview.sample.key.tile_id]
        with np.load(io.BytesIO(files[preview.path]), allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays["image"], original.image)
            assert arrays["image"].shape == (3, 3, 5)
            if original.mask is None:
                assert "mask" not in arrays.files
                assert preview.sample.mask is None
            else:
                np.testing.assert_array_equal(arrays["mask"], original.mask)
    assert any(preview.sample.key.element != "id" for preview in capture.previews)
    pair = next(g for g in capture.plan.galleries if g.family == "same_observation_gsd")
    samples = {preview.variant_id: preview.sample for preview in capture.previews}
    assert {samples[identity].gsd_m for identity in pair.variant_ids} == {(2.0, 3.0), (4.0, 6.0)}
