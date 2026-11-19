"""Tests for frozen dataset preview-gallery rendering."""

import hashlib
import io
import struct

import matplotlib.pyplot as plt
import numpy as np
import pytest
from flight.libs.types import Err, Ok
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tools.ml_models.analysis.artifacts import BundleFile
from tools.ml_models.analysis.config import PlotConfig
from tools.ml_models.analysis.contracts import (
    AvailabilityRecord,
    SampleKey,
    Task,
)
from tools.ml_models.analysis.dataset import DatasetSample, MaskMeasurement
from tools.ml_models.analysis.dataset_previews import (
    DatasetPreview,
    DatasetPreviewCapture,
)
from tools.ml_models.analysis.plots.dataset import RenderedDatasetFigures
from tools.ml_models.analysis.visuals.dataset import render_dataset_visuals
from tools.ml_models.analysis.visuals.selection import DatasetGallery, GalleryPlan
from tools.ml_models.dataset.raw import ObservationMetadata
from tools.ml_models.dataset.store import RowRecord

_CFG = PlotConfig(formats=("png",), dpi=72, width_inches=7.0, height_inches=4.5)


def _npz_bytes(image: np.ndarray, mask: np.ndarray | None) -> bytes:
    """Pack one captured preview entry exactly as the capture stage writes it."""
    stream = io.BytesIO()
    if mask is None:
        np.savez(stream, image=image)
    else:
        np.savez(stream, image=image, mask=mask)
    return stream.getvalue()


def _key(tile_id: str, shard: tuple[int, int], task: Task = "classifier") -> SampleKey:
    return SampleKey(
        dataset_hash="d" * 64,
        task=task,
        split="train",
        spatial_shard=shard,
        row_index=0,
        tile_id=tile_id,
        element="id",
    )


def _sample(
    variant_id: str,
    *,
    shard: tuple[int, int] = (4, 5),
    label: int = 1,
    mask: MaskMeasurement | None = None,
    gsd: tuple[float, float] = (10.0, 10.0),
    observation: str = "obs",
    tile_id: str = "t0",
) -> DatasetSample:
    """Build a sample whose shard, mask state, and mask key stay consistent."""
    row = RowRecord(
        tile_id=tile_id,
        group_id="g0",
        frame_id=None,
        grid_rc=None,
        bin_id="",
        element="id",
        metadata=ObservationMetadata(observation_id=observation),
    )
    return DatasetSample(
        variant_id=variant_id,
        key=_key(tile_id, shard),
        row=row,
        label=label,
        gsd_m=gsd,
        tasks=("classifier",),
        stored_rows=1,
        image_sha256="a" * 64,
        mask=mask,
        gsd_anisotropy=1.0,
        tile_area_m2=6.0,
        mask_key=_key(tile_id, shard, task="segmentor") if mask is not None else None,
    )


def _preview(
    sample: DatasetSample,
    image: np.ndarray,
    mask: np.ndarray | None,
    *,
    indices: tuple[int, ...] = (2, 1, 0),
    label: str = "RGB (RED, GREEN, BLUE)",
) -> tuple[DatasetPreview, bytes]:
    data = _npz_bytes(image, mask)
    return (
        DatasetPreview(
            variant_id=sample.variant_id,
            path=f"previews/{sample.variant_id}.npz",
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
            sample=sample,
            display_indices=indices,
            display_label=label,
        ),
        data,
    )


def _capture(
    entries: tuple[tuple[DatasetPreview, bytes], ...],
    galleries: tuple[DatasetGallery, ...],
    outputs: tuple[AvailabilityRecord, ...] = (),
) -> DatasetPreviewCapture:
    return DatasetPreviewCapture(
        previews=tuple(preview for preview, _ in entries),
        files=tuple(BundleFile(path=preview.path, data=data) for preview, data in entries),
        plan=GalleryPlan(
            galleries=galleries,
            variant_ids=tuple(preview.variant_id for preview, _ in entries),
            reserved_bytes=sum(len(data) for _, data in entries),
            outputs=outputs,
        ),
    )


def _rendered(captured: DatasetPreviewCapture) -> RenderedDatasetFigures:
    result = render_dataset_visuals(captured, _CFG)
    assert isinstance(result, Ok)
    return result.value


def _figure(captured: DatasetPreviewCapture) -> Figure:
    """Build the first gallery figure without exporting, for artist checks."""
    import tools.ml_models.analysis.visuals.dataset as visuals_module

    bytes_by_path = {file.path: file.data for file in captured.files}
    previews_by_id = {preview.variant_id: preview for preview in captured.previews}
    gallery = captured.plan.galleries[0]
    members = []
    for variant_id in gallery.variant_ids:
        loaded = visuals_module._load_preview(previews_by_id[variant_id], bytes_by_path)
        assert isinstance(loaded, Ok)
        members.append((previews_by_id[variant_id], loaded.value))
    if gallery.family == "representative":
        result = visuals_module._representative(gallery, members[0][0], members[0][1], _CFG)
    elif gallery.family == "augmentation":
        result = visuals_module._augmentation(gallery, members[0][0], members[0][1], _CFG)
    else:
        result = visuals_module._gsd_pair(gallery, members, _CFG)
    assert isinstance(result, Ok)
    figure = result.value
    plt.close(figure)
    return figure


def _suptitle(figure: Figure) -> str:
    """Return the figure-level title text."""
    return figure.get_suptitle()


def _image_array(axes: Axes) -> np.ndarray:
    """Return the first panel image array on an axes."""
    array = axes.images[0].get_array()
    assert array is not None
    return array


def _png_pixels(data: bytes) -> tuple[int, int]:
    """Return the PNG IHDR pixel dimensions."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


def _rgb_panel() -> DatasetPreviewCapture:
    """One representative gallery over a distinct-valued BGR input."""
    image = np.zeros((3, 4, 5), dtype=np.float32)
    image[1] = 0.5
    image[2] = 1.0
    sample = _sample("v1")
    return _capture(
        (_preview(sample, image, None),),
        (DatasetGallery("representative_v1", "representative", ("v1",), ()),),
    )


def test_rgb_panel_uses_exact_display_indices() -> None:
    """BGR-ordered bands land in RGB order on the rendered panel."""
    captured = _rgb_panel()
    figure = _figure(captured)
    panel = np.asarray(figure.axes[0].images[0].get_array())
    assert panel.shape == (4, 5, 3)
    assert tuple(panel[0, 0]) == (1.0, 0.5, 0.0)


def test_exported_visual_matches_configured_dimensions() -> None:
    """Visual PNG bytes carry exactly inches times dpi pixels."""
    rendered = _rendered(_rgb_panel())
    by_path = {file.path: file for file in rendered.files}
    assert _png_pixels(by_path["visuals/representative_v1.png"].data) == (504, 324)
    outputs = {record.name: record for record in rendered.outputs}
    assert outputs["visual:representative_v1"].status == "AVAILABLE"


def test_grayscale_panel_uses_supplied_label_and_fixed_scale() -> None:
    """One-band views stay grayscale with the supplied non-RGB label."""
    image = np.full((1, 4, 5), 0.25, dtype=np.float32)
    sample = _sample("v1")
    captured = _capture(
        (
            _preview(
                sample,
                image,
                None,
                indices=(0,),
                label="RED band (unit grayscale)",
            ),
        ),
        (DatasetGallery("representative_v1", "representative", ("v1",), ()),),
    )
    figure = _figure(captured)
    shown = figure.axes[0].images[0]
    assert _image_array(figure.axes[0]).shape == (4, 5)
    assert shown.get_clim() == (0.0, 1.0)
    assert shown.get_cmap().name == "gray"
    assert "RGB" not in figure.axes[0].get_title()
    assert "RED band (unit grayscale)" in figure.axes[0].get_title()


def test_missing_mask_vs_explicit_empty_mask_panels() -> None:
    """Absent mask arrays are labelled; explicit empty masks render empty."""
    image = np.zeros((3, 4, 5), dtype=np.float32)
    no_mask_sample = _sample("v1", mask=None)
    captured = _capture(
        (_preview(no_mask_sample, image, None),),
        (DatasetGallery("representative_v1", "representative", ("v1",), ()),),
    )
    figure = _figure(captured)
    assert not figure.axes[1].images
    assert any(text.get_text() == "Mask unavailable" for text in figure.axes[1].texts)
    assert "label positive" in _suptitle(figure)
    empty = MaskMeasurement(
        area_px=0,
        area_fraction=0.0,
        area_m2=0.0,
        n_components=0,
        component_areas_px=(),
        border_touching=False,
    )
    empty_sample = _sample("v2", label=0, mask=empty)
    captured = _capture(
        (_preview(empty_sample, image, np.zeros((1, 4, 5), dtype=np.uint8)),),
        (DatasetGallery("representative_v2", "representative", ("v2",), ()),),
    )
    figure = _figure(captured)
    assert figure.axes[1].get_title() == "Explicit mask (0 px)"
    assert float(_image_array(figure.axes[1]).sum()) == 0.0
    assert "label negative" in _suptitle(figure)


def test_augmentation_gallery_applies_elements_and_marks_verified() -> None:
    """Legal non-square elements render once each with dims preserved."""
    image = np.zeros((1, 2, 3), dtype=np.float32)
    image[0, 0, 0] = 1.0
    sample = _sample("v1", shard=(2, 3))
    captured = _capture(
        (_preview(sample, image, None, indices=(0,), label="band"),),
        (
            DatasetGallery(
                "augmentation_v1",
                "augmentation",
                ("v1",),
                ("id", "rot180", "flip_h"),
            ),
        ),
    )
    figure = _figure(captured)
    shapes = [_image_array(axes).shape for axes in figure.axes]
    assert shapes == [(2, 3), (2, 3), (2, 3)]
    titles = [axes.get_title() for axes in figure.axes]
    assert titles == ["id", "rot180", "flip_h"]
    assert "verified stored training transforms" in _suptitle(figure)
    rotated = _image_array(figure.axes[1])
    assert rotated[0, 0] == 0.0 and rotated[1, 2] == 1.0


def test_same_observation_gsd_labels_variants_without_independence_claim() -> None:
    """Panels show actual lateral/along GSD and share the recorded observation."""
    image = np.zeros((3, 4, 5), dtype=np.float32)
    near = _sample("v1", gsd=(10.0, 10.0), observation="o42")
    far = _sample("v2", gsd=(15.0, 10.0), observation="o42")
    captured = _capture(
        (_preview(near, image, None), _preview(far, image, None)),
        (
            DatasetGallery(
                "same_observation_gsd_o42",
                "same_observation_gsd",
                ("v1", "v2"),
                (),
            ),
        ),
    )
    figure = _figure(captured)
    assert "o42" in _suptitle(figure)
    titles = [axes.get_title() for axes in figure.axes]
    assert titles == ["lateral 10 m, along 10 m", "lateral 15 m, along 10 m"]
    assert "independent" not in _suptitle(figure)


def test_unavailable_family_renders_indexed_placeholder() -> None:
    """Skipped gallery families surface explicit placeholder bytes and states."""
    outputs = (
        AvailabilityRecord("gallery:representative", "SKIPPED", "budget exhausted"),
        AvailabilityRecord("gallery:augmentation", "SKIPPED", "budget exhausted"),
        AvailabilityRecord(
            "gallery:same_observation_gsd",
            "SKIPPED",
            "no recorded observation spans GSD bins",
        ),
    )
    captured = _capture((), (), outputs)
    rendered = _rendered(captured)
    paths = {file.path for file in rendered.files}
    assert paths == {
        "visuals/gallery_unavailable_representative.png",
        "visuals/gallery_unavailable_augmentation.png",
        "visuals/gallery_unavailable_same_observation_gsd.png",
    }
    by_name = {record.name: record for record in rendered.outputs}
    assert by_name["gallery:representative"].status == "SKIPPED"
    placeholder = by_name["visual:gallery_unavailable_same_observation_gsd"]
    assert placeholder.status == "UNAVAILABLE"
    assert placeholder.reason == "no recorded observation spans GSD bins"


def test_corrupt_or_missing_cache_returns_err() -> None:
    """Byte mismatches and absent files are render errors, not silent renders."""
    image = np.zeros((1, 2, 2), dtype=np.float32)
    sample = _sample("v1", shard=(2, 2))
    preview, data = _preview(sample, image, None, indices=(0,), label="band")
    plan = GalleryPlan(galleries=(), variant_ids=("v1",), reserved_bytes=0, outputs=())
    tampered = DatasetPreviewCapture(
        previews=(preview,),
        files=(BundleFile(path=preview.path, data=b"\x00" + data[1:]),),
        plan=plan,
    )
    result = render_dataset_visuals(tampered, _CFG)
    assert isinstance(result, Err)
    assert "differ" in result.error
    missing = DatasetPreviewCapture(previews=(preview,), files=(), plan=plan)
    result = render_dataset_visuals(missing, _CFG)
    assert isinstance(result, Err)
    assert "differ" in result.error


def test_valid_checksum_but_malformed_arrays_return_err() -> None:
    """Checksum-valid bytes with wrong dtype, shape, or mask state still fail."""
    sample = _sample("v1", shard=(2, 2))
    cases = (
        np.zeros((1, 2, 2), dtype=np.float64),
        np.zeros((1, 3, 3), dtype=np.float32),
        np.full((1, 2, 2), np.float32(2.0)),
    )
    for bad_image in cases:
        preview, data = _preview(sample, bad_image, None, indices=(0,), label="x")
        captured = DatasetPreviewCapture(
            previews=(preview,),
            files=(BundleFile(path=preview.path, data=data),),
            plan=GalleryPlan((), ("v1",), 0, ()),
        )
        result = render_dataset_visuals(captured, _CFG)
        assert isinstance(result, Err), bad_image.shape
    good = np.zeros((1, 2, 2), dtype=np.float32)
    synthesized = MaskMeasurement(
        area_px=1,
        area_fraction=0.25,
        area_m2=1.0,
        n_components=1,
        component_areas_px=(1,),
        border_touching=False,
    )
    no_key = _sample("v1", shard=(2, 2), mask=synthesized)
    no_key_preview, no_key_data = _preview(no_key, good, None, indices=(0,), label="x")
    missing_mask = DatasetPreviewCapture(
        previews=(no_key_preview,),
        files=(BundleFile(path=no_key_preview.path, data=no_key_data),),
        plan=GalleryPlan((), ("v1",), 0, ()),
    )
    assert isinstance(render_dataset_visuals(missing_mask, _CFG), Err)
    keyless = _sample("v1", shard=(2, 2), mask=None)
    keyless_preview, keyless_data = _preview(
        keyless, good, np.zeros((1, 2, 2), dtype=np.uint8), indices=(0,), label="x"
    )
    stray_mask = DatasetPreviewCapture(
        previews=(keyless_preview,),
        files=(BundleFile(path=keyless_preview.path, data=keyless_data),),
        plan=GalleryPlan((), ("v1",), 0, ()),
    )
    assert isinstance(render_dataset_visuals(stray_mask, _CFG), Err)


def test_invalid_display_indices_return_err() -> None:
    """Negative, duplicated-range, or wrong-arity indices are refused."""
    image = np.zeros((3, 2, 2), dtype=np.float32)
    sample = _sample("v1", shard=(2, 2))
    plan = GalleryPlan(galleries=(), variant_ids=("v1",), reserved_bytes=0, outputs=())
    for indices in ((-1, 0, 2), (0, 1), (0, 1, 2, 0), (3,)):
        preview, data = _preview(sample, image, None, indices=indices, label="x")
        captured = DatasetPreviewCapture(
            previews=(preview,),
            files=(BundleFile(path=preview.path, data=data),),
            plan=plan,
        )
        result = render_dataset_visuals(captured, _CFG)
        assert isinstance(result, Err), indices


def test_np_load_only_reads_bytesio_and_no_source_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All np.load calls take BytesIO; dataset/model loaders stay uncalled."""
    import tools.ml_models.dataset.store as store_module

    calls: list[object] = []
    real_load = np.load

    def spy(file: io.BytesIO, allow_pickle: bool = False) -> object:
        calls.append(file)
        return real_load(file, allow_pickle=allow_pickle)

    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("source/model I/O during rendering")

    monkeypatch.setattr(np, "load", spy)
    monkeypatch.setattr(store_module, "read_rows", boom)
    rendered = _rendered(_rgb_panel())
    assert calls and all(isinstance(file, io.BytesIO) for file in calls)
    assert rendered.files
    assert plt.get_fignums() == []


def test_gallery_identifiers_are_deterministic() -> None:
    """Re-rendering the same capture yields identical artifact paths."""
    first = _rendered(_rgb_panel())
    second = _rendered(_rgb_panel())
    assert [file.path for file in first.files] == [file.path for file in second.files]
