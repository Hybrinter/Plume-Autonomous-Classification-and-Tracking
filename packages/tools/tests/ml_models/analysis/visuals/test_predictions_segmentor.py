"""Segmentor galleries render frozen extent panels without re-matching or re-scoring."""

from __future__ import annotations

import hashlib
import io
import math
from dataclasses import replace

import matplotlib
import numpy as np
import pytest
from flight.libs.types import Err, Ok
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from tools.ml_models.analysis.artifacts import BundleFile
from tools.ml_models.analysis.capture import CaptureRow
from tools.ml_models.analysis.config import PlotConfig
from tools.ml_models.analysis.contracts import (
    AvailabilityRecord,
    MetricSupport,
    MetricValue,
    SampleKey,
)
from tools.ml_models.analysis.metrics.boundary import BoundaryRow
from tools.ml_models.analysis.metrics.localization import (
    ComponentMatch,
    LocalizationRow,
    MaskComponent,
)
from tools.ml_models.analysis.metrics.spatial import SpatialRow
from tools.ml_models.analysis.plots.dataset import RenderedDatasetFigures
from tools.ml_models.analysis.prediction_display import segmentation_display_data
from tools.ml_models.analysis.prediction_selections import (
    PredictionGallery,
    prediction_key,
)
from tools.ml_models.analysis.visuals.predictions import (
    PredictionPreview,
    PredictionPreviewCapture,
    render_prediction_visuals,
)

matplotlib.use("Agg")


def frozen_visual_fixture() -> PredictionPreviewCapture:
    """Return exact immutable evidence for the authored 3x6 pixel geometry."""
    image = np.empty((3, 3, 6), dtype=np.float32)
    image[0], image[1], image[2] = 0.1, 0.5, 0.9
    target = np.zeros((1, 3, 6), dtype=np.float32)
    target[0, 0, :2] = 1.0
    target[0, 2, 5] = 1.0
    logits = np.full((1, 3, 6), -1.0, dtype=np.float32)
    logits[0, 0, 1:3] = 1.0
    logits[0, 0, 4:6] = 1.0
    truth = (
        MaskComponent(0, 2, 12.0, 0.5, 0.0, (0, 0, 1, 0)),
        MaskComponent(1, 1, 6.0, 5.0, 2.0, (5, 2, 5, 2)),
    )
    prediction = (
        MaskComponent(0, 2, 12.0, 1.5, 0.0, (1, 0, 2, 0)),
        MaskComponent(1, 2, 12.0, 4.5, 0.0, (4, 0, 5, 0)),
    )
    local = LocalizationRow(
        truth,
        prediction,
        (ComponentMatch(0, 0, 1 / 3, 1.0, 0.0, 1.0, 2.0),),
        (1,),
        (1,),
        0,
        0,
        (2.0, 3.0),
        0.55,
        2,
        0.25,
    )
    boundary = BoundaryRow(
        3,
        4,
        2,
        2,
        1.0,
        "pixel",
        (6 + math.sqrt(5)) / 7,
        2 + 0.7 * (math.sqrt(5) - 2),
        22 / 7,
        6.0,
        None,
        None,
    )
    probability = 1 / (1 + math.exp(-1))
    values = (
        ("true_positive_pixels", 1.0),
        ("false_positive_pixels", 3.0),
        ("true_negative_pixels", 12.0),
        ("false_negative_pixels", 2.0),
        ("target_area_px", 3.0),
        ("predicted_area_px", 4.0),
        ("target_area_m2", 18.0),
        ("predicted_area_m2", 24.0),
        ("foreground_iou", 1 / 6),
        ("foreground_dice", 2 / 7),
        ("binary_cross_entropy", math.log1p(math.exp(-1)) + 5 / 18),
        ("brier_score", (13 * (1 - probability) ** 2 + 5 * probability**2) / 18),
        ("predicted_blobs", 2.0),
    )
    row = CaptureRow(
        key=SampleKey(
            dataset_hash="a" * 64,
            task="segmentor",
            split="val",
            spatial_shard=(3, 6),
            row_index=0,
            tile_id="frozen-toy-plumes",
            element="id",
        ),
        group_id="frozen-toy-group",
        bin_id="toy",
        label=1.0,
        gsd_m=(2.0, 3.0),
        metrics=tuple(
            MetricValue(
                name=name,
                value=value,
                status="AVAILABLE",
                support=MetricSupport(unit="IMAGE", n=1),
            )
            for name, value in values
        ),
        failure_score=5 / 6,
        spatial=SpatialRow(local, boundary),
    )
    stream = io.BytesIO()
    np.savez_compressed(stream, image=image, target=target, logits=logits)
    data = stream.getvalue()
    path = "previews/" + prediction_key(row) + ".npz"
    preview = PredictionPreview(
        row,
        path,
        hashlib.sha256(data).hexdigest(),
        len(data),
        (2, 1, 0),
        "RGB (RED, GREEN, BLUE)",
    )
    gallery = PredictionGallery(
        "val_worst",
        "worst",
        (row,),
        AvailabilityRecord(name="prediction_visual:val:worst", status="AVAILABLE"),
        "Frozen illustrative failure: shifted match, unfiltered tiny truth miss,"
        " retained spurious prediction",
    )
    return PredictionPreviewCapture((preview,), (BundleFile(path, data),), (gallery,))


def _render(captured: PredictionPreviewCapture) -> RenderedDatasetFigures:
    """Render the supplied capture and require success."""
    result = render_prediction_visuals(captured, PlotConfig(dpi=72, formats=("png",)))
    assert isinstance(result, Ok)
    return result.value


def _page_figure(captured: PredictionPreviewCapture) -> Figure:
    """Re-render the single gallery page to inspect its artists directly."""
    import tools.ml_models.analysis.visuals.predictions as module

    previews = {prediction_key(preview.row): preview for preview in captured.previews}
    bytes_by_path = {file.path: file.data for file in captured.files}
    loaded = {}
    for key, preview in previews.items():
        result = module._load(preview, bytes_by_path)
        assert isinstance(result, Ok)
        loaded[key] = result.value
    figures = module._gallery_figures(
        captured.galleries[0], previews, loaded, PlotConfig(dpi=72, formats=("png",))
    )
    assert isinstance(figures, Ok)
    return figures.value[0]


def _panels(figure: Figure) -> list[Axes]:
    """Return the visible panel axes of one rendered page."""
    return [axes for axes in figure.axes if axes.get_visible()]


def test_input_xlabel_wraps_complete_display_label() -> None:
    """The narrow input panel wraps the full display label without truncation."""
    captured = frozen_visual_fixture()
    preview = captured.previews[0]
    labeled = PredictionPreviewCapture(
        (replace(preview, display_label="Display channels pan, red, nir (not RGB)"),),
        captured.files,
        captured.galleries,
    )
    figure = _page_figure(labeled)
    import matplotlib.pyplot as plt

    try:
        xlabel = _panels(figure)[0].get_xlabel()
        assert xlabel.replace("\n", " ") == "Display channels pan, red, nir (not RGB)"
        assert all(len(line) <= 28 for line in xlabel.split("\n"))
    finally:
        plt.close(figure)


def test_extent_panels_copy_frozen_matches_and_counts() -> None:
    captured = frozen_visual_fixture()
    rendered = _render(captured)
    assert {file.path for file in rendered.files} == {"visuals/predictions/val_worst_1.png"}
    output = rendered.outputs[0]
    assert output.status == "AVAILABLE"
    figure = _page_figure(captured)
    try:
        panels = _panels(figure)
        assert len(panels) >= 6
        titles = [axes.get_title() for axes in panels]
        assert "explicit truth" in titles
        assert any(
            title.replace("\n", " ").startswith("raw binary mask p>=0.5") for title in titles
        )
        assert any("pixel errors" in title for title in titles)
        assert any("frozen component" in title for title in titles)
        overlay = panels[5]
        rectangles = [patch for patch in overlay.patches if isinstance(patch, Rectangle)]
        assert len(rectangles) == 4
        bounds = sorted(
            (rect.get_x(), rect.get_y(), rect.get_width(), rect.get_height()) for rect in rectangles
        )
        assert bounds == [
            (-0.5, -0.5, 2.0, 1.0),
            (0.5, -0.5, 2.0, 1.0),
            (3.5, -0.5, 2.0, 1.0),
            (4.5, 1.5, 1.0, 1.0),
        ]
        labels = sorted(text.get_text() for text in overlay.texts)
        assert labels == ["P0", "P1", "T0", "T1"]
        legend = overlay.get_legend()
        assert legend is not None
        legend_text = "\n".join(text.get_text() for text in legend.get_texts())
        assert "truth" in legend_text and "prediction" in legend_text
        assert "match" in legend_text and "unmatched" in legend_text
        anchor = legend.get_bbox_to_anchor()
        assert 0 <= anchor.ymin < overlay.bbox.y0
        assert len(legend.get_texts()) == 4
        matched = [
            line
            for line in overlay.lines
            if line.get_marker() == "None" and len(np.asarray(line.get_xdata(), dtype=float)) == 2
        ]
        assert len(matched) == 1
        np.testing.assert_allclose(np.asarray(matched[0].get_xdata(), dtype=float), [0.5, 1.5])
        np.testing.assert_allclose(np.asarray(matched[0].get_ydata(), dtype=float), [0.0, 0.0])
        unmatched = [line for line in overlay.lines if line.get_marker() == "x"]
        assert len(unmatched) == 2
        footer = "\n".join(text.get_text() for text in figure.texts)
        flat = "".join(footer.split())
        assert "matched 1" in footer
        assert "missed truth 1" in footer
        assert "unmatched pred 1" in footer
        assert "raw-maskp>=0.5" in flat
        assert "blobp>=0.55" in flat
        assert "minarea2" in flat
        assert "matchIoU>=0.25" in flat
        assert "gsd2x3m" in flat
        assert "conditional" in flat and "misses" in flat
        assert "val_worst" in flat and "worst" in flat
        assert "frozen-toy-plumes" in flat
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def test_input_panel_uses_captured_semantic_rgb_mapping() -> None:
    captured = frozen_visual_fixture()
    figure = _page_figure(captured)
    try:
        panels = _panels(figure)
        image = panels[0].images[0].get_array()
        assert image is not None
        assert image.shape == (3, 6, 3)
        np.testing.assert_allclose(image[..., 0], 0.9, atol=1e-6)
        np.testing.assert_allclose(image[..., 1], 0.5, atol=1e-6)
        np.testing.assert_allclose(image[..., 2], 0.1, atol=1e-6)
        assert panels[0].get_title() == "Input"
        assert "RGB (RED, GREEN, BLUE)" in panels[0].get_xlabel()
        footer = "\n".join(text.get_text() for text in figure.texts)
        assert "frozen-toy-plumes" in "".join(footer.split())
        colorbars = [axes for axes in figure.axes if "probability" in axes.get_ylabel()]
        assert colorbars
        truth_image = panels[1].images[0]
        assert truth_image.get_clim() == (0.0, 1.0)
        probability_image = panels[2].images[0]
        assert probability_image.get_clim() == (0.0, 1.0)
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def test_pixel_panels_match_exact_saved_logits() -> None:
    captured = frozen_visual_fixture()
    figure = _page_figure(captured)
    try:
        panels = _panels(figure)
        truth = panels[1].images[0].get_array()
        predicted = panels[3].images[0].get_array()
        errors = panels[4].images[0].get_array()
        assert truth is not None and predicted is not None and errors is not None
        expected_truth = np.zeros((3, 6))
        expected_truth[0, :2] = 1.0
        expected_truth[2, 5] = 1.0
        expected_pred = np.zeros((3, 6))
        expected_pred[0, 1:3] = 1.0
        expected_pred[0, 4:6] = 1.0
        np.testing.assert_array_equal(truth, expected_truth)
        np.testing.assert_array_equal(predicted, expected_pred)
        fp = np.all(errors == (0.85, 0.1, 0.1), axis=-1)
        fn = np.all(errors == (0.1, 0.3, 0.9), axis=-1)
        assert np.count_nonzero(fp) == 3
        assert np.count_nonzero(fn) == 2
        assert fp[0, 2] and fp[0, 4] and fp[0, 5]
        assert fn[0, 0] and fn[2, 5]
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)


def test_mutated_logits_disagreeing_with_frozen_counts_fail() -> None:
    captured = frozen_visual_fixture()
    stream = io.BytesIO()
    image = np.empty((3, 3, 6), dtype=np.float32)
    image[0], image[1], image[2] = 0.1, 0.5, 0.9
    target = np.zeros((1, 3, 6), dtype=np.float32)
    target[0, 0, :2] = 1.0
    target[0, 2, 5] = 1.0
    logits = np.full((1, 3, 6), -1.0, dtype=np.float32)
    logits[0, 0, 1:3] = 1.0
    logits[0, 1, 0] = 1.0
    np.savez(stream, image=image, target=target, logits=logits)
    data = stream.getvalue()
    preview = replace(
        captured.previews[0],
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
    )
    mutated = PredictionPreviewCapture(
        (preview,), (BundleFile(preview.path, data),), captured.galleries
    )
    assert isinstance(render_prediction_visuals(mutated, PlotConfig(dpi=72, formats=("png",))), Err)


@pytest.mark.parametrize("dtype", [np.float64])
def test_segmentor_cache_rejects_non_float32(dtype: np.dtype) -> None:
    captured = frozen_visual_fixture()
    stream = io.BytesIO()
    image = np.empty((3, 3, 6), dtype=np.float32)
    image[0], image[1], image[2] = 0.1, 0.5, 0.9
    target = np.zeros((1, 3, 6), dtype=dtype)
    logits = np.full((1, 3, 6), -1.0, dtype=dtype)
    np.savez(stream, image=image, target=target, logits=logits)
    data = stream.getvalue()
    preview = replace(
        captured.previews[0],
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
    )
    mutated = PredictionPreviewCapture(
        (preview,), (BundleFile(preview.path, data),), captured.galleries
    )
    assert isinstance(render_prediction_visuals(mutated, PlotConfig(dpi=72, formats=("png",))), Err)


def test_wrong_shape_segmentor_arrays_fail() -> None:
    captured = frozen_visual_fixture()
    stream = io.BytesIO()
    image = np.empty((3, 3, 6), dtype=np.float32)
    image[0], image[1], image[2] = 0.1, 0.5, 0.9
    target = np.zeros((3, 6), dtype=np.float32)
    logits = np.full((3, 6), -1.0, dtype=np.float32)
    np.savez(stream, image=image, target=target, logits=logits)
    data = stream.getvalue()
    preview = replace(
        captured.previews[0],
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
    )
    mutated = PredictionPreviewCapture(
        (preview,), (BundleFile(preview.path, data),), captured.galleries
    )
    assert isinstance(render_prediction_visuals(mutated, PlotConfig(dpi=72, formats=("png",))), Err)


def test_each_segmentor_row_occupies_one_page() -> None:
    captured = frozen_visual_fixture()
    first = captured.previews[0].row
    second = replace(
        first,
        key=replace(first.key, row_index=1, tile_id="frozen-toy-plumes-2"),
    )
    data = captured.files[0].data
    path = "previews/" + prediction_key(second) + ".npz"
    preview = replace(captured.previews[0], row=second, path=path)
    gallery = PredictionGallery(
        "val_all",
        "all",
        (first, second),
        AvailabilityRecord(name="prediction_visual:val:all", status="AVAILABLE"),
        "two frozen examples",
    )
    rendered = _render(
        PredictionPreviewCapture(
            (captured.previews[0], preview),
            (captured.files[0], BundleFile(path, data)),
            (gallery,),
        )
    )
    assert {file.path for file in rendered.files} == {
        "visuals/predictions/val_all_1.png",
        "visuals/predictions/val_all_2.png",
    }


def test_omitted_and_empty_segmentor_galleries_render_placeholders() -> None:
    captured = frozen_visual_fixture()
    gallery = captured.galleries[0]
    omitted = PredictionPreviewCapture((), (), (gallery,))
    rendered = render_prediction_visuals(omitted, PlotConfig(dpi=72, formats=("png",)))
    assert isinstance(rendered, Ok)
    output = rendered.value.outputs[0]
    assert output.status == "UNAVAILABLE"
    assert output.reason == "Chosen prediction previews were omitted by the capture budget"
    assert {file.path for file in rendered.value.files} == {"visuals/predictions/val_worst_1.png"}
    empty = replace(
        gallery,
        rows=(),
        availability=AvailabilityRecord(
            name="prediction_visual:val:worst",
            status="UNAVAILABLE",
            reason="No extent-error examples",
        ),
    )
    rendered = render_prediction_visuals(
        PredictionPreviewCapture((), (), (empty,)), PlotConfig(dpi=72, formats=("png",))
    )
    assert isinstance(rendered, Ok)
    output = rendered.value.outputs[0]
    assert output.status == "UNAVAILABLE"
    assert output.reason == "No extent-error examples"


def test_mixed_task_cohort_is_refused() -> None:
    captured = frozen_visual_fixture()
    seg = captured.previews[0].row
    classifier = replace(
        seg,
        key=replace(seg.key, task="classifier", row_index=9),
        spatial=None,
    )
    gallery = replace(
        captured.galleries[0],
        rows=(seg, classifier),
    )
    assert isinstance(
        render_prediction_visuals(
            replace(captured, galleries=(gallery,)), PlotConfig(dpi=72, formats=("png",))
        ),
        Err,
    )


def test_dangling_and_off_image_component_geometry_fail() -> None:
    captured = frozen_visual_fixture()
    row = captured.previews[0].row
    spatial = row.spatial
    assert spatial is not None
    dangling = replace(
        row,
        spatial=replace(
            spatial,
            localization=replace(
                spatial.localization,
                matches=(ComponentMatch(0, 9, 0.5, 0.0, 0.0, 0.0, None),),
            ),
        ),
    )

    local = spatial.localization
    bad_locals = (
        replace(
            local,
            truth=(
                MaskComponent(0, 2, 12.0, 0.5, 0.0, (0, 0, 6, 0)),
                local.truth[1],
            ),
        ),
        replace(
            local,
            truth=(
                MaskComponent(0, 2, 12.0, math.nan, 0.0, (0, 0, 1, 0)),
                local.truth[1],
            ),
        ),
        replace(
            local,
            truth=(
                local.truth[0],
                replace(local.truth[1], component_id=0),
            ),
        ),
        replace(
            local,
            matches=(
                local.matches[0],
                ComponentMatch(0, 1, 0.5, 0.0, 0.0, 0.0, None),
            ),
        ),
        replace(local, unmatched_truth=(0, 1)),
        replace(local, unmatched_truth=()),
        replace(
            local,
            matches=(replace(local.matches[0], distance_px=math.nan),),
        ),
        replace(local, unmatched_prediction=(9,)),
        replace(
            local,
            predicted=(
                MaskComponent(0, 9, 12.0, 1.5, 0.0, (1, 0, 2, 0)),
                local.predicted[1],
            ),
        ),
        replace(local, gsd_m=(math.nan, 3.0)),
        replace(local, match_iou_min=math.inf),
    )
    for bad_row in (
        dangling,
        *(replace(row, spatial=replace(spatial, localization=bad)) for bad in bad_locals),
    ):
        gallery = replace(captured.galleries[0], rows=(bad_row,))
        preview = replace(captured.previews[0], row=bad_row)
        result = render_prediction_visuals(
            PredictionPreviewCapture((preview,), captured.files, (gallery,)),
            PlotConfig(dpi=72, formats=("png",)),
        )
        assert isinstance(result, Err)


def test_known_empty_masks_render_without_errors() -> None:
    captured = frozen_visual_fixture()
    row = captured.previews[0].row
    spatial = row.spatial
    assert spatial is not None
    empty_spatial = replace(
        spatial,
        localization=replace(
            spatial.localization,
            truth=(),
            predicted=(),
            matches=(),
            unmatched_truth=(),
            unmatched_prediction=(),
        ),
        boundary=replace(
            spatial.boundary,
            truth_boundary_pixels=0,
            predicted_boundary_pixels=0,
            truth_hits=0,
            predicted_hits=0,
            asd_px=None,
            hd95_px=None,
            asd_m=None,
            hd95_m=None,
            distance_reason="both masks empty",
            ground_distance_reason="both masks empty",
        ),
    )
    values = (
        ("true_positive_pixels", 0.0),
        ("false_positive_pixels", 0.0),
        ("true_negative_pixels", 18.0),
        ("false_negative_pixels", 0.0),
        ("target_area_px", 0.0),
        ("predicted_area_px", 0.0),
    )
    empty_row = replace(
        row,
        metrics=tuple(
            MetricValue(
                name=name,
                value=value,
                status="AVAILABLE",
                support=MetricSupport(unit="IMAGE", n=1),
            )
            for name, value in values
        ),
        spatial=empty_spatial,
    )
    image = np.empty((3, 3, 6), dtype=np.float32)
    image[0], image[1], image[2] = 0.1, 0.5, 0.9
    target = np.zeros((1, 3, 6), dtype=np.float32)
    logits = np.full((1, 3, 6), -1.0, dtype=np.float32)
    stream = io.BytesIO()
    np.savez(stream, image=image, target=target, logits=logits)
    data = stream.getvalue()
    preview = replace(
        captured.previews[0],
        row=empty_row,
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
    )
    gallery = replace(captured.galleries[0], rows=(empty_row,))
    rendered = _render(
        PredictionPreviewCapture((preview,), (BundleFile(preview.path, data),), (gallery,))
    )
    assert rendered.outputs[0].status == "AVAILABLE"


def test_threshold_endpoint_and_sigmoid_saturation_come_from_helper() -> None:
    captured = frozen_visual_fixture()
    row = captured.previews[0].row
    logits = np.full((1, 3, 6), -1.0, dtype=np.float32)
    logits[0, 0, 1:3] = 1.0
    logits[0, 0, 4:6] = 1.0
    logits[0, 0, 3] = 0.0
    logits[0, 1, 0] = 88.0
    target = np.zeros((1, 3, 6), dtype=np.float32)
    target[0, 0, :2] = 1.0
    target[0, 2, 5] = 1.0
    display = segmentation_display_data(
        replace(
            row,
            metrics=tuple(
                replace(metric, value=value)
                for metric, value in zip(
                    row.metrics,
                    (1.0, 5.0, 10.0, 2.0, 3.0, 6.0)
                    + tuple(metric.value for metric in row.metrics[6:]),
                    strict=True,
                )
            ),
        ),
        target,
        logits,
    )
    assert isinstance(display, Ok)
    assert display.value.predicted[0, 3]
    assert display.value.probability[1, 0] == 1.0


def test_rendering_never_invokes_scoring_or_matching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _bomb(*args: object, **kwargs: object) -> None:
        raise AssertionError("renderer re-ran scoring or matching")

    import tools.ml_models.analysis.metrics.localization as localization
    import tools.ml_models.analysis.metrics.spatial as spatial

    monkeypatch.setattr(spatial, "score_spatial", _bomb)
    monkeypatch.setattr(spatial, "aggregate_spatial", _bomb)
    monkeypatch.setattr(localization, "match_components", _bomb)
    monkeypatch.setattr(localization, "score_localization", _bomb)
    assert isinstance(
        render_prediction_visuals(frozen_visual_fixture(), PlotConfig(dpi=72, formats=("png",))),
        Ok,
    )
