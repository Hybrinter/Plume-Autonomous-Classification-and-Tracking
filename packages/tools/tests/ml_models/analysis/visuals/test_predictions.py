"""Prediction visual rendering checks over immutable verified preview bytes."""

import hashlib
import io
from collections.abc import Callable
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from flight.libs.types import Err, Ok, Result
from tools.ml_models.analysis.artifacts import BundleFile
from tools.ml_models.analysis.capture import CaptureRow
from tools.ml_models.analysis.config import PlotConfig
from tools.ml_models.analysis.contracts import (
    ArtifactKind,
    AvailabilityRecord,
    MetricSupport,
    MetricValue,
    SampleKey,
    SplitEvidence,
)
from tools.ml_models.analysis.plots.dataset import RenderedDatasetFigures
from tools.ml_models.analysis.prediction_selections import (
    PredictionGallery,
    prediction_key,
)
from tools.ml_models.analysis.visuals.predictions import (
    LoadedPrediction,
    PredictionPreview,
    PredictionPreviewCapture,
    render_prediction_visuals,
)

_HASH = "a" * 64
_MANIFEST = "b" * 64


def _metric(name: str, value: float | None) -> MetricValue:
    return MetricValue(
        name=name,
        value=value,
        status="AVAILABLE" if value is not None else "UNAVAILABLE",
        reason=None if value is not None else "missing",
        support=MetricSupport(unit="IMAGE", n=1),
    )


def _rows(count: int = 4) -> tuple[CaptureRow, ...]:
    rows: list[CaptureRow] = []
    for index in range(count):
        label = float(index % 2)
        fp = index % 2 == 0 and index > 0
        loss = 0.1 + index
        rows.append(
            CaptureRow(
                key=SampleKey(
                    dataset_hash=_HASH,
                    task="classifier",
                    split="val",
                    spatial_shard=(8, 8),
                    row_index=index,
                    tile_id=f"tile-{index}",
                    element="I",
                ),
                group_id=str(index),
                bin_id="bin",
                label=label,
                gsd_m=(2.0, 3.0),
                metrics=(
                    _metric("probability", 0.2 + 0.1 * index),
                    _metric("logit", -1.0 + index),
                    _metric("binary_cross_entropy", loss),
                ),
                failure_score=loss,
                false_positive=fp,
                false_negative=index % 2 == 1 and index > 1,
                dataset_manifest_hash=_MANIFEST,
            )
        )
    return tuple(rows)


def _evidence(n: int = 4) -> SplitEvidence:
    return SplitEvidence(
        task="classifier",
        split="val",
        dataset_hash=_HASH,
        dataset_manifest_hash=_MANIFEST,
        checkpoint_hash="c" * 64,
        support=MetricSupport(unit="IMAGE", n=n),
    )


def _preview_bytes(
    row: CaptureRow,
    *,
    image: np.ndarray | None = None,
    target: np.ndarray | None = None,
    logits: np.ndarray | None = None,
) -> bytes:
    if image is None:
        gradient = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
        image = np.stack((gradient, 1 - gradient, gradient * 0.5))
    if target is None:
        target = np.asarray([row.label], dtype=np.float32)
    if logits is None:
        logit = next(m.value for m in row.metrics if m.name == "logit")
        logits = np.asarray([logit], dtype=np.float32)
    stream = io.BytesIO()
    np.savez(stream, image=image, target=target, logits=logits)
    return stream.getvalue()


def _preview(row: CaptureRow, data: bytes | None = None) -> tuple[PredictionPreview, BundleFile]:
    if data is None:
        data = _preview_bytes(row)
    path = "previews/" + prediction_key(row) + ".npz"
    return (
        PredictionPreview(
            row=row,
            path=path,
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
            display_indices=(0, 1, 2),
            display_label="RGB (RED, GREEN, BLUE)",
        ),
        BundleFile(path=path, data=data),
    )


def _gallery(rows: tuple[CaptureRow, ...], family: str = "false_positive") -> PredictionGallery:
    return PredictionGallery(
        identifier="val_" + family,
        family=family,
        rows=rows,
        availability=AvailabilityRecord(
            name="prediction_visual:val:" + family,
            status="AVAILABLE",
        ),
        selection_method="Whole-cohort false positives, descending captured failure score",
    )


def _capture(
    rows: tuple[CaptureRow, ...], galleries: tuple[PredictionGallery, ...] | None = None
) -> PredictionPreviewCapture:
    pairs = tuple(_preview(row) for row in rows)
    if galleries is None:
        galleries = (_gallery(rows),)
    return PredictionPreviewCapture(
        previews=tuple(pair[0] for pair in pairs),
        files=tuple(pair[1] for pair in pairs),
        galleries=galleries,
    )


def _render(captured: PredictionPreviewCapture) -> RenderedDatasetFigures:
    result = render_prediction_visuals(captured, PlotConfig(formats=("png",), dpi=72))
    assert isinstance(result, Ok)
    return result.value


def test_available_gallery_exports_pages_and_selected_ids_unchanged() -> None:
    rows = _rows()
    gallery = _gallery((rows[1], rows[3]))
    rendered = _render(_capture(rows, (gallery,)))
    assert {file.path for file in rendered.files} == {
        "visuals/predictions/val_false_positive_1.png"
    }
    output = next(o for o in rendered.outputs if o.name == "prediction_visual:val:false_positive")
    assert output.status == "AVAILABLE"
    assert gallery.rows == (rows[1], rows[3])


def test_pages_are_bounded_to_four_panels() -> None:
    rows = _rows(6)
    gallery = _gallery(rows, family="representative")
    rendered = _render(_capture(rows, (gallery,)))
    paths = {file.path for file in rendered.files}
    assert paths == {
        "visuals/predictions/val_representative_1.png",
        "visuals/predictions/val_representative_2.png",
    }


def test_panel_text_shows_recorded_prediction_context() -> None:
    rows = _rows()
    gallery = _gallery((rows[0],))
    import tools.ml_models.analysis.visuals.predictions as module

    captured = _capture(rows, (gallery,))
    previews = {prediction_key(p.row): p for p in captured.previews}
    loaded = {}
    for key, preview in previews.items():
        loaded_result = module._load(preview, {f.path: f.data for f in captured.files})
        assert isinstance(loaded_result, Ok)
        loaded[key] = loaded_result.value
    result = module._gallery_figures(gallery, previews, loaded, PlotConfig(dpi=72))
    assert isinstance(result, Ok)
    figure = result.value[0]
    text = (
        "\n".join(text.get_text() for axes in figure.axes for text in axes.texts)
        + "\n"
        + "\n".join(axes.get_title() for axes in figure.axes)
        + "\n"
        + figure.get_suptitle()
    )
    assert "truth=0" in text and "pred=0" in text
    assert "p=0.200" in text and "BCE=0.1" in text
    assert "tile-0" in text
    assert "Whole-cohort false positives" in figure.get_suptitle()
    plt.close(figure)


def test_missing_chosen_preview_renders_omitted_panel_and_unavailable() -> None:
    rows = _rows()
    gallery = _gallery((rows[0], rows[2]))
    captured = _capture(rows, (gallery,))
    captured = replace(
        captured,
        previews=tuple(p for p in captured.previews if p.row != rows[2]),
        files=tuple(f for f in captured.files if prediction_key(rows[2]) not in f.path),
    )
    rendered = _render(captured)
    output = next(o for o in rendered.outputs if o.name == "prediction_visual:val:false_positive")
    assert output.status == "UNAVAILABLE"
    assert output.reason == "Chosen prediction previews were omitted by the capture budget"
    assert gallery.rows == (rows[0], rows[2])


def test_skipped_and_unavailable_families_keep_supplied_state() -> None:
    rows = _rows()
    skipped = replace(
        _gallery(()),
        availability=AvailabilityRecord(
            name="prediction_visual:val:false_positive",
            status="SKIPPED",
            reason="Prediction preview capture disabled",
        ),
    )
    rendered = _render(_capture(rows, (skipped,)))
    output = rendered.outputs[0]
    assert output.status == "SKIPPED"
    assert output.reason == "Prediction preview capture disabled"
    assert any(file.path.startswith("visuals/predictions/") for file in rendered.files)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda row: _preview_bytes(row).replace(b"PK", b"OK", 1),
        lambda row: _preview_bytes(row, image=np.zeros((3, 8, 8), dtype=np.float64)),
        lambda row: _preview_bytes(row, logits=np.asarray([99.0], dtype=np.float32)),
        lambda row: _preview_bytes(row, target=np.asarray([2.0], dtype=np.float32)),
        lambda row: _preview_bytes(row, image=np.zeros((3, 6, 6), dtype=np.float32)),
    ),
)
def test_corrupt_or_mismatched_preview_bytes_fail_closed(
    mutate: Callable[[CaptureRow], bytes],
) -> None:
    rows = _rows()
    row = rows[0]
    data = mutate(row)
    preview = PredictionPreview(
        row=row,
        path="previews/" + prediction_key(row) + ".npz",
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        display_indices=(0, 1, 2),
        display_label="RGB (RED, GREEN, BLUE)",
    )
    captured = PredictionPreviewCapture(
        previews=(preview,),
        files=(BundleFile(preview.path, data),),
        galleries=(_gallery((row,)),),
    )
    assert isinstance(render_prediction_visuals(captured, PlotConfig(dpi=72)), Err)


def test_checksum_mismatch_fails_before_decoding() -> None:
    rows = _rows()
    row = rows[0]
    data = _preview_bytes(row)
    preview = PredictionPreview(
        row=row,
        path="previews/" + prediction_key(row) + ".npz",
        sha256="0" * 64,
        size_bytes=len(data),
        display_indices=(0, 1, 2),
        display_label="RGB (RED, GREEN, BLUE)",
    )
    captured = PredictionPreviewCapture(
        previews=(preview,),
        files=(BundleFile(preview.path, data),),
        galleries=(_gallery((row,)),),
    )
    result = render_prediction_visuals(captured, PlotConfig(dpi=72))
    assert isinstance(result, Err)
    assert "checksum" in result.error


def test_float32_stored_logit_matches_exact_captured_python_float() -> None:
    rows = _rows()
    row = replace(
        rows[0],
        metrics=tuple(
            replace(metric, value=float(np.float32(-0.4))) if metric.name == "logit" else metric
            for metric in rows[0].metrics
        ),
    )
    data = _preview_bytes(row, logits=np.asarray([-0.4], dtype=np.float32))
    preview = PredictionPreview(
        row=row,
        path="previews/" + prediction_key(row) + ".npz",
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        display_indices=(0, 1, 2),
        display_label="RGB (RED, GREEN, BLUE)",
    )
    captured = PredictionPreviewCapture(
        previews=(preview,),
        files=(BundleFile(preview.path, data),),
        galleries=(_gallery((row,)),),
    )
    assert isinstance(render_prediction_visuals(captured, PlotConfig(dpi=72)), Ok)


def test_rounding_cannot_hide_mutated_logit_evidence() -> None:
    row = _rows()[0]
    recorded = float(np.nextafter(-1.0, 0.0))
    altered = replace(
        row,
        metrics=tuple(
            replace(metric, value=recorded) if metric.name == "logit" else metric
            for metric in row.metrics
        ),
    )
    preview, file = _preview(altered, _preview_bytes(row))
    captured = PredictionPreviewCapture((preview,), (file,), (_gallery((altered,)),))
    assert isinstance(render_prediction_visuals(captured, PlotConfig(dpi=72)), Err)


@pytest.mark.parametrize("entry", ("target", "logits"))
def test_non_float32_target_or_logit_is_not_coerced(entry: str) -> None:
    row = _rows()[0]
    data = _preview_bytes(
        row,
        target=np.asarray([row.label], dtype=np.float64) if entry == "target" else None,
        logits=np.asarray([-1.0], dtype=np.float64) if entry == "logits" else None,
    )
    preview, file = _preview(row, data)
    captured = PredictionPreviewCapture((preview,), (file,), (_gallery((row,)),))
    assert isinstance(render_prediction_visuals(captured, PlotConfig(dpi=72)), Err)


def test_out_of_bounds_display_indices_rejected() -> None:
    rows = _rows()
    preview, file = _preview(rows[0])
    preview = replace(preview, display_indices=(0, 1, 9))
    captured = PredictionPreviewCapture(
        previews=(preview,),
        files=(file,),
        galleries=(_gallery((rows[0],)),),
    )
    assert isinstance(render_prediction_visuals(captured, PlotConfig(dpi=72)), Err)


def test_renderer_never_reselects_or_reads_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    def _bomb(*args: object, **kwargs: object) -> None:
        raise AssertionError("renderer touched capture/selection helpers")

    import tools.ml_models.analysis.dataset_previews as previews_module
    import tools.ml_models.analysis.prediction_selections as selections

    monkeypatch.setattr(previews_module, "capture_dataset_previews", _bomb)
    monkeypatch.setattr(selections, "prediction_gallery_data", _bomb)
    assert isinstance(render_prediction_visuals(_capture(_rows()), PlotConfig(dpi=72)), Ok)


def test_duplicate_paths_and_gallery_names_rejected() -> None:
    rows = _rows()
    captured = _capture(rows)
    doubled = replace(captured, files=captured.files + captured.files)
    assert isinstance(render_prediction_visuals(doubled, PlotConfig(dpi=72)), Err)
    gallery = _gallery((rows[0],))
    doubled = replace(captured, galleries=(gallery, gallery))
    assert isinstance(render_prediction_visuals(doubled, PlotConfig(dpi=72)), Err)
    shadow = replace(gallery, identifier="val_other")
    doubled = replace(captured, galleries=(gallery, shadow))
    assert isinstance(render_prediction_visuals(doubled, PlotConfig(dpi=72)), Err)


def test_unsafe_gallery_and_preview_tokens_rejected() -> None:
    rows = _rows()
    captured = _capture(rows)
    unsafe = replace(_gallery((rows[0],)), identifier="a/b")
    assert isinstance(
        render_prediction_visuals(replace(captured, galleries=(unsafe,)), PlotConfig(dpi=72)),
        Err,
    )
    preview, file = _preview(rows[0])
    preview = replace(preview, path="../escape.npz")
    unsafe_capture = PredictionPreviewCapture(
        previews=(preview,), files=(file,), galleries=(_gallery((rows[0],)),)
    )
    assert isinstance(render_prediction_visuals(unsafe_capture, PlotConfig(dpi=72)), Err)


def test_same_key_preview_with_different_scalars_fails() -> None:
    rows = _rows()
    preview, file = _preview(rows[0])
    altered = replace(
        rows[0],
        metrics=tuple(
            replace(metric, value=metric.value + 0.25) if metric.value else metric
            for metric in rows[0].metrics
        ),
    )
    captured = PredictionPreviewCapture(
        previews=(preview,), files=(file,), galleries=(_gallery((altered,)),)
    )
    result = render_prediction_visuals(captured, PlotConfig(dpi=72))
    assert isinstance(result, Err)
    assert "different captured scalars" in result.error


def test_export_failure_closes_all_created_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    import matplotlib.figure
    import tools.ml_models.analysis.plots.common as common
    import tools.ml_models.analysis.visuals.predictions as module

    created: list[matplotlib.figure.Figure] = []
    closed: list[object] = []
    real_figure = matplotlib.figure.Figure

    def _tracking_figure(
        figsize: tuple[float, float] | None = None, dpi: float | None = None
    ) -> matplotlib.figure.Figure:
        figure = real_figure(figsize=figsize, dpi=dpi)
        created.append(figure)
        return figure

    real_close = plt.close

    def _recording_close(figure: matplotlib.figure.Figure | None = None) -> None:
        closed.append(figure)
        real_close(figure)

    real_export = common.export_figure
    calls: list[str] = []

    def _flaky_export(
        figure: matplotlib.figure.Figure,
        identifier: str,
        cfg: PlotConfig,
        *,
        kind: ArtifactKind = "FIGURE",
        population: str | None = None,
    ) -> Result[tuple[BundleFile, ...], str]:
        calls.append(identifier)
        if len(calls) == 2:
            return Err("forced export failure")
        return real_export(figure, identifier, cfg, kind=kind, population=population)

    monkeypatch.setattr(matplotlib.figure, "Figure", _tracking_figure)
    monkeypatch.setattr(plt, "close", _recording_close)
    monkeypatch.setattr(module, "export_figure", _flaky_export)
    rows = _rows(6)
    gallery = _gallery(rows, family="representative")
    result = render_prediction_visuals(_capture(rows, (gallery,)), PlotConfig(dpi=72))
    assert isinstance(result, Err)
    assert len(created) == 2
    assert all(figure in closed for figure in created)


def test_panel_failure_closes_the_created_page(monkeypatch: pytest.MonkeyPatch) -> None:
    import matplotlib.figure
    import tools.ml_models.analysis.visuals.predictions as module
    from matplotlib.axes import Axes

    created: list[matplotlib.figure.Figure] = []
    closed: list[object] = []
    real_figure = matplotlib.figure.Figure

    def _tracking_figure(
        figsize: tuple[float, float] | None = None, dpi: float | None = None
    ) -> matplotlib.figure.Figure:
        figure = real_figure(figsize=figsize, dpi=dpi)
        created.append(figure)
        return figure

    real_close = plt.close

    def _recording_close(figure: matplotlib.figure.Figure | None = None) -> None:
        closed.append(figure)
        real_close(figure)

    real_panel = module._panel
    panels: list[int] = []

    def _failing_panel(
        axes: Axes,
        preview: PredictionPreview | None,
        loaded: LoadedPrediction | None,
        row: CaptureRow,
        cfg: PlotConfig,
    ) -> None:
        panels.append(1)
        if len(panels) == 5:
            raise RuntimeError("forced panel failure")
        real_panel(axes, preview, loaded, row, cfg)

    monkeypatch.setattr(matplotlib.figure, "Figure", _tracking_figure)
    monkeypatch.setattr(plt, "close", _recording_close)
    monkeypatch.setattr(module, "_panel", _failing_panel)
    rows = _rows(6)
    gallery = _gallery(rows, family="representative")
    result = render_prediction_visuals(_capture(rows, (gallery,)), PlotConfig(dpi=72))
    assert isinstance(result, Err)
    assert len(created) == 2
    assert all(figure in closed for figure in created)


def test_figures_are_closed() -> None:
    assert plt.get_fignums() == []
    _render(_capture(_rows()))
    assert plt.get_fignums() == []
    rows = _rows()
    bad = PredictionPreviewCapture(
        previews=(),
        files=(),
        galleries=(_gallery((rows[0],)),),
    )
    assert isinstance(render_prediction_visuals(bad, PlotConfig(dpi=72)), Ok)
    assert plt.get_fignums() == []
