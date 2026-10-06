"""Prediction-overlay visuals over captured evaluation evidence.

Renders the fixed ``PredictionGallery`` selections against immutable
preview bytes supplied by capture. Every chosen panel verifies the NPZ
checksum and size before ``np.load`` on an in-memory buffer
(``allow_pickle=False``), then requires the exact canonical float32
unit image and recorded targets/logits - no coercion, resize,
normalization, substitution, or re-selection. Classifier rows keep
shape-(1,) float32 targets and the exact Python float of the captured
float32 raw logit; segmentor rows take shape-(1,H,W) float32 masks and
logits through the frozen ``segmentation_display_data`` helper, whose
pixel counts must agree with captured metrics. A preview bound to the
same key but carrying different scalars is refused rather than shown
beside the chosen row. Chosen rows whose preview was omitted by the
capture budget render an explicit omitted panel and mark the gallery
output ``UNAVAILABLE``; supplied ``SKIPPED``/``UNAVAILABLE`` family
states are preserved. Classifier pages hold at most four examples;
segmentor pages hold exactly one example in a fixed extent-panel grid.

Contains:
  - PredictionPreview, PredictionPreviewCapture: bound preview records.
  - LoadedPrediction: verified cache arrays plus frozen display data.
  - render_prediction_visuals: export gallery pages into bundle bytes.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import hashlib
import io
import math
import textwrap
import zipfile
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from flight.libs.types import Err, Ok, Result

from tools.ml_models.analysis.artifacts import BundleFile
from tools.ml_models.analysis.capture import CaptureRow
from tools.ml_models.analysis.config import PlotConfig
from tools.ml_models.analysis.contracts import AvailabilityRecord, check_bundle_path
from tools.ml_models.analysis.metrics.localization import LocalizationRow
from tools.ml_models.analysis.model_figures import captured_values
from tools.ml_models.analysis.plots.common import export_figure
from tools.ml_models.analysis.plots.dataset import RenderedDatasetFigures
from tools.ml_models.analysis.prediction_display import (
    SegmentationDisplay,
    segmentation_display_data,
)
from tools.ml_models.analysis.prediction_selections import (
    PredictionGallery,
    prediction_key,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_PAGE_SIZE = 4


@dataclass(frozen=True, slots=True)
class PredictionPreview:
    """Captured preview identity and display mapping bound to one scalar row."""

    row: CaptureRow
    path: str
    sha256: str
    size_bytes: int
    display_indices: tuple[int, ...]
    display_label: str


@dataclass(frozen=True, slots=True)
class PredictionPreviewCapture:
    """Immutable preview files paired with the frozen gallery selections."""

    previews: tuple[PredictionPreview, ...]
    files: tuple[BundleFile, ...]
    galleries: tuple[PredictionGallery, ...]


@dataclass(frozen=True, slots=True)
class LoadedPrediction:
    """Verified cache arrays plus the frozen segmentor display when present."""

    image: npt.NDArray[np.float32]
    target: npt.NDArray[np.float32]
    logits: npt.NDArray[np.float32]
    segmentation: SegmentationDisplay | None


def _wrap(text: str, cfg: PlotConfig) -> str:
    """Wrap caption text to a width bounded by the configured figure size."""
    return textwrap.fill(text, width=max(20, int(float(cfg.width_inches) * 11)))


def _safe_token(value: str, what: str) -> Result[None, str]:
    """Require a single safe path token."""
    if not value or "/" in value or "\\" in value or value in (".", ".."):
        return Err(f"{what} must be a single safe token")
    return Ok(None)


def _load(
    preview: PredictionPreview, bytes_by_path: dict[str, bytes]
) -> Result[LoadedPrediction, str]:
    """Verify identity, dtype, shape, target and raw logit of captured bytes."""
    key = preview.row.key.tile_id
    data = bytes_by_path.get(preview.path)
    if data is None or len(data) != preview.size_bytes:
        return Err(f"preview {key} bytes differ from captured size")
    if hashlib.sha256(data).hexdigest() != preview.sha256:
        return Err(f"preview {key} bytes differ from captured checksum")
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            names = set(archive.files)
            if not {"image", "target", "logits"}.issubset(names):
                return Err(f"preview {key} lacks recorded image/target/logits entries")
            image = archive["image"]
            target = archive["target"]
            logits = archive["logits"]
    except (OSError, EOFError, zipfile.BadZipFile, ValueError) as exc:
        return Err(f"preview {key} is not a decodable NPZ archive: {exc}")
    shape = tuple(int(value) for value in preview.row.key.spatial_shard)
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.float32
        or image.ndim != 3
        or tuple(image.shape[1:]) != shape
    ):
        return Err(f"preview {key} image does not match captured identity")
    if not (np.isfinite(image).all() and image.min() >= 0.0 and image.max() <= 1.0):
        return Err(f"preview {key} image is not finite unit float data")
    if preview.row.key.task == "segmentor":
        if not isinstance(target, np.ndarray) or not isinstance(logits, np.ndarray):
            return Err(f"preview {key} lacks captured target/logit arrays")
        display = segmentation_display_data(preview.row, target, logits)
        if isinstance(display, Err):
            return Err(f"preview {key} segmentation display refused: {display.error}")
        loaded = LoadedPrediction(image, target, logits, display.value)
    else:
        if (
            not isinstance(target, np.ndarray)
            or target.dtype != np.float32
            or target.shape != (1,)
            or float(target[0]) != preview.row.label
        ):
            return Err(f"preview {key} target differs from the captured label")
        recorded_logit = captured_values(preview.row).get("logit")
        if (
            not isinstance(logits, np.ndarray)
            or logits.dtype != np.float32
            or logits.shape != (1,)
            or recorded_logit is None
            or float(logits[0]) != recorded_logit
        ):
            return Err(f"preview {key} raw logit differs from the captured logit")
        loaded = LoadedPrediction(image, target, logits, None)
    if (
        image.shape[0] == 0
        or len(preview.display_indices) not in (1, 3)
        or min(preview.display_indices) < 0
        or max(preview.display_indices) >= image.shape[0]
    ):
        return Err(f"preview {key} display indices out of bounds")
    return Ok(loaded)


def _display_image(axes: Axes, preview: PredictionPreview, image: np.ndarray) -> None:
    """Draw the exact captured array through the supplied channel mapping."""
    indices = preview.display_indices
    if len(indices) == 3:
        axes.imshow(np.transpose(np.asarray(image)[list(indices)], (1, 2, 0)))
    else:
        axes.imshow(image[indices[0]], cmap="gray", vmin=0.0, vmax=1.0)
    axes.set_xticks(())
    axes.set_yticks(())


def _predicted(row: CaptureRow) -> int:
    """Read the recorded decision from truth plus error flags; never re-scores."""
    return (
        1
        if (row.label == 1 and not row.false_negative) or (row.label == 0 and row.false_positive)
        else 0
    )


def _panel(
    axes: Axes,
    preview: PredictionPreview | None,
    loaded: LoadedPrediction | None,
    row: CaptureRow,
    cfg: PlotConfig,
) -> None:
    """Render one chosen row or its explicit omitted placeholder."""
    axes.set_xticks(())
    axes.set_yticks(())
    if preview is None or loaded is None:
        axes.text(
            0.5,
            0.5,
            _wrap(f"{row.key.tile_id}\nPreview omitted by the capture budget", cfg),
            ha="center",
            va="center",
            transform=axes.transAxes,
            color="#666666",
        )
        return
    image = loaded.image
    _display_image(axes, preview, image)
    values = captured_values(row)
    probability = values.get("probability")
    loss = values.get("binary_cross_entropy")
    caption = f"{row.key.tile_id} | gsd {row.gsd_m[0]:g}x{row.gsd_m[1]:g} m"
    if probability is not None and loss is not None:
        caption += (
            f"\ntruth={int(row.label)} pred={_predicted(row)} p={probability:.3f} BCE={loss:.3g}"
        )
    else:
        caption += f"\ntruth={int(row.label)} pred={_predicted(row)}"
    axes.set_title(textwrap.fill(caption, 34), fontsize="x-small")
    axes.set_xlabel(textwrap.fill(preview.display_label, 44), fontsize="x-small")


def _check_spatial(row: CaptureRow) -> LocalizationRow:
    """Reject dangling component IDs or unsafe geometry before plotting."""
    spatial = row.spatial
    if spatial is None:
        raise RuntimeError("segmentor row lacks frozen spatial evidence")
    local = spatial.localization
    for components in (local.truth, local.predicted):
        ids = [component.component_id for component in components]
        if any(
            not isinstance(component_id, int) or isinstance(component_id, bool) or component_id < 0
            for component_id in ids
        ):
            raise RuntimeError("frozen component IDs must be nonnegative integers")
        if len(set(ids)) != len(ids):
            raise RuntimeError("frozen component IDs repeat within one mask")
    truth_ids = {component.component_id for component in local.truth}
    predicted_ids = {component.component_id for component in local.predicted}
    matched_truth = [match.truth_id for match in local.matches]
    matched_prediction = [match.prediction_id for match in local.matches]
    if len(set(matched_truth)) != len(matched_truth) or len(set(matched_prediction)) != len(
        matched_prediction
    ):
        raise RuntimeError("frozen component matches are not one-to-one")
    for matched, unmatched, id_set, name in (
        (matched_truth, local.unmatched_truth, truth_ids, "truth"),
        (matched_prediction, local.unmatched_prediction, predicted_ids, "prediction"),
    ):
        if len(set(unmatched)) != len(unmatched):
            raise RuntimeError(f"frozen unmatched {name} IDs repeat")
        if set(matched) & set(unmatched):
            raise RuntimeError(f"frozen matched {name} IDs are also marked unmatched")
        if set(matched) | set(unmatched) != id_set:
            raise RuntimeError(f"frozen {name} component IDs are incompletely accounted")
    height, width = (int(value) for value in row.key.spatial_shard)
    for component in (*local.truth, *local.predicted):
        x0, y0, x1, y1 = component.bbox_xyxy
        if any(not isinstance(value, int) or isinstance(value, bool) for value in (x0, y0, x1, y1)):
            raise RuntimeError("frozen component bbox bounds must be integers")
        if not (0 <= x0 <= x1 < width and 0 <= y0 <= y1 < height):
            raise RuntimeError("frozen component bbox lies outside the image")
        bbox_area = (x1 - x0 + 1) * (y1 - y0 + 1)
        if (
            not isinstance(component.area_px, int)
            or isinstance(component.area_px, bool)
            or not 0 < component.area_px <= bbox_area
        ):
            raise RuntimeError("frozen component area is not a positive count within its bbox")
        if component.area_m2 is not None and not (
            math.isfinite(component.area_m2) and component.area_m2 > 0
        ):
            raise RuntimeError("frozen component ground area is not a positive finite value")
        if not (
            math.isfinite(component.centroid_x_px)
            and math.isfinite(component.centroid_y_px)
            and x0 <= component.centroid_x_px <= x1
            and y0 <= component.centroid_y_px <= y1
        ):
            raise RuntimeError("frozen component centroid is non-finite or outside its bbox")
    for match in local.matches:
        if not (
            math.isfinite(match.iou)
            and 0 <= match.iou <= 1
            and math.isfinite(match.dx_px)
            and math.isfinite(match.dy_px)
            and math.isfinite(match.distance_px)
            and match.distance_px >= 0
        ):
            raise RuntimeError("frozen match carries non-finite or invalid error evidence")
        if match.distance_m is not None and not (
            math.isfinite(match.distance_m) and match.distance_m >= 0
        ):
            raise RuntimeError("frozen match ground distance is not a nonnegative finite value")
    if not (
        math.isfinite(local.blob_probability_threshold)
        and 0 <= local.blob_probability_threshold <= 1
        and math.isfinite(local.match_iou_min)
        and 0 <= local.match_iou_min <= 1
        and isinstance(local.min_blob_area_px, int)
        and not isinstance(local.min_blob_area_px, bool)
        and local.min_blob_area_px >= 0
    ):
        raise RuntimeError("frozen localization thresholds are not finite declared values")
    if local.gsd_m is not None and not all(
        math.isfinite(value) and value > 0 for value in local.gsd_m
    ):
        raise RuntimeError("frozen localization GSD is not a positive finite pair")
    return local


def _component_label(
    axes: Axes,
    text: str,
    centroid: tuple[float, float],
    offset: tuple[int, int],
    color: str,
) -> None:
    """Label a frozen component near its centroid with a contrast box."""
    axes.annotate(
        text,
        centroid,
        textcoords="offset points",
        xytext=offset,
        fontsize="x-small",
        fontweight="bold",
        color=color,
        bbox={"facecolor": "white", "alpha": 0.65, "pad": 0.6, "linewidth": 0},
    )


def _overlay_panel(axes: Axes, row: CaptureRow) -> None:
    """Draw all frozen components and only the recorded matched pairs."""
    import matplotlib.lines
    import matplotlib.patches

    spatial = row.spatial
    if spatial is None:
        raise RuntimeError("segmentor row lacks frozen spatial evidence")
    local = spatial.localization
    truth = {component.component_id: component for component in local.truth}
    predicted = {component.component_id: component for component in local.predicted}
    for component in local.truth:
        x0, y0, x1, y1 = component.bbox_xyxy
        axes.add_patch(
            matplotlib.patches.Rectangle(
                (x0 - 0.5, y0 - 0.5),
                x1 - x0 + 1,
                y1 - y0 + 1,
                fill=False,
                edgecolor="#0072B2",
                lw=1.0,
            )
        )
        axes.plot(
            [component.centroid_x_px],
            [component.centroid_y_px],
            linestyle="None",
            marker="o",
            ms=4,
            color="#0072B2",
        )
        _component_label(
            axes,
            f"T{component.component_id}",
            (component.centroid_x_px, component.centroid_y_px),
            (-7, -9),
            "#0072B2",
        )
    for component in local.predicted:
        x0, y0, x1, y1 = component.bbox_xyxy
        axes.add_patch(
            matplotlib.patches.Rectangle(
                (x0 - 0.5, y0 - 0.5),
                x1 - x0 + 1,
                y1 - y0 + 1,
                fill=False,
                edgecolor="#E69F00",
                lw=1.0,
                linestyle="--",
            )
        )
        axes.plot(
            [component.centroid_x_px],
            [component.centroid_y_px],
            linestyle="None",
            marker="s",
            ms=4,
            color="#E69F00",
        )
        _component_label(
            axes,
            f"P{component.component_id}",
            (component.centroid_x_px, component.centroid_y_px),
            (-7, 4),
            "#B35C00",
        )
    for match in local.matches:
        pair = (truth[match.truth_id], predicted[match.prediction_id])
        axes.plot(
            [pair[0].centroid_x_px, pair[1].centroid_x_px],
            [pair[0].centroid_y_px, pair[1].centroid_y_px],
            color="#009E73",
            lw=1.2,
        )
    for component_id in local.unmatched_truth:
        component = truth[component_id]
        axes.plot(
            [component.centroid_x_px],
            [component.centroid_y_px],
            linestyle="None",
            marker="x",
            ms=7,
            mew=2,
            color="#0072B2",
        )
    for component_id in local.unmatched_prediction:
        component = predicted[component_id]
        axes.plot(
            [component.centroid_x_px],
            [component.centroid_y_px],
            linestyle="None",
            marker="x",
            ms=7,
            mew=2,
            color="#E69F00",
        )
    handles = (
        matplotlib.patches.Patch(fill=False, edgecolor="#0072B2", label="truth T<id>"),
        matplotlib.patches.Patch(
            fill=False, edgecolor="#E69F00", linestyle="--", label="prediction P<id>"
        ),
        matplotlib.lines.Line2D((), (), color="#009E73", lw=1.2, label="frozen match"),
        matplotlib.lines.Line2D(
            (), (), linestyle="None", marker="x", color="#444444", label="unmatched"
        ),
    )
    axes.legend(
        handles=handles,
        fontsize="small",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncols=2,
        framealpha=0.8,
    )


def _segmentation_page(
    figure: Figure,
    gallery: PredictionGallery,
    row: CaptureRow | None,
    previews: dict[str, PredictionPreview],
    loaded: dict[str, LoadedPrediction],
    cfg: PlotConfig,
) -> None:
    """Render one chosen segmentor example into the fixed 2x3 extent grid."""
    panels: list[Axes] = [figure.add_subplot(2, 3, index + 1) for index in range(6)]
    for axes in panels:
        axes.set_xticks(())
        axes.set_yticks(())
    message = gallery.availability.reason or "No selected examples" if row is None else None
    preview: PredictionPreview | None = None
    prediction: LoadedPrediction | None = None
    if row is not None:
        key = prediction_key(row)
        preview = previews.get(key)
        prediction = loaded.get(key)
        if preview is None or prediction is None:
            message = f"{row.key.tile_id}\nPreview omitted by the capture budget"
    if message is not None or row is None or preview is None or prediction is None:
        panels[0].text(
            0.5,
            0.5,
            _wrap(message or "Unavailable", cfg),
            ha="center",
            va="center",
            transform=panels[0].transAxes,
            color="#666666",
        )
        for axes in panels[1:]:
            axes.set_visible(False)
        return
    segmentation = prediction.segmentation
    if segmentation is None:
        raise RuntimeError("segmentor preview lacks verified display arrays")
    local = _check_spatial(row)
    _display_image(panels[0], preview, prediction.image)
    panels[0].set_title("Input", fontsize="x-small")
    panels[0].set_xlabel(textwrap.fill(preview.display_label, 44), fontsize="x-small")
    panels[1].imshow(segmentation.truth, cmap="gray", vmin=0.0, vmax=1.0)
    panels[1].set_title("explicit truth", fontsize="x-small")
    image = panels[2].imshow(segmentation.probability, cmap="viridis", vmin=0.0, vmax=1.0)
    figure.colorbar(image, ax=panels[2], label="captured-logit probability")
    panels[2].set_title("probability [0,1]", fontsize="x-small")
    panels[3].imshow(segmentation.predicted, cmap="gray", vmin=0.0, vmax=1.0)
    panels[3].set_title(f"raw binary mask\np>={segmentation.threshold:g}", fontsize="x-small")
    errors = np.zeros((*segmentation.truth.shape, 3), dtype=float)
    errors[segmentation.predicted & segmentation.truth] = (0.7, 0.7, 0.7)
    errors[segmentation.false_positive] = (0.85, 0.1, 0.1)
    errors[segmentation.false_negative] = (0.1, 0.3, 0.9)
    panels[4].imshow(errors, vmin=0.0, vmax=1.0)
    panels[4].set_title("pixel errors:\nFP red, FN blue, TP gray", fontsize="x-small")
    _display_image(panels[5], preview, prediction.image)
    _overlay_panel(panels[5], row)
    panels[5].set_title("frozen component matches", fontsize="x-small")
    caption = (
        f"{row.key.tile_id} | raw-mask p>={segmentation.threshold:g}; blob p>="
        f"{local.blob_probability_threshold:g}, min area {local.min_blob_area_px} px; "
        f"match IoU>={local.match_iou_min:g} | matched {len(local.matches)}; "
        f"missed truth {len(local.unmatched_truth)}; unmatched pred "
        f"{len(local.unmatched_prediction)} | gsd {row.gsd_m[0]:g}x{row.gsd_m[1]:g} m | "
        "centroid distances are conditional on matched pairs; misses carry no error"
    )
    figure.supxlabel(_wrap(caption, cfg), fontsize="x-small", color="#444444")


def _gallery_figures(
    gallery: PredictionGallery,
    previews: dict[str, PredictionPreview],
    loaded: dict[str, LoadedPrediction],
    cfg: PlotConfig,
) -> Result[tuple[Figure, ...], str]:
    """Render the fixed chosen rows into bounded pages per task layout."""
    import matplotlib

    tasks = {row.key.task for row in gallery.rows}
    if len(tasks) > 1:
        return Err(f"gallery {gallery.identifier} mixes task cohorts")
    task = next(iter(tasks), "classifier")
    figures: list[Figure] = []
    empty_reason = gallery.availability.reason or "No selected examples"
    rows: tuple[CaptureRow | None, ...] = gallery.rows or (None,)
    pages = len(rows) if task == "segmentor" else (len(rows) + _PAGE_SIZE - 1) // _PAGE_SIZE
    try:
        matplotlib.use("Agg")
        with matplotlib.rc_context({"font.size": float(cfg.font_size)}):
            from matplotlib.figure import Figure

            for page in range(pages):
                figure = Figure(
                    figsize=(float(cfg.width_inches), float(cfg.height_inches)),
                    dpi=int(cfg.dpi),
                )
                figures.append(figure)
                figure.set_layout_engine("constrained")
                figure.suptitle(
                    _wrap(
                        f"{gallery.identifier} - {gallery.selection_method}"
                        + (f" (page {page + 1}/{pages})" if pages > 1 else ""),
                        cfg,
                    )
                )
                if task == "segmentor":
                    _segmentation_page(figure, gallery, rows[page], previews, loaded, cfg)
                    continue
                members = rows[page * _PAGE_SIZE : (page + 1) * _PAGE_SIZE]
                cols = min(len(members), 2)
                grid_rows = (len(members) + cols - 1) // cols
                for index, row in enumerate(members):
                    axes = figure.add_subplot(grid_rows, cols, index + 1)
                    if row is None:
                        axes.text(
                            0.5,
                            0.5,
                            _wrap(empty_reason, cfg),
                            ha="center",
                            va="center",
                            transform=axes.transAxes,
                            color="#666666",
                        )
                        axes.set_xticks(())
                        axes.set_yticks(())
                        continue
                    key = prediction_key(row)
                    _panel(axes, previews.get(key), loaded.get(key), row, cfg)
        return Ok(tuple(figures))
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        for figure in figures:
            _close(figure)
        return Err(f"cannot render {gallery.identifier}: {exc}")


def _close(figure: Figure | None) -> None:
    """Clear and close a partially built figure after a render failure."""
    if figure is not None:
        figure.clf()
        import matplotlib.pyplot as plt

        plt.close(figure)


def render_prediction_visuals(
    captured: PredictionPreviewCapture, cfg: PlotConfig
) -> Result[RenderedDatasetFigures, str]:
    """Export every supplied gallery page into bundle bytes.

    Selection rows are used verbatim: chosen previews that were omitted
    by the capture budget produce omission panels and an ``UNAVAILABLE``
    output instead of substitution. Checksum/shape/content verification
    failures return ``Err`` and nothing is published. A gallery mixing
    classifier and segmentor rows is refused; segmentor galleries draw
    one chosen example per page.
    """
    try:
        chosen_tasks = {row.key.task for gallery in captured.galleries for row in gallery.rows}
        chosen_tasks.update(preview.row.key.task for preview in captured.previews)
        if any(task not in ("classifier", "segmentor") for task in chosen_tasks):
            return Err("prediction visuals require classifier or segmentor task evidence")
        if len({file.path for file in captured.files}) != len(captured.files):
            return Err("prediction files contain duplicate paths")
        for file in captured.files:
            try:
                check_bundle_path(file.path)
            except (ValueError, TypeError) as exc:
                return Err(f"unsafe preview file path {file.path!r}: {exc}")
        if len({gallery.identifier for gallery in captured.galleries}) != len(
            captured.galleries
        ) or len({gallery.availability.name for gallery in captured.galleries}) != len(
            captured.galleries
        ):
            return Err("gallery identifiers and output names must be unique")
        for gallery in captured.galleries:
            safe = _safe_token(gallery.identifier, "gallery identifier")
            if isinstance(safe, Err):
                return safe
        bytes_by_path = {file.path: file.data for file in captured.files}
        previews: dict[str, PredictionPreview] = {}
        for preview in captured.previews:
            try:
                check_bundle_path(preview.path)
            except (ValueError, TypeError) as exc:
                return Err(f"unsafe preview path {preview.path!r}: {exc}")
            key = prediction_key(preview.row)
            if key in previews:
                return Err("duplicate prediction preview for one captured row")
            previews[key] = preview
        loaded: dict[str, LoadedPrediction] = {}
        for key, preview in previews.items():
            arrays = _load(preview, bytes_by_path)
            if isinstance(arrays, Err):
                return arrays
            loaded[key] = arrays.value
        for gallery in captured.galleries:
            for row in gallery.rows:
                bound = previews.get(prediction_key(row))
                if bound is not None and bound.row != row:
                    return Err("preview bound to the chosen key carries different captured scalars")
        files: list[BundleFile] = []
        outputs: list[AvailabilityRecord] = []
        for gallery in captured.galleries:
            omitted = any(prediction_key(row) not in previews for row in gallery.rows)
            figures = _gallery_figures(gallery, previews, loaded, cfg)
            if isinstance(figures, Err):
                return figures
            status = gallery.availability.status
            reason = gallery.availability.reason
            if status == "AVAILABLE" and omitted:
                status = "UNAVAILABLE"
                reason = "Chosen prediction previews were omitted by the capture budget"
            for page, figure in enumerate(figures.value):
                exported = export_figure(
                    figure,
                    f"predictions/{gallery.identifier}_{page + 1}",
                    cfg,
                    kind="VISUAL",
                )
                if isinstance(exported, Err):
                    for pending in figures.value[page:]:
                        _close(pending)
                    return exported
                files.extend(exported.value)
            outputs.append(
                AvailabilityRecord(name=gallery.availability.name, status=status, reason=reason)
            )
        return Ok(RenderedDatasetFigures(files=tuple(files), outputs=tuple(outputs)))
    except (OSError, ValueError, RuntimeError, TypeError, KeyError, IndexError) as exc:
        return Err(f"prediction visual rendering failed: {exc}")
