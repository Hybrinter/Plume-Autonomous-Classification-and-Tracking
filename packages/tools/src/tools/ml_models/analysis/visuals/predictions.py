"""Prediction-overlay visuals over captured evaluation evidence.

Renders the fixed ``PredictionGallery`` selections against immutable
preview bytes supplied by capture. Every chosen panel verifies the NPZ
checksum and size before ``np.load`` on an in-memory buffer
(``allow_pickle=False``), then requires the exact canonical float32
unit image, the recorded binary target, and the exact Python float
of the captured float32 raw logit - no coercion, resize, normalization,
substitution, or re-selection. A preview bound to the same key but
carrying different scalars is refused rather than shown beside the
chosen row. Chosen rows whose preview was omitted by the capture
budget render an explicit omitted panel and mark the gallery output
``UNAVAILABLE``; supplied ``SKIPPED``/``UNAVAILABLE`` family states are
preserved. Classifier families only; segmentation returns ``Err``
until PR14.

Contains:
  - PredictionPreview, PredictionPreviewCapture: bound preview records.
  - render_prediction_visuals: export gallery pages into bundle bytes.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import hashlib
import io
import textwrap
import zipfile
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from flight.libs.types import Err, Ok, Result

from tools.ml_models.analysis.artifacts import BundleFile
from tools.ml_models.analysis.capture import CaptureRow
from tools.ml_models.analysis.config import PlotConfig
from tools.ml_models.analysis.contracts import AvailabilityRecord, check_bundle_path
from tools.ml_models.analysis.model_figures import captured_values
from tools.ml_models.analysis.plots.common import export_figure
from tools.ml_models.analysis.plots.dataset import RenderedDatasetFigures
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
) -> Result[tuple[np.ndarray, float, float], str]:
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
    if (
        image.shape[0] == 0
        or len(preview.display_indices) not in (1, 3)
        or min(preview.display_indices) < 0
        or max(preview.display_indices) >= image.shape[0]
    ):
        return Err(f"preview {key} display indices out of bounds")
    return Ok((image, float(target[0]), float(logits[0])))


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
    loaded: tuple[np.ndarray, float, float] | None,
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
    image, _target, _logit = loaded
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


def _gallery_figures(
    gallery: PredictionGallery,
    previews: dict[str, PredictionPreview],
    loaded: dict[str, tuple[np.ndarray, float, float]],
    cfg: PlotConfig,
) -> Result[tuple[Figure, ...], str]:
    """Render the fixed chosen rows into pages of at most four panels."""
    import matplotlib

    figures: list[Figure] = []
    empty_reason = gallery.availability.reason or "No selected examples"
    rows: tuple[CaptureRow | None, ...] = gallery.rows or (None,)
    pages = (len(rows) + _PAGE_SIZE - 1) // _PAGE_SIZE
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
    failures return ``Err`` and nothing is published. Segmentation
    galleries remain unimplemented until PR14.
    """
    try:
        chosen_tasks = {row.key.task for gallery in captured.galleries for row in gallery.rows}
        chosen_tasks.update(preview.row.key.task for preview in captured.previews)
        if any(task != "classifier" for task in chosen_tasks):
            return Err("segmentation prediction visuals are not implemented until PR14")
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
        loaded: dict[str, tuple[np.ndarray, float, float]] = {}
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
