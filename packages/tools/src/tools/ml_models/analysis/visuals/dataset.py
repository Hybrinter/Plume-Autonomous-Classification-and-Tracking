"""Dataset gallery rendering over captured preview bytes.

Galleries draw the exact captured canonical float32 arrays: bytes are
verified against the frozen preview identity, loaded through ``np.load``
on in-memory buffers only (``allow_pickle=False``), and displayed with
the semantic channel mapping supplied by capture - no normalization,
stretch, resize, source reads, or model calls. Augmentation panels apply
the same stored dihedral transforms the training data used. Gallery
families that are unavailable or skipped still produce indexed
placeholder figures so absence is visible.

Contains:
  - render_dataset_visuals: export every planned gallery into bundle bytes.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import hashlib
import io
import textwrap
from typing import TYPE_CHECKING

import numpy as np
from flight.libs.types import Err, Ok, Result

from tools.ml_models.analysis.artifacts import BundleFile
from tools.ml_models.analysis.config import PlotConfig
from tools.ml_models.analysis.contracts import AvailabilityRecord
from tools.ml_models.analysis.dataset import DatasetSample
from tools.ml_models.analysis.dataset_previews import DatasetPreview, DatasetPreviewCapture
from tools.ml_models.analysis.plots.common import export_figure
from tools.ml_models.analysis.plots.dataset import RenderedDatasetFigures
from tools.ml_models.analysis.visuals.selection import DatasetGallery
from tools.ml_models.dataset.augment import apply_dihedral

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_FAMILY_TITLES: dict[str, str] = {
    "representative": "Representative gallery",
    "augmentation": "Augmentation gallery",
    "same_observation_gsd": "Same-observation GSD gallery",
}


def _wrap(text: str, cfg: PlotConfig) -> str:
    """Wrap caption text to a width bounded by the configured figure size."""
    return textwrap.fill(text, width=max(24, int(float(cfg.width_inches) * 11)))


def _load_preview(
    preview: DatasetPreview, bytes_by_path: dict[str, bytes]
) -> Result[tuple[np.ndarray, np.ndarray | None], str]:
    """Verify identity, dtype, shape, and mask agreement of captured bytes."""
    data = bytes_by_path.get(preview.path)
    if data is None or len(data) != preview.size_bytes:
        return Err(f"preview {preview.variant_id} bytes differ from captured size")
    if hashlib.sha256(data).hexdigest() != preview.sha256:
        return Err(f"preview {preview.variant_id} bytes differ from captured checksum")
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        names = set(archive.files)
        if "image" not in names:
            return Err(f"preview {preview.variant_id} has no image entry")
        image = archive["image"]
        mask = archive["mask"] if "mask" in names else None
    sample = preview.sample
    shape = tuple(int(value) for value in sample.key.spatial_shard)
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.float32
        or image.ndim != 3
        or tuple(image.shape[1:]) != shape
    ):
        return Err(f"preview {preview.variant_id} image does not match captured identity")
    if not (np.isfinite(image).all() and image.min() >= 0.0 and image.max() <= 1.0):
        return Err(f"preview {preview.variant_id} image is not finite unit float data")
    expected_mask = sample.mask_key is not None
    if (mask is None) == expected_mask:
        return Err(f"preview {preview.variant_id} mask entry disagrees with recorded mask key")
    if mask is not None:
        if (
            not isinstance(mask, np.ndarray)
            or mask.dtype != np.uint8
            or mask.ndim != 3
            or mask.shape[0] != 1
            or tuple(mask.shape[1:]) != shape
            or not set(np.unique(mask)).issubset({0, 1})
        ):
            return Err(f"preview {preview.variant_id} mask is not explicit binary geometry")
        if sample.mask is None:
            return Err(
                f"preview {preview.variant_id} mask entry disagrees with recorded mask state"
            )
    if (
        image.shape[0] == 0
        or len(preview.display_indices) not in (1, 3)
        or min(preview.display_indices) < 0
        or max(preview.display_indices) >= image.shape[0]
    ):
        return Err(f"preview {preview.variant_id} display indices out of bounds")
    return Ok((image, mask))


def _display_image(axes: Axes, preview: DatasetPreview, image: np.ndarray) -> None:
    """Draw one captured array with the supplied semantic channel mapping."""
    indices = preview.display_indices
    if len(indices) == 3:
        axes.imshow(np.transpose(np.asarray(image)[list(indices)], (1, 2, 0)))
    elif len(indices) == 1:
        axes.imshow(image[indices[0]], cmap="gray", vmin=0.0, vmax=1.0)
    else:
        axes.text(
            0.5,
            0.5,
            "No display bands recorded",
            ha="center",
            va="center",
            transform=axes.transAxes,
        )
    axes.set_xticks(())
    axes.set_yticks(())


def _mask_panel(axes: Axes, sample: DatasetSample, mask: np.ndarray | None) -> None:
    """Draw the explicit mask, or declare its absence; an empty mask stays empty."""
    if mask is None:
        axes.text(0.5, 0.5, "Mask unavailable", ha="center", va="center", transform=axes.transAxes)
        axes.set_title("Explicit mask")
    else:
        axes.imshow(np.asarray(mask).reshape(mask.shape[-2:]), cmap="gray", vmin=0.0, vmax=1.0)
        state = sample.mask.area_px if sample.mask is not None else 0
        axes.set_title(f"Explicit mask ({state} px)")
    axes.set_xticks(())
    axes.set_yticks(())


def _panel_grid(figure: Figure, count: int) -> list[Axes]:
    """Lay out ``count`` panels in a readable grid inside fixed dimensions."""
    cols = min(count, 4)
    rows = (count + cols - 1) // cols
    return [figure.add_subplot(rows, cols, index + 1) for index in range(count)]


def _new_figure(cfg: PlotConfig, suptitle: str) -> Figure:
    """Create one fixed-dimension headless figure with the configured size."""
    from matplotlib.figure import Figure

    figure = Figure(figsize=(float(cfg.width_inches), float(cfg.height_inches)), dpi=int(cfg.dpi))
    figure.set_layout_engine("constrained")
    figure.suptitle(_wrap(suptitle, cfg))
    return figure


def _close(figure: Figure | None) -> None:
    """Clear and close a partially built figure after a render failure."""
    if figure is not None:
        figure.clf()
        import matplotlib.pyplot as plt

        plt.close(figure)


def _placeholder(family: str, reason: str | None, cfg: PlotConfig) -> Result[Figure, str]:
    """Render a family-level unavailable/skipped placeholder panel."""
    import matplotlib

    figure: Figure | None = None
    try:
        with matplotlib.rc_context({"font.size": float(cfg.font_size)}):
            figure = _new_figure(cfg, _FAMILY_TITLES.get(family, f"{family} gallery"))
            axes = figure.add_subplot()
            axes.set_xticks(())
            axes.set_yticks(())
            axes.text(
                0.5,
                0.5,
                _wrap(f"Gallery unavailable\n{reason or 'No eligible recorded inputs'}", cfg),
                ha="center",
                va="center",
                transform=axes.transAxes,
                color="#666666",
            )
            return Ok(figure)
    except (OSError, ValueError, RuntimeError) as exc:
        _close(figure)
        return Err(f"cannot render {family} gallery placeholder: {exc}")


def _representative(
    gallery: DatasetGallery,
    preview: DatasetPreview,
    loaded: tuple[np.ndarray, np.ndarray | None],
    cfg: PlotConfig,
) -> Result[Figure, str]:
    """Render one canonical input beside its explicit-mask panel."""
    import matplotlib

    figure: Figure | None = None
    try:
        with matplotlib.rc_context({"font.size": float(cfg.font_size)}):
            image, mask = loaded
            sample = preview.sample
            label = "positive" if sample.label else "negative"
            figure = _new_figure(cfg, f"{sample.row.tile_id} - label {label}")
            input_axes, mask_axes = _panel_grid(figure, 2)
            _display_image(input_axes, preview, image)
            input_axes.set_title(_wrap(f"Input - {preview.display_label}", cfg))
            _mask_panel(mask_axes, sample, mask)
            return Ok(figure)
    except (OSError, ValueError, RuntimeError) as exc:
        _close(figure)
        return Err(f"cannot render {gallery.identifier}: {exc}")


def _augmentation(
    gallery: DatasetGallery,
    preview: DatasetPreview,
    loaded: tuple[np.ndarray, np.ndarray | None],
    cfg: PlotConfig,
) -> Result[Figure, str]:
    """Render every stored dihedral element of the canonical input."""
    import matplotlib

    figure: Figure | None = None
    try:
        with matplotlib.rc_context({"font.size": float(cfg.font_size)}):
            image, _ = loaded
            figure = _new_figure(
                cfg,
                f"{preview.sample.row.tile_id} - verified stored training transforms",
            )
            axes = _panel_grid(figure, len(gallery.elements))
            for panel, element in zip(axes, gallery.elements, strict=True):
                _display_image(panel, preview, apply_dihedral(image, element))
                panel.set_title(element)
            return Ok(figure)
    except (OSError, ValueError, RuntimeError) as exc:
        _close(figure)
        return Err(f"cannot render {gallery.identifier}: {exc}")


def _gsd_pair(
    gallery: DatasetGallery,
    previews: list[tuple[DatasetPreview, tuple[np.ndarray, np.ndarray | None]]],
    cfg: PlotConfig,
) -> Result[Figure, str]:
    """Render each supplied variant of one recorded observation at its own GSD."""
    import matplotlib

    figure: Figure | None = None
    try:
        with matplotlib.rc_context({"font.size": float(cfg.font_size)}):
            sample = previews[0][0].sample
            observation = sample.row.metadata.observation_id or "unrecorded"
            figure = _new_figure(cfg, f"Recorded observation {observation}")
            axes = _panel_grid(figure, len(previews))
            for panel, (preview, (image, _)) in zip(axes, previews, strict=True):
                _display_image(panel, preview, image)
                gsd = preview.sample.gsd_m
                panel.set_title(f"lateral {gsd[0]:g} m, along {gsd[1]:g} m")
            return Ok(figure)
    except (OSError, ValueError, RuntimeError) as exc:
        _close(figure)
        return Err(f"cannot render {gallery.identifier}: {exc}")


def render_dataset_visuals(
    captured: DatasetPreviewCapture, cfg: PlotConfig
) -> Result[RenderedDatasetFigures, str]:
    """Export every planned gallery and family placeholder into bundle bytes.

    Captured preview bytes are checksum-verified and shape-validated before
    rendering; corrupt or missing files return ``Err`` and nothing is
    published. Plan outputs pass through unchanged; each rendered gallery
    and each family placeholder adds a uniquely named ``visual:``
    availability record.
    """
    try:
        bytes_by_path = {file.path: file.data for file in captured.files}
        previews_by_id = {preview.variant_id: preview for preview in captured.previews}
        loaded: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
        for preview in captured.previews:
            arrays = _load_preview(preview, bytes_by_path)
            if isinstance(arrays, Err):
                return arrays
            loaded[preview.variant_id] = arrays.value
        files: list[BundleFile] = []
        outputs: list[AvailabilityRecord] = list(captured.plan.outputs)
        for gallery in captured.plan.galleries:
            members = [
                (previews_by_id[variant_id], loaded[variant_id])
                for variant_id in gallery.variant_ids
            ]
            if not members:
                return Err(f"gallery {gallery.identifier} has no preview members")
            if gallery.family == "representative":
                rendered = _representative(gallery, members[0][0], members[0][1], cfg)
            elif gallery.family == "augmentation":
                rendered = _augmentation(gallery, members[0][0], members[0][1], cfg)
            elif gallery.family == "same_observation_gsd":
                rendered = _gsd_pair(gallery, members, cfg)
            else:
                return Err(f"unknown gallery family {gallery.family!r}")
            if isinstance(rendered, Err):
                return rendered
            exported = export_figure(rendered.value, gallery.identifier, cfg, kind="VISUAL")
            if isinstance(exported, Err):
                return exported
            files.extend(exported.value)
            outputs.append(
                AvailabilityRecord(name="visual:" + gallery.identifier, status="AVAILABLE")
            )
        for record in captured.plan.outputs:
            if record.status == "AVAILABLE":
                continue
            family = record.name.split(":", 1)[-1]
            identifier = "gallery_unavailable_" + family
            placeholder = _placeholder(family, record.reason, cfg)
            if isinstance(placeholder, Err):
                return placeholder
            exported = export_figure(placeholder.value, identifier, cfg, kind="VISUAL")
            if isinstance(exported, Err):
                return exported
            files.extend(exported.value)
            outputs.append(
                AvailabilityRecord(
                    name="visual:" + identifier, status="UNAVAILABLE", reason=record.reason
                )
            )
        return Ok(RenderedDatasetFigures(files=tuple(files), outputs=tuple(outputs)))
    except (OSError, ValueError, RuntimeError, TypeError, KeyError, IndexError) as exc:
        return Err(f"dataset gallery rendering failed: {exc}")
