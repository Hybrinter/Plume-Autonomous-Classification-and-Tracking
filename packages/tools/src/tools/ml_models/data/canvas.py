"""Flight-frame scenes pasted from Zenodo chips.

Contains:
  - CanvasConfig: frame size, window, and paste settings.
  - Chip: one source tile.
  - CanvasSample: a full frame or a window.
  - build_scene: mosaic plus an optional annotated plume.
  - take_window: a crop that keeps a positive plume off a forced center.
  - sample_view: one scene and either the full frame or a window.

``config.seed`` and ``full_frame_every`` are stored for the training loop.
``sample_view`` reads the caller's Generator and the ``full_frame`` flag.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from tools.ml_models.data.augment import feather_paste, overlap_window


@dataclass(frozen=True, slots=True)
class CanvasConfig:
    """Geometry and paste settings for one flight scene.

    Attributes:
        frame_hw: Full frame ``(height, width)``. The default is 1544 by 2064.
        window_px: Square crop side. The default is 512.
        full_frame_every: Training steps between full-frame views. The default
            is 8. ``sample_view`` does not read this field.
        chip_side: Spatial side of every chip. The default is 76.
        empty_fraction: Probability that a scene stays empty of plumes.
        max_plumes: Annotated chips pasted when the scene is not empty.
        feather_px: Border blend width of a pasted chip.
        seed: Seed a caller may pass to ``numpy.random.default_rng``.
            ``sample_view`` does not read this field.
    """

    frame_hw: tuple[int, int] = (1544, 2064)
    window_px: int = 512
    full_frame_every: int = 8
    chip_side: int = 76
    empty_fraction: float = 0.5
    max_plumes: int = 1
    feather_px: int = 6
    seed: int = 0


@dataclass(frozen=True, slots=True)
class Chip:
    """One source tile.

    Attributes:
        image: Float32 array ``(C, H, W)``.
        mask: Float32 polygon mask ``(1, H, W)``, or an empty array when the
            tile has no mask.
        label: Presence, ``0`` or ``1``.
        group_id: Location id.
        split: ``train``, ``val``, or ``test``.
        annotated: True when the tile has a polygon annotation.
    """

    image: np.ndarray
    mask: np.ndarray
    label: float
    group_id: str
    split: str
    annotated: bool


@dataclass(frozen=True, slots=True)
class CanvasSample:
    """One training view.

    Attributes:
        image: Float32 frame or crop ``(C, H, W)``.
        mask: Float32 polygon mask ``(1, H, W)``.
        label: ``1`` when any mask pixel is positive, otherwise ``0``.
        full_frame: True when ``image`` is the full scene.
        origin: ``(row, column)`` of the window inside the frame. ``None``
            when ``full_frame`` is True.
    """

    image: np.ndarray
    mask: np.ndarray
    label: float
    full_frame: bool
    origin: tuple[int, int] | None


def _require_int(name: str, value: int, *, minimum: int) -> None:
    """Raise ValueError when ``value`` is not an int at or above ``minimum``.

    Args:
        name: Field name used in the error.
        value: Candidate integer.
        minimum: Inclusive lower bound.

    Returns:
        None.

    Raises:
        ValueError: If ``value`` is a bool or is below ``minimum``.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}; got {value!r}")


def _validate_config(config: CanvasConfig) -> None:
    """Raise ValueError when canvas settings are out of range.

    Args:
        config: Scene settings.

    Returns:
        None.

    Raises:
        ValueError: If a size is below 1 or ``empty_fraction`` is outside
            ``[0, 1]``.
    """
    if len(config.frame_hw) != 2:
        raise ValueError(f"frame_hw must be (height, width); got {config.frame_hw!r}")
    _require_int("frame height", config.frame_hw[0], minimum=1)
    _require_int("frame width", config.frame_hw[1], minimum=1)
    _require_int("window_px", config.window_px, minimum=1)
    _require_int("full_frame_every", config.full_frame_every, minimum=1)
    _require_int("chip_side", config.chip_side, minimum=1)
    _require_int("max_plumes", config.max_plumes, minimum=0)
    _require_int("feather_px", config.feather_px, minimum=0)
    if isinstance(config.seed, bool) or not isinstance(config.seed, int):
        raise ValueError(f"seed must be an int; got {config.seed!r}")
    fraction = config.empty_fraction
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not math.isfinite(float(fraction))
        or not 0.0 <= float(fraction) <= 1.0
    ):
        raise ValueError(f"empty_fraction must be in [0, 1]; got {fraction!r}")


def _validate_chips(chips: Sequence[Chip], config: CanvasConfig) -> None:
    """Raise ValueError when chips do not share ``chip_side`` and channels.

    Args:
        chips: Source tiles.
        config: Scene settings. ``chip_side`` is the required spatial side.

    Returns:
        None.

    Raises:
        ValueError: If an image or mask shape disagrees.
    """
    side = config.chip_side
    channels: int | None = None
    for chip in chips:
        image = chip.image
        if image.ndim != 3:
            raise ValueError(f"chip image must have shape (C, H, W); got {image.shape}")
        if int(image.shape[1]) != side or int(image.shape[2]) != side:
            raise ValueError(f"chip spatial shape {image.shape[1:]} must equal chip_side {side}")
        if image.dtype != np.float32:
            raise ValueError(f"chip image dtype must be float32; got {image.dtype}")
        count = int(image.shape[0])
        if count < 1:
            raise ValueError("chip image needs at least one channel")
        if channels is None:
            channels = count
        elif count != channels:
            raise ValueError("chips must share one channel count")
        if chip.mask.size == 0:
            continue
        if chip.mask.shape != (1, side, side):
            raise ValueError(
                f"chip mask must have shape {(1, side, side)} or be empty; got {chip.mask.shape}"
            )
        if chip.mask.dtype != np.float32:
            raise ValueError(f"chip mask dtype must be float32; got {chip.mask.dtype}")


def _mosaic(
    chips: Sequence[Chip],
    frame_h: int,
    frame_w: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Tile background chips with a random phase.

    Args:
        chips: Label-0 chips that share a spatial size.
        frame_h: Frame height.
        frame_w: Frame width.
        rng: Generator for the phase and the chip choice at each tile.

    Returns:
        np.ndarray[float32, (C, frame_h, frame_w)]: Hard-copy mosaic. Tiles
        that cross the frame border are clipped.
    """
    probe = chips[0].image
    channels = int(probe.shape[0])
    chip_h = int(probe.shape[1])
    chip_w = int(probe.shape[2])
    canvas = np.zeros((channels, frame_h, frame_w), dtype=np.float32)
    phase_y = int(rng.integers(0, chip_h))
    phase_x = int(rng.integers(0, chip_w))
    y = -phase_y
    while y < frame_h:
        x = -phase_x
        while x < frame_w:
            choice = chips[int(rng.integers(0, len(chips)))]
            _blit(canvas, choice.image, y, x)
            x += chip_w
        y += chip_h
    return canvas


def _blit(canvas: np.ndarray, chip: np.ndarray, top: int, left: int) -> None:
    """Copy the overlapping part of ``chip`` onto ``canvas``.

    Args:
        canvas: Destination ``(C, H, W)``.
        chip: Source ``(C, h, w)``.
        top: Destination row of the chip origin.
        left: Destination column of the chip origin.

    Returns:
        None.
    """
    window = overlap_window(
        (int(canvas.shape[1]), int(canvas.shape[2])),
        (int(chip.shape[1]), int(chip.shape[2])),
        top,
        left,
    )
    if window is None:
        return
    src_y, src_x, dst_y, dst_x, height, width = window
    canvas[:, dst_y : dst_y + height, dst_x : dst_x + width] = chip[
        :, src_y : src_y + height, src_x : src_x + width
    ]


def _paste_mask(canvas_mask: np.ndarray, chip_mask: np.ndarray, top: int, left: int) -> None:
    """Write polygon pixels that land inside the frame.

    Args:
        canvas_mask: Scene mask ``(1, H, W)``, updated in place.
        chip_mask: Polygon mask ``(1, h, w)``. Values are not feathered.
        top: Destination row of the chip origin.
        left: Destination column of the chip origin.

    Returns:
        None.
    """
    window = overlap_window(
        (int(canvas_mask.shape[1]), int(canvas_mask.shape[2])),
        (int(chip_mask.shape[1]), int(chip_mask.shape[2])),
        top,
        left,
    )
    if window is None:
        return
    src_y, src_x, dst_y, dst_x, height, width = window
    dst = canvas_mask[:, dst_y : dst_y + height, dst_x : dst_x + width]
    src = chip_mask[:, src_y : src_y + height, src_x : src_x + width]
    np.maximum(dst, src, out=dst)


def _paste_annotated(
    canvas: np.ndarray,
    canvas_mask: np.ndarray,
    chip: Chip,
    rng: np.random.Generator,
    feather_px: int,
) -> None:
    """Paste one annotated chip at a random clipped offset.

    Args:
        canvas: Scene image.
        canvas_mask: Scene mask.
        chip: Annotated chip whose mask contains a polygon.
        rng: Generator for the offset.
        feather_px: Image border blend. The mask is not feathered.

    Returns:
        None.

    Raises:
        ValueError: If the chip is unannotated or the mask has no positive pixel.
    """
    if not chip.annotated:
        raise ValueError("unannotated chip cannot be pasted")
    chip_h = int(chip.image.shape[1])
    chip_w = int(chip.image.shape[2])
    if chip.mask.shape != (1, chip_h, chip_w):
        raise ValueError("annotated chip mask must match the chip image")
    if not bool(np.any(chip.mask > 0.0)):
        raise ValueError("annotated chip mask has no polygon pixel")
    frame_h = int(canvas.shape[1])
    frame_w = int(canvas.shape[2])
    top = int(rng.integers(-chip_h + 1, frame_h))
    left = int(rng.integers(-chip_w + 1, frame_w))
    feather_paste(canvas, chip.image, top, left, feather_px)
    _paste_mask(canvas_mask, chip.mask, top, left)


def build_scene(
    chips: Sequence[Chip],
    config: CanvasConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build one flight frame from background chips and an optional plume.

    Args:
        chips: Tiles for one scene. Background tiles have ``label == 0``.
            Every tile in the scene shares one split.
        config: Frame size and paste settings.
        rng: Generator for the mosaic phase, the empty draw, and the paste
            offset.

    Returns:
        tuple[np.ndarray, np.ndarray, float]: Image ``(C, H, W)``, mask
        ``(1, H, W)``, and label. The label is ``1`` when any mask pixel is
        positive.

    Raises:
        ValueError: If the background split is mixed, a positive chip uses
            another split, or a paste is requested from an unannotated chip.
    """
    _validate_config(config)
    if not chips:
        raise ValueError("scene needs chips")
    _validate_chips(chips, config)
    background = [chip for chip in chips if chip.label <= 0.0]
    positives = [chip for chip in chips if chip.label > 0.0]
    if not background:
        raise ValueError("scene needs a background chip with label 0")
    splits = {chip.split for chip in background}
    if len(splits) != 1:
        raise ValueError("background chips must share one split")
    background_split = next(iter(splits))
    for chip in positives:
        if chip.split != background_split:
            raise ValueError(
                f"positive chip split {chip.split!r} differs from the background split "
                f"{background_split!r}"
            )
    frame_h, frame_w = config.frame_hw
    image = _mosaic(background, frame_h, frame_w, rng)
    mask = np.zeros((1, frame_h, frame_w), dtype=np.float32)
    if config.max_plumes > 0 and float(rng.random()) >= float(config.empty_fraction):
        annotated = [chip for chip in positives if chip.annotated]
        if any(not chip.annotated for chip in positives) and not annotated:
            raise ValueError("unannotated chip cannot be pasted")
        if not annotated:
            raise ValueError("no annotated chip to paste")
        for _ in range(config.max_plumes):
            choice = annotated[int(rng.integers(0, len(annotated)))]
            _paste_annotated(image, mask, choice, rng, config.feather_px)
    label = 1.0 if bool(np.any(mask > 0.0)) else 0.0
    return image, mask, label


def take_window(
    scene_image: np.ndarray,
    scene_mask: np.ndarray,
    config: CanvasConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Crop a window from a scene.

    Args:
        scene_image: Frame ``(C, H, W)``.
        scene_mask: Frame mask ``(1, H, W)``.
        config: ``window_px`` is the crop side.
        rng: Generator for the empty-window origin or the positive-pixel anchor.

    Returns:
        tuple[np.ndarray, np.ndarray, tuple[int, int]]: Crop, crop mask, and
        ``(row, column)`` of the crop origin. When ``window_px`` is at least
        the frame height or the frame width, the crop is the full frame and
        the origin is ``(0, 0)``.

    Raises:
        ValueError: If the mask shape disagrees with the image.

    Notes:
        An empty mask uses a uniform origin inside the frame. A positive mask
        picks one positive pixel and places it at a uniform coordinate inside
        the window, including the window border. The origin is then clamped
        so the window stays inside the frame.
    """
    _validate_config(config)
    if scene_image.ndim != 3 or scene_mask.ndim != 3:
        raise ValueError("scene image must be (C, H, W) and mask (1, H, W)")
    frame_h = int(scene_image.shape[1])
    frame_w = int(scene_image.shape[2])
    if scene_mask.shape != (1, frame_h, frame_w):
        raise ValueError("scene mask spatial shape must match the image")
    window = config.window_px
    if window >= frame_h or window >= frame_w:
        return (
            np.array(scene_image, dtype=np.float32, copy=True),
            np.array(scene_mask, dtype=np.float32, copy=True),
            (0, 0),
        )
    positive = np.argwhere(scene_mask[0] > 0.0)  # np.ndarray[int, (N, 2)]
    if positive.shape[0] == 0:
        top = int(rng.integers(0, frame_h - window + 1))
        left = int(rng.integers(0, frame_w - window + 1))
    else:
        pick = int(rng.integers(0, int(positive.shape[0])))
        py = int(positive[pick, 0])
        px = int(positive[pick, 1])
        anchor_y = int(rng.integers(0, window))
        anchor_x = int(rng.integers(0, window))
        top = _clamp(py - anchor_y, 0, frame_h - window)
        left = _clamp(px - anchor_x, 0, frame_w - window)
    crop = np.array(
        scene_image[:, top : top + window, left : left + window],
        dtype=np.float32,
        copy=True,
    )
    crop_mask = np.array(
        scene_mask[:, top : top + window, left : left + window],
        dtype=np.float32,
        copy=True,
    )
    return crop, crop_mask, (top, left)


def _clamp(value: int, low: int, high: int) -> int:
    """Return ``value`` clamped to ``[low, high]``.

    Args:
        value: Candidate integer.
        low: Inclusive lower bound.
        high: Inclusive upper bound.

    Returns:
        int: Clamped value.
    """
    if value < low:
        return low
    if value > high:
        return high
    return value


def sample_view(
    chips: Sequence[Chip],
    config: CanvasConfig,
    rng: np.random.Generator,
    *,
    full_frame: bool,
) -> CanvasSample:
    """Build a scene and return the full frame or a window.

    Args:
        chips: Tiles passed to :func:`build_scene`.
        config: Frame and window settings.
        rng: Generator shared by the scene and the window.
        full_frame: When True, return the scene. When False, return
            :func:`take_window`.

    Returns:
        CanvasSample: Image, mask, label, full-frame flag, and window origin.
        The origin is ``None`` for a full frame.

    Raises:
        ValueError: If :func:`build_scene` rejects the chips.
    """
    image, mask, label = build_scene(chips, config, rng)
    if full_frame:
        return CanvasSample(image=image, mask=mask, label=label, full_frame=True, origin=None)
    crop, crop_mask, origin = take_window(image, mask, config, rng)
    view_label = 1.0 if bool(np.any(crop_mask > 0.0)) else 0.0
    return CanvasSample(
        image=crop,
        mask=crop_mask,
        label=view_label,
        full_frame=False,
        origin=origin,
    )
