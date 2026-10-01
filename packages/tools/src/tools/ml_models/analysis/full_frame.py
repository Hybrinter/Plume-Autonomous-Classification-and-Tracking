"""Tile-stitch scoring for a camera frame, plus canvas eval scenes.

The frame is an 8 by 8 grid. On the 1544 by 2064 camera frame each tile is
193 by 258. One batch covers the 64 tiles. The classifier gate is the max
tile logit at or above 0. Blob pixels come from the stitched probability
mask through ``extract_blobs``. Probability and area defaults are the
controller vision gates on ``PactConfig``.

Contains:
  - tile_hw_for_frame: tile size for an 8 by 8 grid.
  - slice_tiles: row-major tiles, along-track then lateral.
  - stitch_tiles: tile probabilities back to the frame.
  - TiledFrameScore: gate result, max tile logit, and the stitched plane.
  - score_tiled_frame: one batch, stitched mask, gate, and blobs.
  - FullFrameScore: hit, empty-frame false positive, and an optional placement.
  - score_full_frame: gate and blob overlap on one probability plane.
  - placement_of: center, corner, edge, or empty from a mask centroid.
  - EvalScene: one canvas view plus the source chip.
  - build_eval_scenes: test-split scenes from ``sample_view``.
  - FrameEval: one scored scene with chip scores and tile scores.
  - summarize_frames: hit rate, empty-frame rate, chip scores, and tile scores.
  - score_dry_run: fake tile logits for a scene.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from flight.libs.config import PactConfig
from flight.payload.blobs import extract_blobs

from tools.ml_models.data.canvas import CanvasConfig, Chip, sample_view
from tools.ml_models.data.prism import FRAME_HW, TILE_GRID, TILE_HW

_VISION = PactConfig().controller.vision
DEFAULT_PROB_THRESHOLD = float(_VISION.confidence_gate)
DEFAULT_MIN_AREA = int(_VISION.min_blob_area_px)
DEFAULT_LOGIT_THRESHOLD = 0.0
_PLACEMENTS = ("center", "corner", "edge")


@dataclass(frozen=True, slots=True)
class FullFrameScore:
    """One frame after the classifier gate and blob overlap test.

    Attributes:
        hit: True when the gate is open and a blob overlaps the ground-truth mask.
        empty_false_positive: True when the ground truth is empty, the gate is
            open, and a blob is present.
        placement: Optional ``center``, ``corner``, ``edge``, or ``empty``.
    """

    hit: bool
    empty_false_positive: bool
    placement: str | None = None


@dataclass(frozen=True, slots=True)
class EvalScene:
    """One full-frame canvas view built from a pack.

    Attributes:
        image: Float32 scene ``(C, H, W)``.
        mask: Float32 mask ``(1, H, W)``.
        label: ``1`` when any mask pixel is positive.
        placement: ``center``, ``corner``, ``edge``, or ``empty``.
        chip_image: Annotated source chip ``(C, H, W)``, or ``None``.
        chip_mask: Matching chip mask, or ``None``.
    """

    image: np.ndarray
    mask: np.ndarray
    label: float
    placement: str
    chip_image: np.ndarray | None
    chip_mask: np.ndarray | None


@dataclass(frozen=True, slots=True)
class TiledFrameScore:
    """One frame after a single tile batch and a stitched probability mask.

    Attributes:
        score: Gate and blob result on the stitched mask.
        tile_logit: Max classifier logit across the tile batch.
        probability: Stitched probability plane ``(H, W)``.
    """

    score: FullFrameScore
    tile_logit: float
    probability: np.ndarray


@dataclass(frozen=True, slots=True)
class FrameEval:
    """One scored scene plus separate chip scores and tile scores.

    Attributes:
        score: Gate and blob result on the stitched frame.
        chip_logit: Classifier logit on the source chip.
        tile_logit: Max classifier logit across the tile batch.
        chip_iou: Overlap on the source chip. ``None`` when the chip has no
            positive ground truth. This value is not a pass or fail field.
        tile_iou: Overlap of the stitched probability mask and the frame
            ground truth. ``None`` when the frame ground truth is empty.
    """

    score: FullFrameScore
    chip_logit: float
    tile_logit: float
    chip_iou: float | None = None
    tile_iou: float | None = None


def _logit_value(logits: object) -> float:
    """Return the maximum classifier logit."""
    array = np.asarray(logits, dtype=np.float64)
    if array.size == 0:
        raise ValueError("classifier logits are empty")
    return float(array.reshape(-1).max())


def _plane(mask: object, name: str) -> np.ndarray:
    """Return a float32 ``(H, W)`` plane."""
    array = np.asarray(mask, dtype=np.float32)
    while array.ndim > 2 and int(array.shape[0]) == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"{name} must be (H, W); got {array.shape}")
    return array


def _blob_overlaps(bbox: tuple[int, int, int, int], gt: np.ndarray) -> bool:
    """Return True when a positive ground-truth pixel lies inside ``bbox``."""
    x0, y0, x1, y1 = bbox
    window = gt[y0 : y1 + 1, x0 : x1 + 1]
    return bool(np.any(window > 0.0))


def placement_of(mask: object) -> str:
    """Label a mask centroid as center, corner, edge, or empty.

    Args:
        mask: Ground-truth mask ``(H, W)`` or with a leading singleton axis.

    Returns:
        str: ``empty`` when no pixel is positive. Otherwise ``corner`` when the
        centroid is near both borders, ``edge`` when it is near one border,
        and ``center`` otherwise. The border band is the outer quarter of the
        frame.
    """
    plane = _plane(mask, "mask")
    ys, xs = np.where(plane > 0.0)
    if ys.size == 0:
        return "empty"
    height, width = plane.shape
    cy = float(ys.mean()) / float(max(height - 1, 1))
    cx = float(xs.mean()) / float(max(width - 1, 1))
    margin = 0.25
    near_y = cy <= margin or cy >= 1.0 - margin
    near_x = cx <= margin or cx >= 1.0 - margin
    if near_y and near_x:
        return "corner"
    if near_y or near_x:
        return "edge"
    return "center"


def score_full_frame(
    logits_classifier: object,
    prob_mask: object,
    gt_mask: object,
    *,
    logit_threshold: float = DEFAULT_LOGIT_THRESHOLD,
    prob_threshold: float = DEFAULT_PROB_THRESHOLD,
    min_area: int = DEFAULT_MIN_AREA,
    placement: str | None = None,
) -> FullFrameScore:
    """Score one full frame with the classifier gate and blob overlap.

    Args:
        logits_classifier: Classifier logit or logits. The gate reads the max.
        prob_mask: Probability mask. Sigmoid is already applied.
        gt_mask: Ground-truth mask. A positive pixel is above 0.
        logit_threshold: Gate opens at this logit. The default is 0.
        prob_threshold: Pixel threshold passed to ``extract_blobs``. The
            default is the controller vision ``confidence_gate``.
        min_area: Minimum blob area passed to ``extract_blobs``. The default
            is the controller vision ``min_blob_area_px``.
        placement: Optional label stored on the score.

    Returns:
        FullFrameScore: ``hit`` is true only when the gate is open and a blob
        overlaps the ground truth. ``empty_false_positive`` is true when the
        ground truth is empty, the gate is open, and a blob is present.

    Raises:
        ValueError: If a mask is not a plane or the logit array is empty.
    """
    logit = _logit_value(logits_classifier)
    prob = _plane(prob_mask, "prob_mask")
    gt = _plane(gt_mask, "gt_mask")
    if prob.shape != gt.shape:
        raise ValueError(f"prob_mask shape {prob.shape} != gt_mask shape {gt.shape}")
    gate_open = logit >= float(logit_threshold)
    blobs = extract_blobs(prob, float(prob_threshold), int(min_area))
    empty = not bool(np.any(gt > 0.0))
    overlaps = any(_blob_overlaps(blob.bbox, gt) for blob in blobs)
    hit = bool(gate_open and overlaps and not empty)
    empty_false_positive = bool(empty and gate_open and blobs)
    return FullFrameScore(
        hit=hit,
        empty_false_positive=empty_false_positive,
        placement=placement,
    )


def tile_hw_for_frame(frame_hw: tuple[int, int]) -> tuple[int, int]:
    """Return the tile size for an 8 by 8 grid on ``frame_hw``.

    Args:
        frame_hw: Frame ``(height, width)``.

    Returns:
        tuple[int, int]: ``(height / 8, width / 8)``. The camera frame
        ``(1544, 2064)`` returns ``(193, 258)``.

    Raises:
        ValueError: If either side is not a positive multiple of 8, or the
        camera frame does not yield the flight tile.
    """
    rows, cols = TILE_GRID
    height, width = int(frame_hw[0]), int(frame_hw[1])
    if height <= 0 or width <= 0 or height % rows != 0 or width % cols != 0:
        raise ValueError(f"frame {(height, width)} must divide into a {rows} by {cols} tile grid")
    tile = (height // rows, width // cols)
    if (height, width) == FRAME_HW and tile != TILE_HW:
        raise ValueError(f"flight tile {tile} must equal {TILE_HW} on frame {FRAME_HW}")
    return tile


def slice_tiles(frame: np.ndarray, tile_hw: tuple[int, int] | None = None) -> np.ndarray:
    """Slice a frame into non-overlapping tiles, row-major.

    Args:
        frame: ``(H, W)`` or ``(C, H, W)``.
        tile_hw: Tile ``(height, width)``. ``None`` uses
            :func:`tile_hw_for_frame`.

    Returns:
        np.ndarray: ``(N, tile_h, tile_w)`` or ``(N, C, tile_h, tile_w)``.
        Index ``row * n_cols + col`` is the tile at that grid cell. ``row``
        is along-track. ``col`` is lateral. The camera frame returns
        ``N == 64`` tiles of 193 by 258.

    Raises:
        ValueError: If ``frame`` is not a plane or a channel stack, or the
        frame does not divide into the tile grid.
    """
    array = np.ascontiguousarray(np.asarray(frame))
    if array.ndim not in (2, 3):
        raise ValueError(f"frame must be (H, W) or (C, H, W); got {array.shape}")
    frame_hw = (int(array.shape[-2]), int(array.shape[-1]))
    if tile_hw is None:
        tile_h, tile_w = tile_hw_for_frame(frame_hw)
    else:
        tile_h, tile_w = int(tile_hw[0]), int(tile_hw[1])
    if tile_h < 1 or tile_w < 1 or frame_hw[0] % tile_h != 0 or frame_hw[1] % tile_w != 0:
        raise ValueError(f"frame {frame_hw} does not divide into tiles {(tile_h, tile_w)}")
    n_rows = frame_hw[0] // tile_h
    n_cols = frame_hw[1] // tile_w
    if array.ndim == 2:
        tiles = array.reshape(n_rows, tile_h, n_cols, tile_w)
        tiles = np.transpose(tiles, (0, 2, 1, 3))
        return np.ascontiguousarray(tiles.reshape(n_rows * n_cols, tile_h, tile_w))
    channels = int(array.shape[0])
    tiles = array.reshape(channels, n_rows, tile_h, n_cols, tile_w)
    tiles = np.transpose(tiles, (1, 3, 0, 2, 4))
    return np.ascontiguousarray(tiles.reshape(n_rows * n_cols, channels, tile_h, tile_w))


def stitch_tiles(tiles: np.ndarray, frame_hw: tuple[int, int]) -> np.ndarray:
    """Stitch row-major tiles back to a frame.

    Args:
        tiles: ``(N, tile_h, tile_w)`` or ``(N, C, tile_h, tile_w)``.
        frame_hw: Frame ``(height, width)``.

    Returns:
        np.ndarray: ``(H, W)`` or ``(C, H, W)`` with each tile written to
        ``[row * tile_h, col * tile_w]``.

    Raises:
        ValueError: If the tile count does not fill ``frame_hw``.
    """
    array = np.ascontiguousarray(np.asarray(tiles))
    if array.ndim not in (3, 4):
        raise ValueError(f"tiles must be (N, H, W) or (N, C, H, W); got {array.shape}")
    height, width = int(frame_hw[0]), int(frame_hw[1])
    count = int(array.shape[0])
    tile_h = int(array.shape[-2])
    tile_w = int(array.shape[-1])
    if tile_h < 1 or tile_w < 1 or height % tile_h != 0 or width % tile_w != 0:
        raise ValueError(f"frame {(height, width)} does not divide into tiles {(tile_h, tile_w)}")
    n_rows = height // tile_h
    n_cols = width // tile_w
    if count != n_rows * n_cols:
        raise ValueError(f"need {n_rows * n_cols} tiles for frame {(height, width)}; got {count}")
    if array.ndim == 3:
        view = np.transpose(array.reshape(n_rows, n_cols, tile_h, tile_w), (0, 2, 1, 3))
        return np.ascontiguousarray(view.reshape(height, width))
    channels = int(array.shape[1])
    view = np.transpose(
        array.reshape(n_rows, n_cols, channels, tile_h, tile_w),
        (2, 0, 3, 1, 4),
    )
    return np.ascontiguousarray(view.reshape(channels, height, width))


def score_tiled_frame(
    image: np.ndarray,
    gt_mask: object,
    forward: Callable[[np.ndarray], tuple[object, object]],
    *,
    logit_threshold: float = DEFAULT_LOGIT_THRESHOLD,
    prob_threshold: float = DEFAULT_PROB_THRESHOLD,
    min_area: int = DEFAULT_MIN_AREA,
    placement: str | None = None,
) -> TiledFrameScore:
    """Score one frame from a single batch of its tiles.

    Args:
        image: Scene ``(C, H, W)``. The camera frame is 1544 by 2064.
        gt_mask: Ground-truth mask. A positive pixel is above 0.
        forward: Maps the tile batch ``(N, C, tile_h, tile_w)`` to classifier
            logits and a sigmoid probability batch. The probability batch is
            ``(N, tile_h, tile_w)`` or ``(N, 1, tile_h, tile_w)``.
        logit_threshold: Gate opens at this logit. The default is 0.
        prob_threshold: Pixel threshold passed to ``extract_blobs``.
        min_area: Minimum blob area passed to ``extract_blobs``.
        placement: Optional label stored on the score.

    Returns:
        TiledFrameScore: The gate reads the max tile logit. Blobs come from
        the stitched probability plane.

    Raises:
        ValueError: If the image is not ``(C, H, W)``, the frame does not
        divide into the tile grid, or the probability batch length disagrees
        with the tile batch.
    """
    frame = np.ascontiguousarray(np.asarray(image, dtype=np.float32))
    if frame.ndim != 3:
        raise ValueError(f"image must be (C, H, W); got {frame.shape}")
    frame_hw = (int(frame.shape[-2]), int(frame.shape[-1]))
    tiles = slice_tiles(frame)
    logits, probs = forward(tiles)
    prob_batch = np.asarray(probs, dtype=np.float32)
    batch = int(tiles.shape[0])
    got = int(prob_batch.shape[0]) if prob_batch.ndim >= 1 else 0
    if prob_batch.ndim < 1 or got != batch:
        raise ValueError(f"probability batch {got} != tile batch {batch}")
    stitched = stitch_tiles(prob_batch, frame_hw)
    score = score_full_frame(
        logits,
        stitched,
        gt_mask,
        logit_threshold=logit_threshold,
        prob_threshold=prob_threshold,
        min_area=min_area,
        placement=placement,
    )
    return TiledFrameScore(
        score=score,
        tile_logit=_logit_value(logits),
        probability=_plane(stitched, "prob_mask"),
    )


def _iou(prob: np.ndarray, gt: np.ndarray, threshold: float) -> float:
    """Return positive-class IoU of a thresholded probability mask."""
    pred = prob >= threshold
    truth = gt > 0.0
    intersection = float(np.logical_and(pred, truth).sum())
    union = float(np.logical_or(pred, truth).sum())
    if union == 0.0:
        return 1.0
    return intersection / union


def _read_splits(path: Path) -> dict[str, tuple[int, ...]]:
    """Return train, val, and test index tuples from ``splits.json``."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("splits.json must be an object")
    splits: dict[str, tuple[int, ...]] = {}
    for name in ("train", "val", "test"):
        rows = payload.get(name)
        if not isinstance(rows, list):
            raise ValueError(f"splits.json missing {name}")
        splits[name] = tuple(int(item) for item in rows)
    return splits


def _chips(
    images: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    indices: Sequence[int],
) -> tuple[Chip, ...]:
    """Return chips for ``indices``."""
    chips: list[Chip] = []
    for index in indices:
        image = np.array(images[index], dtype=np.float32, copy=True)
        mask = np.array(masks[index], dtype=np.float32, copy=True)
        if mask.ndim == 2:
            mask = mask[None, :, :]
        label = float(np.asarray(labels[index], dtype=np.float32).reshape(-1)[0])
        chips.append(
            Chip(
                image=image,
                mask=mask,
                label=label,
                group_id=str(index),
                split="test",
                annotated=bool(np.any(mask > 0.0)),
            )
        )
    return tuple(chips)


def default_canvas(chip_side: int, frame_h: int = 1544, frame_w: int = 2064) -> CanvasConfig:
    """Return a canvas whose frame defaults to 1544 by 2064.

    Args:
        chip_side: Spatial side of every chip.
        frame_h: Frame height. The default is 1544.
        frame_w: Frame width. The default is 2064.

    Returns:
        CanvasConfig: Frame, chip side, and a window that fits the frame.
    """
    window = min(512, frame_h, frame_w, chip_side)
    feather = min(6, max(chip_side // 8, 0))
    return CanvasConfig(
        frame_hw=(frame_h, frame_w),
        window_px=max(window, 1),
        chip_side=chip_side,
        feather_px=feather,
        empty_fraction=0.0,
        max_plumes=1,
    )


def build_eval_scenes(
    pack_dir: str | Path,
    canvas: CanvasConfig | None = None,
    *,
    limit: int = 1,
    seed: int = 0,
    frame_h: int | None = None,
    frame_w: int | None = None,
) -> tuple[EvalScene, ...]:
    """Build full-frame scenes from the pack test split.

    Args:
        pack_dir: Directory with ``images.npy``, ``masks.npy``, ``labels.npy``,
            and ``splits.json``.
        canvas: Scene settings. ``None`` builds :func:`default_canvas` from
            the pack chip side. ``frame_h`` and ``frame_w`` set the frame when
            ``canvas`` is omitted. An omitted canvas returns ``limit`` plume
            scenes and ``limit`` empty scenes. An explicit canvas returns
            ``limit`` draws from that canvas.
        limit: Draws per default cohort, or draws from an explicit canvas.
            Must be at least 1.
        seed: Generator seed. The default cohorts share one generator.
        frame_h: Frame height used when ``canvas`` is omitted.
        frame_w: Frame width used when ``canvas`` is omitted.

    Returns:
        tuple[EvalScene, ...]: Full-frame views. An omitted canvas returns
        ``limit`` plume scenes and ``limit`` empty scenes. An explicit canvas
        returns ``limit`` scenes. Each scene's placement comes from the mask
        centroid.

    Raises:
        ValueError: If the test split is empty, the pack has no background
            chip, or ``limit`` is below 1.
        FileNotFoundError: If a pack file is missing.
    """
    if limit < 1:
        raise ValueError(f"limit must be >= 1; got {limit}")
    root = Path(pack_dir)
    images = np.load(root / "images.npy")
    masks = np.load(root / "masks.npy")
    labels = np.load(root / "labels.npy")
    test_index = _read_splits(root / "splits.json")["test"]
    if not test_index:
        raise ValueError("test split is empty")
    chips = _chips(images, masks, labels, test_index)
    if canvas is None:
        side = int(images.shape[-1])
        if int(images.shape[-2]) != side:
            raise ValueError("pack chips must be square")
        height = 1544 if frame_h is None else int(frame_h)
        width = 2064 if frame_w is None else int(frame_w)
        base = default_canvas(side, height, width)
        canvases: tuple[CanvasConfig, ...] = (
            replace(base, empty_fraction=0.0),
            replace(base, empty_fraction=1.0),
        )
    else:
        canvases = (canvas,)
    annotated = [chip for chip in chips if chip.annotated and chip.label > 0.0]
    chip_ref = annotated[0] if annotated else None
    rng = np.random.default_rng(seed)
    scenes: list[EvalScene] = []
    for view in canvases:
        for _ in range(limit):
            sample = sample_view(chips, view, rng, full_frame=True)
            label = float(sample.label)
            chip_image = None
            chip_mask = None
            if chip_ref is not None and label > 0.0:
                chip_image = chip_ref.image
                chip_mask = chip_ref.mask
            scenes.append(
                EvalScene(
                    image=sample.image,
                    mask=sample.mask,
                    label=label,
                    placement=placement_of(sample.mask),
                    chip_image=chip_image,
                    chip_mask=chip_mask,
                )
            )
    return tuple(scenes)


def _chip_iou(mask: np.ndarray | None) -> float | None:
    """Return chip IoU, or ``None`` when the chip has no positive pixels."""
    if mask is None:
        return None
    plane = _plane(mask, "chip_mask")
    if not bool(np.any(plane > 0.0)):
        return None
    return _iou(np.clip(plane, 0.0, 1.0), plane, DEFAULT_PROB_THRESHOLD)


def _mean(values: Sequence[float]) -> float | None:
    """Return the mean of ``values``, or ``None`` when ``values`` is empty."""
    if not values:
        return None
    return float(sum(values) / len(values))


def score_dry_run(scene: EvalScene) -> FrameEval:
    """Score one scene by stitching a fake 64-tile batch.

    Args:
        scene: Canvas view. The frame must divide into an 8 by 8 tile grid.

    Returns:
        FrameEval: A positive scene uses tile logit 1 and chip logit 1.5.
        The segmentor probabilities are the ground-truth mask, sliced and
        stitched. An empty scene uses tile logit 1, chip logit 0, and paints
        a 4 by 4 blob on the stitched mask.
    """
    gt = _plane(scene.mask, "mask")
    positive = float(scene.label) > 0.0
    if positive:
        prob = np.clip(gt, 0.0, 1.0)
        chip_logit = 1.5
        chip_iou = _chip_iou(scene.chip_mask)
    else:
        prob = np.zeros_like(gt)
        prob[:4, :4] = 1.0
        chip_logit = 0.0
        chip_iou = None
    prob_tiles = slice_tiles(prob)

    def forward(tiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if int(tiles.shape[0]) != int(prob_tiles.shape[0]):
            raise ValueError(
                f"tile batch {tiles.shape[0]} != frame tile count {prob_tiles.shape[0]}"
            )
        logits = np.ones((int(tiles.shape[0]),), dtype=np.float64)
        return logits, prob_tiles

    tiled = score_tiled_frame(scene.image, gt, forward, placement=scene.placement)
    tile_iou = _iou(tiled.probability, gt, DEFAULT_PROB_THRESHOLD) if positive else None
    return FrameEval(
        score=tiled.score,
        chip_logit=chip_logit,
        tile_logit=tiled.tile_logit,
        chip_iou=chip_iou,
        tile_iou=tile_iou,
    )


def summarize_frames(records: Sequence[FrameEval]) -> dict[str, object]:
    """Summarize hit rate, empty-frame false positives, and separate scores.

    Args:
        records: Scored scenes.

    Returns:
        dict[str, object]: ``full_frame_hit_rate`` is the fraction of
        non-empty scenes that hit. ``hit_rate_by_placement`` maps center,
        corner, and edge. ``empty_frame_false_positive_rate`` is the fraction
        of empty scenes with a false positive. ``chip_scores`` and
        ``tile_scores`` each carry ``logit`` and ``iou`` means over non-empty
        scenes. ``chip_iou`` repeats the chip IoU mean and is not a pass or
        fail field.
    """
    by_place: dict[str, list[bool]] = {name: [] for name in _PLACEMENTS}
    empty_flags: list[bool] = []
    plume_hits: list[bool] = []
    chip_logits: list[float] = []
    tile_logits: list[float] = []
    chip_ious: list[float] = []
    tile_ious: list[float] = []
    for record in records:
        placement = record.score.placement
        if placement == "empty":
            empty_flags.append(record.score.empty_false_positive)
            continue
        plume_hits.append(record.score.hit)
        chip_logits.append(record.chip_logit)
        tile_logits.append(record.tile_logit)
        if placement in by_place:
            by_place[placement].append(record.score.hit)
        if record.chip_iou is not None:
            chip_ious.append(record.chip_iou)
        if record.tile_iou is not None:
            tile_ious.append(record.tile_iou)
    rates: dict[str, float | None] = {}
    for name in _PLACEMENTS:
        hits = by_place[name]
        rates[name] = None if not hits else float(sum(hits) / len(hits))
    chip_iou = _mean(chip_ious)
    return {
        "full_frame_hit_rate": None if not plume_hits else float(sum(plume_hits) / len(plume_hits)),
        "hit_rate_by_placement": rates,
        "empty_frame_false_positive_rate": (
            None if not empty_flags else float(sum(empty_flags) / len(empty_flags))
        ),
        "chip_scores": {"logit": _mean(chip_logits), "iou": chip_iou},
        "tile_scores": {"logit": _mean(tile_logits), "iou": _mean(tile_ious)},
        "chip_iou": chip_iou,
        "n_scenes": len(records),
    }
