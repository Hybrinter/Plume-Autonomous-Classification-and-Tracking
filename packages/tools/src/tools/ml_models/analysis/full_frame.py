"""Full-frame hit, empty-frame false positives, and canvas eval scenes.

The classifier gate is a logit at or above 0. Blob pixels come from a
probability mask through ``extract_blobs``. Probability and area defaults
are the controller vision gates on ``PactConfig``.

Contains:
  - FullFrameScore: hit, empty-frame false positive, and an optional placement.
  - score_full_frame: one frame's gate and blob overlap.
  - placement_of: center, corner, edge, or empty from a mask centroid.
  - EvalScene: one canvas view plus the source chip.
  - build_eval_scenes: test-split scenes from ``sample_view``.
  - FrameEval: one scored scene with chip and frame logits.
  - summarize_frames: hit rate, empty-frame rate, and logit margin.
  - score_dry_run: fake logits for a scene.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from flight.libs.config import PactConfig
from flight.payload.blobs import extract_blobs

from tools.ml_models.data.canvas import CanvasConfig, Chip, sample_view

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
class FrameEval:
    """One scored scene plus the logits used for the margin.

    Attributes:
        score: Gate and blob result.
        chip_logit: Max logit on the source chip.
        frame_logit: Classifier logit on the full frame.
        chip_iou: Overlap of the probability mask and the ground truth.
            ``None`` when the ground truth is empty.
    """

    score: FullFrameScore
    chip_logit: float
    frame_logit: float
    chip_iou: float | None = None


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


def score_dry_run(scene: EvalScene) -> FrameEval:
    """Score one scene with fixed logits and the scene mask as probabilities.

    Args:
        scene: Canvas view.

    Returns:
        FrameEval: A positive scene uses frame logit 1 and chip logit 1.5.
        An empty scene uses frame logit 1 and paints a 4 by 4 blob.
    """
    gt = np.asarray(scene.mask[0], dtype=np.float32)
    if float(scene.label) > 0.0:
        frame_logit = 1.0
        chip_logit = 1.5
        prob = np.clip(gt, 0.0, 1.0)
        iou = _iou(prob, gt, DEFAULT_PROB_THRESHOLD)
    else:
        frame_logit = 1.0
        chip_logit = 0.0
        prob = np.zeros_like(gt)
        prob[:4, :4] = 1.0
        iou = None
    scored = score_full_frame(
        frame_logit,
        prob,
        gt,
        placement=scene.placement,
    )
    return FrameEval(
        score=scored,
        chip_logit=chip_logit,
        frame_logit=frame_logit,
        chip_iou=iou,
    )


def summarize_frames(records: Sequence[FrameEval]) -> dict[str, object]:
    """Summarize hit rate, empty-frame false positives, and logit margin.

    Args:
        records: Scored scenes.

    Returns:
        dict[str, object]: ``full_frame_hit_rate`` is the fraction of
        non-empty scenes that hit. ``hit_rate_by_placement`` maps center,
        corner, and edge. ``empty_frame_false_positive_rate`` is the fraction
        of empty scenes with a false positive. ``chip_vs_frame_logit_margin``
        is the mean of chip logit minus frame logit. ``chip_iou`` is the mean
        overlap and is not a pass or fail field.
    """
    by_place: dict[str, list[bool]] = {name: [] for name in _PLACEMENTS}
    empty_flags: list[bool] = []
    plume_hits: list[bool] = []
    margins: list[float] = []
    ious: list[float] = []
    for record in records:
        placement = record.score.placement
        if placement == "empty":
            empty_flags.append(record.score.empty_false_positive)
            continue
        plume_hits.append(record.score.hit)
        margins.append(record.chip_logit - record.frame_logit)
        if placement in by_place:
            by_place[placement].append(record.score.hit)
        if record.chip_iou is not None:
            ious.append(record.chip_iou)
    rates: dict[str, float | None] = {}
    for name in _PLACEMENTS:
        hits = by_place[name]
        rates[name] = None if not hits else float(sum(hits) / len(hits))
    return {
        "full_frame_hit_rate": None if not plume_hits else float(sum(plume_hits) / len(plume_hits)),
        "hit_rate_by_placement": rates,
        "empty_frame_false_positive_rate": (
            None if not empty_flags else float(sum(empty_flags) / len(empty_flags))
        ),
        "chip_vs_frame_logit_margin": None if not margins else float(sum(margins) / len(margins)),
        "chip_iou": None if not ious else float(sum(ious) / len(ious)),
        "n_scenes": len(records),
    }
