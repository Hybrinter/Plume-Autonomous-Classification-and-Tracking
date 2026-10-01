"""Evaluate real finished flight frames as 64 GSD-conditioned tiles.

Only complete classifier frames are scored. Full-frame mask scores require
all 64 ground masks; partial annotations contribute only to per-tile metrics.
No synthetic scene composition or resizing is performed.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tools.ml_models.dataset.geometry import frame_hw, slice_frame, stitch_tiles, tile_hw
from tools.ml_models.dataset.manifest import load_manifest
from tools.ml_models.dataset.preprocess import dequantize_unit, to_model_gsd
from tools.ml_models.dataset.store import read_rows
from tools.ml_models.train.metrics import classifier_metrics, compute_dice, compute_iou


@dataclass(frozen=True, slots=True)
class TiledScore:
    """Frame probability mask, row-major logits, and row-major positive flags."""

    mask: np.ndarray
    logits: np.ndarray
    positive: np.ndarray


def score_tiled_frame(
    classifier: nn.Module,
    segmentor: nn.Module,
    frame: np.ndarray,
    tile_gsd_m: np.ndarray,
    *,
    gsd_reference_m: float = 15.87,
    logit_threshold: float = 0.0,
) -> TiledScore:
    """Classify one full 3-band unit frame and segment only selected tiles."""
    if not math.isfinite(logit_threshold):
        raise ValueError("classifier threshold must be finite")
    tiles = slice_frame(np.asarray(frame, dtype=np.float32)[None])
    gsd = np.asarray(tile_gsd_m, dtype=np.float32)
    if gsd.shape != (64, 2):
        raise ValueError("flight frame requires 64 GSD pairs")
    encoded = to_model_gsd(gsd, gsd_reference_m)
    images = torch.from_numpy(tiles)
    condition = torch.from_numpy(encoded)
    old_classifier, old_segmentor = classifier.training, segmentor.training
    classifier.eval()
    segmentor.eval()
    try:
        with torch.no_grad():
            output = classifier(images, condition)
            if output.shape != (64, 1) or not torch.isfinite(output).all():
                raise ValueError("invalid classifier output")
            logits = output[:, 0].cpu().numpy()
            positive = logits >= logit_threshold
            masks = np.zeros((64, 1, *tile_hw()), dtype=np.float32)
            indices = np.flatnonzero(positive)
            if len(indices):
                segmentation = segmentor(images[indices], condition[indices])
                if (
                    segmentation.shape != (len(indices), 1, *tile_hw())
                    or not torch.isfinite(segmentation).all()
                ):
                    raise ValueError("invalid segmentor output")
                masks[indices] = torch.sigmoid(segmentation).cpu().numpy()
            return TiledScore(stitch_tiles(masks), np.asarray(logits), positive)
    finally:
        classifier.train(old_classifier)
        segmentor.train(old_segmentor)


def evaluate_flight_frames(
    dataset: str | Path,
    classifier: nn.Module,
    segmentor: nn.Module,
    *,
    logit_threshold: float = 0.0,
) -> dict[str, object]:
    """Score complete 64-tile test frames; preserve unknown masks as unknown."""
    root = Path(dataset)
    manifest = load_manifest(root / "dataset.json")
    if manifest.source != "flight":
        raise ValueError("full-frame evaluation requires a finished flight dataset")
    frames: dict[str, dict[int, tuple[np.ndarray, np.ndarray, float, str, str]]] = defaultdict(dict)
    masks: dict[str, np.ndarray] = {}
    for shard in manifest.shards:
        if shard.split != "test":
            continue
        directory = root / shard.task / "test" / f"{shard.height}x{shard.width}"
        rows = read_rows(directory)
        if (shard.height, shard.width) != tile_hw():
            raise ValueError("flight evaluation found non-flight tile dimensions")
        if shard.task == "segmentor":
            gold = np.load(directory / "masks.npy", mmap_mode="r")
            for index, row in enumerate(rows):
                if row.element != "id" or row.tile_id in masks:
                    raise ValueError("test masks contain duplicates or augmented rows")
                masks[row.tile_id] = np.asarray(gold[index, 0])
            continue
        images = np.load(directory / "images.npy", mmap_mode="r")
        gsds = np.load(directory / "gsd.npy", mmap_mode="r")
        labels = np.load(directory / "labels.npy", mmap_mode="r")
        for index, row in enumerate(rows):
            if row.frame_id is None or row.grid_rc is None or row.element != "id":
                raise ValueError("test tile lacks original frame coordinates")
            r, c = row.grid_rc
            if not (0 <= r < 8 and 0 <= c < 8) or r * 8 + c in frames[row.frame_id]:
                raise ValueError("test frame has invalid or repeated tile coordinates")
            frames[row.frame_id][r * 8 + c] = (
                dequantize_unit(images[index]),
                np.asarray(gsds[index]),
                float(labels[index, 0]),
                row.tile_id,
                row.bin_id or "unbinned",
            )
    reports: list[dict[str, object]] = []
    bins: dict[str, list[tuple[float, float, float | None, float | None]]] = defaultdict(list)
    skipped: list[str] = []
    for frame_id, records in sorted(frames.items()):
        if set(records) != set(range(64)):
            skipped.append(frame_id)
            continue
        ordered = [records[index] for index in range(64)]
        image_tiles = np.stack([row[0] for row in ordered])
        frame = stitch_tiles(image_tiles)
        gsd = np.stack([row[1] for row in ordered])
        labels = np.asarray([row[2] for row in ordered])
        scored = score_tiled_frame(
            classifier,
            segmentor,
            frame,
            gsd,
            gsd_reference_m=manifest.gsd_reference_m,
            logit_threshold=logit_threshold,
        )
        predicted = slice_frame(scored.mask[None, None])[:, 0]
        annotated = 0
        for index, (_, _, label, tile_id, bin_id) in enumerate(ordered):
            gold = masks.get(tile_id)
            iou = dice = None
            if gold is not None:
                annotated += 1
                iou = compute_iou(predicted[index], gold)
                dice = compute_dice(predicted[index], gold)
            bins[bin_id].append((float(scored.logits[index]), label, iou, dice))
        whole_iou = whole_dice = None
        if annotated == 64:
            gold_tiles = np.stack([masks[row[3]][None] for row in ordered])
            gold_frame = stitch_tiles(gold_tiles)
            whole_iou = compute_iou(scored.mask, gold_frame)
            whole_dice = compute_dice(scored.mask, gold_frame)
        reports.append(
            {
                "frame_id": frame_id,
                "annotated_tiles": annotated,
                "classifier": asdict(classifier_metrics(scored.logits, labels, logit_threshold)),
                "full_frame_iou": whole_iou,
                "full_frame_dice": whole_dice,
                "segmented_tiles": int(scored.positive.sum()),
            }
        )
    bin_reports: dict[str, object] = {}
    for name, bin_rows in sorted(bins.items()):
        overlap = [item for item in bin_rows if item[2] is not None]
        bin_reports[name] = {
            "classifier": asdict(
                classifier_metrics(
                    np.asarray([item[0] for item in bin_rows]),
                    np.asarray([item[1] for item in bin_rows]),
                    logit_threshold,
                )
            ),
            "annotated_tiles": len(overlap),
            "mean_tile_iou": (
                sum(float(item[2]) for item in overlap if item[2] is not None) / len(overlap)
                if overlap
                else None
            ),
            "mean_tile_dice": (
                sum(float(item[3]) for item in overlap if item[3] is not None) / len(overlap)
                if overlap
                else None
            ),
        }
    return {
        "dataset_hash": manifest.dataset_hash,
        "frame_hw": frame_hw(),
        "frames": reports,
        "incomplete_frame_ids": skipped,
        "bins": bin_reports,
    }
