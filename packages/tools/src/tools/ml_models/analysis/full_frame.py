"""Evaluate finished flight frames through flight's shared tile inference path.

Only complete classifier frames are scored. Full-frame mask scores require
every configured ground mask; partial annotations contribute per-tile metrics.
No synthetic scene composition or resizing is performed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from flight.libs.types import Err, FaultCode, Ok, Result
from flight.payload.gimbal.footprint import to_model_gsd
from flight.payload.inference.classifier import TileClassifierBackend
from flight.payload.inference.detector import infer_tiles
from flight.payload.inference.segmentor import TileSegmentorBackend
from flight.payload.preprocess.tiling import slice_frame, stitch_tiles
from torch import nn

from tools.ml_models.dataset.geometry import (
    GSD_REFERENCE_M,
    INPUT_BANDS,
    frame_hw,
    grid_hw,
    tile_hw,
)
from tools.ml_models.dataset.manifest import load_manifest
from tools.ml_models.dataset.store import read_rows
from tools.ml_models.train.metrics import classifier_metrics, compute_dice, compute_iou


@dataclass(frozen=True, slots=True)
class TorchTileClassifier:
    """Adapt a torch classifier to flight's NumPy tile backend contract."""

    model: nn.Module

    def classify_tiles(self, images: np.ndarray, gsd: np.ndarray) -> Result[np.ndarray, FaultCode]:
        """Return raw logits as a NumPy vector, reporting malformed outputs."""
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        try:
            with torch.no_grad():
                output = self.model(
                    torch.as_tensor(images, dtype=torch.float32, device=device),
                    torch.as_tensor(gsd, dtype=torch.float32, device=device),
                )
            if output.shape == (images.shape[0], 1):
                output = output[:, 0]
            if output.shape != (images.shape[0],) or not bool(torch.isfinite(output).all()):
                return Err(FaultCode.INFERENCE_NAN)
            return Ok(output.detach().cpu().numpy().astype(np.float32, copy=False))
        except RuntimeError, TypeError, ValueError:
            return Err(FaultCode.FRAME_MALFORMED)


@dataclass(frozen=True, slots=True)
class TorchTileSegmentor:
    """Adapt torch segmentation logits to flight's probability mask contract."""

    model: nn.Module

    def segment_tiles(self, images: np.ndarray, gsd: np.ndarray) -> Result[np.ndarray, FaultCode]:
        """Return sigmoid probabilities in ``(N,1,h,w)`` layout."""
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        try:
            with torch.no_grad():
                output = self.model(
                    torch.as_tensor(images, dtype=torch.float32, device=device),
                    torch.as_tensor(gsd, dtype=torch.float32, device=device),
                )
            expected = (images.shape[0], 1, images.shape[2], images.shape[3])
            if output.shape != expected or not bool(torch.isfinite(output).all()):
                return Err(FaultCode.INFERENCE_NAN)
            probabilities = torch.sigmoid(output)
            return Ok(probabilities.detach().cpu().numpy().astype(np.float32, copy=False))
        except RuntimeError, TypeError, ValueError:
            return Err(FaultCode.FRAME_MALFORMED)


@dataclass(frozen=True, slots=True)
class TiledScore:
    """Stitched probability mask, row-major logits and positive flags."""

    mask: np.ndarray
    logits: np.ndarray
    positive: np.ndarray


def score_tiled_frame(
    classifier: nn.Module,
    segmentor: nn.Module,
    frame: np.ndarray,
    tile_gsd_m: np.ndarray,
    *,
    gsd_reference_m: float = GSD_REFERENCE_M,
    logit_threshold: float = 0.0,
) -> TiledScore:
    """Classify one full 3-band unit frame through shared flight inference."""
    if not np.isfinite(logit_threshold):
        raise ValueError("classifier threshold must be finite")
    image = np.asarray(frame, dtype=np.float32)
    if image.ndim != 3 or image.shape != (len(INPUT_BANDS), *frame_hw()):
        raise ValueError("flight frame requires a complete frame with configured bands")
    grid = grid_hw()
    tile_count = grid[0] * grid[1]
    tiles_result = slice_frame(image[None], grid)
    if isinstance(tiles_result, Err):
        raise ValueError("flight frame cannot be divided into tiles")
    gsd = np.asarray(tile_gsd_m, dtype=np.float32)
    if gsd.shape != (tile_count, 2):
        raise ValueError(f"flight frame requires {tile_count} GSD pairs")
    encoded = to_model_gsd(gsd, gsd_reference_m)
    if isinstance(encoded, Err):
        raise ValueError("flight frame contains invalid GSD")

    old_classifier, old_segmentor = classifier.training, segmentor.training
    classifier.eval()
    segmentor.eval()
    try:
        classifier_backend: TileClassifierBackend = TorchTileClassifier(classifier)
        segmentor_backend: TileSegmentorBackend = TorchTileSegmentor(segmentor)
        scored = infer_tiles(
            classifier_backend,
            segmentor_backend,
            tiles_result.value,
            encoded.value,
            float(logit_threshold),
        )
    finally:
        classifier.train(old_classifier)
        segmentor.train(old_segmentor)
    if isinstance(scored, Err):
        raise ValueError(f"flight tile inference failed: {scored.error}")
    stitched = stitch_tiles(scored.value.masks, grid)
    if isinstance(stitched, Err):
        raise ValueError("flight tile masks could not be stitched")
    return TiledScore(stitched.value, scored.value.logits, scored.value.positive)


def evaluate_flight_frames(
    dataset: str | Path,
    classifier: nn.Module,
    segmentor: nn.Module,
    *,
    logit_threshold: float = 0.0,
) -> dict[str, object]:
    """Score complete test frames and preserve unannotated masks as unknown."""
    root = Path(dataset)
    manifest = load_manifest(root / "dataset.json")
    if manifest.source != "flight":
        raise ValueError("full-frame evaluation requires a finished flight dataset")
    grid = grid_hw()
    grid_rows, grid_cols = grid
    tile_count = grid_rows * grid_cols
    frames: dict[str, dict[int, tuple[np.ndarray, np.ndarray, float, str, str]]] = defaultdict(dict)
    masks: dict[str, np.ndarray] = {}
    for shard in manifest.shards:
        if shard.split != "test":
            continue
        directory = root / shard.task / "test" / f"{shard.height}x{shard.width}"
        row_records = read_rows(directory)
        if (shard.height, shard.width) != tile_hw():
            raise ValueError("flight evaluation found non-flight tile dimensions")
        if shard.task == "segmentor":
            gold = np.load(directory / "masks.npy", mmap_mode="r")
            for index, row in enumerate(row_records):
                if row.element != "id" or row.tile_id in masks:
                    raise ValueError("test masks contain duplicates or augmented rows")
                masks[row.tile_id] = np.asarray(gold[index, 0])
            continue
        images = np.load(directory / "images.npy", mmap_mode="r")
        gsds = np.load(directory / "gsd.npy", mmap_mode="r")
        labels = np.load(directory / "labels.npy", mmap_mode="r")
        for index, row in enumerate(row_records):
            if row.frame_id is None or row.grid_rc is None or row.element != "id":
                raise ValueError("test tile lacks original frame coordinates")
            r, c = row.grid_rc
            if (
                not (0 <= r < grid_rows and 0 <= c < grid_cols)
                or r * grid_cols + c in frames[row.frame_id]
            ):
                raise ValueError("test frame has invalid or repeated tile coordinates")
            frames[row.frame_id][r * grid_cols + c] = (
                np.asarray(images[index], dtype=np.float32),
                np.asarray(gsds[index]),
                float(labels[index, 0]),
                row.tile_id,
                row.bin_id or "unbinned",
            )
    reports: list[dict[str, object]] = []
    bins: dict[str, list[tuple[float, float, float | None, float | None]]] = defaultdict(list)
    skipped: list[str] = []
    for frame_id, records in sorted(frames.items()):
        if set(records) != set(range(tile_count)):
            skipped.append(frame_id)
            continue
        ordered = [records[index] for index in range(tile_count)]
        image_tiles = np.stack([row[0] for row in ordered])
        frame_result = stitch_tiles(image_tiles, grid)
        if isinstance(frame_result, Err):
            raise ValueError("stored frame tiles could not be stitched")
        frame = frame_result.value
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
        predicted_result = slice_frame(scored.mask[None], grid)
        if isinstance(predicted_result, Err):
            raise ValueError("scored frame mask could not be divided into tiles")
        predicted = predicted_result.value[:, 0]
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
        if annotated == tile_count:
            gold_tiles = np.stack([masks[row[3]][None] for row in ordered])
            gold_result = stitch_tiles(gold_tiles, grid)
            if isinstance(gold_result, Err):
                raise ValueError("stored ground masks could not be stitched")
            whole_iou = compute_iou(scored.mask, gold_result.value)
            whole_dice = compute_dice(scored.mask, gold_result.value)
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
