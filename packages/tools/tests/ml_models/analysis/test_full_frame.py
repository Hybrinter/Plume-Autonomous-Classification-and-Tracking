"""Tests for the 64-tile full-frame evaluation."""

from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from flight.libs.types import Ok
from flight.payload.preprocess.tiling import slice_frame as flight_slice_frame
from tools.ml_models.analysis.full_frame import evaluate_flight_frames, score_tiled_frame
from tools.ml_models.dataset.augment import AugmentRecipe
from tools.ml_models.dataset.geometry import (
    INPUT_BANDS,
    TILE_H_PX,
    TILE_W_PX,
    frame_hw,
)
from tools.ml_models.dataset.manifest import (
    DatasetManifest,
    ShardCount,
    compute_dataset_hash,
    load_manifest,
    write_manifest,
)
from tools.ml_models.dataset.split import SplitRecipe
from tools.ml_models.dataset.store import RowRecord, ShardWriter
from torch import nn

_GSD_M = np.array([16.0, 16.0], dtype=np.float32)
_POSITIVE_INDICES = (1, 5, 63)


class _StubClassifier(nn.Module):
    """Positive exactly where the tile mean is high."""

    def forward(self, image: torch.Tensor, gsd: torch.Tensor) -> torch.Tensor:
        """Return +4 where the tile mean exceeds half scale, else -4."""
        mean = image.mean(dim=(1, 2, 3))
        return torch.where(mean > 0.5, 4.0, -4.0).unsqueeze(1)


class _StubSegmentor(nn.Module):
    """Emit a large positive logit mask for every input tile."""

    def forward(self, image: torch.Tensor, gsd: torch.Tensor) -> torch.Tensor:
        """Return +30 logits so sigmoid probabilities sit near 1."""
        return torch.full(
            (image.shape[0], 1, image.shape[2], image.shape[3]),
            30.0,
            dtype=image.dtype,
        )


def _tile_image(positive: bool) -> np.ndarray:
    """Return a float32 unit tile filled for the stub classifier."""
    fill = 1.0 if positive else 0.0
    return np.full((3, TILE_H_PX, TILE_W_PX), fill, dtype=np.float32)


def _row(
    frame_id: str,
    index: int,
    *,
    bin_id: str = "elevation15",
    tile_suffix: str = "",
) -> RowRecord:
    """Build one unaugmented test row for ``frame_id`` at flat grid index."""
    return RowRecord(
        tile_id=f"{frame_id}-{index}{tile_suffix}",
        group_id=frame_id,
        frame_id=frame_id,
        grid_rc=(index // 8, index % 8),
        bin_id=bin_id,
        element="id",
        theta_g_deg=15.0,
    )


def _write_dataset(
    root: Path,
    frames: dict[str, list[int]],
    annotated: dict[str, set[int]],
) -> None:
    """Write classifier and segmentor test shards plus ``dataset.json``.

    Args:
        root: Finished dataset directory.
        frames: frame_id -> flat grid indices present in the classifier shard.
        annotated: frame_id -> grid indices whose tiles carry a ground mask.
    """
    classifier_rows: list[tuple[np.ndarray, RowRecord]] = []
    segmentor_rows: list[tuple[np.ndarray, np.ndarray, RowRecord]] = []
    for frame_id, indices in frames.items():
        for index in indices:
            bin_id = "elevation15" if index < 32 else "elevation45"
            row = _row(frame_id, index, bin_id=bin_id)
            image = _tile_image(index in _POSITIVE_INDICES)
            classifier_rows.append((image, row))
            if index in annotated.get(frame_id, set()):
                fill = 1 if index in _POSITIVE_INDICES else 0
                mask = np.full((1, TILE_H_PX, TILE_W_PX), fill, dtype=np.uint8)
                segmentor_rows.append((image, mask, row))
    root.mkdir(parents=True)
    counts: list[ShardCount] = []
    classifier_dir = root / "classifier" / "test" / f"{TILE_H_PX}x{TILE_W_PX}"
    writer = ShardWriter(
        classifier_dir,
        len(classifier_rows),
        TILE_H_PX,
        TILE_W_PX,
        channels=3,
        with_masks=False,
    )
    n_positive = 0
    for image, row in classifier_rows:
        label = (
            1.0
            if row.grid_rc is not None
            and (row.grid_rc[0] * 8 + row.grid_rc[1]) in _POSITIVE_INDICES
            else 0.0
        )
        n_positive += int(label)
        writer.append(image, _GSD_M, label, None, row)
    writer.close()
    counts.append(
        ShardCount(
            task="classifier",
            split="test",
            height=TILE_H_PX,
            width=TILE_W_PX,
            n=len(classifier_rows),
            n_positive=n_positive,
        )
    )
    if segmentor_rows:
        segmentor_dir = root / "segmentor" / "test" / f"{TILE_H_PX}x{TILE_W_PX}"
        writer = ShardWriter(
            segmentor_dir,
            len(segmentor_rows),
            TILE_H_PX,
            TILE_W_PX,
            channels=3,
            with_masks=True,
        )
        for image, mask, row in segmentor_rows:
            writer.append(image, _GSD_M, 0.0, mask, row)
        writer.close()
        counts.append(
            ShardCount(
                task="segmentor",
                split="test",
                height=TILE_H_PX,
                width=TILE_W_PX,
                n=len(segmentor_rows),
                n_positive=0,
            )
        )
    manifest = DatasetManifest(
        source="flight",
        source_ref="full-frame-test",
        weight_table_id="",
        band_names=INPUT_BANDS,
        norm="unit",
        image_dtype="float32",
        gsd_reference_m=15.87,
        split=SplitRecipe(),
        augment=AugmentRecipe(),
        bins=(),
        shards=tuple(counts),
        gsd_lateral_min_m=16.0,
        gsd_lateral_max_m=16.0,
        gsd_along_min_m=16.0,
        gsd_along_max_m=16.0,
        dataset_hash=compute_dataset_hash(root),
        schema_version=2,
    )
    write_manifest(root / "dataset.json", manifest)


def test_complete_frames_score_and_incomplete_reported(tmp_path: Path) -> None:
    """Complete frames score; the incomplete frame is reported, not scored."""
    root = tmp_path / "ds"
    _write_dataset(
        root,
        {
            "f-full": list(range(64)),
            "f-part": list(range(64)),
            "f-skip": list(range(63)),
        },
        {
            "f-full": set(range(64)),
            "f-part": {0, 1, 5},
        },
    )
    report = evaluate_flight_frames(root, _StubClassifier(), _StubSegmentor())
    assert report["incomplete_frame_ids"] == ["f-skip"]
    entries = cast("list[dict[str, object]]", report["frames"])
    frames = {str(entry["frame_id"]): entry for entry in entries}
    assert set(frames) == {"f-full", "f-part"}
    full = frames["f-full"]
    assert full["annotated_tiles"] == 64
    assert full["segmented_tiles"] == len(_POSITIVE_INDICES)
    classifier = cast("dict[str, object]", full["classifier"])
    assert classifier["n"] == 64
    assert classifier["accuracy"] == 1.0
    assert full["full_frame_iou"] == pytest.approx(1.0)
    assert full["full_frame_dice"] == pytest.approx(1.0)
    part = frames["f-part"]
    assert part["annotated_tiles"] == 3
    assert part["full_frame_iou"] is None
    assert part["full_frame_dice"] is None
    assert part["segmented_tiles"] == len(_POSITIVE_INDICES)
    bins = cast("dict[str, dict[str, object]]", report["bins"])
    assert set(bins) == {"elevation15", "elevation45"}
    for entry in bins.values():
        bin_classifier = cast("dict[str, object]", entry["classifier"])
        assert cast("int", bin_classifier["n"]) > 0
        assert cast("int", entry["annotated_tiles"]) > 0
        assert entry["mean_tile_iou"] == pytest.approx(1.0)
        assert entry["mean_tile_dice"] == pytest.approx(1.0)


def test_duplicate_coordinates_rejected(tmp_path: Path) -> None:
    """A repeated (frame, row, col) pair is an error."""
    root = tmp_path / "ds"
    _write_dataset(root, {"f-dup": list(range(64))}, {})
    shard = root / "classifier" / "test" / f"{TILE_H_PX}x{TILE_W_PX}"
    rows_path = shard / "rows.jsonl"
    lines = rows_path.read_text(encoding="utf-8").splitlines()
    lines.append(lines[0])
    rows_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    images = np.load(shard / "images.npy")
    np.save(
        shard / "images.npy",
        np.concatenate([images, images[:1]]),
    )
    labels = np.load(shard / "labels.npy")
    np.save(shard / "labels.npy", np.concatenate([labels, labels[:1]]))
    gsd = np.load(shard / "gsd.npy")
    np.save(shard / "gsd.npy", np.concatenate([gsd, gsd[:1]]))
    manifest = load_manifest(root / "dataset.json", verify=False)
    write_manifest(
        root / "dataset.json",
        replace(manifest, dataset_hash=compute_dataset_hash(root)),
    )
    with pytest.raises(ValueError, match="invalid or repeated"):
        evaluate_flight_frames(root, _StubClassifier(), _StubSegmentor())


def test_score_tiled_frame_segments_only_positive_tiles() -> None:
    """The segmentor runs only on classifier-positive tiles, in order."""
    seen: list[int] = []

    class _RecordingSegmentor(_StubSegmentor):
        def forward(self, image: torch.Tensor, gsd: torch.Tensor) -> torch.Tensor:
            """Record the batch size the segmentor received."""
            seen.append(image.shape[0])
            return super().forward(image, gsd)

    frame = np.zeros((3, *frame_hw()), dtype=np.float32)
    for index in _POSITIVE_INDICES:
        row, col = index // 8, index % 8
        frame[
            :,
            row * TILE_H_PX : (row + 1) * TILE_H_PX,
            col * TILE_W_PX : (col + 1) * TILE_W_PX,
        ] = 1.0
    gsd = np.tile(_GSD_M, (64, 1))
    score = score_tiled_frame(_StubClassifier(), _RecordingSegmentor(), frame, gsd)
    assert seen == [len(_POSITIVE_INDICES)]
    assert score.positive.tolist() == [index in _POSITIVE_INDICES for index in range(64)]
    result = flight_slice_frame(score.mask[None])
    assert isinstance(result, Ok)
    tiles = result.value[:, 0]
    for index in range(64):
        expected = 1.0 if index in _POSITIVE_INDICES else 0.0
        assert float(tiles[index].mean()) == pytest.approx(expected)


def test_score_tiled_frame_restores_training_mode() -> None:
    """The scorer leaves each model in the mode it found."""
    classifier = _StubClassifier()
    segmentor = _StubSegmentor()
    classifier.train()
    segmentor.train()
    frame = np.zeros((3, *frame_hw()), dtype=np.float32)
    gsd = np.tile(_GSD_M, (64, 1))
    score_tiled_frame(classifier, segmentor, frame, gsd)
    assert classifier.training
    assert segmentor.training


def test_score_tiled_frame_rejects_bad_inputs() -> None:
    """Wrong GSD count and a non-finite threshold are rejected."""
    frame = np.zeros((3, *frame_hw()), dtype=np.float32)
    with pytest.raises(ValueError, match="64 GSD pairs"):
        score_tiled_frame(_StubClassifier(), _StubSegmentor(), frame, np.zeros((3, 2)))
    with pytest.raises(ValueError, match="threshold"):
        score_tiled_frame(
            _StubClassifier(),
            _StubSegmentor(),
            frame,
            np.tile(_GSD_M, (64, 1)),
            logit_threshold=float("nan"),
        )
