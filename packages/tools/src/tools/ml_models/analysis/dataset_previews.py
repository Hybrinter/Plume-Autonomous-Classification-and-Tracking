"""Identity-bound compact dataset preview capture, without re-measurement.

Selected inputs are copied from exact image/mask keys and inverted to source
orientation. Only explicit stored masks are retained. Semantic display channel
selection never alters the captured float32 unit pixels.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from flight.libs.types import Err, Ok, Result

from tools.ml_models.analysis.artifacts import BundleFile, dataset_identity
from tools.ml_models.analysis.config import CaptureConfig
from tools.ml_models.analysis.dataset import (
    DatasetMeasurement,
    DatasetSample,
    _array_hash,
    _inverse,
)
from tools.ml_models.analysis.visuals.selection import GalleryPlan, select_dataset_galleries
from tools.ml_models.dataset.manifest import shard_dir


@dataclass(frozen=True, slots=True)
class DatasetPreview:
    """Captured input identity and display mapping; arrays live only in referenced NPZ bytes."""

    variant_id: str
    path: str
    sha256: str
    size_bytes: int
    sample: DatasetSample
    display_indices: tuple[int, ...]
    display_label: str


@dataclass(frozen=True, slots=True)
class DatasetPreviewCapture:
    """Bounded immutable preview files and the exact deterministic gallery plan."""

    previews: tuple[DatasetPreview, ...]
    files: tuple[BundleFile, ...]
    plan: GalleryPlan
    method: str = "exact_canonical_float32_unit_inputs_explicit_uint8_masks_v1"


def display_channels(band_names: tuple[str, ...]) -> tuple[tuple[int, ...], str]:
    """Return semantic RGB indices only when each named RGB band is unambiguous.

    Other three-channel views explicitly say not RGB. One/two-band inputs
    display their first recorded band in grayscale. No contrast stretch,
    pixel normalization, resizing or synthesis is performed.
    """
    normalized = tuple(name.upper() for name in band_names)
    if all(normalized.count(name) == 1 for name in ("RED", "GREEN", "BLUE")):
        return tuple(
            normalized.index(name) for name in ("RED", "GREEN", "BLUE")
        ), "RGB (RED, GREEN, BLUE)"
    if len(band_names) >= 3:
        return (0, 1, 2), "Display channels " + ", ".join(band_names[:3]) + " (not RGB)"
    if band_names:
        return (0,), band_names[0] + " band (unit grayscale)"
    return (), "No display bands recorded"


def capture_dataset_previews(
    root: Path,
    measured: DatasetMeasurement,
    cfg: CaptureConfig,
) -> Result[DatasetPreviewCapture, str]:
    """Capture selected exact canonical inputs under a shared image/byte budget.

    Source identity is verified before and after capture. NPZ has no pickle
    objects; absent masks have no array entry, never a synthetic empty mask.
    Failures preserve no user files because this boundary returns bytes only.
    """
    try:
        identity = dataset_identity(root)
        if isinstance(identity, Err):
            return identity
        if identity.value != measured.identity:
            return Err("dataset preview source identity differs from frozen measurements")
        plan = select_dataset_galleries(measured, cfg)
        by_id = {sample.variant_id: sample for sample in measured.samples}
        previews: list[DatasetPreview] = []
        files: list[BundleFile] = []
        total_bytes = 0
        mapping, display_label = display_channels(measured.identity.band_names)
        for variant_id in plan.variant_ids:
            sample = by_id[variant_id]
            key = sample.key
            directory = shard_dir(root, key.task, key.split, *key.spatial_shard)
            images = np.load(directory / "images.npy", mmap_mode="r", allow_pickle=False)
            image = _inverse(images[key.row_index], key.element)
            if _array_hash(image) != sample.image_sha256:
                return Err("captured canonical pixels differ from frozen image identity")
            mask = None
            if sample.mask_key is not None:
                mask_key = sample.mask_key
                directory = shard_dir(root, mask_key.task, mask_key.split, *mask_key.spatial_shard)
                masks = np.load(directory / "masks.npy", mmap_mode="r", allow_pickle=False)
                mask = _inverse(masks[mask_key.row_index], mask_key.element)
            stream = io.BytesIO()
            if mask is None:
                np.savez(stream, image=image)
            else:
                np.savez(stream, image=image, mask=mask)
            data = stream.getvalue()
            total_bytes += len(data)
            if total_bytes > cfg.max_capture_bytes or total_bytes > plan.reserved_bytes:
                return Err(
                    "preview codec exceeded its configured or conservatively reserved byte budget"
                )
            path = "previews/" + variant_id + ".npz"
            files.append(BundleFile(path=path, data=data))
            previews.append(
                DatasetPreview(
                    variant_id,
                    path,
                    hashlib.sha256(data).hexdigest(),
                    len(data),
                    sample,
                    mapping,
                    display_label,
                )
            )
        final_identity = dataset_identity(root)
        if isinstance(final_identity, Err):
            return final_identity
        if final_identity.value != measured.identity:
            return Err("dataset source changed during preview capture")
        return Ok(DatasetPreviewCapture(tuple(previews), tuple(files), plan))
    except (OSError, ValueError, TypeError, RuntimeError, OverflowError) as exc:
        return Err(f"dataset preview capture failed: {exc}")
