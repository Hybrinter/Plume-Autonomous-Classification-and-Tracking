"""Bounded tiled inference: classify all tiles, segment positives, stitch masks.

Satisfies: REQ-AIML-COMP-001.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

import numpy as np

from flight.libs.messages import BlobMeta, InferenceResultMsg, ProcessedFrameMsg
from flight.libs.types import Err, FaultCode, FrameUsabilityTag, MessageType, Ok, Result
from flight.payload.blobs import extract_blobs
from flight.payload.gimbal.footprint import GSD_REFERENCE_M, to_model_gsd
from flight.payload.inference.classifier import (
    OnnxClassifier,
    ScriptedClassifier,
    TileClassifierBackend,
)
from flight.payload.inference.segmentor import (
    OnnxSegmentor,
    ScriptedSegmentor,
    TileSegmentorBackend,
)
from flight.payload.inference.verify import check_inference_latency
from flight.payload.preprocess.tiling import slice_frame, stitch_tiles


@dataclass(frozen=True, slots=True)
class TiledScore:
    """Raw classifier logits, positive gate decisions, and tile probability masks."""

    logits: np.ndarray  # (N,) float32
    positive: np.ndarray  # (N,) bool
    masks: np.ndarray  # (N, 1, h, w) float32 probabilities


@runtime_checkable
class DetectorBackend(Protocol):
    """Public flight detector entry point."""

    def detect(self, frame: ProcessedFrameMsg) -> Result[InferenceResultMsg, FaultCode]:
        """Run tiled inference on a preprocessed full frame."""
        ...


def infer_tiles(
    classifier: TileClassifierBackend,
    segmentor: TileSegmentorBackend,
    images: np.ndarray,
    gsd: np.ndarray,
    logit_threshold: float = 0.0,
) -> Result[TiledScore, FaultCode]:
    """Run shared pure tile orchestration for flight and ground adapters.

    ``gsd`` is the model encoding, with shape ``(N, 2)``. The classifier sees
    every tile; the segmentor sees only positive tiles. Its outputs scatter
    into an all-zero mask array, so a frame with no positives makes no segment
    call.
    """
    if not _valid_inputs(images, gsd) or not np.isfinite(logit_threshold):
        return Err(FaultCode.FRAME_MALFORMED)
    classified = classifier.classify_tiles(images, gsd)
    if isinstance(classified, Err):
        return classified
    try:
        raw_logits = np.asarray(classified.value)
    except TypeError, ValueError:
        return Err(FaultCode.FRAME_MALFORMED)
    if raw_logits.shape != (images.shape[0],) or not _is_real_numeric(raw_logits):
        return Err(FaultCode.FRAME_MALFORMED)
    logits = raw_logits.astype(np.float32)
    if not bool(np.isfinite(logits).all()):
        return Err(FaultCode.INFERENCE_NAN)
    positive = logits >= logit_threshold
    masks = np.zeros((images.shape[0], 1, images.shape[2], images.shape[3]), dtype=np.float32)
    positive_indices = np.flatnonzero(positive)
    if positive_indices.size:
        segmented = segmentor.segment_tiles(images[positive_indices], gsd[positive_indices])
        if isinstance(segmented, Err):
            return segmented
        try:
            raw_values = np.asarray(segmented.value)
        except TypeError, ValueError:
            return Err(FaultCode.FRAME_MALFORMED)
        expected = (positive_indices.size, 1, images.shape[2], images.shape[3])
        if raw_values.shape != expected or not _is_real_numeric(raw_values):
            return Err(FaultCode.FRAME_MALFORMED)
        values = raw_values.astype(np.float32)
        if not bool(np.isfinite(values).all()):
            return Err(FaultCode.INFERENCE_NAN)
        if np.any(values < 0) or np.any(values > 1):
            return Err(FaultCode.INFERENCE_NAN)
        masks[positive_indices] = values
    return Ok(TiledScore(logits=logits, positive=positive, masks=masks))


class Detector:
    """Compose tile classification, gated segmentation, mask stitching, and blobs."""

    def __init__(
        self,
        classifier: TileClassifierBackend,
        segmentor: TileSegmentorBackend,
        confidence_gate: float = 0.55,
        min_blob_area_px: int = 15,
        model_version: str = "unknown",
        latency_budget_ms: float = 0.0,
        record_wall_clock: bool = True,
        grid: tuple[int, int] = (8, 8),
        gsd_reference_m: float = GSD_REFERENCE_M,
        logit_threshold: float = 0.0,
    ) -> None:
        self._classifier = classifier
        self._segmentor = segmentor
        self._confidence_gate = confidence_gate
        self._min_blob_area_px = min_blob_area_px
        self._model_version = model_version
        self._latency_budget_ms = latency_budget_ms
        self._record_wall_clock = record_wall_clock
        self._grid = grid
        self._gsd_reference_m = gsd_reference_m
        self._logit_threshold = logit_threshold

    def detect(self, frame: ProcessedFrameMsg) -> Result[InferenceResultMsg, FaultCode]:
        """Slice, classify, selectively segment, stitch, and extract plume blobs."""
        start = time.perf_counter() if self._record_wall_clock else 0.0
        try:
            raw_tensor = np.asarray(frame.tensor)
        except TypeError, ValueError:
            return Err(FaultCode.FRAME_MALFORMED)
        if not _is_real_numeric(raw_tensor):
            return Err(FaultCode.FRAME_MALFORMED)
        tensor = raw_tensor.astype(np.float32)
        tiles_result = slice_frame(tensor, self._grid)
        if isinstance(tiles_result, Err):
            return tiles_result
        tiles = tiles_result.value
        if not bool(np.isfinite(tiles).all()):
            return Err(FaultCode.FRAME_MALFORMED)

        actual_gsd = frame.tile_gsd_m
        if actual_gsd is None:
            return Err(FaultCode.FRAME_MALFORMED)
        try:
            raw_gsd = np.asarray(actual_gsd)
        except TypeError, ValueError:
            return Err(FaultCode.FRAME_MALFORMED)
        if not _is_real_numeric(raw_gsd):
            return Err(FaultCode.FRAME_MALFORMED)
        actual = raw_gsd.astype(np.float32)
        if actual.shape != (tiles.shape[0], 2):
            return Err(FaultCode.FRAME_MALFORMED)
        encoded_result = to_model_gsd(actual, self._gsd_reference_m)
        if isinstance(encoded_result, Err):
            return encoded_result
        score = infer_tiles(
            self._classifier,
            self._segmentor,
            tiles,
            encoded_result.value,
            self._logit_threshold,
        )
        if isinstance(score, Err):
            return score
        stitched = stitch_tiles(score.value.masks, self._grid)
        if isinstance(stitched, Err):
            return stitched
        full_mask = stitched.value[0]
        blobs: tuple[BlobMeta, ...] = extract_blobs(
            full_mask, self._confidence_gate, self._min_blob_area_px
        )
        inference_ms = (time.perf_counter() - start) * 1000.0 if self._record_wall_clock else 0.0
        latency = check_inference_latency(inference_ms, self._latency_budget_ms)
        if isinstance(latency, Err):
            return Err(latency.error)
        tile_gsd_meta = tuple((float(pair[0]), float(pair[1])) for pair in actual)
        return Ok(
            InferenceResultMsg(
                msg_type=MessageType.INFERENCE_RESULT,
                timestamp_utc=frame.timestamp_utc,
                frame_id=frame.frame_id,
                mask=full_mask,
                blobs=blobs,
                model_version=self._model_version,
                inference_ms=inference_ms,
                mode_flags=0,
                quality_flags=frame.quality_flags,
                tile_logits=tuple(float(value) for value in score.value.logits),
                tile_positive=tuple(bool(value) for value in score.value.positive),
                tile_gsd_m=tile_gsd_meta,
            )
        )


class ScriptedDetector(Detector):
    """Deterministic scripted detector; defaults to one tile for small SIL fixtures."""

    def __init__(
        self,
        prob_mask: np.ndarray,
        confidence_gate: float = 0.55,
        min_blob_area_px: int = 15,
        model_version: str = "scripted",
        classifier_positive: bool = True,
        latency_budget_ms: float = 0.0,
        grid: tuple[int, int] = (1, 1),
        gsd_reference_m: float = GSD_REFERENCE_M,
    ) -> None:
        self._grid = grid
        self._scripted_classifier = ScriptedClassifier(positive=classifier_positive)
        tiled_mask = _slice_scripted_mask(prob_mask, grid)
        self._scripted_segmentor = ScriptedSegmentor(tiled_mask)
        super().__init__(
            classifier=self._scripted_classifier,
            segmentor=self._scripted_segmentor,
            confidence_gate=confidence_gate,
            min_blob_area_px=min_blob_area_px,
            model_version=model_version,
            latency_budget_ms=latency_budget_ms,
            record_wall_clock=False,
            grid=grid,
            gsd_reference_m=gsd_reference_m,
        )

    def detect(self, frame: ProcessedFrameMsg) -> Result[InferenceResultMsg, FaultCode]:
        """Allow old standalone scripted fixtures to omit physical GSD."""
        if frame.tile_gsd_m is not None:
            return super().detect(frame)
        tensor = np.asarray(frame.tensor)
        tiles = slice_frame(tensor, self._grid)
        if isinstance(tiles, Err):
            return tiles
        count = tiles.value.shape[0]
        # This fallback is deliberately confined to the scripted test/SIL backend.
        synthetic = np.full((count, 2), self._gsd_reference_m, dtype=np.float32)
        frame_with_gsd = replace(
            frame,
            quality_flags=frame.quality_flags | frozenset({FrameUsabilityTag.GSD_NOMINAL}),
            tile_gsd_m=synthetic,
        )
        return super().detect(frame_with_gsd)

    def load_mask(self, prob_mask: np.ndarray) -> None:
        """Replace the fixed segmentation mask used on the next positive tiles."""
        self._scripted_segmentor.load_mask(_slice_scripted_mask(prob_mask, self._grid))


class OnnxDetector(Detector):
    """Flight detector over strict image-plus-GSD ONNX classifier and segmentor graphs."""

    def __init__(
        self,
        segmentor_model_path: str,
        classifier_model_path: str,
        confidence_gate: float = 0.55,
        min_blob_area_px: int = 15,
        model_version: str = "unknown",
        logit_threshold: float = 0.0,
        classifier_sha256: str | None = None,
        segmentor_sha256: str | None = None,
        latency_budget_ms: float = 0.0,
        expected_input_shape: tuple[int | None, ...] | None = None,
        expected_segmentor_output_shape: tuple[int | None, ...] | None = None,
        expected_classifier_output_shape: tuple[int | None, ...] | None = None,
        expected_gsd_shape: tuple[int | None, ...] | None = (None, 2),
        grid: tuple[int, int] = (8, 8),
        gsd_reference_m: float = GSD_REFERENCE_M,
    ) -> None:
        classifier = OnnxClassifier(
            classifier_model_path,
            logit_threshold=logit_threshold,
            expected_sha256=classifier_sha256,
            expected_input_shape=expected_input_shape,
            expected_output_shape=expected_classifier_output_shape,
            expected_gsd_shape=expected_gsd_shape,
        )
        segmentor = OnnxSegmentor(
            segmentor_model_path,
            expected_sha256=segmentor_sha256,
            expected_input_shape=expected_input_shape,
            expected_output_shape=expected_segmentor_output_shape,
            expected_gsd_shape=expected_gsd_shape,
        )
        super().__init__(
            classifier=classifier,
            segmentor=segmentor,
            confidence_gate=confidence_gate,
            min_blob_area_px=min_blob_area_px,
            model_version=model_version,
            latency_budget_ms=latency_budget_ms,
            grid=grid,
            gsd_reference_m=gsd_reference_m,
            logit_threshold=logit_threshold,
        )


def _valid_inputs(images: np.ndarray, gsd: np.ndarray) -> bool:
    """Validate NCHW tiles and corresponding finite GSD encodings."""
    return bool(
        isinstance(images, np.ndarray)
        and images.ndim == 4
        and images.shape[0] > 0
        and min(images.shape[1:]) > 0
        and _is_real_numeric(images)
        and np.isfinite(images).all()
        and isinstance(gsd, np.ndarray)
        and gsd.shape == (images.shape[0], 2)
        and _is_real_numeric(gsd)
        and np.isfinite(gsd).all()
    )


def _is_real_numeric(values: np.ndarray) -> bool:
    """Reject object, string, and complex arrays before numeric operations."""
    return bool(
        isinstance(values, np.ndarray)
        and np.issubdtype(values.dtype, np.number)
        and not np.issubdtype(values.dtype, np.complexfloating)
    )


def _slice_scripted_mask(prob_mask: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """Turn one full-frame scripted mask into row-major tile masks."""
    mask = np.asarray(prob_mask, dtype=np.float32)
    if mask.ndim != 2 or not bool(np.isfinite(mask).all()) or np.any(mask < 0) or np.any(mask > 1):
        raise ValueError("scripted mask must be a finite (H, W) probability array")
    result = slice_frame(mask[None, None], grid)
    if isinstance(result, Err):
        raise ValueError("scripted mask dimensions must divide evenly by the configured grid")
    return result.value
