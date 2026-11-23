"""Tile-level segmentation backends for the onboard plume inference pipeline.

Satisfies: REQ-AIML-COMP-001.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from flight.libs.messages import ProcessedFrameMsg
from flight.libs.types import Err, FaultCode, Ok, Result
from flight.payload.inference.onnx_session import OnnxInferenceSession, load_onnx_session


@runtime_checkable
class TileSegmentorBackend(Protocol):
    """Segmentor which returns probability masks for selected positive tiles."""

    def segment_tiles(self, images: np.ndarray, gsd: np.ndarray) -> Result[np.ndarray, FaultCode]:
        """Return probability masks with shape ``(N, 1, h, w)``."""
        ...


@runtime_checkable
class SegmentorBackend(Protocol):
    """Legacy frame-level segmentor interface."""

    def segment(self, frame: ProcessedFrameMsg) -> Result[np.ndarray, FaultCode]:
        """Segment a processed frame."""
        ...


class ScriptedSegmentor:
    """Deterministic segmentor backed by a fixed probability mask."""

    def __init__(self, prob_mask: np.ndarray) -> None:
        self._prob_mask = np.array(prob_mask, dtype=np.float32, copy=True)

    def load_mask(self, prob_mask: np.ndarray) -> None:
        """Replace the stored fixed probability mask."""
        self._prob_mask = np.array(prob_mask, dtype=np.float32, copy=True)

    def segment_tiles(self, images: np.ndarray, gsd: np.ndarray) -> Result[np.ndarray, FaultCode]:
        """Repeat the fixed mask for each tile when its dimensions match."""
        if not _valid_batch(images, gsd):
            return Err(FaultCode.FRAME_MALFORMED)
        if self._prob_mask.ndim == 4:
            if self._prob_mask.shape != (
                images.shape[0],
                1,
                images.shape[2],
                images.shape[3],
            ):
                return Err(FaultCode.FRAME_MALFORMED)
            mask = self._prob_mask
            exact_batch = True
        elif self._prob_mask.ndim == 2:
            mask = self._prob_mask[None, None]
            exact_batch = False
        elif self._prob_mask.ndim == 3 and self._prob_mask.shape[0] == 1:
            mask = self._prob_mask[None]
            exact_batch = False
        else:
            return Err(FaultCode.FRAME_MALFORMED)
        if mask.shape[2:] != images.shape[2:] or not np.isfinite(mask).all():
            return Err(FaultCode.FRAME_MALFORMED)
        if np.any(mask < 0) or np.any(mask > 1):
            return Err(FaultCode.FRAME_MALFORMED)
        if exact_batch:
            return Ok(mask.astype(np.float32, copy=True))
        return Ok(np.repeat(mask, images.shape[0], axis=0).astype(np.float32))

    def segment(self, frame: ProcessedFrameMsg) -> Result[np.ndarray, FaultCode]:
        """Legacy frame-level wrapper returning one ``(H, W)`` mask."""
        del frame
        return Ok(self._prob_mask)


class OnnxSegmentor:
    """ONNX segmentor using named image and encoded-GSD inputs."""

    def __init__(
        self,
        model_path: str,
        expected_sha256: str | None = None,
        expected_input_shape: tuple[int | None, ...] | None = None,
        expected_output_shape: tuple[int | None, ...] | None = None,
        expected_gsd_shape: tuple[int | None, ...] | None = (None, 2),
    ) -> None:
        self._session: OnnxInferenceSession = load_onnx_session(
            model_path,
            expected_sha256=expected_sha256,
            expected_input_shape=expected_input_shape,
            expected_output_shape=expected_output_shape,
            expected_gsd_shape=expected_gsd_shape,
        )

    def segment_tiles(self, images: np.ndarray, gsd: np.ndarray) -> Result[np.ndarray, FaultCode]:
        """Run segmentation and apply a numerically stable sigmoid to logits."""
        if not _valid_batch(images, gsd):
            return Err(FaultCode.FRAME_MALFORMED)
        try:
            raw_logits = np.asarray(
                self._session.run(
                    None,
                    {
                        "image": np.asarray(images, dtype=np.float32),
                        "gsd": np.asarray(gsd, dtype=np.float32),
                    },
                )[0]
            )
        except Exception:
            return Err(FaultCode.INFERENCE_NAN)
        expected = (images.shape[0], 1, images.shape[2], images.shape[3])
        if (
            raw_logits.shape != expected
            or not np.issubdtype(raw_logits.dtype, np.number)
            or np.issubdtype(raw_logits.dtype, np.complexfloating)
        ):
            return Err(FaultCode.FRAME_MALFORMED)
        logits = raw_logits.astype(np.float32)
        if not bool(np.isfinite(logits).all()):
            return Err(FaultCode.INFERENCE_NAN)
        probs = np.empty_like(logits, dtype=np.float32)
        positive = logits >= 0
        probs[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exp_values = np.exp(logits[~positive])
        probs[~positive] = exp_values / (1.0 + exp_values)
        return Ok(probs)

    def segment(self, frame: ProcessedFrameMsg) -> Result[np.ndarray, FaultCode]:
        """Legacy one-image wrapper returning a ``(H, W)`` probability mask."""
        image = np.asarray(frame.tensor, dtype=np.float32)
        if image.ndim == 3:
            image = image[None]
        if image.ndim != 4 or image.shape[0] != 1 or frame.tile_gsd_m is None:
            return Err(FaultCode.FRAME_MALFORMED)
        from flight.payload.gimbal.footprint import to_model_gsd

        encoded = to_model_gsd(np.asarray(frame.tile_gsd_m, dtype=np.float32))
        if isinstance(encoded, Err):
            return encoded
        result = self.segment_tiles(image, encoded.value)
        if isinstance(result, Err):
            return result
        return Ok(result.value[0, 0])


def _valid_batch(images: np.ndarray, gsd: np.ndarray) -> bool:
    """Validate real numeric tile images and corresponding encoded GSD."""
    return bool(
        isinstance(images, np.ndarray)
        and images.ndim == 4
        and images.shape[0] > 0
        and min(images.shape[1:]) > 0
        and np.issubdtype(images.dtype, np.number)
        and not np.issubdtype(images.dtype, np.complexfloating)
        and np.isfinite(images).all()
        and isinstance(gsd, np.ndarray)
        and gsd.shape == (images.shape[0], 2)
        and np.issubdtype(gsd.dtype, np.number)
        and not np.issubdtype(gsd.dtype, np.complexfloating)
        and np.isfinite(gsd).all()
    )
