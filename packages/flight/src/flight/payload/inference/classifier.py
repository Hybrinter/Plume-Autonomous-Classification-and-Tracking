"""Tile-level classifier backends for the onboard plume inference pipeline.

Satisfies: REQ-AIML-COMP-001.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from flight.libs.messages import ProcessedFrameMsg
from flight.libs.types import Err, FaultCode, Ok, Result
from flight.payload.inference.onnx_session import OnnxInferenceSession, load_onnx_session


@dataclass(frozen=True, slots=True)
class ClassifierDecision:
    """Legacy single-frame decision retained for callers during migration."""

    logit: float
    positive: bool


@runtime_checkable
class TileClassifierBackend(Protocol):
    """Classifier which returns one raw plume-presence logit per tile."""

    def classify_tiles(self, images: np.ndarray, gsd: np.ndarray) -> Result[np.ndarray, FaultCode]:
        """Return finite raw logits with shape ``(N,)``."""
        ...


@runtime_checkable
class ClassifierBackend(Protocol):
    """Legacy single-frame backend interface."""

    def classify(self, frame: ProcessedFrameMsg) -> Result[ClassifierDecision, FaultCode]:
        """Classify a processed frame."""
        ...


class ScriptedClassifier:
    """Deterministic tile classifier for SIL and tests."""

    def __init__(self, positive: bool = True, logit: float = 1.0) -> None:
        self._positive = positive
        self._logit = float(logit)

    def classify_tiles(self, images: np.ndarray, gsd: np.ndarray) -> Result[np.ndarray, FaultCode]:
        """Return the configured finite logit once for each valid tile."""
        if not _valid_batch(images, gsd):
            return Err(FaultCode.FRAME_MALFORMED)
        # Preserve the historical boolean override for scripted SIL decisions.
        logit = abs(self._logit) if self._positive else -max(abs(self._logit), 1e-6)
        return Ok(np.full((images.shape[0],), logit, dtype=np.float32))

    def classify(self, frame: ProcessedFrameMsg) -> Result[ClassifierDecision, FaultCode]:
        """Return the configured legacy decision, ignoring frame contents."""
        del frame
        return Ok(ClassifierDecision(logit=self._logit, positive=self._positive))


class OnnxClassifier:
    """ONNX binary classifier using named ``image`` and encoded ``gsd`` inputs."""

    def __init__(
        self,
        model_path: str,
        logit_threshold: float = 0.0,
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
        self._logit_threshold = logit_threshold

    def classify_tiles(self, images: np.ndarray, gsd: np.ndarray) -> Result[np.ndarray, FaultCode]:
        """Return the model's raw logits for all input tiles."""
        if not _valid_batch(images, gsd):
            return Err(FaultCode.FRAME_MALFORMED)
        try:
            raw = self._session.run(
                None,
                {
                    "image": np.asarray(images, dtype=np.float32),
                    "gsd": np.asarray(gsd, dtype=np.float32),
                },
            )[0]
        except Exception:
            return Err(FaultCode.INFERENCE_NAN)
        logits = np.asarray(raw)
        if (
            logits.shape != (images.shape[0], 1)
            or not np.issubdtype(logits.dtype, np.number)
            or np.issubdtype(logits.dtype, np.complexfloating)
        ):
            return Err(FaultCode.FRAME_MALFORMED)
        logits = logits.astype(np.float32)
        logits = logits[:, 0]
        if not bool(np.isfinite(logits).all()):
            return Err(FaultCode.INFERENCE_NAN)
        return Ok(logits)

    def classify(self, frame: ProcessedFrameMsg) -> Result[ClassifierDecision, FaultCode]:
        """Legacy one-image wrapper; callers should use ``classify_tiles``."""
        image = np.asarray(frame.tensor, dtype=np.float32)
        if image.ndim == 3:
            image = image[None]
        if image.ndim != 4 or image.shape[0] != 1 or frame.tile_gsd_m is None:
            return Err(FaultCode.FRAME_MALFORMED)
        from flight.payload.gimbal.footprint import to_model_gsd

        encoded = to_model_gsd(np.asarray(frame.tile_gsd_m, dtype=np.float32))
        if isinstance(encoded, Err):
            return encoded
        result = self.classify_tiles(image, encoded.value)
        if isinstance(result, Err):
            return result
        logit = float(result.value[0])
        return Ok(ClassifierDecision(logit, logit >= self._logit_threshold))


def _valid_batch(images: np.ndarray, gsd: np.ndarray) -> bool:
    """Validate finite NCHW imagery and matching encoded ``(N, 2)`` GSD."""
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
