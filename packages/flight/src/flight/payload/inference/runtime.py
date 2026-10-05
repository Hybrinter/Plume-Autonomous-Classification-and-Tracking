"""Lazy inference runtime: sessions, factories, and the verified-session holder.

A RuntimeSession pairs a DetectorBackend with a stable identity string and a
bounded warm-up. A RuntimeFactory loads a session on demand (file digests,
session construction, I/O contract) without doing that work at construction.
InferenceRuntime is the lock-protected holder the payload shell owns: it
starts empty on the real path, installs only through ``install_verified``
(called by the control owner after INIT verification), and exposes a stable
snapshot for the capture path. ``from_scripted`` is the explicit sim/test
composition seam: it installs a known scripted session up front and also keeps
a ScriptedRuntimeFactory so the INIT MODEL_LOAD path still runs, without any
raw-backend compatibility union in PayloadApp.

The observed digest pair that forms an OnnxRuntimeSession identity is a
fingerprint of the files read at load time, not trusted manifest
authenticity: it identifies exactly which artifacts were loaded and lets
same-load consistency checks detect divergence, nothing more.

Satisfies: REQ-AIML-COMP-001, REQ-AIML-HIGH-004.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from flight.libs.config import PactConfig
from flight.libs.types import Err, FaultCode, Ok, Result
from flight.payload.inference.artifact_path import resolve_quantized_path
from flight.payload.inference.detector import (
    Detector,
    DetectorBackend,
    OnnxDetector,
)
from flight.payload.inference.verify import compute_sha256


@runtime_checkable
class RuntimeSession(Protocol):
    """One usable inference runtime: identity, backend, and bounded warm-up."""

    @property
    def identity(self) -> str:
        """Stable model identity: ordered classifier/segmentor digest pair, or
        the explicit scripted identity for test/sim sessions."""
        ...

    @property
    def backend(self) -> DetectorBackend:
        """The frame-level detect entry point used by the capture path."""
        ...

    def warm_up(self, cancel: threading.Event) -> Result[None, FaultCode]:
        """Exercise both models once on synthetic input, honoring cancellation."""
        ...


@runtime_checkable
class RuntimeFactory(Protocol):
    """Loads a RuntimeSession on demand; construction performs no I/O."""

    def load(self, cancel: threading.Event) -> Result[RuntimeSession, FaultCode]:
        """Load, verify, and return one session; Err on failure/cancellation."""
        ...


@dataclass(frozen=True, slots=True)
class OnnxRuntimeSession:
    """ONNX-backed session; warm-up runs one synthetic tile through both models.

    Attributes:
        identity: Ordered ``onnx:<classifier_sha256>:<segmentor_sha256>``
            fingerprint of the loaded artifacts.
        backend: The loaded OnnxDetector used for frame detection.
        bands: Model input channel count.
        tile_hw: Model tile height/width in pixels.
    """

    identity: str
    backend: Detector
    bands: int
    tile_hw: tuple[int, int]

    def warm_up(self, cancel: threading.Event) -> Result[None, FaultCode]:
        """Run classifier and segmentor once on zeroed synthetic tile tensors.

        Cancellation is checked before the call and between the classifier and
        segmentor stages inside ``Detector.warm_up``. The synthetic input is the
        exact conditioned one-tile shape ``(1, bands, tile_h, tile_w)`` float32
        with a reference-encoded all-zero ``(1, 2)`` GSD row, so warm-up never
        consumes or gates on a production frame.
        """
        if cancel.is_set():
            return Err(FaultCode.INFERENCE_TIMEOUT)
        tile_h, tile_w = self.tile_hw
        images = np.zeros((1, self.bands, tile_h, tile_w), dtype=np.float32)
        gsd = np.zeros((1, 2), dtype=np.float32)
        try:
            warmed = self.backend.warm_up(images, gsd, cancel=cancel)
        except Exception:
            return Err(FaultCode.MODEL_CORRUPT)
        if cancel.is_set():
            return Err(FaultCode.INFERENCE_TIMEOUT)
        return warmed


@dataclass(frozen=True, slots=True)
class ScriptedRuntimeSession:
    """Explicit test/sim session over an injected DetectorBackend.

    warm_up is a no-op: scripted backends carry no SDK state and warm-up must
    never consume scripted frames.
    """

    identity: str
    backend: DetectorBackend

    def warm_up(self, cancel: threading.Event) -> Result[None, FaultCode]:
        """No-op success; scripted sessions need no SDK priming."""
        if cancel.is_set():
            return Err(FaultCode.INFERENCE_TIMEOUT)
        return Ok(None)


class ScriptedRuntimeFactory:
    """Explicit factory returning a scripted session over an injected backend.

    Composition-only seam for SIL/tests; the real path never selects it.
    """

    def __init__(self, backend: DetectorBackend, identity: str = "scripted") -> None:
        """Wrap the injected backend under an explicit session identity.

        Inputs:
            backend: The scripted DetectorBackend the session exposes.
            identity: Stable identity string; non-empty.
        """
        self._backend = backend
        self._identity = identity

    def load(self, cancel: threading.Event) -> Result[RuntimeSession, FaultCode]:
        """Return the scripted session without touching the backend."""
        if cancel.is_set():
            return Err(FaultCode.INFERENCE_TIMEOUT)
        return Ok(ScriptedRuntimeSession(identity=self._identity, backend=self._backend))


class OnnxRuntimeFactory:
    """Config-driven ONNX session factory; ``__init__`` performs no SDK or file I/O.

    ``load`` resolves the quantized artifact paths, hashes both files
    (unreadable artifacts are MODEL_CORRUPT), constructs the OnnxDetector with
    those observed digests as the pinned expectations, and returns a session
    whose identity is the ordered digest pair. Broad SDK-boundary exceptions
    and contract failures map to Err(MODEL_CORRUPT); cancellation checkpoints
    map to Err(INFERENCE_TIMEOUT). No model or threshold value is invented:
    every option comes from the supplied PactConfig.
    """

    def __init__(self, config: PactConfig) -> None:
        """Store the typed config; nothing is read or loaded here.

        Inputs:
            config: The validated flight configuration.
        """
        self._config = config

    def load(self, cancel: threading.Event) -> Result[RuntimeSession, FaultCode]:
        """Load and verify the configured model pair, then wrap it as a session.

        Inputs:
            cancel: Cooperative cancellation flag checked between stages.

        Outputs:
            Result[RuntimeSession, FaultCode]: The verified session, or
                Err(MODEL_CORRUPT) for unreadable artifacts, missing/failed
                SDK load, or contract mismatch; Err(INFERENCE_TIMEOUT) when
                cancelled between checkpoints.
        """
        cfg = self._config
        inf = cfg.inference
        classifier_path = resolve_quantized_path(inf.classifier_model_path, inf.use_int8)
        segmentor_path = resolve_quantized_path(inf.segmentor_model_path, inf.use_int8)
        if cancel.is_set():
            return Err(FaultCode.INFERENCE_TIMEOUT)
        try:
            classifier_sha = compute_sha256(classifier_path)
            segmentor_sha = compute_sha256(segmentor_path)
        except OSError:
            return Err(FaultCode.MODEL_CORRUPT)
        if cancel.is_set():
            return Err(FaultCode.INFERENCE_TIMEOUT)
        bands = len(inf.input_bands)
        tile_height = inf.input_height_px // inf.tile_rows
        tile_width = inf.input_width_px // inf.tile_cols
        try:
            detector = OnnxDetector(
                segmentor_model_path=segmentor_path,
                classifier_model_path=classifier_path,
                confidence_gate=cfg.controller.vision.confidence_gate,
                min_blob_area_px=cfg.controller.vision.min_blob_area_px,
                logit_threshold=inf.classifier_logit_threshold,
                classifier_sha256=classifier_sha,
                segmentor_sha256=segmentor_sha,
                latency_budget_ms=cfg.fault.inference_timeout_ms,
                grid=(inf.tile_rows, inf.tile_cols),
                gsd_reference_m=inf.gsd_reference_m,
                expected_input_shape=(None, bands, tile_height, tile_width),
                expected_gsd_shape=(None, 2),
                expected_classifier_output_shape=(None, 1),
                expected_segmentor_output_shape=(None, 1, tile_height, tile_width),
            )
        except Exception:
            return Err(FaultCode.MODEL_CORRUPT)
        if cancel.is_set():
            return Err(FaultCode.INFERENCE_TIMEOUT)
        return Ok(
            OnnxRuntimeSession(
                identity=f"onnx:{classifier_sha}:{segmentor_sha}",
                backend=detector,
                bands=bands,
                tile_hw=(tile_height, tile_width),
            )
        )


class InferenceRuntime:
    """Lock-protected holder for the currently usable inference session.

    The real path constructs it with a factory and no session; the session is
    installed only by the control owner through ``install_verified`` after the
    INIT lifecycle verifies the candidate. There is no automatic load, no
    detect fallback, and no union with a raw backend.
    """

    def __init__(self, factory: RuntimeFactory | None = None) -> None:
        """Start empty, optionally carrying the lazy factory for INIT load.

        Inputs:
            factory: The session factory used by the MODEL_LOAD effect, or
                None when no load is possible (candidate then comes from the
                already-installed session).
        """
        self._factory = factory
        self._session: RuntimeSession | None = None
        self._lock = threading.Lock()

    @property
    def factory(self) -> RuntimeFactory | None:
        """The configured session factory, or None."""
        return self._factory

    @property
    def identity(self) -> str:
        """The installed session's identity, or "" when empty."""
        with self._lock:
            return "" if self._session is None else self._session.identity

    def snapshot(self) -> RuntimeSession | None:
        """Return the installed session for one consistent detection pass."""
        with self._lock:
            return self._session

    def install_verified(self, session: RuntimeSession) -> Result[None, FaultCode]:
        """Install a session that passed INIT verification.

        Inputs:
            session: The verified candidate; must carry a nonempty identity.

        Outputs:
            Result[None, FaultCode]: Ok(None) once installed;
                Err(MODEL_CORRUPT) when the session identity is empty.
        """
        if not session.identity:
            return Err(FaultCode.MODEL_CORRUPT)
        with self._lock:
            self._session = session
        return Ok(None)

    @classmethod
    def from_scripted(
        cls, backend: DetectorBackend, identity: str = "scripted"
    ) -> InferenceRuntime:
        """Build the explicit sim/test runtime: scripted session preinstalled.

        The session is installed through the same ``install_verified`` seam the
        control owner uses, and a ScriptedRuntimeFactory is kept so the INIT
        MODEL_LOAD effect still exercises the real load path. This helper is
        for explicit sim/test composition only; the real driver selection never
        calls it.

        Inputs:
            backend: The scripted DetectorBackend to expose.
            identity: Session identity; defaults to "scripted".

        Outputs:
            InferenceRuntime: Holder with the scripted session installed.
        """
        holder = cls(factory=ScriptedRuntimeFactory(backend, identity=identity))
        installed = holder.install_verified(
            ScriptedRuntimeSession(identity=identity, backend=backend)
        )
        if isinstance(installed, Err):
            raise ValueError(f"scripted runtime identity invalid: {installed.error.value}")
        return holder
