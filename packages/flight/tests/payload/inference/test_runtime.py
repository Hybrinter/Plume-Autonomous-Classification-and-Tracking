"""InferenceRuntime holder, scripted session, and ONNX factory/session tests."""

import sys
import threading

import numpy as np
from flight.libs.config import PactConfig
from flight.libs.messages import InferenceResultMsg, ProcessedFrameMsg
from flight.libs.types import Err, FaultCode, Ok, Result
from flight.payload.inference import (
    Detector,
    DetectorBackend,
    InferenceRuntime,
    OnnxRuntimeFactory,
    OnnxRuntimeSession,
    RuntimeSession,
    ScriptedRuntimeSession,
)


class _CountingBackend:
    """DetectorBackend spy: counts detect calls; never produces a result."""

    def __init__(self) -> None:
        self.detect_calls = 0

    def detect(self, frame: ProcessedFrameMsg) -> Result[InferenceResultMsg, FaultCode]:
        """Count and fail so no scripted queue is involved."""
        del frame
        self.detect_calls += 1
        return Err(FaultCode.INFERENCE_NAN)


class _FixedClassifier:
    """Tile classifier returning one configured logit per tile."""

    def __init__(self, logit: float) -> None:
        self.logit = logit
        self.calls = 0

    def classify_tiles(self, images: np.ndarray, gsd: np.ndarray) -> Result[np.ndarray, FaultCode]:
        """Return the fixed logit for each tile row."""
        del gsd
        self.calls += 1
        return Ok(np.full(images.shape[0], self.logit, dtype=np.float32))


class _CountingSegmentor:
    """Tile segmentor returning probability masks; counts invocations."""

    def __init__(self) -> None:
        self.calls = 0

    def segment_tiles(self, images: np.ndarray, gsd: np.ndarray) -> Result[np.ndarray, FaultCode]:
        """Return a half-probability mask per positive tile."""
        del gsd
        self.calls += 1
        return Ok(np.full((images.shape[0], 1, images.shape[2], images.shape[3]), 0.5, np.float32))


class _FailingSegmentor:
    """Tile segmentor that always fails typed."""

    def segment_tiles(self, images: np.ndarray, gsd: np.ndarray) -> Result[np.ndarray, FaultCode]:
        """Fail with a contract fault."""
        del images, gsd
        return Err(FaultCode.MODEL_CORRUPT)


class _RaisingClassifier:
    """Tile classifier raising like an unguarded SDK boundary."""

    def classify_tiles(self, images: np.ndarray, gsd: np.ndarray) -> Result[np.ndarray, FaultCode]:
        """Raise instead of returning a typed failure."""
        del images, gsd
        raise RuntimeError("sdk session died")


class _CancelSettingClassifier:
    """Classifier that sets the cancel flag during its call."""

    def __init__(self, cancel: threading.Event) -> None:
        self._cancel = cancel

    def classify_tiles(self, images: np.ndarray, gsd: np.ndarray) -> Result[np.ndarray, FaultCode]:
        """Return a valid logit row, then signal cancellation."""
        del gsd
        self._cancel.set()
        return Ok(np.zeros(images.shape[0], dtype=np.float32))


class _CancelSettingSegmentor:
    """Segmentor that sets the cancel flag during its call."""

    def __init__(self, cancel: threading.Event) -> None:
        self._cancel = cancel
        self.calls = 0

    def segment_tiles(self, images: np.ndarray, gsd: np.ndarray) -> Result[np.ndarray, FaultCode]:
        """Return a valid mask, then signal cancellation."""
        del gsd
        self.calls += 1
        self._cancel.set()
        return Ok(np.full((images.shape[0], 1, images.shape[2], images.shape[3]), 0.5, np.float32))


class _NullFactory:
    """RuntimeFactory whose load count is observable."""

    def __init__(self) -> None:
        self.calls = 0

    def load(self, cancel: threading.Event) -> Result[RuntimeSession, FaultCode]:
        """Never expected to be called by the holder itself."""
        del cancel
        self.calls += 1
        return Err(FaultCode.MODEL_CORRUPT)


def _scripted_session(identity: str, backend: DetectorBackend) -> RuntimeSession:
    """A RuntimeSession double: explicit identity, inert warm-up."""
    return ScriptedRuntimeSession(identity=identity, backend=backend)


def test_holder_starts_empty_with_factory() -> None:
    """A real-path holder carries the factory but installs nothing eagerly."""
    factory = _NullFactory()
    runtime = InferenceRuntime(factory=factory)
    assert runtime.snapshot() is None
    assert runtime.identity == ""
    assert factory.calls == 0


def test_install_verified_rejects_empty_identity() -> None:
    """A session without an identity cannot install."""
    runtime = InferenceRuntime()
    result = runtime.install_verified(_scripted_session("", _CountingBackend()))
    assert isinstance(result, Err)
    assert result.error is FaultCode.MODEL_CORRUPT
    assert runtime.snapshot() is None


def test_install_verified_exposes_session_and_identity() -> None:
    """install_verified publishes the session through snapshot/identity."""
    runtime = InferenceRuntime()
    backend = _CountingBackend()
    result = runtime.install_verified(_scripted_session("model-a", backend))
    assert isinstance(result, Ok)
    assert runtime.identity == "model-a"
    session = runtime.snapshot()
    assert session is not None
    assert session.backend is backend


def test_install_replaces_session_atomically() -> None:
    """A newer verified install replaces the snapshot wholesale."""
    runtime = InferenceRuntime()
    first = _scripted_session("model-a", _CountingBackend())
    second = _scripted_session("model-b", _CountingBackend())
    assert isinstance(runtime.install_verified(first), Ok)
    assert isinstance(runtime.install_verified(second), Ok)
    assert runtime.snapshot() is second
    assert runtime.identity == "model-b"


def test_from_scripted_installs_and_keeps_factory() -> None:
    """from_scripted exposes the scripted session AND a MODEL_LOAD factory."""
    backend = _CountingBackend()
    runtime = InferenceRuntime.from_scripted(backend, identity="test-scripted")
    assert runtime.identity == "test-scripted"
    session = runtime.snapshot()
    assert session is not None
    assert session.backend is backend
    factory = runtime.factory
    assert factory is not None
    loaded = factory.load(threading.Event())
    assert isinstance(loaded, Ok)
    assert loaded.value.backend is backend
    assert loaded.value.identity == "test-scripted"


def test_scripted_session_warm_up_consumes_nothing() -> None:
    """Scripted warm_up is a no-op: no detect calls, no scripted frames spent."""
    backend = _CountingBackend()
    runtime = InferenceRuntime.from_scripted(backend)
    session = runtime.snapshot()
    assert session is not None
    assert isinstance(session.warm_up(threading.Event()), Ok)
    assert backend.detect_calls == 0


def test_scripted_session_warm_up_honors_cancel() -> None:
    """A pre-set cancel flag fails scripted warm-up with INFERENCE_TIMEOUT."""
    session = ScriptedRuntimeSession(identity="s", backend=_CountingBackend())
    cancel = threading.Event()
    cancel.set()
    result = session.warm_up(cancel)
    assert isinstance(result, Err)
    assert result.error is FaultCode.INFERENCE_TIMEOUT


def test_onnx_session_warm_up_runs_segmentor_on_negative_classifier() -> None:
    """Warm-up exercises BOTH model paths even when the classifier is negative."""
    classifier = _FixedClassifier(logit=-5.0)
    segmentor = _CountingSegmentor()
    detector = Detector(classifier, segmentor, grid=(1, 1), record_wall_clock=False)
    session = OnnxRuntimeSession(identity="onnx:test", backend=detector, bands=3, tile_hw=(8, 8))
    result = session.warm_up(threading.Event())
    assert isinstance(result, Ok)
    assert classifier.calls == 1
    assert segmentor.calls == 1


def test_onnx_session_warm_up_reports_model_failure() -> None:
    """A typed backend failure surfaces as Err from warm_up."""
    detector = Detector(
        _FixedClassifier(1.0), _FailingSegmentor(), grid=(1, 1), record_wall_clock=False
    )
    session = OnnxRuntimeSession(identity="onnx:test", backend=detector, bands=3, tile_hw=(8, 8))
    result = session.warm_up(threading.Event())
    assert isinstance(result, Err)
    assert result.error is FaultCode.MODEL_CORRUPT


def test_onnx_session_warm_up_cancels_before_work() -> None:
    """A pre-set cancel flag aborts warm-up before either model runs."""
    classifier = _FixedClassifier(0.0)
    segmentor = _CountingSegmentor()
    detector = Detector(classifier, segmentor, grid=(1, 1), record_wall_clock=False)
    session = OnnxRuntimeSession(identity="onnx:test", backend=detector, bands=3, tile_hw=(8, 8))
    cancel = threading.Event()
    cancel.set()
    result = session.warm_up(cancel)
    assert isinstance(result, Err)
    assert result.error is FaultCode.INFERENCE_TIMEOUT
    assert classifier.calls == 0
    assert segmentor.calls == 0


def test_onnx_session_warm_up_maps_backend_exception() -> None:
    """An unguarded SDK exception becomes Err(MODEL_CORRUPT), never propagates."""
    detector = Detector(
        _RaisingClassifier(), _CountingSegmentor(), grid=(1, 1), record_wall_clock=False
    )
    session = OnnxRuntimeSession(identity="onnx:test", backend=detector, bands=3, tile_hw=(8, 8))
    result = session.warm_up(threading.Event())
    assert isinstance(result, Err)
    assert result.error is FaultCode.MODEL_CORRUPT


def test_onnx_session_warm_up_cancels_between_models() -> None:
    """Cancellation after the classifier skips the segmentor entirely."""
    cancel = threading.Event()
    segmentor = _CountingSegmentor()
    detector = Detector(
        _CancelSettingClassifier(cancel), segmentor, grid=(1, 1), record_wall_clock=False
    )
    session = OnnxRuntimeSession(identity="onnx:test", backend=detector, bands=3, tile_hw=(8, 8))
    result = session.warm_up(cancel)
    assert isinstance(result, Err)
    assert result.error is FaultCode.INFERENCE_TIMEOUT
    assert segmentor.calls == 0


def test_onnx_session_warm_up_reports_cancel_after_models() -> None:
    """A cancel arriving during the segmentor still reports cancellation, not Ok."""
    cancel = threading.Event()
    segmentor = _CancelSettingSegmentor(cancel)
    detector = Detector(_FixedClassifier(1.0), segmentor, grid=(1, 1), record_wall_clock=False)
    session = OnnxRuntimeSession(identity="onnx:test", backend=detector, bands=3, tile_hw=(8, 8))
    result = session.warm_up(cancel)
    assert isinstance(result, Err)
    assert result.error is FaultCode.INFERENCE_TIMEOUT
    assert segmentor.calls == 1


def test_runtime_module_imports_sdk_free() -> None:
    """Importing the runtime module never pulls ML/camera SDKs into sys.modules."""
    assert "onnxruntime" not in sys.modules
    assert "PySpin" not in sys.modules


def test_onnx_factory_constructs_nothing_until_load() -> None:
    """OnnxRuntimeFactory.__init__ performs no file reads or SDK work.

    The configured artifact paths intentionally do not exist here: if the
    constructor touched them, this test would see an error path. The load
    behavior itself is covered in tests/core/test_select_drivers.py where
    real digests, cancellation, and SDK failures are exercised.
    """
    factory = OnnxRuntimeFactory(PactConfig())
    assert isinstance(factory, OnnxRuntimeFactory)
