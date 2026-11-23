"""Payload inference: classifier, segmentor composer, and artifact verification."""

from flight.payload.inference.classifier import (
    ClassifierBackend,
    ClassifierDecision,
    OnnxClassifier,
    ScriptedClassifier,
    TileClassifierBackend,
)
from flight.payload.inference.detector import (
    Detector,
    DetectorBackend,
    OnnxDetector,
    ScriptedDetector,
    TiledScore,
    infer_tiles,
)
from flight.payload.inference.runtime import (
    InferenceRuntime,
    OnnxRuntimeFactory,
    OnnxRuntimeSession,
    RuntimeFactory,
    RuntimeSession,
    ScriptedRuntimeFactory,
    ScriptedRuntimeSession,
)
from flight.payload.inference.segmentor import (
    OnnxSegmentor,
    ScriptedSegmentor,
    SegmentorBackend,
    TileSegmentorBackend,
)
from flight.payload.inference.verify import (
    check_inference_latency,
    compute_sha256,
    verify_io_contract,
    verify_model_hash,
)

__all__ = [
    "ClassifierBackend",
    "ClassifierDecision",
    "Detector",
    "DetectorBackend",
    "InferenceRuntime",
    "OnnxClassifier",
    "OnnxDetector",
    "OnnxRuntimeFactory",
    "OnnxRuntimeSession",
    "OnnxSegmentor",
    "RuntimeFactory",
    "RuntimeSession",
    "ScriptedClassifier",
    "ScriptedDetector",
    "ScriptedRuntimeFactory",
    "ScriptedRuntimeSession",
    "ScriptedSegmentor",
    "SegmentorBackend",
    "TileClassifierBackend",
    "TileSegmentorBackend",
    "TiledScore",
    "check_inference_latency",
    "compute_sha256",
    "infer_tiles",
    "verify_io_contract",
    "verify_model_hash",
]
