"""Payload inference: classifier, segmentor composer, and artifact verification."""

from flight.payload.inference.classifier import (
    ClassifierBackend,
    ClassifierDecision,
    OnnxClassifier,
    ScriptedClassifier,
)
from flight.payload.inference.detector import (
    Detector,
    DetectorBackend,
    OnnxDetector,
    ScriptedDetector,
)
from flight.payload.inference.segmentor import OnnxSegmentor, ScriptedSegmentor, SegmentorBackend
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
    "OnnxClassifier",
    "OnnxDetector",
    "OnnxSegmentor",
    "ScriptedClassifier",
    "ScriptedDetector",
    "ScriptedSegmentor",
    "SegmentorBackend",
    "check_inference_latency",
    "compute_sha256",
    "verify_io_contract",
    "verify_model_hash",
]
