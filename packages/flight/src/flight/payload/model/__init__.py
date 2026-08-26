"""Payload detection: classifier, segmentor, blob extraction, and artifact verification."""

from flight.payload.model.blobs import extract_blobs
from flight.payload.model.classifier import (
    ClassifierBackend,
    ClassifierDecision,
    OnnxClassifier,
    ScriptedClassifier,
)
from flight.payload.model.detector import Detector, DetectorBackend, OnnxDetector, ScriptedDetector
from flight.payload.model.segmentor import OnnxSegmentor, ScriptedSegmentor, SegmentorBackend
from flight.payload.model.verify import (
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
    "extract_blobs",
    "verify_io_contract",
    "verify_model_hash",
]
