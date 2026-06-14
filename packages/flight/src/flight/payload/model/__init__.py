"""Payload detection: swappable backends + shared blob extraction + artifact verification."""

from flight.payload.model.blobs import extract_blobs
from flight.payload.model.detector import DetectorBackend, OnnxDetector, ScriptedDetector
from flight.payload.model.verify import (
    check_inference_latency,
    compute_sha256,
    verify_io_contract,
    verify_model_hash,
)

__all__ = [
    "DetectorBackend",
    "OnnxDetector",
    "ScriptedDetector",
    "check_inference_latency",
    "compute_sha256",
    "extract_blobs",
    "verify_io_contract",
    "verify_model_hash",
]
