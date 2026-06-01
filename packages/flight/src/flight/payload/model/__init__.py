"""Payload detection: swappable backends + shared blob extraction."""

from flight.payload.model.blobs import extract_blobs
from flight.payload.model.detector import DetectorBackend, OnnxDetector, ScriptedDetector

__all__ = ["DetectorBackend", "OnnxDetector", "ScriptedDetector", "extract_blobs"]
