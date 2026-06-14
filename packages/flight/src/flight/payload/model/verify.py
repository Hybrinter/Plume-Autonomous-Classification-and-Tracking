"""Pure model-artifact verification helpers (no onnxruntime, no I/O beyond hashing a file).

OnnxDetector uses these at load time and per frame; the model-acceptance gate (tools/) reuses the
same checks. They are pure/deterministic so they are fully unit-tested even though onnxruntime is
not present in this repo (the OnnxDetector session itself is exercised only on hardware):

  - verify_model_hash: the artifact's SHA-256 must equal the manifest digest (MODEL_CORRUPT).
  - verify_io_contract: the model's input/output tensor shapes must equal the flight contract
    ((1, C, H, W) in, (1, 1, H, W) out), with None entries treated as wildcards (dynamic dims).
  - check_inference_latency: a per-frame wall-clock budget (INFERENCE_TIMEOUT when exceeded).

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

# stdlib
import hashlib
from pathlib import Path

# internal
from flight.libs.types import Err, FaultCode, Ok, Result

Shape = tuple[int | None, ...]


def compute_sha256(path: str) -> str:
    """Return the SHA-256 hex digest of a file's bytes.

    Args:
        path: Filesystem path to the artifact.

    Returns:
        The lowercase hex SHA-256 digest.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_model_hash(path: str, expected_sha256: str) -> Result[None, FaultCode]:
    """Verify an artifact's SHA-256 matches the expected digest.

    Args:
        path: Filesystem path to the artifact.
        expected_sha256: The expected lowercase hex SHA-256 digest.

    Returns:
        Ok(None) on a match, else Err(FaultCode.MODEL_CORRUPT) (mismatch or unreadable file).
    """
    try:
        actual = compute_sha256(path)
    except OSError:
        return Err(FaultCode.MODEL_CORRUPT)
    if actual.lower() != expected_sha256.lower():
        return Err(FaultCode.MODEL_CORRUPT)
    return Ok(None)


def _shape_matches(actual: Shape, expected: Shape) -> bool:
    """Return True iff shapes have equal rank and each dim matches (None == wildcard)."""
    if len(actual) != len(expected):
        return False
    return all(e is None or a is None or a == e for a, e in zip(actual, expected, strict=True))


def verify_io_contract(
    actual_input: Shape,
    actual_output: Shape,
    expected_input: Shape,
    expected_output: Shape,
) -> Result[None, FaultCode]:
    """Verify a model's input/output shapes match the flight inference contract.

    Args:
        actual_input: The model's declared input shape (None entries are dynamic dims).
        actual_output: The model's declared output shape.
        expected_input: The required input shape (e.g. (1, 4, 256, 256)).
        expected_output: The required output shape (e.g. (1, 1, 256, 256)).

    Returns:
        Ok(None) when both shapes match (treating None as a wildcard), else
        Err(FaultCode.MODEL_CORRUPT).
    """
    if _shape_matches(actual_input, expected_input) and _shape_matches(
        actual_output, expected_output
    ):
        return Ok(None)
    return Err(FaultCode.MODEL_CORRUPT)


def check_inference_latency(elapsed_ms: float, budget_ms: float) -> Result[None, FaultCode]:
    """Check a per-frame inference time against the configured budget.

    Args:
        elapsed_ms: Measured inference wall-clock time in milliseconds.
        budget_ms: The per-frame latency budget in milliseconds (<= 0 disables the check).

    Returns:
        Ok(None) within budget (or when budget_ms <= 0), else Err(FaultCode.INFERENCE_TIMEOUT).
    """
    if budget_ms > 0.0 and elapsed_ms > budget_ms:
        return Err(FaultCode.INFERENCE_TIMEOUT)
    return Ok(None)
