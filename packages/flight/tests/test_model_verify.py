"""Model-artifact verification helper tests (hash, I/O contract, latency budget)."""

from pathlib import Path

from flight.libs.types import Err, FaultCode, Ok
from flight.payload.model.verify import (
    check_inference_latency,
    compute_sha256,
    verify_io_contract,
    verify_model_hash,
)


def test_verify_hash_match_and_mismatch(tmp_path: Path) -> None:
    """verify_model_hash accepts the true digest and rejects a wrong one."""
    artifact = tmp_path / "m.onnx"
    artifact.write_bytes(b"model-bytes")
    digest = compute_sha256(str(artifact))
    assert isinstance(verify_model_hash(str(artifact), digest), Ok)
    bad = verify_model_hash(str(artifact), "0" * 64)
    assert isinstance(bad, Err) and bad.error is FaultCode.MODEL_CORRUPT


def test_verify_hash_missing_file_is_corrupt(tmp_path: Path) -> None:
    """A missing artifact reports MODEL_CORRUPT rather than raising."""
    result = verify_model_hash(str(tmp_path / "absent.onnx"), "0" * 64)
    assert isinstance(result, Err) and result.error is FaultCode.MODEL_CORRUPT


def test_io_contract_match_mismatch_and_wildcard() -> None:
    """I/O contract matches exactly, allows None wildcards, and rejects a true mismatch."""
    exp_in, exp_out = (1, 4, 256, 256), (1, 1, 256, 256)
    assert isinstance(verify_io_contract(exp_in, exp_out, exp_in, exp_out), Ok)
    # A dynamic batch dim (None) is a wildcard.
    assert isinstance(verify_io_contract((None, 4, 256, 256), exp_out, exp_in, exp_out), Ok)
    bad = verify_io_contract((1, 3, 256, 256), exp_out, exp_in, exp_out)
    assert isinstance(bad, Err) and bad.error is FaultCode.MODEL_CORRUPT


def test_latency_budget() -> None:
    """check_inference_latency passes within budget and flags INFERENCE_TIMEOUT over it."""
    assert isinstance(check_inference_latency(100.0, 500.0), Ok)
    over = check_inference_latency(600.0, 500.0)
    assert isinstance(over, Err) and over.error is FaultCode.INFERENCE_TIMEOUT
    assert isinstance(check_inference_latency(9999.0, 0.0), Ok)  # budget <= 0 disables the check
