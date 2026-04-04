"""Unit tests for pact.fault.detector — detect_faults().

Satisfies: §6.2 of PACT_SW_ARCH.md — Fault Detection subsystem unit tests.
REQ-SAFE-HIGH-002, GOAL-006
"""

from __future__ import annotations

# stdlib
import time

# third-party
import numpy as np
import pytest

# module under test
from pact.fault.detector import detect_faults

# pact types
from pact.types.enums import FaultCode, MessageType
from pact.types.messages import BlobMeta, FaultEventMsg, InferenceResultMsg
from pact.types.config import FaultConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_inference_result(
    frame_id: int = 1,
    mask: object = None,
    blobs: tuple[BlobMeta, ...] = (),
) -> InferenceResultMsg:
    """Construct a minimal InferenceResultMsg for fault detector tests."""
    if mask is None:
        mask = np.zeros((256, 256), dtype=np.float32)
    return InferenceResultMsg(
        msg_type=MessageType.INFERENCE_RESULT,
        timestamp_utc="2026-04-03T00:00:00.000Z",
        frame_id=frame_id,
        mask=mask,
        blobs=blobs,
        model_version="test-v0",
        inference_ms=50.0,
        mode_flags=0,
    )


def default_fault_config() -> FaultConfig:
    """Return a FaultConfig with default thresholds."""
    return FaultConfig(
        watchdog_interval_s=5.0,
        watchdog_max_miss_count=3,
        inference_timeout_ms=2000.0,
        thermal_limit_c=80.0,
        power_limit_w=55.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_faults_clean_result() -> None:
    """detect_faults on a clean, timely inference result must return an empty list."""
    result = make_inference_result()
    config = default_fault_config()
    now = time.time()
    start_time = now - 0.050  # 50ms elapsed — well within 2000ms timeout

    faults = detect_faults(
        inference_result=result,
        inference_start_time=start_time,
        config=config,
    )
    assert faults == [], f"Expected no faults for clean result, got {faults}"


def test_nan_in_mask_raises_inference_nan() -> None:
    """detect_faults must emit FaultCode.INFERENCE_NAN when the mask contains NaN."""
    nan_mask = np.full((256, 256), float("nan"), dtype=np.float32)
    result = make_inference_result(mask=nan_mask)
    config = default_fault_config()
    start_time = time.time() - 0.050

    faults = detect_faults(
        inference_result=result,
        inference_start_time=start_time,
        config=config,
    )

    fault_codes = [f.fault_code for f in faults]
    assert FaultCode.INFERENCE_NAN in fault_codes, (
        f"Expected INFERENCE_NAN fault for NaN mask, got {fault_codes}"
    )


def test_timeout_raises_inference_timeout() -> None:
    """detect_faults must emit FaultCode.INFERENCE_TIMEOUT when elapsed time exceeds budget."""
    result = make_inference_result()
    config = default_fault_config()
    # Start time 3000ms ago — exceeds the 2000ms timeout budget
    start_time = time.time() - 3.0

    faults = detect_faults(
        inference_result=result,
        inference_start_time=start_time,
        config=config,
    )

    fault_codes = [f.fault_code for f in faults]
    assert FaultCode.INFERENCE_TIMEOUT in fault_codes, (
        f"Expected INFERENCE_TIMEOUT fault for 3s elapsed (budget=2s), got {fault_codes}"
    )


def test_none_result_no_crash() -> None:
    """detect_faults with inference_result=None must not raise and must return a list."""
    config = default_fault_config()
    start_time = time.time() - 0.050

    # Should not raise; returns either empty or a fault list
    faults = detect_faults(
        inference_result=None,
        inference_start_time=start_time,
        config=config,
    )
    assert isinstance(faults, list), f"Expected list, got {type(faults)}"


def test_faults_are_fault_event_msgs() -> None:
    """All items returned by detect_faults must be FaultEventMsg instances."""
    nan_mask = np.full((256, 256), float("nan"), dtype=np.float32)
    result = make_inference_result(mask=nan_mask)
    config = default_fault_config()
    start_time = time.time() - 3.0  # also triggers timeout

    faults = detect_faults(
        inference_result=result,
        inference_start_time=start_time,
        config=config,
    )
    for fault in faults:
        assert isinstance(fault, FaultEventMsg), (
            f"Expected FaultEventMsg, got {type(fault)}: {fault}"
        )
