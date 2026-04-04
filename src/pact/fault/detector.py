"""Fault detector — checks inference results for pathological conditions.

Produces FaultEventMsg for:
  - INFERENCE_NAN:     output mask contains NaN or Inf values.
  - INFERENCE_TIMEOUT: inference wall-clock time exceeded config.inference_timeout_ms.

This module does not handle thermal or power faults (those require hardware sensor
data not yet available in Phase I — see fault/CLAUDE.md known gaps).

Satisfies: REQ-SAFE-HIGH-002.
"""

from __future__ import annotations

# stdlib
from datetime import datetime, timezone
from typing import Optional

# third-party
import numpy as np

# internal
from pact.types.config import FaultConfig
from pact.types.enums import FaultCode, MessageType
from pact.types.messages import FaultEventMsg, InferenceResultMsg


def detect_faults(
    inference_result: Optional[InferenceResultMsg],
    inference_start_time: float,
    config: FaultConfig,
) -> list[FaultEventMsg]:
    """Check an inference result for NaN, timeout, and other failure modes.

    Args:
        inference_result:     The result to inspect. If None, no faults are raised
                              (the caller is responsible for watchdog-based timeout detection).
        inference_start_time: Unix timestamp (float, seconds) at which inference began.
                              Used to compute elapsed time if inference_result is not None.
        config:               FaultConfig containing inference_timeout_ms threshold.

    Returns:
        A list of FaultEventMsg (may be empty). Multiple faults may be raised for a
        single result (e.g. both NaN and timeout on the same frame).
    """
    if inference_result is None:
        return []

    now_str = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    faults: list[FaultEventMsg] = []

    # --- NaN / Inf check ---
    mask = inference_result.mask
    if isinstance(mask, np.ndarray) and (np.isnan(mask).any() or np.isinf(mask).any()):
        faults.append(
            FaultEventMsg(
                msg_type=MessageType.FAULT_EVENT,
                timestamp_utc=now_str,
                fault_code=FaultCode.INFERENCE_NAN,
                subsystem="inference",
                detail=(
                    f"mask contains NaN or Inf for frame_id={inference_result.frame_id}"
                ),
            )
        )

    # --- latency timeout check ---
    if inference_result.inference_ms > config.inference_timeout_ms:
        faults.append(
            FaultEventMsg(
                msg_type=MessageType.FAULT_EVENT,
                timestamp_utc=now_str,
                fault_code=FaultCode.INFERENCE_TIMEOUT,
                subsystem="inference",
                detail=(
                    f"inference_ms={inference_result.inference_ms:.1f} exceeded "
                    f"timeout={config.inference_timeout_ms:.1f} ms "
                    f"for frame_id={inference_result.frame_id}"
                ),
            )
        )

    return faults
