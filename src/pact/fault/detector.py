"""Fault detector — checks inference results for pathological conditions.

Produces FaultEventMsg for:
  - INFERENCE_NAN:     output mask contains NaN or Inf values.
  - INFERENCE_TIMEOUT: inference wall-clock time exceeded config.inference_timeout_ms.
  - THERMAL_OVER_LIMIT: temperature exceeds cfg.thermal_limit_c.
  - POWER_OVER_LIMIT:   power draw exceeds cfg.power_limit_w.

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
from pact.types.messages import FaultEventMsg, InferenceResultMsg, utc_now_iso


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


def check_thermal(
    temp_c: float, cfg: FaultConfig,
) -> Optional[FaultEventMsg]:
    """Return a FaultEventMsg if temp_c exceeds cfg.thermal_limit_c."""
    if temp_c > cfg.thermal_limit_c:
        return FaultEventMsg(
            msg_type=MessageType.FAULT_EVENT,
            timestamp_utc=utc_now_iso(),
            fault_code=FaultCode.THERMAL_OVER_LIMIT,
            subsystem="fault",
            detail=(
                f"thermal limit exceeded: "
                f"{temp_c:.1f}C > {cfg.thermal_limit_c:.1f}C"
            ),
        )
    return None


def check_power(
    watts: float, cfg: FaultConfig,
) -> Optional[FaultEventMsg]:
    """Return a FaultEventMsg if watts exceeds cfg.power_limit_w."""
    if watts > cfg.power_limit_w:
        return FaultEventMsg(
            msg_type=MessageType.FAULT_EVENT,
            timestamp_utc=utc_now_iso(),
            fault_code=FaultCode.POWER_OVER_LIMIT,
            subsystem="fault",
            detail=(
                f"power limit exceeded: "
                f"{watts:.1f}W > {cfg.power_limit_w:.1f}W"
            ),
        )
    return None
