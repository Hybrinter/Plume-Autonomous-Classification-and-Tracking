"""Fault handlers — one function per FaultCode, dispatched by the fault process.

Each handler receives a FaultEventMsg and returns either:
  - ModeChangeMsg  — if the fault requires a system mode transition (e.g. SAFE)
  - None           — if the fault is self-healing or handled locally

FAULT_HANDLERS maps every FaultCode to its handler.  Completeness is asserted at
startup by run_fault_process() (see fault/process.py).

Satisfies: REQ-SAFE-HIGH-002, REQ-GIMB-HIGH-003, GOAL-006.
"""

from __future__ import annotations

# stdlib
from datetime import datetime, timezone
from typing import Callable, Final, Optional

# internal
from pact.types.enums import FaultCode, MessageType, SystemMode
from pact.types.messages import FaultEventMsg, ModeChangeMsg

import structlog

log = structlog.get_logger().bind(subsystem="fault")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _safe_mode_msg(reason: str) -> ModeChangeMsg:
    """Construct a ModeChangeMsg requesting SAFE mode."""
    return ModeChangeMsg(
        msg_type=MessageType.MODE_CHANGE,
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        new_mode=SystemMode.SAFE,
        requested_by=f"fault_handler:{reason}",
    )


# ---------------------------------------------------------------------------
# Handler functions — one per FaultCode
# ---------------------------------------------------------------------------


def handle_none(event: FaultEventMsg) -> Optional[ModeChangeMsg]:
    """FaultCode.NONE — no-op; should never be dispatched in practice."""
    log.warning("handle_none_called", detail=event.detail)
    return None


def handle_inference_timeout(event: FaultEventMsg) -> Optional[ModeChangeMsg]:
    """FaultCode.INFERENCE_TIMEOUT — inference exceeded latency budget.

    A single timeout is logged but does not trigger safe mode.  Repeated timeouts
    are handled by the watchdog (WATCHDOG_EXPIRE → PROCESS_DIED chain).
    # TODO: add a consecutive-timeout counter and enter SAFE after N timeouts.
    """
    log.warning("inference_timeout", subsystem=event.subsystem, detail=event.detail)
    return None


def handle_inference_nan(event: FaultEventMsg) -> Optional[ModeChangeMsg]:
    """FaultCode.INFERENCE_NAN — mask output contains NaN/Inf.

    NaN output indicates a corrupted model or numerical instability.  Enter SAFE
    mode immediately to prevent bad inference results driving the gimbal.
    """
    log.error("inference_nan", subsystem=event.subsystem, detail=event.detail)
    return _safe_mode_msg("inference_nan")


def handle_camera_stall(event: FaultEventMsg) -> Optional[ModeChangeMsg]:
    """FaultCode.CAMERA_STALL — no frame received within the stall timeout.

    Camera stall halts the imaging pipeline.  Enter SAFE mode until ground can
    assess and command recovery.
    # TODO: attempt camera soft-reset before entering SAFE (Phase II).
    """
    log.error("camera_stall", subsystem=event.subsystem, detail=event.detail)
    return _safe_mode_msg("camera_stall")


def handle_storage_full(event: FaultEventMsg) -> Optional[ModeChangeMsg]:
    """FaultCode.STORAGE_FULL — disk write failed due to insufficient space.

    Storage full is not immediately mission-ending — the system can continue
    inference and telemetry even if new frames cannot be stored.  Log the fault
    but do not enter SAFE mode.
    # TODO: implement LRU eviction policy in Phase II.
    """
    log.error("storage_full", subsystem=event.subsystem, detail=event.detail)
    return None


def handle_thermal_over_limit(event: FaultEventMsg) -> Optional[ModeChangeMsg]:
    """FaultCode.THERMAL_OVER_LIMIT — hardware temperature exceeded config limit.

    Thermal limit breach is a safety-critical condition.  Enter SAFE mode
    immediately to reduce compute load.
    """
    log.error("thermal_over_limit", subsystem=event.subsystem, detail=event.detail)
    return _safe_mode_msg("thermal_over_limit")


def handle_power_over_limit(event: FaultEventMsg) -> Optional[ModeChangeMsg]:
    """FaultCode.POWER_OVER_LIMIT — power draw exceeded config limit.

    Power limit breach requires reducing system activity.  Enter SAFE mode.
    """
    log.error("power_over_limit", subsystem=event.subsystem, detail=event.detail)
    return _safe_mode_msg("power_over_limit")


def handle_gimbal_runaway(event: FaultEventMsg) -> Optional[ModeChangeMsg]:
    """FaultCode.GIMBAL_RUNAWAY — commanded displacement exceeded max deadband.

    A runaway gimbal command is safety-critical (REQ-GIMB-HIGH-003).
    Enter SAFE mode immediately to halt further gimbal commands.
    """
    log.error("gimbal_runaway", subsystem=event.subsystem, detail=event.detail)
    return _safe_mode_msg("gimbal_runaway")


def handle_comm_timeout(event: FaultEventMsg) -> Optional[ModeChangeMsg]:
    """FaultCode.COMM_TIMEOUT — communications subsystem missed its window.

    A single comm timeout is non-critical (TDRSS windows are not guaranteed).
    Log and continue.  The daily budget accounting handles repeated misses.
    """
    log.warning("comm_timeout", subsystem=event.subsystem, detail=event.detail)
    return None


def handle_watchdog_expire(event: FaultEventMsg) -> Optional[ModeChangeMsg]:
    """FaultCode.WATCHDOG_EXPIRE — a subsystem missed too many heartbeats.

    A silent process is considered dead.  Enter SAFE mode immediately.
    # TODO: attempt process restart before entering SAFE (Phase II).
    """
    log.error("watchdog_expire", subsystem=event.subsystem, detail=event.detail)
    return _safe_mode_msg("watchdog_expire")


def handle_model_corrupt(event: FaultEventMsg) -> Optional[ModeChangeMsg]:
    """FaultCode.MODEL_CORRUPT — model file CRC-32 verification failed on uplink.

    A corrupt model must never be activated.  Enter SAFE mode and await ground
    command to retry the uplink.
    """
    log.error("model_corrupt", subsystem=event.subsystem, detail=event.detail)
    return _safe_mode_msg("model_corrupt")


def handle_process_died(event: FaultEventMsg) -> Optional[ModeChangeMsg]:
    """FaultCode.PROCESS_DIED — a subsystem process exited unexpectedly.

    An unexpected process exit is always fatal in Phase I.  Enter SAFE mode.
    # TODO: attempt process restart in Phase II.
    """
    log.error("process_died", subsystem=event.subsystem, detail=event.detail)
    return _safe_mode_msg("process_died")


# ---------------------------------------------------------------------------
# Dispatch table — must cover every FaultCode member
# ---------------------------------------------------------------------------


FAULT_HANDLERS: Final[
    dict[FaultCode, Callable[[FaultEventMsg], Optional[ModeChangeMsg]]]
] = {
    FaultCode.NONE:                 handle_none,
    FaultCode.INFERENCE_TIMEOUT:    handle_inference_timeout,
    FaultCode.INFERENCE_NAN:        handle_inference_nan,
    FaultCode.CAMERA_STALL:         handle_camera_stall,
    FaultCode.STORAGE_FULL:         handle_storage_full,
    FaultCode.THERMAL_OVER_LIMIT:   handle_thermal_over_limit,
    FaultCode.POWER_OVER_LIMIT:     handle_power_over_limit,
    FaultCode.GIMBAL_RUNAWAY:       handle_gimbal_runaway,
    FaultCode.COMM_TIMEOUT:         handle_comm_timeout,
    FaultCode.WATCHDOG_EXPIRE:      handle_watchdog_expire,
    FaultCode.MODEL_CORRUPT:        handle_model_corrupt,
    FaultCode.PROCESS_DIED:         handle_process_died,
}
