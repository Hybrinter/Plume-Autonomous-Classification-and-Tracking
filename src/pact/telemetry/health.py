"""System health snapshot dataclass.

Defines the immutable SystemHealthSnapshot that captures a point-in-time view of the
PACT system's operational health.  Produced by the telemetry reporter and serialised
into a CCSDS packet for downlink.

Satisfies: REQ-OPER-HIGH-001.
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass

# internal
from pact.types.enums import FaultCode, GimbalState, ModelDeployState, SystemMode


@dataclass(frozen=True)
class SystemHealthSnapshot:
    """Snapshot of system health for periodic monitoring.  REQ-OPER-HIGH-001.

    All fields are populated at construction time; no incremental updates.
    This dataclass is frozen (immutable) — mirroring a Rust struct.

    Fields:
        timestamp_utc:              ISO 8601 timestamp of snapshot creation.
        system_mode:                Current top-level system mode.
        gimbal_state:               Current gimbal arbiter state.
        active_faults:              Set of currently active (uncleared) fault codes.
        frames_captured_today:      Count of frames captured in the current UTC day.
        bytes_downlinked_today:     Bytes downlinked in the current UTC day.
        bytes_remaining_today:      Remaining daily downlink budget in bytes.
        model_version:              Identifier string of the active model checkpoint.
        model_deploy_state:         Deployment lifecycle state of the active model.
        inference_latency_ms_mean:  Mean inference latency over the last N frames.
        inference_latency_ms_max:   Maximum inference latency over the last N frames.
        storage_bytes_used:         Total bytes used in the data_root directory.
    """

    timestamp_utc: str
    system_mode: SystemMode
    gimbal_state: GimbalState
    active_faults: frozenset[FaultCode]
    frames_captured_today: int
    bytes_downlinked_today: int
    bytes_remaining_today: int
    model_version: str
    model_deploy_state: ModelDeployState
    inference_latency_ms_mean: float
    inference_latency_ms_max: float
    storage_bytes_used: int
