"""PACT typed configuration dataclasses. §4.3 of PACT_SW_ARCH.

All tunable parameters for every subsystem are represented here as frozen dataclasses.
Default field values exactly match config/default.toml so that an unmodified load
produces identical results to constructing PactConfig() with no arguments.

ops/config_loader.py is the sole entry point for populating these classes from TOML.
No subsystem reads TOML directly — each receives its typed config dataclass argument.

Satisfies: REQ-AIML-COMP-001 (type-safe configuration throughout the system).
No other pact submodule is imported here.
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Per-subsystem config dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ControllerConfig:
    """Configuration for the gimbal controller and safety arbiter subsystem."""

    confidence_gate: float = 0.55               # minimum mean blob confidence to accept
    ema_alpha: float = 0.4                       # EMA smoothing factor (0 < alpha <= 1)
    min_deadband_px: int = 20                    # minimum displacement to issue a command
    max_deadband_px: int = 250                   # maximum displacement before GIMBAL_RUNAWAY
    max_deadband_strike_count: int = 3           # consecutive max violations before fault
    retarget_rate_limit_hz: float = 0.5          # maximum gimbal command rate (Hz)
    max_slew_rate_deg_per_s: float = 2.0         # maximum slew rate (degrees per second)
    acquire_persistence_frames: int = 3          # frames needed to enter TRACKING from ACQUIRING
    release_persistence_frames: int = 5          # consecutive miss frames before IDLE
    scan_entry_idle_seconds: float = 60.0        # idle duration before entering SCAN mode
    scan_slew_rate_deg_per_s: float = 0.5        # slew rate during nadir scan
    blob_iou_match_threshold: float = 0.25       # minimum IoU for blob association across frames
    min_blob_area_px: int = 15                   # minimum blob area in pixels to accept
    # Kalman filter parameters
    kalman_dt_s: float = 0.1                     # state propagation timestep (seconds)
    kalman_process_noise: float = 1e-2           # scalar process noise variance (Q = I * value)
    kalman_measurement_noise: float = 1e-1       # scalar measurement noise variance (R = I * value)
    # LQR controller parameters
    lqr_Q_diag: tuple[float, ...] = (10.0, 10.0, 1.0, 1.0)  # state cost weights [pan, tilt, dpan, dtilt]
    lqr_R_diag: tuple[float, ...] = (1.0, 1.0)              # control cost weights [pan_cmd, tilt_cmd]
    max_slew_deg_s: float = 2.0                  # maximum LQR output clamp (deg/s)


@dataclass(frozen=True)
class InferenceConfig:
    """Configuration for the inference subsystem and model deployment."""

    model_path: str = "data/models/active.pt"           # path to active model checkpoint
    rollback_model_path: str = "data/models/rollback.pt"  # path to rollback model checkpoint
    input_bands: tuple[str, ...] = ("B2", "B3", "B4", "B8")  # spectral bands to select
    input_height_px: int = 256                          # model input height in pixels
    input_width_px: int = 256                           # model input width in pixels
    use_int8: bool = False                              # enable INT8 quantization (flight only)
    latency_budget_ms: float = 500.0                    # maximum inference latency (TBD: update after Jetson benchmark)


@dataclass(frozen=True)
class CommsConfig:
    """Configuration for CCSDS communications, downlink, and uplink subsystems."""

    max_downlink_rate_bps: int = 5_000_000              # 5 Mbps TDRSS downlink limit
    max_uplink_rate_bps: int = 2_000_000                # 2 Mbps TDRSS uplink limit
    max_daily_downlink_bytes: int = 1_073_741_824        # 1 GB daily downlink cap
    max_daily_uplink_bytes: int = 104_857_600            # 100 MB daily uplink cap
    comm_window_days: tuple[str, ...] = ("MON", "TUE", "WED", "THU", "FRI")  # weekdays only
    ccsds_apid: int = 0x001                             # CCSDS Application Process Identifier
    staged_model_path: str = "data/models/staged.pt"   # staging path for uploaded model chunks


@dataclass(frozen=True)
class StorageConfig:
    """Configuration for the frame storage subsystem."""

    data_root: str = "data/flight"                      # root directory for all stored data
    max_storage_bytes: int = 107_374_182_400             # 100 GB storage limit (placeholder)
    checksum_algorithm: str = "sha256"                  # hash algorithm for file integrity


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration for the preprocessing quality-flag subsystem."""

    saturation_fraction_threshold: float = 0.05   # fraction of pixels above 0.95 → SATURATED flag
    nir_red_ratio_threshold: float = 3.0           # NIR/Red mean ratio above this → CLOUD_CONTAMINATED
    sunglint_nir_mean_threshold: float = 0.6       # mean NIR above this → SUNGLINT flag
    motion_smear_exposure_us: float = 5000.0       # exposure time above this (µs) → MOTION_SMEAR flag


@dataclass(frozen=True)
class FaultConfig:
    """Configuration for the fault detection and watchdog subsystem."""

    watchdog_interval_s: float = 5.0                    # heartbeat check interval (seconds)
    watchdog_max_miss_count: int = 3                    # missed heartbeats before fault
    inference_timeout_ms: float = 2000.0                # inference timeout before fault
    thermal_limit_c: float = 80.0                       # thermal limit in degrees Celsius
    power_limit_w: float = 55.0                         # power consumption limit in Watts


@dataclass(frozen=True)
class PactConfig:
    """Top-level PACT configuration. Composes all per-subsystem configs.

    Constructed by ops/config_loader.py from a TOML file. Default field values
    produce a fully functional development configuration with no arguments.
    """

    controller: ControllerConfig = field(default_factory=ControllerConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    comms: CommsConfig = field(default_factory=CommsConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    fault: FaultConfig = field(default_factory=FaultConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
