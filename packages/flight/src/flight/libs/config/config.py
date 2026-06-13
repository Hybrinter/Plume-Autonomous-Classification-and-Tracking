"""Flight typed configuration dataclasses. Section 4.3 of PACT_SW_ARCH.

All tunable parameters for every subsystem are represented here as frozen dataclasses.
Default field values exactly match config/default.toml so that an unmodified load
produces identical results to constructing PactConfig() with no arguments.

config_loader is the sole entry point for populating these classes from TOML.
No subsystem reads TOML directly -- each receives its typed config dataclass argument.

Satisfies: REQ-AIML-COMP-001 (type-safe configuration throughout the system).
No other flight module is imported here.
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

    confidence_gate: float = 0.55  # minimum mean blob confidence to accept
    ema_alpha: float = 0.4  # EMA smoothing factor (0 < alpha <= 1)
    min_deadband_px: int = 20  # minimum displacement to issue a command
    max_deadband_px: int = 250  # maximum displacement before GIMBAL_RUNAWAY
    max_deadband_strike_count: int = 3  # consecutive max violations before fault
    retarget_rate_limit_hz: float = 0.5  # maximum gimbal command rate (Hz)
    max_slew_rate_deg_per_s: float = 2.0  # maximum slew rate (degrees per second)
    acquire_persistence_frames: int = 3  # frames needed to enter TRACKING from ACQUIRING
    release_persistence_frames: int = 5  # consecutive miss frames before IDLE
    scan_entry_idle_seconds: float = 60.0  # idle duration before entering SCAN mode
    scan_slew_rate_deg_per_s: float = 0.5  # slew rate during nadir scan
    blob_iou_match_threshold: float = 0.25  # minimum IoU for blob association across frames
    min_blob_area_px: int = 15  # minimum blob area in pixels to accept
    # Kalman filter parameters
    kalman_dt_s: float = 0.1  # state propagation timestep (seconds)
    kalman_process_noise: float = 1e-2  # scalar process noise variance (Q = I * value)
    kalman_measurement_noise: float = 1e-1  # scalar measurement noise variance (R = I * value)
    # LQR controller parameters
    lqr_Q_diag: tuple[float, ...] = (10.0, 10.0, 1.0, 1.0)  # noqa: N815  state cost weights
    lqr_R_diag: tuple[float, ...] = (1.0, 1.0)  # noqa: N815  control cost weights
    max_slew_deg_s: float = 2.0  # maximum LQR output clamp (deg/s)
    # Encoder-runaway tuning
    runaway_rate_tolerance_deg_per_s: float = 1.0  # commanded-vs-encoder rate divergence limit
    runaway_strike_count: int = 3  # consecutive divergent frames before GIMBAL_RUNAWAY


@dataclass(frozen=True)
class InferenceConfig:
    """Configuration for the inference subsystem and model deployment."""

    model_path: str = "data/models/active.pt"  # path to active model checkpoint
    rollback_model_path: str = "data/models/rollback.pt"  # path to rollback model checkpoint
    input_bands: tuple[str, ...] = ("BLUE", "GREEN", "RED", "NIR")  # bands to select
    input_height_px: int = 256  # model input height in pixels
    input_width_px: int = 256  # model input width in pixels
    use_int8: bool = False  # enable INT8 quantization (flight only)
    latency_budget_ms: float = 500.0  # max inference latency (TBD: tune after Jetson benchmark)


@dataclass(frozen=True)
class CommsConfig:
    """Configuration for CCSDS communications, downlink, and uplink subsystems."""

    max_downlink_rate_bps: int = 5_000_000  # 5 Mbps TDRSS downlink limit
    max_uplink_rate_bps: int = 2_000_000  # 2 Mbps TDRSS uplink limit
    max_daily_downlink_bytes: int = 1_073_741_824  # 1 GB daily downlink cap
    max_daily_uplink_bytes: int = 104_857_600  # 100 MB daily uplink cap
    comm_window_days: tuple[str, ...] = ("MON", "TUE", "WED", "THU", "FRI")  # weekdays only
    ccsds_apid: int = 0x001  # CCSDS Application Process Identifier
    staged_model_path: str = "data/models/staged.pt"  # staging path for uploaded model chunks


@dataclass(frozen=True)
class StorageConfig:
    """Configuration for the frame storage subsystem."""

    data_root: str = "data/flight"  # root directory for all stored data
    max_storage_bytes: int = 107_374_182_400  # 100 GB storage limit (placeholder)
    checksum_algorithm: str = "sha256"  # hash algorithm for file integrity


@dataclass(frozen=True)
class SensorConfig:
    """Configuration for the imaging sensor and its 2x2 mosaic filter optics.

    Geometry and optics constants for the FLIR Blackfly S (12-bit Sony IMX-class)
    behind a custom 2x2 mosaic filter (BLUE/GREEN/RED/NIR ~ Sentinel-2 B2/B3/B4/B8).
    These values drive demosaic, normalization, quality gates, and the composition
    root's calibration-load decision.

    Satisfies: REQ-AIML-IMAG-001.
    """

    width_px: int = 1024  # mosaic plane width in pixels (must be even)
    height_px: int = 1024  # mosaic plane height in pixels (must be even)
    bit_depth: int = 12  # ADC bit depth; full scale = 2**bit_depth - 1 DN
    # Row-major band name per 2x2 cell: (0,0), (0,1), (1,0), (1,1).
    mosaic_layout: tuple[str, ...] = ("BLUE", "GREEN", "RED", "NIR")
    # Per band-plane pixel; 1024 @ 0.02 keeps FOV parity with the previous 512 @ 0.04.
    ifov_deg_per_px: float = 0.02  # instantaneous field of view per band-plane pixel
    default_exposure_us: float = 1000.0  # exposure commanded at startup
    default_gain_db: float = 0.0  # gain commanded at startup
    calibration_dir: str = ""  # dir of dark/flat/bad-pixel artifacts; "" -> identity (SIL only)


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration for the preprocessing quality-flag subsystem."""

    saturation_fraction_threshold: float = 0.05  # fraction of pixels above 0.95 -> SATURATED flag
    nir_red_ratio_threshold: float = 3.0  # NIR/Red mean ratio above this -> CLOUD_CONTAMINATED
    sunglint_nir_mean_threshold: float = 0.6  # mean NIR above this -> SUNGLINT flag
    max_motion_smear_px: float = 1.0  # predicted smear (slew x exposure / IFOV) above this -> flag


@dataclass(frozen=True)
class FaultConfig:
    """Configuration for the fault detection and watchdog subsystem."""

    watchdog_interval_s: float = 5.0  # heartbeat check interval (seconds)
    watchdog_max_miss_count: int = 3  # missed heartbeats before fault
    inference_timeout_ms: float = 2000.0  # inference timeout before fault
    thermal_limit_c: float = 80.0  # thermal limit in degrees Celsius
    power_limit_w: float = 55.0  # power consumption limit in Watts


@dataclass(frozen=True)
class GimbalConfig:
    """Configuration for the gimbal hardware envelope, poses, sim dynamics, and link.

    Fields cover the travel limits, configured stow/home poses, SimGimbal first-order
    dynamics parameters for SIL, and the serial link for the real PTU driver.

    Satisfies: REQ-AIML-GIMB-001, REQ-GIMB-HIGH-001.
    """

    az_min_deg: float = -90.0  # travel limit, azimuth minimum
    az_max_deg: float = 90.0  # travel limit, azimuth maximum
    el_min_deg: float = -45.0  # travel limit, elevation minimum
    el_max_deg: float = 45.0  # travel limit, elevation maximum
    max_hw_slew_rate_deg_per_s: float = 10.0  # hardware slew envelope (driver-enforced)
    stow_az_deg: float = 0.0  # stow pose azimuth (inside travel limits)
    stow_el_deg: float = -45.0  # stow pose elevation (inside travel limits)
    home_az_deg: float = 0.0  # home pose azimuth
    home_el_deg: float = 0.0  # home pose elevation
    sim_time_constant_s: float = 0.2  # SimGimbal first-order response time constant
    sim_encoder_noise_deg: float = 0.005  # SimGimbal encoder read noise (1-sigma)
    sim_seed: int = 0  # SimGimbal noise RNG seed (SIL determinism)
    serial_port: str = ""  # PTU serial port; "" -> RealGimbal unavailable (startup error)
    serial_baud: int = 9600  # PTU serial baud rate
    counts_per_deg: float = 77.6  # PTU encoder counts per degree (E46-class resolution)


@dataclass(frozen=True)
class LinkConfig:
    """Station data-link transport config: CCSDS endpoints + APIDs.

    Commands arrive as CCSDS TC packets over a TCP server socket the payload binds; telemetry
    and products are sent as CCSDS TM packets over UDP to the station endpoint. Sockets open
    lazily in the real driver; SIL uses the byte-level sim link and ignores host/port.
    """

    command_tcp_host: str = "127.0.0.1"  # bind address for inbound TC server socket
    command_tcp_port: int = 50501  # TCP port the payload listens on for commands
    telemetry_udp_host: str = "127.0.0.1"  # station endpoint for outbound TM
    telemetry_udp_port: int = 50502  # UDP port for outbound telemetry/products
    socket_timeout_s: float = 1.0  # accept/recv timeout so the link thread can stop promptly
    tc_apid: int = 0x001  # CCSDS APID for inbound telecommands
    tm_apid: int = 0x002  # CCSDS APID for outbound telemetry


@dataclass(frozen=True)
class CommandIngressConfig:
    """Command-ingress integrity + authentication config.

    The HMAC key is loaded from hmac_key_path by the composition root and injected into
    iss_iface (not read by the app). accepted_sources is the command-origin allow-list; the
    per-source replay guard (reject seq <= last accepted seq per source) is enforced in the
    ingress pipeline state, not here.
    """

    hmac_key_path: str = "data/keys/uplink_hmac.key"  # path to the shared HMAC secret
    require_auth: bool = True  # if False, skip HMAC verification (test/bench only)
    accepted_sources: tuple[str, ...] = ("ground", "station_ops")  # allowed command origins


@dataclass(frozen=True)
class PactConfig:
    """Top-level PACT configuration. Composes all per-subsystem configs.

    Constructed by config_loader from a TOML file. Default field values
    produce a fully functional development configuration with no arguments.
    """

    controller: ControllerConfig = field(default_factory=ControllerConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    comms: CommsConfig = field(default_factory=CommsConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    fault: FaultConfig = field(default_factory=FaultConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    gimbal: GimbalConfig = field(default_factory=GimbalConfig)
    link: LinkConfig = field(default_factory=LinkConfig)
    command_ingress: CommandIngressConfig = field(default_factory=CommandIngressConfig)
