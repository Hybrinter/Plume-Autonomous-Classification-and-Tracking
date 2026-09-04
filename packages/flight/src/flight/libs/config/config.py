"""Flight typed configuration dataclasses.

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
from dataclasses import field
from typing import Literal, Self

# third-party
from pydantic import ConfigDict, Field, field_validator, model_validator
from pydantic.dataclasses import dataclass

_SCHEMA = ConfigDict(extra="forbid")
_MOSAIC_BANDS: frozenset[str] = frozenset({"BLUE", "GREEN", "RED", "NIR"})

# ---------------------------------------------------------------------------
# Per-subsystem config dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, config=_SCHEMA)
class ControllerConfig:
    """Configuration for the gimbal controller and safety arbiter subsystem."""

    confidence_gate: float = Field(default=0.55, ge=0.0, le=1.0)
    ema_alpha: float = Field(default=0.4, gt=0.0, le=1.0)
    retarget_rate_limit_hz: float = Field(default=0.5, gt=0.0)
    acquire_persistence_frames: int = Field(default=3, ge=1)
    release_persistence_frames: int = Field(default=5, ge=1)
    blob_iou_match_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    min_blob_area_px: int = 15
    kalman_dt_s: float = 0.1
    kalman_process_noise: float = Field(default=1e-2, gt=0.0)
    kalman_measurement_noise: float = Field(default=1e-1, gt=0.0)
    lqr_Q_diag: tuple[float, ...] = Field(default=(10.0, 10.0, 1.0, 1.0))  # noqa: N815
    lqr_R_diag: tuple[float, ...] = Field(default=(1.0, 1.0))  # noqa: N815
    runaway_rate_tolerance_deg_per_s: float = Field(default=1.0, gt=0.0)
    runaway_strike_count: int = Field(default=3, ge=1)


@dataclass(frozen=True, config=_SCHEMA)
class InferenceConfig:
    """Configuration for the inference subsystem and model deployment."""

    segmentor_model_path: str = "data/models/active_segmentor.onnx"
    classifier_model_path: str = "data/models/active_classifier.onnx"
    segmentor_rollback_model_path: str = "data/models/rollback_segmentor.onnx"
    classifier_rollback_model_path: str = "data/models/rollback_classifier.onnx"
    classifier_logit_threshold: float = 0.0
    input_bands: tuple[str, ...] = Field(default=("BLUE", "GREEN", "RED", "NIR"), min_length=1)
    input_height_px: int = Field(default=1024, gt=0)
    input_width_px: int = Field(default=1224, gt=0)
    use_int8: bool = False
    latency_budget_ms: float = Field(default=4.0, gt=0.0)


@dataclass(frozen=True, config=_SCHEMA)
class CommsConfig:
    """Configuration for CCSDS communications, downlink, and uplink subsystems."""

    max_downlink_rate_bps: int = Field(default=5_000_000, gt=0)
    max_uplink_rate_bps: int = Field(default=2_000_000, gt=0)
    max_daily_downlink_bytes: int = Field(default=1_073_741_824, gt=0)
    max_daily_uplink_bytes: int = Field(default=104_857_600, gt=0)
    comm_window_days: tuple[str, ...] = ("MON", "TUE", "WED", "THU", "FRI")
    ccsds_apid: int = Field(default=0x001, ge=0, le=0x7FF)
    staged_segmentor_model_path: str = "data/models/staged_segmentor.onnx"
    staged_classifier_model_path: str = "data/models/staged_classifier.onnx"
    downlink_max_bytes_per_pass: int = Field(default=1_048_576, gt=0)


@dataclass(frozen=True, config=_SCHEMA)
class StorageConfig:
    """Configuration for the frame storage subsystem."""

    data_root: str = "data/flight"
    max_storage_bytes: int = Field(default=107_374_182_400, gt=0)
    checksum_algorithm: str = Field(default="sha256", min_length=1)


@dataclass(frozen=True, config=_SCHEMA)
class SensorConfig:
    """Configuration for the imaging sensor and its 2x2 mosaic filter optics.

    Geometry and optics constants for the FLIR Blackfly S BFS-U3-50S5M-C (Sony IMX264)
    behind a 150 mm f/4 athermal lens and a custom 2x2 mosaic filter
    (BLUE/GREEN/RED/NIR ~ Sentinel-2 B2/B3/B4/B8). width_px is lateral (cross-track);
    height_px is along-track. IFOV is a fixed optic property. Pointing and smear use
    ifov_band_deg_per_px (2x2 demosaic). These values drive demosaic, normalization,
    quality gates, and the composition root's calibration-load decision.

    Satisfies: REQ-AIML-IMAG-001.
    """

    part_number: str = "BFS-U3-50S5M-C"
    sensor_name: str = "Sony IMX264"
    width_px: int = Field(default=2448, gt=0)
    height_px: int = Field(default=2048, gt=0)
    bit_depth: int = Field(default=12, ge=1, le=16)
    mosaic_layout: tuple[str, ...] = ("BLUE", "GREEN", "RED", "NIR")
    pixel_um: float = Field(default=3.45, gt=0.0)
    focal_length_mm: float = Field(default=150.0, gt=0.0)
    f_number: float = Field(default=4.0, gt=0.0)
    lens_distortion_pct: float = Field(default=0.66, ge=0.0)
    ifov_mosaic_deg_per_px: float = Field(default=0.001318, gt=0.0)
    ifov_band_deg_per_px: float = Field(default=0.002636, gt=0.0)
    fov_lateral_deg: float = Field(default=3.204, gt=0.0)
    fov_along_deg: float = Field(default=2.681, gt=0.0)
    datasheet_hfov_2_3_deg: float = Field(default=3.36, gt=0.0)
    qe_530_pct: float = Field(default=62.51, gt=0.0)
    saturation_capacity_e: float = Field(default=10824.0, gt=0.0)
    temporal_dark_noise_e: float = Field(default=2.27, ge=0.0)
    dynamic_range_db: float = Field(default=71.83, gt=0.0)
    max_frame_rate_hz: float = Field(default=35.0, gt=0.0)
    exposure_min_us: float = Field(default=13.0, gt=0.0)
    exposure_max_us: float = Field(default=30_000_000.0, gt=0.0)
    initial_exposure_us: float = Field(default=13.0, gt=0.0)
    gain_min_db: float = Field(default=0.0, ge=0.0)
    gain_max_db: float = Field(default=47.0, ge=0.0)
    initial_gain_db: float = Field(default=0.0, ge=0.0)
    calibration_dir: str = ""

    @field_validator("width_px", "height_px")
    @classmethod
    def _even_mosaic_dim(cls, value: int) -> int:
        """Reject odd mosaic-plane dimensions (2x2 CFA separation)."""
        if value % 2:
            raise ValueError("must be even (2x2 mosaic separation)")
        return value

    @model_validator(mode="after")
    def _mosaic_permutation(self) -> Self:
        """Reject a mosaic_layout that is not a permutation of BLUE/GREEN/RED/NIR."""
        if frozenset(self.mosaic_layout) != _MOSAIC_BANDS or len(self.mosaic_layout) != len(
            _MOSAIC_BANDS
        ):
            raise ValueError(
                "sensor.mosaic_layout must name each Band (BLUE/GREEN/RED/NIR) exactly once"
            )
        return self

    @model_validator(mode="after")
    def _exposure_gain_range(self) -> Self:
        """Reject inverted exposure/gain ranges or initials outside the range."""
        if self.exposure_min_us >= self.exposure_max_us:
            raise ValueError("exposure_min_us must be < exposure_max_us")
        if not (self.exposure_min_us <= self.initial_exposure_us <= self.exposure_max_us):
            raise ValueError(
                "initial_exposure_us must be within [exposure_min_us, exposure_max_us]"
            )
        if self.gain_min_db > self.gain_max_db:
            raise ValueError("gain_min_db must be <= gain_max_db")
        if not (self.gain_min_db <= self.initial_gain_db <= self.gain_max_db):
            raise ValueError("initial_gain_db must be within [gain_min_db, gain_max_db]")
        return self


@dataclass(frozen=True, config=_SCHEMA)
class PreprocessingConfig:
    """Configuration for the preprocessing quality-flag subsystem."""

    saturation_fraction_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    nir_red_ratio_threshold: float = Field(default=3.0, gt=0.0)
    sunglint_nir_mean_threshold: float = Field(default=0.6, gt=0.0)
    max_motion_smear_px: float = Field(default=1.0, gt=0.0)


@dataclass(frozen=True, config=_SCHEMA)
class FaultConfig:
    """Configuration for the fault detection and watchdog subsystem."""

    watchdog_interval_s: float = Field(default=5.0, gt=0.0)
    watchdog_max_miss_count: int = Field(default=3, ge=1)
    inference_timeout_ms: float = Field(default=20.0, gt=0.0)
    power_limit_w: float = Field(default=55.0, gt=0.0)


@dataclass(frozen=True, config=_SCHEMA)
class ThermalConfig:
    """Record-only per-component temperature limits in degrees Celsius.

    Housekeeping publishes a single scalar sample and does not compare against these
    limits until per-component sensors exist.
    """

    camera_min_c: float = 0.0
    camera_max_c: float = 50.0
    lens_min_c: float = -10.0
    lens_max_c: float = 50.0
    gimbal_min_c: float = -20.0
    gimbal_max_c: float = 70.0
    compute_min_c: float = 0.0
    compute_max_c: float = 80.0

    @model_validator(mode="after")
    def _limit_order(self) -> Self:
        """Reject inverted min/max pairs."""
        pairs = (
            ("camera", self.camera_min_c, self.camera_max_c),
            ("lens", self.lens_min_c, self.lens_max_c),
            ("gimbal", self.gimbal_min_c, self.gimbal_max_c),
            ("compute", self.compute_min_c, self.compute_max_c),
        )
        for name, lo, hi in pairs:
            if lo >= hi:
                raise ValueError(f"{name}_min_c must be < {name}_max_c")
        return self


@dataclass(frozen=True, config=_SCHEMA)
class GimbalConfig:
    """Configuration for the single-axis gimbal envelope, poses, sim dynamics, and link.

    Elevation is signed off-nadir degrees: 0 at geocentric nadir, positive along-track
    (velocity), negative look-back. Hardware travel, science imaging window, and stow/home
    poses are distinct. SimGimbal first-order dynamics and the serial PTU link live here.

    Satisfies: REQ-AIML-GIMB-001, REQ-GIMB-HIGH-001.
    """

    el_hw_min_deg: float = -45.0
    el_hw_max_deg: float = 90.0
    el_science_min_deg: float = 0.0
    el_science_max_deg: float = 45.0
    max_hw_slew_rate_deg_per_s: float = Field(default=10.0, gt=0.0)
    stow_el_deg: float = -45.0
    home_el_deg: float = 45.0
    sim_time_constant_s: float = 0.2
    sim_encoder_noise_deg: float = 0.005
    sim_seed: int = 0
    serial_port: str = ""
    serial_baud: int = 9600
    counts_per_deg: float = 77.6

    @model_validator(mode="after")
    def _travel_envelope(self) -> Self:
        """Reject inverted envelopes or poses outside hardware travel."""
        if self.el_hw_min_deg >= self.el_hw_max_deg:
            raise ValueError("el_hw_min_deg must be < el_hw_max_deg")
        if self.el_science_min_deg >= self.el_science_max_deg:
            raise ValueError("el_science_min_deg must be < el_science_max_deg")
        if not (
            self.el_hw_min_deg <= self.el_science_min_deg
            and self.el_science_max_deg <= self.el_hw_max_deg
        ):
            raise ValueError("science elevation window must lie within hardware travel")
        if not (self.el_hw_min_deg <= self.stow_el_deg <= self.el_hw_max_deg):
            raise ValueError("stow_el_deg must be within [el_hw_min_deg, el_hw_max_deg]")
        if not (self.el_hw_min_deg <= self.home_el_deg <= self.el_hw_max_deg):
            raise ValueError("home_el_deg must be within [el_hw_min_deg, el_hw_max_deg]")
        return self


@dataclass(frozen=True, config=_SCHEMA)
class LinkConfig:
    """Station data-link transport config: CCSDS endpoints + APIDs.

    Commands arrive as CCSDS TC packets over a TCP server socket the payload binds; telemetry
    and products are sent as CCSDS TM packets over UDP to the station endpoint. Sockets open
    lazily in the real driver; SIL uses the byte-level sim link and ignores host/port.
    """

    command_tcp_host: str = "127.0.0.1"
    command_tcp_port: int = Field(default=50501, ge=1, le=65535)
    telemetry_udp_host: str = "127.0.0.1"
    telemetry_udp_port: int = Field(default=50502, ge=1, le=65535)
    socket_timeout_s: float = Field(default=1.0, gt=0.0)
    tc_apid: int = Field(default=0x001, ge=0, le=0x7FF)
    tm_apid: int = Field(default=0x002, ge=0, le=0x7FF)


@dataclass(frozen=True, config=_SCHEMA)
class CommandIngressConfig:
    """Command-ingress integrity + authentication config.

    The HMAC key is loaded from hmac_key_path by the composition root and injected into
    iss_iface (not read by the app). accepted_sources is the command-origin allow-list; the
    per-source replay guard (reject seq <= last accepted seq per source) is enforced in the
    ingress pipeline state, not here.
    """

    hmac_key_path: str = "data/keys/uplink_hmac.key"
    require_auth: bool = True
    accepted_sources: tuple[str, ...] = Field(default=("ground", "station_ops"), min_length=1)

    @model_validator(mode="after")
    def _auth_key_path(self) -> Self:
        """Reject an empty HMAC path when authentication is required."""
        if self.require_auth and not self.hmac_key_path:
            raise ValueError("hmac_key_path must be set when require_auth is true")
        return self


@dataclass(frozen=True, config=_SCHEMA)
class CommandRouterConfig:
    """Configuration for the core command router (routing + ARM/EXECUTE two-step).

    arm_window_s bounds how long a hazardous command stays armed before its EXECUTE must
    arrive; an EXECUTE after the window lapses is rejected and re-arming is required. The
    routable-target set and the hazardous-command set are derived from the command dictionary
    (flight.libs.commands), not configured here, so they stay in sync with the dictionary.
    """

    arm_window_s: float = Field(default=30.0, gt=0.0)


# A deployment axis is wired to either a sim stand-in or the real device/driver.
AxisMode = Literal["sim", "real"]


@dataclass(frozen=True, config=_SCHEMA)
class EnvironmentConfig:
    """Per-axis sim/real wiring selector for the composition root.

    Each field names a deployment axis the composition root must resolve to a
    concrete driver: 'sim' selects an in-process stand-in, 'real' selects the
    flight driver/device. host is a free-form label for the target machine
    (provenance only; not acted on). The 'lock' (LaunchLock) axis is intentionally
    absent: there is no LaunchLock device, so it is a permanent VCRM gap, not a
    config field. The clock axis is informational here -- the composition root
    chooses RealClock vs ManualClock from it BEFORE building drivers.

    Satisfies: REQ-OPER-HIGH-002 (validated startup config selects the deployment axes).
    """

    sensor: AxisMode = "real"
    gimbal: AxisMode = "real"
    compute: AxisMode = "real"
    link: AxisMode = "real"
    clock: AxisMode = "real"
    host: str = "jetson_aarch64"


@dataclass(frozen=True, config=_SCHEMA)
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
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    gimbal: GimbalConfig = field(default_factory=GimbalConfig)
    link: LinkConfig = field(default_factory=LinkConfig)
    command_ingress: CommandIngressConfig = field(default_factory=CommandIngressConfig)
    command_router: CommandRouterConfig = field(default_factory=CommandRouterConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)

    @model_validator(mode="after")
    def _input_bands_in_mosaic(self) -> Self:
        """Reject inference input bands that are absent from the sensor mosaic."""
        mosaic_set = set(self.sensor.mosaic_layout)
        for band in self.inference.input_bands:
            if band not in mosaic_set:
                raise ValueError(
                    f"inference.input_bands entry {band!r} is not present in sensor.mosaic_layout"
                )
        return self

    @model_validator(mode="after")
    def _inference_matches_band_plane(self) -> Self:
        """Reject inference input size that is not the full demosaiced band plane."""
        plane_h = self.sensor.height_px // 2
        plane_w = self.sensor.width_px // 2
        if self.inference.input_height_px != plane_h or self.inference.input_width_px != plane_w:
            raise ValueError(
                f"inference input size must equal the demosaiced band plane ({plane_h} x {plane_w})"
            )
        return self
