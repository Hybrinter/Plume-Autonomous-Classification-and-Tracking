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
_SYMMETRY_TOL = 1e-12


def _require_len_4(name: str, values: tuple[float, ...]) -> None:
    """Reject a row-major 2x2 that is not length 4."""
    if len(values) != 4:
        raise ValueError(f"{name} must have length 4 (row-major 2x2)")


def _require_spd_2x2(name: str, values: tuple[float, ...]) -> None:
    """Reject a 2x2 inertia that is not symmetric positive definite."""
    _require_len_4(name, values)
    a, b, c, d = values
    if abs(b - c) > _SYMMETRY_TOL:
        raise ValueError(f"{name} must be symmetric")
    det = a * d - b * c
    if a <= 0.0 or det <= 0.0:
        raise ValueError(f"{name} must be positive definite")


# ---------------------------------------------------------------------------
# Per-subsystem config dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, config=_SCHEMA)
class ControllerConfig:
    """Configuration for the gimbal controller and safety arbiter subsystem."""

    confidence_gate: float = Field(default=0.55, ge=0.0, le=1.0)
    ema_alpha: float = Field(default=0.4, gt=0.0, le=1.0)
    min_deadband_px: int = Field(default=20, ge=0)
    max_deadband_px: int = 250
    max_deadband_strike_count: int = Field(default=3, ge=1)
    retarget_rate_limit_hz: float = Field(default=0.5, gt=0.0)
    max_slew_rate_deg_per_s: float = Field(default=2.0, gt=0.0)
    acquire_persistence_frames: int = Field(default=3, ge=1)
    release_persistence_frames: int = Field(default=5, ge=1)
    scan_entry_idle_seconds: float = 60.0
    scan_slew_rate_deg_per_s: float = Field(default=0.5, gt=0.0)
    blob_iou_match_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    min_blob_area_px: int = 15
    kalman_dt_s: float = 0.1
    kalman_process_noise: float = Field(default=1e-2, gt=0.0)
    kalman_measurement_noise: float = Field(default=1e-1, gt=0.0)
    lqr_Q_diag: tuple[float, ...] = Field(default=(10.0, 10.0, 1.0, 1.0))  # noqa: N815
    lqr_R_diag: tuple[float, ...] = Field(default=(1.0, 1.0))  # noqa: N815
    max_slew_deg_s: float = Field(default=2.0, gt=0.0)
    runaway_rate_tolerance_deg_per_s: float = Field(default=1.0, gt=0.0)
    runaway_strike_count: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def _deadband_order(self) -> Self:
        """Reject a max deadband that is not strictly greater than the min."""
        if self.max_deadband_px <= self.min_deadband_px:
            raise ValueError("max_deadband_px must be > min_deadband_px")
        return self


@dataclass(frozen=True, config=_SCHEMA)
class InferenceConfig:
    """Configuration for the inference subsystem and model deployment."""

    segmentor_model_path: str = "data/models/active_segmentor.onnx"
    classifier_model_path: str = "data/models/active_classifier.onnx"
    segmentor_rollback_model_path: str = "data/models/rollback_segmentor.onnx"
    classifier_rollback_model_path: str = "data/models/rollback_classifier.onnx"
    classifier_logit_threshold: float = 0.0
    input_bands: tuple[str, ...] = Field(default=("BLUE", "GREEN", "RED", "NIR"), min_length=1)
    input_height_px: int = Field(default=256, gt=0)
    input_width_px: int = Field(default=256, gt=0)
    use_int8: bool = False
    latency_budget_ms: float = Field(default=500.0, gt=0.0)


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

    Geometry and optics constants for the FLIR Blackfly S (12-bit Sony IMX-class)
    behind a custom 2x2 mosaic filter (BLUE/GREEN/RED/NIR ~ Sentinel-2 B2/B3/B4/B8).
    These values drive demosaic, normalization, quality gates, and the composition
    root's calibration-load decision.

    Satisfies: REQ-AIML-IMAG-001.
    """

    width_px: int = Field(default=1024, gt=0)
    height_px: int = Field(default=1024, gt=0)
    bit_depth: int = Field(default=12, ge=1, le=16)
    mosaic_layout: tuple[str, ...] = ("BLUE", "GREEN", "RED", "NIR")
    ifov_deg_per_px: float = Field(default=0.02, gt=0.0)
    default_exposure_us: float = Field(default=1000.0, gt=0.0)
    default_gain_db: float = Field(default=0.0, ge=0.0)
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
    inference_timeout_ms: float = Field(default=2000.0, gt=0.0)
    thermal_limit_c: float = Field(default=80.0, gt=0.0)
    power_limit_w: float = Field(default=55.0, gt=0.0)


@dataclass(frozen=True, config=_SCHEMA)
class GimbalConfig:
    """Configuration for the gimbal hardware envelope, poses, sim dynamics, and link.

    Fields cover the travel limits, configured stow/home poses, SimGimbal inertia and
    first-order dynamics parameters for SIL, and the serial link for the real PTU driver.

    Satisfies: REQ-AIML-GIMB-001, REQ-GIMB-HIGH-001.
    """

    az_min_deg: float = -90.0
    az_max_deg: float = 90.0
    el_min_deg: float = -45.0
    el_max_deg: float = 45.0
    max_hw_slew_rate_deg_per_s: float = Field(default=10.0, gt=0.0)
    stow_az_deg: float = 0.0
    stow_el_deg: float = -45.0
    home_az_deg: float = 0.0
    home_el_deg: float = 0.0
    sim_time_constant_s: float = 0.2
    sim_encoder_noise_deg: float = 0.005
    sim_seed: int = 0
    serial_port: str = ""
    serial_baud: int = 9600
    counts_per_deg: float = 77.6
    J_kg_m2: tuple[float, ...] = Field(default=(1.0, 0.0, 0.0, 1.0))  # noqa: N815
    B_nms_per_rad: tuple[float, ...] = Field(default=(0.1, 0.0, 0.0, 0.1))  # noqa: N815

    @model_validator(mode="after")
    def _travel_envelope(self) -> Self:
        """Reject inverted travel limits, poses outside the envelope, or a bad inertia matrix."""
        if self.az_min_deg >= self.az_max_deg:
            raise ValueError("az_min_deg must be < az_max_deg")
        if self.el_min_deg >= self.el_max_deg:
            raise ValueError("el_min_deg must be < el_max_deg")
        if not (self.az_min_deg <= self.stow_az_deg <= self.az_max_deg):
            raise ValueError("stow_az_deg must be within [az_min_deg, az_max_deg]")
        if not (self.el_min_deg <= self.stow_el_deg <= self.el_max_deg):
            raise ValueError("stow_el_deg must be within [el_min_deg, el_max_deg]")
        if not (self.az_min_deg <= self.home_az_deg <= self.az_max_deg):
            raise ValueError("home_az_deg must be within [az_min_deg, az_max_deg]")
        if not (self.el_min_deg <= self.home_el_deg <= self.el_max_deg):
            raise ValueError("home_el_deg must be within [el_min_deg, el_max_deg]")
        _require_spd_2x2("J_kg_m2", self.J_kg_m2)
        _require_len_4("B_nms_per_rad", self.B_nms_per_rad)
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
