"""Configuration loader.

Merges config/default.toml with an optional override file (e.g. config/flight.toml).
Override values take precedence over defaults.  Result is mapped field-by-field into
the PactConfig hierarchy of frozen dataclasses.

No subsystem reads TOML directly -- each receives a typed config dataclass argument.

Satisfies: REQ-OPER-HIGH-002 (validated config at startup), REQ-CONFIG-INTEGRITY-001.
"""

from __future__ import annotations

# stdlib
import dataclasses
import tomllib
from typing import Any

# internal
from flight.libs.config import (
    AxisMode,
    CommandIngressConfig,
    CommandRouterConfig,
    CommsConfig,
    ControllerConfig,
    EnvironmentConfig,
    FaultConfig,
    GimbalConfig,
    InferenceConfig,
    LinkConfig,
    PactConfig,
    PreprocessingConfig,
    SensorConfig,
    StorageConfig,
)
from flight.libs.types import Band, Err, Ok, Result

# Maps each TOML section name to the frozen config dataclass that backs it. Used to reject
# unknown sections and unknown per-section keys (typos must fail loudly at startup, not be
# silently ignored -- the one place raising/Err on a bad config is correct).
_SECTION_TO_CLASS: dict[str, type] = {
    "controller": ControllerConfig,
    "inference": InferenceConfig,
    "comms": CommsConfig,
    "storage": StorageConfig,
    "fault": FaultConfig,
    "preprocessing": PreprocessingConfig,
    "sensor": SensorConfig,
    "gimbal": GimbalConfig,
    "link": LinkConfig,
    "command_ingress": CommandIngressConfig,
    "command_router": CommandRouterConfig,
    "environment": EnvironmentConfig,
}


def load_config(
    config_path: str,
    override_path: str | None = None,
) -> Result[PactConfig, str]:
    """Load a TOML config file and populate PactConfig.

    Merge logic:
      1. Load config/default.toml (always required).
      2. If override_path is provided, load it and deep-merge on top of defaults.
         Keys present in the override replace defaults; absent keys retain defaults.
      3. Map the merged dict field-by-field into the PactConfig dataclass hierarchy.

    Returns Ok(PactConfig) on success.
    Returns Err(str) if any file is missing, malformed, or out-of-range.
    """
    try:
        with open(config_path, "rb") as fh:
            data: dict[str, Any] = tomllib.load(fh)
    except FileNotFoundError:
        return Err(f"config file not found: {config_path}")
    except tomllib.TOMLDecodeError as exc:
        return Err(f"TOML parse error in {config_path}: {exc}")

    if override_path is not None:
        try:
            with open(override_path, "rb") as fh:
                override: dict[str, Any] = tomllib.load(fh)
            data = _deep_merge(data, override)
        except FileNotFoundError:
            return Err(f"override config not found: {override_path}")
        except tomllib.TOMLDecodeError as exc:
            return Err(f"TOML parse error in {override_path}: {exc}")

    validation_error = _validate(data)
    if validation_error is not None:
        return Err(validation_error)

    try:
        config = _build_pact_config(data)
    except (KeyError, TypeError, ValueError) as exc:
        return Err(f"config mapping error: {exc}")

    return Ok(config)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base.  Override values win at every level."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _validate(data: dict[str, Any]) -> str | None:
    """Validate the merged config dict. Return the first violation string, or None if valid.

    Runs, in order: unknown-section/unknown-key rejection (typo guard), per-section range
    checks, then cross-field checks (gimbal travel envelope, mosaic/band consistency). All
    checks run against the merged dict (default.toml + optional override), so every section
    and field is present with at least its default value.

    Args:
        data: The merged TOML dict to validate.

    Returns:
        A human-readable violation string for the first failed check, or None if the config
        is internally consistent.

    Notes:
        APID values must fit in 11 bits (0..0x7FF) per the CCSDS primary-header layout; ports
        in 1..65535; mosaic dimensions even (2x2 CFA separation); mosaic_layout a permutation
        of the four Band names; input_bands a subset of mosaic_layout. Unknown keys are
        rejected so a typo'd field fails at startup rather than silently using the default.
    """
    for check in (_unknown_key_violation, _range_violation, _cross_field_violation):
        violation = check(data)
        if violation is not None:
            return violation
    return None


def _unknown_key_violation(data: dict[str, Any]) -> str | None:
    """Reject any top-level section or per-section key not backed by a config dataclass."""
    for section, value in data.items():
        cls = _SECTION_TO_CLASS.get(section)
        if cls is None:
            return f"unknown config section: {section!r}"
        if not isinstance(value, dict):
            return f"config section {section!r} must be a table"
        known = {f.name for f in dataclasses.fields(cls)}
        for key in value:
            if key not in known:
                return f"unknown config key {section}.{key}"
    return None


def _num(data: dict[str, Any], section: str, key: str, default: float) -> float:
    """Read a numeric config value from the merged dict, falling back to a default."""
    return float(data.get(section, {}).get(key, default))


def _range_violation(data: dict[str, Any]) -> str | None:
    """Per-section range checks. Return the first out-of-range field, or None."""
    ctrl = data.get("controller", {})
    if not (0.0 < _num(data, "controller", "ema_alpha", 0.4) <= 1.0):
        return "controller.ema_alpha must be in (0, 1]"
    if not (0.0 <= _num(data, "controller", "confidence_gate", 0.55) <= 1.0):
        return "controller.confidence_gate must be in [0, 1]"
    if not (0.0 <= _num(data, "controller", "blob_iou_match_threshold", 0.25) <= 1.0):
        return "controller.blob_iou_match_threshold must be in [0, 1]"
    if int(ctrl.get("min_deadband_px", 20)) < 0:
        return "controller.min_deadband_px must be >= 0"
    if int(ctrl.get("max_deadband_px", 250)) <= int(ctrl.get("min_deadband_px", 20)):
        return "controller.max_deadband_px must be > min_deadband_px"
    for count_key in (
        "max_deadband_strike_count",
        "acquire_persistence_frames",
        "release_persistence_frames",
        "runaway_strike_count",
    ):
        if int(ctrl.get(count_key, 1)) < 1:
            return f"controller.{count_key} must be >= 1"
    for rate_key in (
        "retarget_rate_limit_hz",
        "max_slew_rate_deg_per_s",
        "scan_slew_rate_deg_per_s",
        "max_slew_deg_s",
        "kalman_process_noise",
        "kalman_measurement_noise",
        "runaway_rate_tolerance_deg_per_s",
    ):
        if _num(data, "controller", rate_key, 1.0) <= 0.0:
            return f"controller.{rate_key} must be > 0"

    if _num(data, "inference", "latency_budget_ms", 500.0) <= 0.0:
        return "inference.latency_budget_ms must be > 0"
    for dim_key in ("input_height_px", "input_width_px"):
        if int(data.get("inference", {}).get(dim_key, 256)) <= 0:
            return f"inference.{dim_key} must be > 0"

    sensor = data.get("sensor", {})
    for dim_key in ("width_px", "height_px"):
        dim = int(sensor.get(dim_key, 1024))
        if dim <= 0:
            return f"sensor.{dim_key} must be > 0"
        if dim % 2:
            return f"sensor.{dim_key} must be even (2x2 mosaic separation)"
    if not (1 <= int(sensor.get("bit_depth", 12)) <= 16):
        return "sensor.bit_depth must be in 1..16"
    if _num(data, "sensor", "ifov_deg_per_px", 0.02) <= 0.0:
        return "sensor.ifov_deg_per_px must be > 0"
    if _num(data, "sensor", "default_exposure_us", 1000.0) <= 0.0:
        return "sensor.default_exposure_us must be > 0"
    if _num(data, "sensor", "default_gain_db", 0.0) < 0.0:
        return "sensor.default_gain_db must be >= 0"

    if _num(data, "fault", "watchdog_interval_s", 5.0) <= 0.0:
        return "fault.watchdog_interval_s must be > 0"
    if int(data.get("fault", {}).get("watchdog_max_miss_count", 3)) < 1:
        return "fault.watchdog_max_miss_count must be >= 1"
    for pos_key in ("inference_timeout_ms", "thermal_limit_c", "power_limit_w"):
        if _num(data, "fault", pos_key, 1.0) <= 0.0:
            return f"fault.{pos_key} must be > 0"

    if int(data.get("storage", {}).get("max_storage_bytes", 1)) <= 0:
        return "storage.max_storage_bytes must be > 0"
    if not str(data.get("storage", {}).get("checksum_algorithm", "sha256")):
        return "storage.checksum_algorithm must be non-empty"

    for rate_key in ("max_downlink_rate_bps", "max_uplink_rate_bps"):
        if int(data.get("comms", {}).get(rate_key, 1)) <= 0:
            return f"comms.{rate_key} must be > 0"
    for cap_key in ("max_daily_downlink_bytes", "max_daily_uplink_bytes"):
        if int(data.get("comms", {}).get(cap_key, 1)) <= 0:
            return f"comms.{cap_key} must be > 0"
    if int(data.get("comms", {}).get("downlink_max_bytes_per_pass", 1)) <= 0:
        return "comms.downlink_max_bytes_per_pass must be > 0"
    if not (0 <= int(data.get("comms", {}).get("ccsds_apid", 1)) <= 0x7FF):
        return "comms.ccsds_apid must fit in 11 bits (0..0x7FF)"

    if not (0.0 <= _num(data, "preprocessing", "saturation_fraction_threshold", 0.05) <= 1.0):
        return "preprocessing.saturation_fraction_threshold must be in [0, 1]"
    for pos_key in (
        "nir_red_ratio_threshold",
        "sunglint_nir_mean_threshold",
        "max_motion_smear_px",
    ):
        if _num(data, "preprocessing", pos_key, 1.0) <= 0.0:
            return f"preprocessing.{pos_key} must be > 0"

    link = data.get("link", {})
    for apid_key in ("tc_apid", "tm_apid"):
        if not (0 <= int(link.get(apid_key, 0)) <= 0x7FF):
            return f"link.{apid_key} must fit in 11 bits (0..0x7FF)"
    for port_key in ("command_tcp_port", "telemetry_udp_port"):
        if not (1 <= int(link.get(port_key, 0)) <= 65535):
            return f"link.{port_key} must be in 1..65535"
    if _num(data, "link", "socket_timeout_s", 1.0) <= 0.0:
        return "link.socket_timeout_s must be > 0"

    ingress = data.get("command_ingress", {})
    if bool(ingress.get("require_auth", True)) and not str(ingress.get("hmac_key_path", "")):
        return "command_ingress.hmac_key_path must be set when require_auth is true"
    if not tuple(ingress.get("accepted_sources", ("ground",))):
        return "command_ingress.accepted_sources must be non-empty"

    if _num(data, "command_router", "arm_window_s", 30.0) <= 0.0:
        return "command_router.arm_window_s must be > 0"
    return None


def _cross_field_violation(data: dict[str, Any]) -> str | None:
    """Cross-field consistency checks (gimbal envelope + mosaic/band agreement)."""
    az_min = _num(data, "gimbal", "az_min_deg", -90.0)
    az_max = _num(data, "gimbal", "az_max_deg", 90.0)
    el_min = _num(data, "gimbal", "el_min_deg", -45.0)
    el_max = _num(data, "gimbal", "el_max_deg", 45.0)
    if az_min >= az_max:
        return "gimbal.az_min_deg must be < az_max_deg"
    if el_min >= el_max:
        return "gimbal.el_min_deg must be < el_max_deg"
    if _num(data, "gimbal", "max_hw_slew_rate_deg_per_s", 10.0) <= 0.0:
        return "gimbal.max_hw_slew_rate_deg_per_s must be > 0"
    for pose in ("stow", "home"):
        pose_az = _num(data, "gimbal", f"{pose}_az_deg", 0.0)
        pose_el = _num(data, "gimbal", f"{pose}_el_deg", 0.0)
        if not (az_min <= pose_az <= az_max):
            return f"gimbal.{pose}_az_deg must be within [az_min_deg, az_max_deg]"
        if not (el_min <= pose_el <= el_max):
            return f"gimbal.{pose}_el_deg must be within [el_min_deg, el_max_deg]"

    band_names = {b.value for b in Band}
    mosaic = [str(v) for v in data.get("sensor", {}).get("mosaic_layout", sorted(band_names))]
    if sorted(mosaic) != sorted(band_names):
        return "sensor.mosaic_layout must name each Band (BLUE/GREEN/RED/NIR) exactly once"
    input_bands = [str(v) for v in data.get("inference", {}).get("input_bands", mosaic)]
    if not input_bands:
        return "inference.input_bands must be non-empty"
    mosaic_set = set(mosaic)
    for band in input_bands:
        if band not in mosaic_set:
            return f"inference.input_bands entry {band!r} is not present in sensor.mosaic_layout"
    return None


def _axis_mode(section: dict[str, Any], key: str, default: str) -> AxisMode:
    """Resolve and validate one environment axis to the 'sim'/'real' literal.

    Args:
        section: The parsed [environment] TOML dict (or {} when absent).
        key: The axis field name (e.g. "sensor").
        default: The dataclass default for the axis ("sim" or "real").

    Returns:
        The validated AxisMode literal ("sim" or "real").

    Raises:
        ValueError: If the configured value is neither "sim" nor "real". Explicit
            branches (no cast) keep the return statically typed as AxisMode under
            mypy --strict.
    """
    raw = str(section.get(key, default))
    if raw == "sim":
        return "sim"
    if raw == "real":
        return "real"
    raise ValueError(f"environment.{key} must be 'sim' or 'real', got {raw!r}")


def _build_pact_config(data: dict[str, Any]) -> PactConfig:
    """Map a parsed TOML dict to a PactConfig dataclass.

    Each TOML section maps to one sub-config dataclass.  get() with a default mirrors
    the dataclass field default, ensuring an empty section yields the same result as
    constructing the dataclass with no arguments. All sections are explicitly mapped so
    mismatches between TOML keys and dataclass fields are caught at load time.

    Args:
        data: The merged TOML dict (default + optional override).

    Returns:
        A fully populated PactConfig with all subsystem configs.
    """
    ctrl = data.get("controller", {})
    controller_config = ControllerConfig(
        confidence_gate=float(ctrl.get("confidence_gate", ControllerConfig.confidence_gate)),
        ema_alpha=float(ctrl.get("ema_alpha", ControllerConfig.ema_alpha)),
        min_deadband_px=int(ctrl.get("min_deadband_px", ControllerConfig.min_deadband_px)),
        max_deadband_px=int(ctrl.get("max_deadband_px", ControllerConfig.max_deadband_px)),
        max_deadband_strike_count=int(
            ctrl.get("max_deadband_strike_count", ControllerConfig.max_deadband_strike_count)
        ),
        retarget_rate_limit_hz=float(
            ctrl.get("retarget_rate_limit_hz", ControllerConfig.retarget_rate_limit_hz)
        ),
        max_slew_rate_deg_per_s=float(
            ctrl.get("max_slew_rate_deg_per_s", ControllerConfig.max_slew_rate_deg_per_s)
        ),
        acquire_persistence_frames=int(
            ctrl.get("acquire_persistence_frames", ControllerConfig.acquire_persistence_frames)
        ),
        release_persistence_frames=int(
            ctrl.get("release_persistence_frames", ControllerConfig.release_persistence_frames)
        ),
        scan_entry_idle_seconds=float(
            ctrl.get("scan_entry_idle_seconds", ControllerConfig.scan_entry_idle_seconds)
        ),
        scan_slew_rate_deg_per_s=float(
            ctrl.get("scan_slew_rate_deg_per_s", ControllerConfig.scan_slew_rate_deg_per_s)
        ),
        blob_iou_match_threshold=float(
            ctrl.get("blob_iou_match_threshold", ControllerConfig.blob_iou_match_threshold)
        ),
        min_blob_area_px=int(ctrl.get("min_blob_area_px", ControllerConfig.min_blob_area_px)),
        kalman_dt_s=float(ctrl.get("kalman_dt_s", ControllerConfig.kalman_dt_s)),
        kalman_process_noise=float(
            ctrl.get("kalman_process_noise", ControllerConfig.kalman_process_noise)
        ),
        kalman_measurement_noise=float(
            ctrl.get("kalman_measurement_noise", ControllerConfig.kalman_measurement_noise)
        ),
        lqr_Q_diag=tuple(
            float(v) for v in ctrl.get("lqr_Q_diag", list(ControllerConfig.lqr_Q_diag))
        ),
        lqr_R_diag=tuple(
            float(v) for v in ctrl.get("lqr_R_diag", list(ControllerConfig.lqr_R_diag))
        ),
        max_slew_deg_s=float(ctrl.get("max_slew_deg_s", ControllerConfig.max_slew_deg_s)),
        runaway_rate_tolerance_deg_per_s=float(
            ctrl.get(
                "runaway_rate_tolerance_deg_per_s",
                ControllerConfig.runaway_rate_tolerance_deg_per_s,
            )
        ),
        runaway_strike_count=int(
            ctrl.get("runaway_strike_count", ControllerConfig.runaway_strike_count)
        ),
    )

    inf = data.get("inference", {})
    inference_config = InferenceConfig(
        model_path=str(inf.get("model_path", InferenceConfig.model_path)),
        rollback_model_path=str(
            inf.get("rollback_model_path", InferenceConfig.rollback_model_path)
        ),
        input_bands=tuple(inf.get("input_bands", list(InferenceConfig.input_bands))),
        input_height_px=int(inf.get("input_height_px", InferenceConfig.input_height_px)),
        input_width_px=int(inf.get("input_width_px", InferenceConfig.input_width_px)),
        use_int8=bool(inf.get("use_int8", InferenceConfig.use_int8)),
        latency_budget_ms=float(inf.get("latency_budget_ms", InferenceConfig.latency_budget_ms)),
    )

    comms = data.get("comms", {})
    comms_config = CommsConfig(
        max_downlink_rate_bps=int(
            comms.get("max_downlink_rate_bps", CommsConfig.max_downlink_rate_bps)
        ),
        max_uplink_rate_bps=int(comms.get("max_uplink_rate_bps", CommsConfig.max_uplink_rate_bps)),
        max_daily_downlink_bytes=int(
            comms.get("max_daily_downlink_bytes", CommsConfig.max_daily_downlink_bytes)
        ),
        max_daily_uplink_bytes=int(
            comms.get("max_daily_uplink_bytes", CommsConfig.max_daily_uplink_bytes)
        ),
        comm_window_days=tuple(comms.get("comm_window_days", list(CommsConfig.comm_window_days))),
        ccsds_apid=int(comms.get("ccsds_apid", CommsConfig.ccsds_apid)),
        staged_model_path=str(comms.get("staged_model_path", CommsConfig.staged_model_path)),
        downlink_max_bytes_per_pass=int(
            comms.get("downlink_max_bytes_per_pass", CommsConfig.downlink_max_bytes_per_pass)
        ),
    )

    stor = data.get("storage", {})
    storage_config = StorageConfig(
        data_root=str(stor.get("data_root", StorageConfig.data_root)),
        max_storage_bytes=int(stor.get("max_storage_bytes", StorageConfig.max_storage_bytes)),
        checksum_algorithm=str(stor.get("checksum_algorithm", StorageConfig.checksum_algorithm)),
    )

    flt = data.get("fault", {})
    fault_config = FaultConfig(
        watchdog_interval_s=float(flt.get("watchdog_interval_s", FaultConfig.watchdog_interval_s)),
        watchdog_max_miss_count=int(
            flt.get("watchdog_max_miss_count", FaultConfig.watchdog_max_miss_count)
        ),
        inference_timeout_ms=float(
            flt.get("inference_timeout_ms", FaultConfig.inference_timeout_ms)
        ),
        thermal_limit_c=float(flt.get("thermal_limit_c", FaultConfig.thermal_limit_c)),
        power_limit_w=float(flt.get("power_limit_w", FaultConfig.power_limit_w)),
    )

    prep = data.get("preprocessing", {})
    preprocessing_config = PreprocessingConfig(
        saturation_fraction_threshold=float(
            prep.get(
                "saturation_fraction_threshold", PreprocessingConfig.saturation_fraction_threshold
            )
        ),
        nir_red_ratio_threshold=float(
            prep.get("nir_red_ratio_threshold", PreprocessingConfig.nir_red_ratio_threshold)
        ),
        sunglint_nir_mean_threshold=float(
            prep.get("sunglint_nir_mean_threshold", PreprocessingConfig.sunglint_nir_mean_threshold)
        ),
        max_motion_smear_px=float(
            prep.get("max_motion_smear_px", PreprocessingConfig.max_motion_smear_px)
        ),
    )

    sens = data.get("sensor", {})
    sensor_config = SensorConfig(
        width_px=int(sens.get("width_px", SensorConfig.width_px)),
        height_px=int(sens.get("height_px", SensorConfig.height_px)),
        bit_depth=int(sens.get("bit_depth", SensorConfig.bit_depth)),
        mosaic_layout=tuple(
            str(v) for v in sens.get("mosaic_layout", list(SensorConfig.mosaic_layout))
        ),
        ifov_deg_per_px=float(sens.get("ifov_deg_per_px", SensorConfig.ifov_deg_per_px)),
        default_exposure_us=float(
            sens.get("default_exposure_us", SensorConfig.default_exposure_us)
        ),
        default_gain_db=float(sens.get("default_gain_db", SensorConfig.default_gain_db)),
        calibration_dir=str(sens.get("calibration_dir", SensorConfig.calibration_dir)),
    )

    gimb = data.get("gimbal", {})
    gimbal_config = GimbalConfig(
        az_min_deg=float(gimb.get("az_min_deg", GimbalConfig.az_min_deg)),
        az_max_deg=float(gimb.get("az_max_deg", GimbalConfig.az_max_deg)),
        el_min_deg=float(gimb.get("el_min_deg", GimbalConfig.el_min_deg)),
        el_max_deg=float(gimb.get("el_max_deg", GimbalConfig.el_max_deg)),
        max_hw_slew_rate_deg_per_s=float(
            gimb.get("max_hw_slew_rate_deg_per_s", GimbalConfig.max_hw_slew_rate_deg_per_s)
        ),
        stow_az_deg=float(gimb.get("stow_az_deg", GimbalConfig.stow_az_deg)),
        stow_el_deg=float(gimb.get("stow_el_deg", GimbalConfig.stow_el_deg)),
        home_az_deg=float(gimb.get("home_az_deg", GimbalConfig.home_az_deg)),
        home_el_deg=float(gimb.get("home_el_deg", GimbalConfig.home_el_deg)),
        sim_time_constant_s=float(
            gimb.get("sim_time_constant_s", GimbalConfig.sim_time_constant_s)
        ),
        sim_encoder_noise_deg=float(
            gimb.get("sim_encoder_noise_deg", GimbalConfig.sim_encoder_noise_deg)
        ),
        sim_seed=int(gimb.get("sim_seed", GimbalConfig.sim_seed)),
        serial_port=str(gimb.get("serial_port", GimbalConfig.serial_port)),
        serial_baud=int(gimb.get("serial_baud", GimbalConfig.serial_baud)),
        counts_per_deg=float(gimb.get("counts_per_deg", GimbalConfig.counts_per_deg)),
    )

    link_sect = data.get("link", {})
    link_config = LinkConfig(
        command_tcp_host=str(link_sect.get("command_tcp_host", LinkConfig.command_tcp_host)),
        command_tcp_port=int(link_sect.get("command_tcp_port", LinkConfig.command_tcp_port)),
        telemetry_udp_host=str(link_sect.get("telemetry_udp_host", LinkConfig.telemetry_udp_host)),
        telemetry_udp_port=int(link_sect.get("telemetry_udp_port", LinkConfig.telemetry_udp_port)),
        socket_timeout_s=float(link_sect.get("socket_timeout_s", LinkConfig.socket_timeout_s)),
        tc_apid=int(link_sect.get("tc_apid", LinkConfig.tc_apid)),
        tm_apid=int(link_sect.get("tm_apid", LinkConfig.tm_apid)),
    )

    ingress_sect = data.get("command_ingress", {})
    command_ingress_config = CommandIngressConfig(
        hmac_key_path=str(ingress_sect.get("hmac_key_path", CommandIngressConfig.hmac_key_path)),
        require_auth=bool(ingress_sect.get("require_auth", CommandIngressConfig.require_auth)),
        accepted_sources=tuple(
            str(s)
            for s in ingress_sect.get("accepted_sources", CommandIngressConfig.accepted_sources)
        ),
    )

    router_sect = data.get("command_router", {})
    command_router_config = CommandRouterConfig(
        arm_window_s=float(router_sect.get("arm_window_s", CommandRouterConfig.arm_window_s)),
    )

    env = data.get("environment", {})
    environment_config = EnvironmentConfig(
        sensor=_axis_mode(env, "sensor", EnvironmentConfig.sensor),
        gimbal=_axis_mode(env, "gimbal", EnvironmentConfig.gimbal),
        compute=_axis_mode(env, "compute", EnvironmentConfig.compute),
        link=_axis_mode(env, "link", EnvironmentConfig.link),
        clock=_axis_mode(env, "clock", EnvironmentConfig.clock),
        host=str(env.get("host", EnvironmentConfig.host)),
    )

    return PactConfig(
        controller=controller_config,
        inference=inference_config,
        comms=comms_config,
        storage=storage_config,
        fault=fault_config,
        preprocessing=preprocessing_config,
        sensor=sensor_config,
        gimbal=gimbal_config,
        link=link_config,
        command_ingress=command_ingress_config,
        command_router=command_router_config,
        environment=environment_config,
    )
