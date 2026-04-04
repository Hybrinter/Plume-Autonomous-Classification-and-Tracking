"""Configuration loader.

Merges config/default.toml with an optional override file (e.g. config/flight.toml).
Override values take precedence over defaults.  Result is mapped field-by-field into
the PactConfig hierarchy of frozen dataclasses.

No subsystem reads TOML directly -- each receives a typed config dataclass argument.

Satisfies: REQ-OPER-HIGH-002 (type-safe, validated config at startup).
"""

from __future__ import annotations

# stdlib
import tomllib
from typing import Any

# internal
from pact.types.config import (
    CommsConfig,
    ControllerConfig,
    FaultConfig,
    InferenceConfig,
    PactConfig,
    StorageConfig,
)
from pact.types.enums import Ok, Err, Result  # type: ignore[attr-defined]


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

    # TODO: implement TOML field mapping
    # TODO: implement range validation in _validate() for all numeric fields.
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
    """Validate top-level config values.  Return an error string or None if valid.

    # TODO: implement range validation for all numeric fields:
    #   - controller.confidence_gate must be in (0.0, 1.0)
    #   - controller.ema_alpha must be in (0.0, 1.0]
    #   - inference.latency_budget_ms must be > 0
    #   - fault.watchdog_interval_s must be > 0
    #   - fault.watchdog_max_miss_count must be >= 1
    #   - comms.ccsds_apid must fit in 11 bits (0x000 to 0x7FF)
    """
    return None  # placeholder -- no validation yet


def _build_pact_config(data: dict[str, Any]) -> PactConfig:
    """Map a parsed TOML dict to a PactConfig dataclass.

    Each TOML section maps to one sub-config dataclass.  get() with a default mirrors
    the dataclass field default, ensuring an empty section yields the same result as
    constructing the dataclass with no arguments.

    # TODO: implement TOML field mapping -- replace each .get() default with an
    #        explicit reference so mismatches between TOML keys and dataclass fields
    #        are caught at load time rather than silently using the Python default.
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
        latency_budget_ms=float(
            inf.get("latency_budget_ms", InferenceConfig.latency_budget_ms)
        ),
    )

    comms = data.get("comms", {})
    comms_config = CommsConfig(
        max_downlink_rate_bps=int(
            comms.get("max_downlink_rate_bps", CommsConfig.max_downlink_rate_bps)
        ),
        max_uplink_rate_bps=int(
            comms.get("max_uplink_rate_bps", CommsConfig.max_uplink_rate_bps)
        ),
        max_daily_downlink_bytes=int(
            comms.get("max_daily_downlink_bytes", CommsConfig.max_daily_downlink_bytes)
        ),
        max_daily_uplink_bytes=int(
            comms.get("max_daily_uplink_bytes", CommsConfig.max_daily_uplink_bytes)
        ),
        comm_window_days=tuple(
            comms.get("comm_window_days", list(CommsConfig.comm_window_days))
        ),
        ccsds_apid=int(comms.get("ccsds_apid", CommsConfig.ccsds_apid)),
    )

    stor = data.get("storage", {})
    storage_config = StorageConfig(
        data_root=str(stor.get("data_root", StorageConfig.data_root)),
        max_storage_bytes=int(
            stor.get("max_storage_bytes", StorageConfig.max_storage_bytes)
        ),
        checksum_algorithm=str(
            stor.get("checksum_algorithm", StorageConfig.checksum_algorithm)
        ),
    )

    flt = data.get("fault", {})
    fault_config = FaultConfig(
        watchdog_interval_s=float(
            flt.get("watchdog_interval_s", FaultConfig.watchdog_interval_s)
        ),
        watchdog_max_miss_count=int(
            flt.get("watchdog_max_miss_count", FaultConfig.watchdog_max_miss_count)
        ),
        inference_timeout_ms=float(
            flt.get("inference_timeout_ms", FaultConfig.inference_timeout_ms)
        ),
        thermal_limit_c=float(flt.get("thermal_limit_c", FaultConfig.thermal_limit_c)),
        power_limit_w=float(flt.get("power_limit_w", FaultConfig.power_limit_w)),
    )

    return PactConfig(
        controller=controller_config,
        inference=inference_config,
        comms=comms_config,
        storage=storage_config,
        fault=fault_config,
    )
