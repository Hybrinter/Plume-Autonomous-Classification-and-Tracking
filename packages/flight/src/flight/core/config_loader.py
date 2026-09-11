"""Configuration loader.

Merges config/default.toml with an optional override file (e.g. config/flight.toml).
Override values take precedence over defaults. The merged dict is validated into
the PactConfig schema.

No subsystem reads TOML directly -- each receives a typed config dataclass argument.

Satisfies: REQ-OPER-HIGH-002 (validated config at startup), REQ-CONFIG-INTEGRITY-001.
"""

from __future__ import annotations

# stdlib
import tomllib
from typing import Any

# third-party
from pydantic import TypeAdapter, ValidationError

# internal
from flight.libs.config import PactConfig
from flight.libs.types import Err, Ok, Result

_PACT_CONFIG_ADAPTER = TypeAdapter(PactConfig)


def load_config(
    config_path: str,
    override_path: str | None = None,
) -> Result[PactConfig, str]:
    """Load a TOML config file and populate PactConfig.

    Merge logic:
      1. Load config/default.toml (always required).
      2. If override_path is provided, load it and deep-merge on top of defaults.
         Keys present in the override replace defaults; absent keys retain defaults.
      3. Validate the merged dict into the PactConfig schema.

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

    try:
        config = _PACT_CONFIG_ADAPTER.validate_python(data)
    except ValidationError as exc:
        return Err(_format_validation_error(exc))

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


def _format_validation_error(exc: ValidationError) -> str:
    """Flatten a ValidationError into a single human-readable string.

    Each issue is ``<dotted.loc>: <message>`` so callers and tests can match on
    the field name (``tc_apid``, ``ema_alpha``, ``wdith_px``) without parsing
    Pydantic's structured error list.
    """
    parts: list[str] = []
    for error in exc.errors():
        loc = ".".join(str(item) for item in error["loc"])
        msg = str(error["msg"])
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(parts) if parts else str(exc)
