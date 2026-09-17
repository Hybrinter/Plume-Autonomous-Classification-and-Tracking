"""Load EnvironmentConfig from TOML. Sim-only; not PactConfig."""

from __future__ import annotations

# stdlib
import tomllib
from typing import Any

# internal
from flight.libs.types import Err, Result

from sim.environment.config import EnvironmentConfig, parse_environment_config


def load_environment_config(path: str) -> Result[EnvironmentConfig, str]:
    """Read a TOML file into EnvironmentConfig.

    Args:
        path: Filesystem path. Nested table [environment] is optional; a flat
            file of axis names is also valid.

    Returns:
        Ok(EnvironmentConfig) or Err with a parse/validation message.
    """
    try:
        with open(path, "rb") as fh:
            data: dict[str, Any] = tomllib.load(fh)
    except FileNotFoundError:
        return Err(f"environment config not found: {path}")
    except tomllib.TOMLDecodeError as exc:
        return Err(f"TOML parse error in {path}: {exc}")
    if "environment" in data and isinstance(data["environment"], dict):
        payload = data["environment"]
    else:
        payload = data
    return parse_environment_config(payload)
