"""Asserts config dataclass defaults match config/default.toml exactly.

This is the check CLAUDE.md flagged as missing: silent divergence between the
typed defaults and the shipped TOML would make behavior depend on which path set
a value. TOML arrays load as lists, so they are normalized to tuples for
comparison against the dataclass tuple defaults.
"""

import dataclasses
import tomllib
from pathlib import Path

from flight.libs.config import (
    CommandIngressConfig,
    CommandRouterConfig,
    CommsConfig,
    ControllerConfig,
    EnvironmentConfig,
    EphemerisConfig,
    FaultConfig,
    GimbalConfig,
    InferenceConfig,
    LinkConfig,
    PreprocessingConfig,
    SensorConfig,
    StorageConfig,
    ThermalConfig,
)


def _repo_root() -> Path:
    """Walk up to the directory that holds config/default.toml."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config" / "default.toml").exists():
            return parent
    msg = "could not locate repo root (config/default.toml) above the test file"
    raise FileNotFoundError(msg)


_REPO_ROOT = _repo_root()
_DEFAULT_TOML = _REPO_ROOT / "config" / "default.toml"

_SECTION_TO_DATACLASS = {
    "controller": ControllerConfig,
    "inference": InferenceConfig,
    "comms": CommsConfig,
    "storage": StorageConfig,
    "preprocessing": PreprocessingConfig,
    "fault": FaultConfig,
    "thermal": ThermalConfig,
    "sensor": SensorConfig,
    "gimbal": GimbalConfig,
    "link": LinkConfig,
    "command_ingress": CommandIngressConfig,
    "command_router": CommandRouterConfig,
    "ephemeris": EphemerisConfig,
    "environment": EnvironmentConfig,
}


def _normalize(value: object) -> object:
    """TOML arrays load as lists; normalize to tuple for comparison."""
    if isinstance(value, list):
        return tuple(value)
    return value


def _compare_defaults(
    path: str, dataclass_value: object, toml_value: object, mismatches: list[str]
) -> None:
    """Recurse into nested config dataclasses so nested TOML tables are checked."""
    if dataclasses.is_dataclass(dataclass_value) and not isinstance(dataclass_value, type):
        if not isinstance(toml_value, dict):
            mismatches.append(f"{path}: expected nested table, got {toml_value!r}")
            return
        for field in dataclasses.fields(type(dataclass_value)):
            key = f"{path}.{field.name}"
            if field.name not in toml_value:
                mismatches.append(f"{key}: missing from TOML")
                continue
            _compare_defaults(
                key, getattr(dataclass_value, field.name), toml_value[field.name], mismatches
            )
        return
    left = _normalize(dataclass_value)
    right = _normalize(toml_value)
    if left != right:
        mismatches.append(f"{path}: dataclass={left!r} toml={right!r}")


def test_config_defaults_match_default_toml() -> None:
    """Every config dataclass field default equals its config/default.toml value."""
    with _DEFAULT_TOML.open("rb") as fh:
        toml_data = tomllib.load(fh)

    mismatches: list[str] = []
    for section, dataclass_type in _SECTION_TO_DATACLASS.items():
        defaults = dataclass_type()
        toml_section = toml_data.get(section, {})
        _compare_defaults(section, defaults, toml_section, mismatches)

    assert not mismatches, "config default divergence:\n" + "\n".join(mismatches)
