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
    CommsConfig,
    ControllerConfig,
    FaultConfig,
    GimbalConfig,
    InferenceConfig,
    LinkConfig,
    PreprocessingConfig,
    SensorConfig,
    StorageConfig,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_TOML = _REPO_ROOT / "config" / "default.toml"

_SECTION_TO_DATACLASS = {
    "controller": ControllerConfig,
    "inference": InferenceConfig,
    "comms": CommsConfig,
    "storage": StorageConfig,
    "preprocessing": PreprocessingConfig,
    "fault": FaultConfig,
    "sensor": SensorConfig,
    "gimbal": GimbalConfig,
    "link": LinkConfig,
    "command_ingress": CommandIngressConfig,
}


def _normalize(value: object) -> object:
    """TOML arrays load as lists; normalize to tuple for comparison."""
    if isinstance(value, list):
        return tuple(value)
    return value


def test_config_defaults_match_default_toml() -> None:
    """Every config dataclass field default equals its config/default.toml value."""
    with _DEFAULT_TOML.open("rb") as fh:
        toml_data = tomllib.load(fh)

    mismatches: list[str] = []
    for section, dataclass_type in _SECTION_TO_DATACLASS.items():
        defaults = dataclass_type()
        toml_section = toml_data.get(section, {})
        for field in dataclasses.fields(dataclass_type):
            if field.name not in toml_section:
                mismatches.append(f"{section}.{field.name}: missing from TOML")
                continue
            dataclass_value = _normalize(getattr(defaults, field.name))
            toml_value = _normalize(toml_section[field.name])
            if dataclass_value != toml_value:
                mismatches.append(
                    f"{section}.{field.name}: dataclass={dataclass_value!r} toml={toml_value!r}"
                )

    assert not mismatches, "config default divergence:\n" + "\n".join(mismatches)
