"""Verifies the [environment] config axis defaults and loader mapping."""

import dataclasses
import tomllib
from pathlib import Path

from flight.core.config_loader import load_config
from flight.libs.config import EnvironmentConfig, PactConfig
from flight.libs.types import Ok

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_TOML = _REPO_ROOT / "config" / "default.toml"


def test_environment_defaults_all_real() -> None:
    """A bare EnvironmentConfig has every axis 'real' and the jetson host."""
    env = EnvironmentConfig()
    assert (env.sensor, env.gimbal, env.compute, env.link, env.clock) == (
        "real",
        "real",
        "real",
        "real",
        "real",
    )
    assert env.host == "jetson_aarch64"


def test_pactconfig_has_environment_field_last() -> None:
    """PactConfig exposes an environment field, declared last."""
    field_names = [f.name for f in dataclasses.fields(PactConfig)]
    assert field_names[-1] == "environment"
    assert PactConfig().environment == EnvironmentConfig()


def test_environment_defaults_match_default_toml() -> None:
    """The [environment] section of default.toml equals the dataclass defaults."""
    with _DEFAULT_TOML.open("rb") as fh:
        toml_data = tomllib.load(fh)
    section = toml_data["environment"]
    defaults = EnvironmentConfig()
    for field in dataclasses.fields(EnvironmentConfig):
        assert section[field.name] == getattr(defaults, field.name), field.name


def test_loader_maps_environment_section() -> None:
    """load_config populates environment from default.toml (all axes real)."""
    result = load_config(str(_DEFAULT_TOML))
    assert isinstance(result, Ok)
    assert result.value.environment == EnvironmentConfig()
