"""Verifies the [drivers] config axis defaults and loader mapping."""

import dataclasses
import tomllib
from pathlib import Path

from flight.core.config_loader import load_config
from flight.libs.config import DriverConfig, PactConfig
from flight.libs.types import Ok


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


def test_driver_defaults_all_real() -> None:
    """A bare DriverConfig has every axis 'real' and the jetson host."""
    drivers = DriverConfig()
    assert (drivers.sensor, drivers.gimbal, drivers.compute, drivers.link, drivers.clock) == (
        "real",
        "real",
        "real",
        "real",
        "real",
    )
    assert drivers.host == "jetson_aarch64"


def test_pactconfig_has_drivers_field_last() -> None:
    """PactConfig exposes a drivers field, declared last."""
    field_names = [f.name for f in dataclasses.fields(PactConfig)]
    assert field_names[-1] == "drivers"
    assert PactConfig().drivers == DriverConfig()


def test_driver_defaults_match_default_toml() -> None:
    """The [drivers] section of default.toml equals the dataclass defaults."""
    with _DEFAULT_TOML.open("rb") as fh:
        toml_data = tomllib.load(fh)
    section = toml_data["drivers"]
    defaults = DriverConfig()
    for field in dataclasses.fields(DriverConfig):
        assert section[field.name] == getattr(defaults, field.name), field.name


def test_loader_maps_drivers_section() -> None:
    """load_config populates drivers from default.toml (all axes real)."""
    result = load_config(str(_DEFAULT_TOML))
    assert isinstance(result, Ok)
    assert result.value.drivers == DriverConfig()
