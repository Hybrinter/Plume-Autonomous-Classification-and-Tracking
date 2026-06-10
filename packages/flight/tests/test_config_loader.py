"""Tests for the migrated config loader."""

from pathlib import Path

from flight.core import load_config
from flight.libs.config import PactConfig
from flight.libs.types import Err, Ok

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_TOML = str(_REPO_ROOT / "config" / "default.toml")
_FLIGHT_TOML = str(_REPO_ROOT / "config" / "flight.toml")


def test_loads_default_config() -> None:
    """load_config returns Ok(PactConfig) for the default TOML."""
    result = load_config(_DEFAULT_TOML)
    assert isinstance(result, Ok)
    assert isinstance(result.value, PactConfig)


def test_flight_override_merges() -> None:
    """The flight.toml override (use_int8 = true) merges over defaults."""
    result = load_config(_DEFAULT_TOML, _FLIGHT_TOML)
    assert isinstance(result, Ok)
    assert result.value.inference.use_int8 is True


def test_missing_file_returns_err() -> None:
    """A missing config path returns Err, not an exception."""
    result = load_config(str(_REPO_ROOT / "config" / "does_not_exist.toml"))
    assert isinstance(result, Err)


def test_sensor_section_loads() -> None:
    """[sensor] TOML section maps into SensorConfig."""
    result = load_config(_DEFAULT_TOML)
    assert isinstance(result, Ok)
    sensor = result.value.sensor
    assert sensor.width_px == 512
    assert sensor.height_px == 512
    assert sensor.bit_depth == 12
    assert sensor.mosaic_layout == ("BLUE", "GREEN", "RED", "NIR")
    assert sensor.ifov_deg_per_px == 0.04
    assert sensor.calibration_dir == ""
