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
    assert sensor.width_px == 1024
    assert sensor.height_px == 1024
    assert sensor.bit_depth == 12
    assert sensor.mosaic_layout == ("BLUE", "GREEN", "RED", "NIR")
    assert sensor.ifov_deg_per_px == 0.02
    assert sensor.calibration_dir == ""


def test_gimbal_section_loads() -> None:
    """[gimbal] TOML section maps into GimbalConfig."""
    result = load_config(_DEFAULT_TOML)
    assert isinstance(result, Ok)
    g = result.value.gimbal
    assert g.az_min_deg == -90.0
    assert g.az_max_deg == 90.0
    assert g.el_min_deg == -45.0
    assert g.el_max_deg == 45.0
    assert g.max_hw_slew_rate_deg_per_s == 10.0
    assert g.stow_el_deg == -45.0
    assert g.serial_port == ""


def test_controller_runaway_fields_load() -> None:
    """Encoder-runaway tuning fields map into ControllerConfig."""
    result = load_config(_DEFAULT_TOML)
    assert isinstance(result, Ok)
    c = result.value.controller
    assert c.runaway_rate_tolerance_deg_per_s == 1.0
    assert c.runaway_strike_count == 3


def test_link_section_loads() -> None:
    """[link] TOML section maps into LinkConfig."""
    result = load_config(_DEFAULT_TOML)
    assert isinstance(result, Ok)
    lnk = result.value.link
    assert lnk.command_tcp_host == "127.0.0.1"
    assert lnk.command_tcp_port == 50501
    assert lnk.telemetry_udp_host == "127.0.0.1"
    assert lnk.telemetry_udp_port == 50502
    assert lnk.socket_timeout_s == 1.0
    assert lnk.tc_apid == 0x001
    assert lnk.tm_apid == 0x002


def test_command_ingress_section_loads() -> None:
    """[command_ingress] TOML section maps into CommandIngressConfig."""
    result = load_config(_DEFAULT_TOML)
    assert isinstance(result, Ok)
    ing = result.value.command_ingress
    assert ing.hmac_key_path == "data/keys/uplink_hmac.key"
    assert ing.require_auth is True
    assert ing.accepted_sources == ("ground", "station_ops")


def test_out_of_range_apid_returns_err(tmp_path: Path) -> None:
    """An APID exceeding 11 bits causes load_config to return Err."""
    bad_toml = tmp_path / "bad.toml"
    bad_toml.write_text(
        "[link]\ntc_apid = 4096\ntm_apid = 2\ncommand_tcp_port = 50501\n"
        "telemetry_udp_port = 50502\nsocket_timeout_s = 1.0\n"
        'command_tcp_host = "127.0.0.1"\ntelemetry_udp_host = "127.0.0.1"\n',
        encoding="utf-8",
    )
    result = load_config(_DEFAULT_TOML, str(bad_toml))
    assert isinstance(result, Err)
    assert "tc_apid" in result.error


def test_out_of_range_port_returns_err(tmp_path: Path) -> None:
    """A TCP port of 0 causes load_config to return Err."""
    bad_toml = tmp_path / "bad_port.toml"
    bad_toml.write_text(
        "[link]\ncommand_tcp_port = 0\n",
        encoding="utf-8",
    )
    result = load_config(_DEFAULT_TOML, str(bad_toml))
    assert isinstance(result, Err)
    assert "command_tcp_port" in result.error
