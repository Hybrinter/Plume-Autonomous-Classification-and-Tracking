"""Verifies the deployment profiles override the [environment] axes correctly."""

from pathlib import Path

import pytest
from flight.core.config_loader import load_config
from flight.libs.types import Ok

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT = str(_REPO_ROOT / "config" / "default.toml")


def _profile(name: str) -> str:
    """Absolute path to a repo-root profile override."""
    return str(_REPO_ROOT / "profiles" / name)


def test_sil_profile_all_axes_sim() -> None:
    """profiles/sil.toml forces every environment axis to 'sim'."""
    result = load_config(_DEFAULT, _profile("sil.toml"))
    assert isinstance(result, Ok)
    env = result.value.environment
    assert (env.sensor, env.gimbal, env.compute, env.link, env.clock) == (
        "sim",
        "sim",
        "sim",
        "sim",
        "sim",
    )
    assert env.host == "x86_64"


def test_sil_link_real_profile_only_link_real() -> None:
    """profiles/sil-link-real.toml sets link='real', leaving the others 'sim'."""
    result = load_config(_DEFAULT, _profile("sil-link-real.toml"))
    assert isinstance(result, Ok)
    env = result.value.environment
    assert env.link == "real"
    assert (env.sensor, env.gimbal, env.compute, env.clock) == (
        "sim",
        "sim",
        "sim",
        "sim",
    )


@pytest.mark.parametrize(
    "name, sensor, link, clock, host",
    [
        ("pil.toml", "sim", "real", "real", "jetson_aarch64"),
        ("hil.toml", "real", "real", "real", "jetson_aarch64"),
    ],
)
def test_defined_not_run_profiles_load(
    name: str, sensor: str, link: str, clock: str, host: str
) -> None:
    """The DEFINED-NOT-RUN profiles still load and override their axes."""
    result = load_config(_DEFAULT, _profile(name))
    assert isinstance(result, Ok)
    env = result.value.environment
    assert env.sensor == sensor
    assert env.link == link
    assert env.clock == clock
    assert env.host == host
