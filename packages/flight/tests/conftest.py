"""Shared fixtures for the flight test suite."""

from pathlib import Path

import pytest
from flight.core import load_config
from flight.libs.config import PactConfig
from flight.libs.types import GimbalState, Ok
from flight.payload.gimbal import ArbiterState


def _repo_root() -> Path:
    """Walk up to the directory that holds config/default.toml."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config" / "default.toml").exists():
            return parent
    msg = "could not locate repo root (config/default.toml) above the test file"
    raise FileNotFoundError(msg)


_REPO_ROOT = _repo_root()


@pytest.fixture
def default_config() -> PactConfig:
    """PactConfig loaded from config/default.toml (frozen; use replace() to modify)."""
    result = load_config(str(_REPO_ROOT / "config" / "default.toml"))
    assert isinstance(result, Ok)
    return result.value


@pytest.fixture
def arbiter_idle_state() -> ArbiterState:
    """An ArbiterState in GimbalState.IDLE with no tracked blobs."""
    return ArbiterState(
        gimbal_state=GimbalState.IDLE,
        tracked_blobs=(),
        idle_duration_s=0.0,
        last_command_time=0.0,
        current_target_id=None,
    )
