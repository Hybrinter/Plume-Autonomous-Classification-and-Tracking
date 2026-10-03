"""Deterministic profile fixtures for nominal GSE cadence regressions."""

from pathlib import Path

import pytest


@pytest.fixture
def deterministic_profiles(tmp_path: Path) -> dict[str, Path]:
    """Copy test profiles with exact feedback, leaving driver selection and defaults intact."""
    profiles: dict[str, Path] = {}
    for name in ("sil", "sil-link-real"):
        destination = tmp_path / f"{name}.toml"
        destination.write_text(
            Path(f"profiles/{name}.toml").read_text(encoding="utf-8")
            + "\n[gimbal.simulation]\nencoder_noise_deg = 0.0\n",
            encoding="utf-8",
        )
        profiles[name] = destination
    return profiles
