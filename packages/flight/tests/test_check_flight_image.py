"""Unit tests for scripts/check_flight_image.py pure helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _repo_root() -> Path:
    """Walk up to the directory holding scripts/check_flight_image.py."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "scripts" / "check_flight_image.py").exists():
            return parent
    msg = "could not locate scripts/check_flight_image.py above the test file"
    raise FileNotFoundError(msg)


def _load_check_flight_image() -> ModuleType:
    """Import check_flight_image from scripts/ without installing it as a package."""
    script = _repo_root() / "scripts" / "check_flight_image.py"
    module_name = "check_flight_image"
    spec = importlib.util.spec_from_file_location(module_name, script)
    if spec is None or spec.loader is None:
        msg = f"could not load module spec from {script}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_required_extras_declared_in_real_pyprojects() -> None:
    """Flight and tools pyprojects declare every required role extra."""
    mod = _load_check_flight_image()
    assert mod.check_extras_declared(_repo_root()) == []


def test_missing_extras_reports_difference() -> None:
    """missing_extras returns required names absent from the declared set."""
    mod = _load_check_flight_image()
    declared = frozenset({"inference", "camera"})
    required = frozenset({"inference", "camera", "gimbal"})
    assert mod.missing_extras(declared, required) == frozenset({"gimbal"})


def test_denied_constants_cover_expected_names() -> None:
    """Denied distribution and module sets match the lean-image policy."""
    mod = _load_check_flight_image()
    assert mod.DENIED_DISTRIBUTIONS == frozenset({"pact-tools", "pact-sim", "pact-gse"})
    assert mod.DENIED_MODULES == frozenset(
        {"torch", "torchvision", "tensorflow", "jax", "jaxlib", "flax", "keras", "mlx"}
    )
    assert mod.FLIGHT_EXTRAS == frozenset({"inference", "camera", "gimbal"})
    assert mod.TOOLS_EXTRAS == frozenset({"export"})


def test_load_optional_dependency_keys_reads_pyproject(tmp_path: Path) -> None:
    """load_optional_dependency_keys parses [project.optional-dependencies] from TOML."""
    mod = _load_check_flight_image()
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "[project.optional-dependencies]\ninference = []\ncamera = []\n",
        encoding="utf-8",
    )
    assert mod.load_optional_dependency_keys(pyproject) == frozenset({"inference", "camera"})
