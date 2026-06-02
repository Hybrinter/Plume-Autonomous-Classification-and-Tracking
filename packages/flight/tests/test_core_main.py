"""Smoke tests for the flight entry module (importable + exposes its API).

build_flight_system() and main() construct real drivers (RealSensor lazy-imports
PySpin; OnnxDetector lazy-imports onnxruntime), so they are NOT executed here -- CI has
neither SDK. These tests assert the module imports and exposes the expected callables;
the wiring itself is exercised by the SIL integration (Phase 10) via build_apps.
"""

from flight.core import build_flight_system, main
from flight.core.scheduler import Scheduler


def test_core_exposes_entry_callables() -> None:
    """flight.core re-exports the flight entry points."""
    assert callable(build_flight_system)
    assert callable(main)


def test_scheduler_is_importable_from_core() -> None:
    """The scheduler type used by main is importable (sanity for the entry wiring)."""
    assert Scheduler is not None
