"""Tests for the libs.telemetry logging helper."""

import structlog
from flight.libs.telemetry import configure_logging, get_logger


def test_configure_logging_dev_mode_runs() -> None:
    """configure_logging(dev) sets up structlog without error."""
    configure_logging(flight_mode=False)
    assert structlog.is_configured()


def test_get_logger_binds_subsystem() -> None:
    """get_logger binds the subsystem field into the event context."""
    configure_logging(flight_mode=True)
    log = get_logger("payload")
    bound = log.bind()
    context = structlog.get_context(bound)
    assert context.get("subsystem") == "payload"
