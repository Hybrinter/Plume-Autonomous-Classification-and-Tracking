"""Structured logging configuration for flight software.

Implements the project logging convention: a JSON renderer for flight (one object
per line, suitable for downlink/parsing) and a human-readable console renderer for
development, with every entry bound to a `subsystem` field. The `event` field is
the first positional argument to each log call by structlog convention.
"""

from typing import cast

import structlog
from structlog.typing import FilteringBoundLogger, Processor


def configure_logging(flight_mode: bool) -> None:
    """Configure structlog process-wide. Call once at process startup.

    Args:
        flight_mode: When True, render each entry as a JSON object (flight/downlink);
            when False, use the colorized console renderer (development).

    Notes:
        Reconfigures global structlog state; intended to be called exactly once
        before any logger is obtained.
    """
    renderer: Processor = (
        structlog.processors.JSONRenderer() if flight_mode else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(subsystem: str) -> FilteringBoundLogger:
    """Return a structlog logger bound to the given subsystem.

    Args:
        subsystem: Subsystem name recorded in every log entry (e.g. "payload").

    Returns:
        A bound structlog logger carrying the `subsystem` field.
    """
    return cast(FilteringBoundLogger, structlog.get_logger().bind(subsystem=subsystem))
