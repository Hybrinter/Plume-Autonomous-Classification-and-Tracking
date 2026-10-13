"""Bounded prediction/evidence persistence boundary.

Evaluation hands per-row predictions to a ``CaptureSink`` implementation,
which owns how much is persisted. Sink implementations land with the capture
phase; only the typed protocol is declared here.

Contains:
  - CaptureSink: sink protocol with an explicit Result-returning close.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from flight.libs.types import Result


@runtime_checkable
class CaptureSink(Protocol):
    """Destination for captured evidence rows."""

    def close(self) -> Result[None, str]:
        """Flush and finish the sink.

        Returns:
            Result[None, str]: Ok on a clean finish, Err with the failure
            reason otherwise.
        """
        ...
