"""Thread-based scheduler for subsystem apps.

The message bus is in-process (queue.Queue transport), so subsystem apps run as
daemon threads that share it. The scheduler owns one stop Event, launches each app's
run(stop_event) in a named thread, and joins them on stop. Each app owns its own loop
and internal state; the scheduler only starts and stops them.

Contains:
  - RunnableApp: the Protocol every schedulable app satisfies (run(stop_event)).
  - Scheduler: start() launches threads, stop() signals + joins, is_running() reports liveness.
"""

from __future__ import annotations

# stdlib
import threading
from typing import Protocol, runtime_checkable


@runtime_checkable
class RunnableApp(Protocol):
    """A subsystem app the scheduler can run: a single blocking run(stop_event) loop."""

    def run(self, stop_event: threading.Event) -> None:
        """Run until stop_event is set."""
        ...


class Scheduler:
    """Runs each registered app's run(stop_event) in its own daemon thread."""

    def __init__(self, apps: list[tuple[str, RunnableApp]]) -> None:
        """Register the apps to schedule.

        Args:
            apps: (name, app) pairs; name labels the thread for diagnostics.
        """
        self._apps = apps
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        """Launch each app's run(stop_event) in a named daemon thread."""
        for name, app in self._apps:
            thread = threading.Thread(target=app.run, args=(self._stop,), name=name, daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self, timeout: float = 5.0) -> None:
        """Signal every app to stop and join each thread up to timeout seconds.

        The joined (now-dead) threads are retained so is_running() reports liveness
        honestly after stop rather than reading an emptied list.
        """
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=timeout)

    def is_running(self) -> bool:
        """Return True if any scheduled thread is still alive."""
        return any(thread.is_alive() for thread in self._threads)
