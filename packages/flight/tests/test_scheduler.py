"""Tests for the thread-based subsystem scheduler."""

import threading

from flight.core.scheduler import RunnableApp, Scheduler


class _BlockingApp:
    """RunnableApp that signals it started, then blocks until stopped."""

    def __init__(self, started: threading.Event) -> None:
        self._started = started

    def run(self, stop_event: threading.Event) -> None:
        """Signal startup, then wait until the scheduler sets stop_event."""
        self._started.set()
        stop_event.wait()


def test_blocking_app_satisfies_runnable_protocol() -> None:
    """_BlockingApp conforms to RunnableApp at runtime."""
    app: RunnableApp = _BlockingApp(threading.Event())
    assert isinstance(app, RunnableApp)


def test_scheduler_starts_apps_in_threads() -> None:
    """start() launches each app's run() in a thread that actually executes."""
    started = threading.Event()
    scheduler = Scheduler([("worker", _BlockingApp(started))])
    scheduler.start()
    try:
        assert started.wait(timeout=2.0)  # the worker thread ran
        assert scheduler.is_running()
    finally:
        scheduler.stop(timeout=2.0)


def test_scheduler_stop_joins_threads() -> None:
    """stop() sets the shared stop event and joins all threads."""
    started = threading.Event()
    scheduler = Scheduler([("worker", _BlockingApp(started))])
    scheduler.start()
    assert started.wait(timeout=2.0)
    scheduler.stop(timeout=2.0)
    assert not scheduler.is_running()
