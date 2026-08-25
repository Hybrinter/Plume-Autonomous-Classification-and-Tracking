"""Tests for the thread-based subsystem scheduler."""

import threading
import time

import pytest
from flight.core.scheduler import RunnableApp, Scheduler, next_supervision_action
from flight.libs.bus import MessageBus
from flight.libs.messages import FaultEventMsg
from flight.libs.types import FaultCode


class _BlockingApp:
    """RunnableApp that signals it started, then blocks until stopped."""

    def __init__(self, started: threading.Event) -> None:
        self._started = started

    def run(self, stop_event: threading.Event) -> None:
        """Signal startup, then wait until the scheduler sets stop_event."""
        self._started.set()
        stop_event.wait()


class _CrashingApp:
    """RunnableApp whose run() raises immediately (an unhandled-exception death)."""

    def run(self, stop_event: threading.Event) -> None:
        """Raise immediately to simulate an app thread dying."""
        raise RuntimeError("boom")


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


def test_supervision_action_decision() -> None:
    """The pure restart-vs-give-up decision covers alive/stopping/restart/give-up."""
    assert (
        next_supervision_action(alive=True, stopping=False, restart_count=0, restart_limit=3)
        == "none"
    )
    assert (
        next_supervision_action(alive=False, stopping=True, restart_count=0, restart_limit=3)
        == "none"
    )
    assert (
        next_supervision_action(alive=False, stopping=False, restart_count=0, restart_limit=3)
        == "restart"
    )
    assert (
        next_supervision_action(alive=False, stopping=False, restart_count=3, restart_limit=3)
        == "give_up"
    )


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_crashed_app_restarted_then_gives_up_to_safe() -> None:
    """A repeatedly crashing app is restarted up to the limit, then PROCESS_DIED is published."""
    bus = MessageBus()
    faults = bus.subscribe(FaultEventMsg)
    scheduler = Scheduler([("crasher", _CrashingApp())], bus=bus, restart_limit=2)
    scheduler.start()
    try:
        for _ in range(200):
            scheduler.check()
            if scheduler.gave_up_on("crasher"):
                break
            time.sleep(0.02)
        assert scheduler.gave_up_on("crasher")
        assert scheduler.restart_count("crasher") == 2
        process_died = [f for f in _drain_faults(faults) if f.fault_code is FaultCode.PROCESS_DIED]
        assert process_died and process_died[0].subsystem == "crasher"
    finally:
        scheduler.stop(timeout=2.0)


def _drain_faults(sub: object) -> list[FaultEventMsg]:
    """Drain a fault subscription into a list."""
    out: list[FaultEventMsg] = []
    while not sub.empty():  # type: ignore[attr-defined]
        out.append(sub.get_nowait())  # type: ignore[attr-defined]
    return out
