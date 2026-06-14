"""Thread-based scheduler for subsystem apps, with crash supervision.

The message bus is in-process (queue.Queue transport), so subsystem apps run as daemon threads
that share it. The scheduler owns one stop Event, launches each app's run(stop_event) in a named
thread, and joins them on stop. Each app owns its own loop and internal state.

Supervision (spec Section 7): when a thread dies unexpectedly (an unhandled exception, not a
normal stop), the scheduler restarts it up to a configured restart limit; once a thread exhausts
its restarts the scheduler publishes a PROCESS_DIED FaultEventMsg (which FDIR routes to SAFE) and
stops restarting it. Whole-process death is the external supervisor's job (out of scope here).

Contains:
  - RunnableApp: the Protocol every schedulable app satisfies (run(stop_event)).
  - next_supervision_action: the pure restart-vs-give-up decision.
  - Scheduler: start(); stop(); is_running(); check() (one supervision pass); supervise() (loop).

Satisfies: REQ-OPER-HIGH-002.
"""

from __future__ import annotations

# stdlib
import threading
from typing import Literal, Protocol, runtime_checkable

# internal
from flight.libs.bus import MessageBus
from flight.libs.messages import FaultEventMsg, utc_now_iso
from flight.libs.types import FaultCode, MessageType

SupervisionAction = Literal["none", "restart", "give_up"]


@runtime_checkable
class RunnableApp(Protocol):
    """A subsystem app the scheduler can run: a single blocking run(stop_event) loop."""

    def run(self, stop_event: threading.Event) -> None:
        """Run until stop_event is set."""
        ...


def next_supervision_action(
    alive: bool, stopping: bool, restart_count: int, restart_limit: int
) -> SupervisionAction:
    """Decide how to supervise one app thread (pure).

    Args:
        alive: Whether the app's thread is currently alive.
        stopping: Whether the scheduler is shutting down (a dead thread is then expected).
        restart_count: How many times this app has already been restarted.
        restart_limit: The maximum number of restarts before giving up.

    Returns:
        "none" if the thread is alive or the scheduler is stopping; "restart" if it died and
        has restarts remaining; "give_up" once it has exhausted its restarts.
    """
    if stopping or alive:
        return "none"
    if restart_count < restart_limit:
        return "restart"
    return "give_up"


class Scheduler:
    """Runs each registered app's run(stop_event) in its own daemon thread, with supervision."""

    def __init__(
        self,
        apps: list[tuple[str, RunnableApp]],
        bus: MessageBus | None = None,
        restart_limit: int = 3,
    ) -> None:
        """Register the apps to schedule.

        Args:
            apps: (name, app) pairs; name labels the thread for diagnostics.
            bus: Optional MessageBus used to publish PROCESS_DIED when an app exhausts restarts.
            restart_limit: Maximum unexpected-death restarts per app before giving up -> SAFE.
        """
        self._apps = apps
        self._bus = bus
        self._restart_limit = restart_limit
        self._stop = threading.Event()
        self._threads: dict[str, threading.Thread] = {}
        self._restart_count: dict[str, int] = {}
        self._gave_up: set[str] = set()

    def start(self) -> None:
        """Launch each app's run(stop_event) in a named daemon thread."""
        for name, app in self._apps:
            self._launch(name, app)

    def _launch(self, name: str, app: RunnableApp) -> None:
        """(Re)launch one app's run loop in a fresh daemon thread."""
        thread = threading.Thread(target=app.run, args=(self._stop,), name=name, daemon=True)
        thread.start()
        self._threads[name] = thread

    def check(self) -> None:
        """Run one supervision pass: restart unexpectedly-dead threads, or give up -> PROCESS_DIED.

        A thread that died while the scheduler is not stopping is restarted up to restart_limit
        times; once a thread exhausts its restarts a PROCESS_DIED FaultEventMsg is published once
        (FDIR routes it to SAFE) and the thread is no longer restarted.
        """
        for name, app in self._apps:
            if name in self._gave_up:
                continue
            thread = self._threads.get(name)
            alive = thread.is_alive() if thread is not None else False
            action = next_supervision_action(
                alive, self._stop.is_set(), self._restart_count.get(name, 0), self._restart_limit
            )
            if action == "restart":
                self._restart_count[name] = self._restart_count.get(name, 0) + 1
                self._launch(name, app)
            elif action == "give_up":
                self._gave_up.add(name)
                self._publish_process_died(name)

    def supervise(self, stop_event: threading.Event, poll_interval_s: float = 1.0) -> None:
        """Run supervision passes until stop_event is set (the flight main-loop supervisor)."""
        while not stop_event.is_set():
            self.check()
            stop_event.wait(timeout=poll_interval_s)

    def restart_count(self, name: str) -> int:
        """Return how many times the named app has been restarted (diagnostics/tests)."""
        return self._restart_count.get(name, 0)

    def gave_up_on(self, name: str) -> bool:
        """Return True if the named app exhausted its restarts and PROCESS_DIED was published."""
        return name in self._gave_up

    def _publish_process_died(self, name: str) -> None:
        """Publish a PROCESS_DIED FaultEventMsg for an app that exhausted its restarts."""
        if self._bus is None:
            return
        self._bus.publish(
            FaultEventMsg(
                msg_type=MessageType.FAULT_EVENT,
                timestamp_utc=utc_now_iso(),
                fault_code=FaultCode.PROCESS_DIED,
                subsystem=name,
                detail=f"{name} thread died and exhausted {self._restart_limit} restarts",
            )
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Signal every app to stop and join each thread up to timeout seconds.

        The joined (now-dead) threads are retained so is_running() reports liveness
        honestly after stop rather than reading an emptied list.
        """
        self._stop.set()
        for thread in self._threads.values():
            thread.join(timeout=timeout)

    def is_running(self) -> bool:
        """Return True if any scheduled thread is still alive."""
        return any(thread.is_alive() for thread in self._threads.values())
