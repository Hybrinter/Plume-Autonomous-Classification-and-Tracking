"""In-process typed pub/sub message bus.

Generalizes the per-channel queue pattern into one bus routed by exact message type:
publish(msg) delivers msg to every Subscription registered for type(msg). The
composition root owns the bus and injects Subscriptions into apps; apps never
construct queues themselves. Transport is in-process queue.Queue (what unit tests
and single-process SIL use); a multiprocessing-backed transport can replace the
queue factory later without changing this API.
"""

import threading
from queue import Queue
from typing import Generic, TypeVar, cast

_T = TypeVar("_T")


class Subscription(Generic[_T]):  # noqa: UP046  (keep explicit Generic form for Rust-idiomatic parity)
    """A typed receive handle for one subscribed message type."""

    def __init__(self, queue: Queue[_T]) -> None:
        """Wrap the backing queue for a single subscription."""
        self._queue = queue

    def get(self, timeout: float | None = None) -> _T:
        """Block for the next message, optionally up to timeout seconds.

        Raises:
            queue.Empty: If timeout elapses with no message.
        """
        return self._queue.get(timeout=timeout)

    def get_nowait(self) -> _T:
        """Return the next message immediately.

        Raises:
            queue.Empty: If no message is queued.
        """
        return self._queue.get_nowait()

    def empty(self) -> bool:
        """Return True if no message is currently queued."""
        return self._queue.empty()


class MessageBus:
    """Typed pub/sub bus routed by exact message type (in-process)."""

    def __init__(self, maxsize: int = 0) -> None:
        """Create an empty bus.

        Args:
            maxsize: Per-subscription queue bound (0 = unbounded).
        """
        self._maxsize = maxsize
        self._subscribers: dict[type, list[Queue[object]]] = {}
        self._lock = threading.Lock()

    def subscribe(self, message_type: type[_T]) -> Subscription[_T]:
        """Register interest in a message type and return a receive handle."""
        queue: Queue[object] = Queue(maxsize=self._maxsize)
        with self._lock:
            self._subscribers.setdefault(message_type, []).append(queue)
        return Subscription(cast("Queue[_T]", queue))

    def publish(self, message: object) -> None:
        """Deliver message to every Subscription registered for its exact type."""
        with self._lock:
            queues = list(self._subscribers.get(type(message), []))
        for queue in queues:
            queue.put(message)
