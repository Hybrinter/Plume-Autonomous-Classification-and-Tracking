"""In-process typed pub/sub message bus with per-type bounded queues + overflow policy.

publish(msg) delivers msg to every Subscription registered for type(msg). The composition root
owns the bus and injects Subscriptions into apps; apps never construct queues themselves.
Transport is in-process queue.Queue.

Per spec Section 7, the bus enforces per-message-type queue bounds with an explicit overflow
policy (message-agnostic: the policy is a dict keyed by message type, built by the composition
root which knows the message types -- the bus itself imports no message class):

  - DROP_OLDEST: a bounded queue; on overflow the oldest queued message is discarded and a
    per-type drop counter is incremented (the telemetry policy -- shed stale data, keep flowing).
  - NEVER_DROP: an unbounded queue; the message is never discarded, but exceeding the configured
    soft bound increments a per-type overflow counter so the anomaly is observable as a fault
    (the command/fault policy -- losing a command or a fault is never acceptable).

The default policy is unbounded DROP_OLDEST (maxsize 0), so an unconfigured bus behaves exactly
as before (the deterministic SIL keeps an unbounded bus; only the flight entry installs bounds).

Satisfies: REQ-PLAT-QUEUE-001.
"""

import enum
import threading
from dataclasses import dataclass
from queue import Empty, Full, Queue
from typing import Generic, TypeVar, cast

_T = TypeVar("_T")


class OverflowPolicy(enum.Enum):
    """How a per-type queue handles overflow.

    String values mirror member names (log readability convention).
    """

    DROP_OLDEST = "DROP_OLDEST"  # bounded; discard the oldest on overflow (telemetry)
    NEVER_DROP = "NEVER_DROP"  # unbounded; count soft-bound exceedance (commands/faults)


@dataclass(frozen=True, slots=True)
class QueuePolicy:
    """Per-message-type queue configuration.

    Fields:
        maxsize: The bound (DROP_OLDEST) or soft bound (NEVER_DROP). 0 means unbounded with no
            overflow accounting.
        overflow: The overflow policy applied when the bound is reached.
    """

    maxsize: int
    overflow: OverflowPolicy


_UNBOUNDED = QueuePolicy(maxsize=0, overflow=OverflowPolicy.DROP_OLDEST)


class Subscription(Generic[_T]):  # noqa: UP046  (explicit Generic form retained intentionally)
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
    """Typed pub/sub bus routed by exact message type, with per-type bounds + overflow policy."""

    def __init__(self, policy: dict[type, QueuePolicy] | None = None) -> None:
        """Create an empty bus.

        Args:
            policy: Optional per-message-type queue policy. Types absent from the map (and an
                omitted map entirely) use the unbounded default, preserving prior behavior.
        """
        self._policy: dict[type, QueuePolicy] = dict(policy) if policy else {}
        self._subscribers: dict[type, list[Queue[object]]] = {}
        self._dropped: dict[str, int] = {}
        self._overflow: dict[str, int] = {}
        self._lock = threading.Lock()

    def _policy_for(self, message_type: type) -> QueuePolicy:
        """Return the configured policy for a message type, or the unbounded default."""
        return self._policy.get(message_type, _UNBOUNDED)

    def subscribe(self, message_type: type[_T]) -> Subscription[_T]:
        """Register interest in a message type and return a receive handle.

        DROP_OLDEST queues are created with a hard maxsize so overflow is detectable; NEVER_DROP
        (and unbounded) queues are created unbounded so a message is never blocked or lost.
        """
        pol = self._policy_for(message_type)
        hard_max = pol.maxsize if pol.overflow is OverflowPolicy.DROP_OLDEST else 0
        queue: Queue[object] = Queue(maxsize=hard_max)
        with self._lock:
            self._subscribers.setdefault(message_type, []).append(queue)
        return Subscription(cast("Queue[_T]", queue))

    def publish(self, message: object) -> None:
        """Deliver message to every Subscription registered for its exact type, per policy."""
        message_type = type(message)
        pol = self._policy_for(message_type)
        with self._lock:
            queues = list(self._subscribers.get(message_type, []))
        for queue in queues:
            self._deliver(queue, message, message_type, pol)

    def _deliver(
        self, queue: Queue[object], message: object, message_type: type, pol: QueuePolicy
    ) -> None:
        """Deliver one message to one queue, applying the type's overflow policy."""
        if pol.maxsize <= 0:
            queue.put(message)
            return
        if pol.overflow is OverflowPolicy.DROP_OLDEST:
            try:
                queue.put_nowait(message)
            except Full:
                try:
                    queue.get_nowait()  # discard the oldest
                except Empty:  # pragma: no cover - racing consumer drained it
                    pass
                self._dropped[message_type.__name__] = (
                    self._dropped.get(message_type.__name__, 0) + 1
                )
                try:
                    queue.put_nowait(message)
                except Full:  # pragma: no cover - racing producer refilled it
                    pass
        else:  # NEVER_DROP: unbounded queue; count soft-bound exceedance, never discard.
            if queue.qsize() >= pol.maxsize:
                self._overflow[message_type.__name__] = (
                    self._overflow.get(message_type.__name__, 0) + 1
                )
            queue.put(message)

    def dropped_count(self, message_type: type) -> int:
        """Return how many messages of message_type were dropped (DROP_OLDEST overflow)."""
        return self._dropped.get(message_type.__name__, 0)

    def overflow_count(self, message_type: type) -> int:
        """Return how many times message_type exceeded its soft bound (NEVER_DROP overflow)."""
        return self._overflow.get(message_type.__name__, 0)

    def total_dropped(self) -> int:
        """Return the total dropped-message count across all DROP_OLDEST types."""
        return sum(self._dropped.values())

    def total_overflow(self) -> int:
        """Return the total soft-bound-overflow count across all NEVER_DROP types."""
        return sum(self._overflow.values())
