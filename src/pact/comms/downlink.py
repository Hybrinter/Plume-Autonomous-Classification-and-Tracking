"""
Priority-ordered downlink queue for PACT comms subsystem.

Wraps queue.PriorityQueue to provide a typed, budget-aware, window-aware downlink queue.
Items are ordered by DownlinkPriority (lower integer value = higher priority, per enum).

Budget and window enforcement:
- dequeue() returns None if the communication window is closed.
- dequeue() returns None if the daily byte budget is exhausted.
- enqueue() always accepts items (bounded by maxsize); budget is checked on dequeue only.

Satisfies: REQ-COMM-HIGH-001, REQ-COMM-HIGH-002, GOAL-008
"""

from __future__ import annotations

import queue
from datetime import datetime
from typing import Optional

from pact.comms.scheduler import bytes_remaining_today, is_comm_window_open
from pact.types.enums import DownlinkPriority
from pact.types.messages import DownlinkItemMsg


class DownlinkQueue:
    """Priority-ordered downlink queue with daily byte budget enforcement.

    Items are prioritised by DownlinkPriority value (HEALTH_TELEMETRY=0 is highest priority).
    The queue is thread-safe (queue.PriorityQueue uses a Lock internally).

    Parameters
    ----------
    daily_limit_bytes:
        Maximum bytes that may be dequeued in a single UTC day.
        Sourced from CommsConfig.max_daily_downlink_bytes (default 1 GB).
    allowed_comm_days:
        Weekday abbreviations on which the comm window is open.
        Sourced from CommsConfig.comm_window_days.
    maxsize:
        Maximum number of items the queue can hold. 0 = unbounded.
        Default 256 to bound memory usage for low-bandwidth scenarios.
    """

    def __init__(
        self,
        daily_limit_bytes: int,
        allowed_comm_days: tuple[str, ...],
        maxsize: int = 256,
    ) -> None:
        self._queue: "queue.PriorityQueue[tuple[int, DownlinkItemMsg]]" = (
            queue.PriorityQueue(maxsize=maxsize)
        )
        self._daily_limit_bytes = daily_limit_bytes
        self._allowed_comm_days = allowed_comm_days
        self._bytes_used_today: int = 0
        self._current_day: Optional[int] = None  # UTC weekday (0=MON … 6=SUN)

    def enqueue(self, item: DownlinkItemMsg) -> None:
        """Add an item to the queue. Blocks if queue is full (maxsize reached).

        Priority is determined by item.priority.value (lower int = dequeued first).

        Parameters
        ----------
        item:
            DownlinkItemMsg to enqueue. Must have a valid priority field.
        """
        priority_key = item.priority.value  # int, lower = higher priority
        self._queue.put((priority_key, item))

    def dequeue(self, utc_now: Optional[datetime] = None) -> Optional[DownlinkItemMsg]:
        """Remove and return the highest-priority item, subject to window and budget checks.

        Returns None (without dequeuing) if:
        - The communication window is closed (non-allowed weekday).
        - The daily byte budget is exhausted.
        - The queue is empty.

        Parameters
        ----------
        utc_now:
            Current UTC datetime for window and budget checks. If None, uses datetime.utcnow().

        Returns
        -------
        Optional[DownlinkItemMsg]
            The highest-priority item, or None if window/budget/empty prevents dequeue.
        """
        now = utc_now if utc_now is not None else datetime.utcnow()

        # Reset daily byte counter at day rollover
        current_weekday = now.weekday()
        if self._current_day is None or self._current_day != current_weekday:
            self._bytes_used_today = 0
            self._current_day = current_weekday

        # Gate 1: communication window
        if not is_comm_window_open(now, self._allowed_comm_days):
            return None

        # Gate 2: daily byte budget
        if bytes_remaining_today(self._bytes_used_today, self._daily_limit_bytes) == 0:
            return None

        # Gate 3: queue empty check (non-blocking)
        try:
            _priority, item = self._queue.get_nowait()
        except queue.Empty:
            return None

        # Track bytes consumed
        self._bytes_used_today += len(item.payload_bytes)
        return item

    @property
    def bytes_used_today(self) -> int:
        """Bytes dequeued (and presumably transmitted) in the current UTC day."""
        return self._bytes_used_today

    @property
    def bytes_remaining_today(self) -> int:
        """Remaining byte budget for the current UTC day."""
        return bytes_remaining_today(self._bytes_used_today, self._daily_limit_bytes)

    def qsize(self) -> int:
        """Return approximate number of items currently in the queue."""
        return self._queue.qsize()
