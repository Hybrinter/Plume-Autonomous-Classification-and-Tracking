"""Unit tests for pact.comms.downlink — DownlinkQueue priority, budget, and window enforcement.

Satisfies: §6.2 of PACT_SW_ARCH.md — Comms subsystem unit tests.
REQ-COMM-HIGH-001 (weekday-only comm window), REQ-COMM-HIGH-002 (daily byte budget),
GOAL-008 (priority-ordered downlink).
"""

from __future__ import annotations

# stdlib
from datetime import datetime, timezone

# third-party
import pytest

# module under test
from pact.comms.ccsds import compute_crc32
from pact.comms.downlink import DownlinkQueue

# pact types
from pact.types.enums import DownlinkPriority, MessageType
from pact.types.messages import DownlinkItemMsg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WEEKDAYS: tuple[str, ...] = ("MON", "TUE", "WED", "THU", "FRI")

_MONDAY_UTC = datetime(2026, 4, 6, 12, 0, 0, tzinfo=timezone.utc)
_TUESDAY_UTC = datetime(2026, 4, 7, 12, 0, 0, tzinfo=timezone.utc)
_SATURDAY_UTC = datetime(2026, 4, 11, 12, 0, 0, tzinfo=timezone.utc)

# 1 GB — effectively unlimited for most tests
_LARGE_BUDGET: int = 1_073_741_824


def make_downlink_item(
    priority: DownlinkPriority,
    payload_bytes: bytes,
    item_id: str = "test-item",
) -> DownlinkItemMsg:
    """Construct a DownlinkItemMsg for testing."""
    return DownlinkItemMsg(
        msg_type=MessageType.DOWNLINK_ITEM,
        timestamp_utc="2026-04-06T12:00:00.000Z",
        priority=priority,
        payload_bytes=payload_bytes,
        crc32=compute_crc32(payload_bytes),
        item_id=item_id,
    )


# ---------------------------------------------------------------------------
# Basic enqueue / dequeue
# ---------------------------------------------------------------------------


def test_enqueue_dequeue_returns_item() -> None:
    """A single item enqueued and dequeued on a weekday within budget must be returned."""
    dq = DownlinkQueue(daily_limit_bytes=_LARGE_BUDGET, allowed_comm_days=_WEEKDAYS)
    item = make_downlink_item(DownlinkPriority.HEALTH_TELEMETRY, b"telemetry payload")
    dq.enqueue(item)

    result = dq.dequeue(utc_now=_MONDAY_UTC)

    assert result is not None, "dequeue returned None — expected the enqueued item"
    assert result.item_id == item.item_id


def test_dequeue_empty_queue_returns_none() -> None:
    """dequeue on an empty queue must return None without error."""
    dq = DownlinkQueue(daily_limit_bytes=_LARGE_BUDGET, allowed_comm_days=_WEEKDAYS)
    result = dq.dequeue(utc_now=_MONDAY_UTC)
    assert result is None, f"Expected None for empty queue, got {result}"


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------


def test_priority_ordering() -> None:
    """HEALTH_TELEMETRY (.value=0) must be returned before RAW_IMAGERY (.value=3),
    regardless of enqueue order.
    """
    dq = DownlinkQueue(daily_limit_bytes=_LARGE_BUDGET, allowed_comm_days=_WEEKDAYS)
    raw_item = make_downlink_item(DownlinkPriority.RAW_IMAGERY, b"raw", item_id="raw")
    health_item = make_downlink_item(
        DownlinkPriority.HEALTH_TELEMETRY, b"health", item_id="health"
    )

    # Enqueue lower-priority first to confirm ordering is not insertion-order
    dq.enqueue(raw_item)
    dq.enqueue(health_item)

    first = dq.dequeue(utc_now=_MONDAY_UTC)
    second = dq.dequeue(utc_now=_MONDAY_UTC)

    assert first is not None
    assert second is not None
    assert first.priority == DownlinkPriority.HEALTH_TELEMETRY, (
        f"Expected HEALTH_TELEMETRY first, got {first.priority}"
    )
    assert second.priority == DownlinkPriority.RAW_IMAGERY


def test_all_priorities_ordered() -> None:
    """All four DownlinkPriority levels must be dequeued in ascending priority order."""
    dq = DownlinkQueue(daily_limit_bytes=_LARGE_BUDGET, allowed_comm_days=_WEEKDAYS)
    # Enqueue in worst-first (reverse priority) order
    for prio in reversed(list(DownlinkPriority)):
        dq.enqueue(make_downlink_item(prio, b"payload", item_id=prio.name))

    expected_order = [
        DownlinkPriority.HEALTH_TELEMETRY,
        DownlinkPriority.SCIENCE_DATA,
        DownlinkPriority.COMPRESSED_IMAGERY,
        DownlinkPriority.RAW_IMAGERY,
    ]
    for expected_prio in expected_order:
        item = dq.dequeue(utc_now=_MONDAY_UTC)
        assert item is not None, f"Queue unexpectedly empty before {expected_prio}"
        assert item.priority == expected_prio, (
            f"Expected {expected_prio.name}, got {item.priority.name}"
        )


# ---------------------------------------------------------------------------
# Comm window enforcement
# ---------------------------------------------------------------------------


def test_dequeue_closed_window_returns_none() -> None:
    """dequeue on a non-allowed day (Saturday) must return None without consuming the item."""
    dq = DownlinkQueue(daily_limit_bytes=_LARGE_BUDGET, allowed_comm_days=_WEEKDAYS)
    dq.enqueue(make_downlink_item(DownlinkPriority.HEALTH_TELEMETRY, b"data"))

    result = dq.dequeue(utc_now=_SATURDAY_UTC)

    assert result is None, "dequeue returned an item on a closed comm window day"
    assert dq.qsize() == 1, "Item should remain in queue after window-closed rejection"


# ---------------------------------------------------------------------------
# Daily byte budget enforcement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("payload,limit", [
    (b"hi",    2),   # payload == limit: dequeue consumes exactly the budget
    (b"hello", 5),   # payload == limit: same with larger payload
])
def test_dequeue_budget_single_item_fits(payload: bytes, limit: int) -> None:
    """An item whose payload exactly equals the daily limit must be deliverable."""
    dq = DownlinkQueue(daily_limit_bytes=limit, allowed_comm_days=_WEEKDAYS)
    dq.enqueue(make_downlink_item(DownlinkPriority.HEALTH_TELEMETRY, payload))

    result = dq.dequeue(utc_now=_MONDAY_UTC)

    assert result is not None, (
        f"Expected item with budget {limit} and payload {len(payload)}B"
    )


def test_dequeue_budget_exhausted_returns_none() -> None:
    """dequeue must return None when the daily byte budget is fully consumed."""
    payload = b"hello"  # 5 bytes
    daily_limit = len(payload)  # budget covers exactly one item
    dq = DownlinkQueue(daily_limit_bytes=daily_limit, allowed_comm_days=_WEEKDAYS)

    dq.enqueue(make_downlink_item(DownlinkPriority.HEALTH_TELEMETRY, payload, item_id="a"))
    first = dq.dequeue(utc_now=_MONDAY_UTC)
    assert first is not None, "First dequeue should succeed within budget"
    assert dq.bytes_remaining_today == 0

    dq.enqueue(make_downlink_item(DownlinkPriority.HEALTH_TELEMETRY, b"x", item_id="b"))
    second = dq.dequeue(utc_now=_MONDAY_UTC)
    assert second is None, "dequeue should return None when daily budget is exhausted"


# ---------------------------------------------------------------------------
# Day rollover
# ---------------------------------------------------------------------------


def test_day_rollover_resets_bytes_used() -> None:
    """bytes_used_today must reset to 0 when the UTC day changes between dequeue calls."""
    payload = b"hello"
    daily_limit = len(payload)  # tight budget: one item per day
    dq = DownlinkQueue(daily_limit_bytes=daily_limit, allowed_comm_days=_WEEKDAYS)

    # Monday: exhaust the budget
    dq.enqueue(make_downlink_item(DownlinkPriority.HEALTH_TELEMETRY, payload, item_id="mon"))
    first = dq.dequeue(utc_now=_MONDAY_UTC)
    assert first is not None
    assert dq.bytes_used_today == len(payload)

    # Tuesday: budget resets — new item should be deliverable
    dq.enqueue(make_downlink_item(DownlinkPriority.HEALTH_TELEMETRY, payload, item_id="tue"))
    second = dq.dequeue(utc_now=_TUESDAY_UTC)
    assert second is not None, "dequeue should succeed after daily budget resets at day rollover"
    assert dq.bytes_used_today == len(payload)  # re-consumed on the new day


# ---------------------------------------------------------------------------
# Byte tracking properties
# ---------------------------------------------------------------------------


def test_bytes_used_today_tracks_payload() -> None:
    """bytes_used_today must increment by len(payload_bytes) on each successful dequeue."""
    dq = DownlinkQueue(daily_limit_bytes=_LARGE_BUDGET, allowed_comm_days=_WEEKDAYS)
    assert dq.bytes_used_today == 0

    payloads = [b"abc", b"defgh", b"ij"]
    for i, p in enumerate(payloads):
        dq.enqueue(make_downlink_item(DownlinkPriority.HEALTH_TELEMETRY, p, item_id=str(i)))

    expected_total = 0
    for p in payloads:
        dq.dequeue(utc_now=_MONDAY_UTC)
        expected_total += len(p)
        assert dq.bytes_used_today == expected_total, (
            f"After consuming {len(p)}B: expected bytes_used_today={expected_total}, "
            f"got {dq.bytes_used_today}"
        )


def test_bytes_remaining_today_property() -> None:
    """bytes_remaining_today must equal daily_limit - bytes_used_today."""
    limit = 1_000
    dq = DownlinkQueue(daily_limit_bytes=limit, allowed_comm_days=_WEEKDAYS)
    assert dq.bytes_remaining_today == limit

    payload = b"x" * 100
    dq.enqueue(make_downlink_item(DownlinkPriority.HEALTH_TELEMETRY, payload))
    dq.dequeue(utc_now=_MONDAY_UTC)
    assert dq.bytes_remaining_today == limit - len(payload)


# ---------------------------------------------------------------------------
# qsize
# ---------------------------------------------------------------------------


def test_qsize_reflects_enqueued() -> None:
    """qsize() must reflect the approximate number of items currently in the queue."""
    dq = DownlinkQueue(daily_limit_bytes=_LARGE_BUDGET, allowed_comm_days=_WEEKDAYS)
    assert dq.qsize() == 0

    for i in range(3):
        dq.enqueue(
            make_downlink_item(DownlinkPriority.SCIENCE_DATA, b"data", item_id=str(i))
        )
    assert dq.qsize() == 3

    dq.dequeue(utc_now=_MONDAY_UTC)
    assert dq.qsize() == 2
