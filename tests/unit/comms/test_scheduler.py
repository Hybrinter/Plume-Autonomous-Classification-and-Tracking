"""Unit tests for pact.comms.scheduler — is_comm_window_open() and bytes_remaining_today().

Satisfies: §6.2 of PACT_SW_ARCH.md — Comms subsystem unit tests.
REQ-COMM-HIGH-001 (weekday-only comm window), REQ-COMM-HIGH-002 (daily byte budget)
"""

from __future__ import annotations

# stdlib
from datetime import datetime, timezone

# third-party
import pytest

# module under test
from pact.comms.scheduler import bytes_remaining_today, is_comm_window_open


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Default allowed comm days (MON–FRI) per config/default.toml
_WEEKDAYS: tuple[str, ...] = ("MON", "TUE", "WED", "THU", "FRI")

# Monday 2026-04-06 12:00 UTC (a known weekday)
_MONDAY_UTC = datetime(2026, 4, 6, 12, 0, 0, tzinfo=timezone.utc)
# Saturday 2026-04-11 12:00 UTC (a known weekend day)
_SATURDAY_UTC = datetime(2026, 4, 11, 12, 0, 0, tzinfo=timezone.utc)
# Sunday 2026-04-12 12:00 UTC
_SUNDAY_UTC = datetime(2026, 4, 12, 12, 0, 0, tzinfo=timezone.utc)

# Daily downlink budget: 1 GB per config/default.toml
_DAILY_LIMIT: int = 1_073_741_824


# ---------------------------------------------------------------------------
# is_comm_window_open tests
# ---------------------------------------------------------------------------


def test_weekday_window_open() -> None:
    """is_comm_window_open must return True on a weekday when that day is allowed."""
    assert is_comm_window_open(_MONDAY_UTC, _WEEKDAYS) is True, (
        "Comm window should be open on Monday (MON is in allowed_days)"
    )


def test_weekend_window_closed() -> None:
    """is_comm_window_open must return False on Saturday and Sunday."""
    assert is_comm_window_open(_SATURDAY_UTC, _WEEKDAYS) is False, (
        "Comm window should be closed on Saturday"
    )
    assert is_comm_window_open(_SUNDAY_UTC, _WEEKDAYS) is False, (
        "Comm window should be closed on Sunday"
    )


@pytest.mark.parametrize("day_abbrev,should_be_open", [
    ("MON", True),
    ("TUE", True),
    ("WED", True),
    ("THU", True),
    ("FRI", True),
    ("SAT", False),
    ("SUN", False),
])
def test_window_all_days(day_abbrev: str, should_be_open: bool) -> None:
    """is_comm_window_open parametrized over all days of the week.

    Uses Monday 2026-04-06 as a reference and shifts by offset to reach each day.
    """
    # Offsets from Monday (0) to reach each day
    day_offsets = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6}
    from datetime import timedelta
    target_dt = _MONDAY_UTC + timedelta(days=day_offsets[day_abbrev])
    result = is_comm_window_open(target_dt, _WEEKDAYS)
    assert result == should_be_open, (
        f"Day {day_abbrev}: expected {'open' if should_be_open else 'closed'}, "
        f"got {'open' if result else 'closed'}"
    )


def test_empty_allowed_days_always_closed() -> None:
    """is_comm_window_open with no allowed days must always return False."""
    assert is_comm_window_open(_MONDAY_UTC, ()) is False


# ---------------------------------------------------------------------------
# bytes_remaining_today tests
# ---------------------------------------------------------------------------


def test_bytes_remaining_full_budget() -> None:
    """bytes_remaining_today with 0 bytes used must return the full daily limit."""
    remaining = bytes_remaining_today(bytes_used=0, daily_limit=_DAILY_LIMIT)
    assert remaining == _DAILY_LIMIT, (
        f"Expected {_DAILY_LIMIT} bytes remaining with 0 used, got {remaining}"
    )


def test_bytes_remaining_over_limit() -> None:
    """bytes_remaining_today when bytes_used exceeds daily_limit must return 0 (not negative)."""
    over_budget = _DAILY_LIMIT + 1_000_000
    remaining = bytes_remaining_today(bytes_used=over_budget, daily_limit=_DAILY_LIMIT)
    assert remaining == 0, (
        f"Expected 0 remaining when over budget, got {remaining}"
    )


def test_bytes_remaining_partial_use() -> None:
    """bytes_remaining_today with partial use must return the correct remainder."""
    used = 500_000_000  # half a GB
    remaining = bytes_remaining_today(bytes_used=used, daily_limit=_DAILY_LIMIT)
    assert remaining == _DAILY_LIMIT - used, (
        f"Expected {_DAILY_LIMIT - used}, got {remaining}"
    )


def test_bytes_remaining_exactly_at_limit() -> None:
    """bytes_remaining_today exactly at the limit must return 0."""
    remaining = bytes_remaining_today(bytes_used=_DAILY_LIMIT, daily_limit=_DAILY_LIMIT)
    assert remaining == 0


def test_bytes_remaining_returns_non_negative() -> None:
    """bytes_remaining_today must never return a negative value."""
    remaining = bytes_remaining_today(bytes_used=_DAILY_LIMIT * 10, daily_limit=_DAILY_LIMIT)
    assert remaining >= 0, f"bytes_remaining_today returned negative: {remaining}"
