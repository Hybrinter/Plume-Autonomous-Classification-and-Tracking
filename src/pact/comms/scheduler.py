"""
Communication pass window scheduler for PACT.

Determines whether a communication window is currently open based on UTC weekday,
and tracks the remaining daily byte budget for downlink and uplink.

Phase I uses a simple weekday-only window (MON–FRI) with no orbital mechanics.
No Doppler correction or link margin calculations are performed in software.

Satisfies: REQ-COMM-HIGH-001, REQ-COMM-HIGH-002
"""

from __future__ import annotations

from datetime import datetime


# Day-of-week abbreviations used by CommsConfig.comm_window_days.
# Python's datetime.strftime("%a") returns 3-letter abbreviations matching these.
_DAY_ABBREV_MAP: dict[int, str] = {
    0: "MON",
    1: "TUE",
    2: "WED",
    3: "THU",
    4: "FRI",
    5: "SAT",
    6: "SUN",
}


def is_comm_window_open(
    utc_now: datetime,
    allowed_days: tuple[str, ...],
) -> bool:
    """Return True if the current UTC day is an allowed communication day. REQ-COMM-HIGH-001.

    Parameters
    ----------
    utc_now:
        Current UTC datetime. The caller is responsible for providing a timezone-aware
        or naive UTC datetime; this function uses only the weekday().
    allowed_days:
        Tuple of 3-letter weekday abbreviations that are allowed comm days.
        Example: ("MON", "TUE", "WED", "THU", "FRI")
        Sourced from CommsConfig.comm_window_days.

    Returns
    -------
    bool
        True if utc_now.weekday() maps to an abbreviation present in allowed_days.

    Notes
    -----
    This function performs no orbital mechanics or contact window calculations.
    It is a simple weekday gate. Future work should integrate a contact window
    prediction library (e.g., Skyfield) for higher fidelity scheduling.
    """
    current_day_abbrev = _DAY_ABBREV_MAP[utc_now.weekday()]
    return current_day_abbrev in allowed_days


def bytes_remaining_today(
    bytes_used: int,
    daily_limit: int,
) -> int:
    """Return the remaining byte budget for today's communication window. REQ-COMM-HIGH-002.

    The daily budget resets at midnight UTC (tracked externally by the comms process).
    This function is stateless — it only computes the arithmetic.

    Parameters
    ----------
    bytes_used:
        Number of bytes already transmitted (downlink) or received (uplink) today.
        Must be >= 0.
    daily_limit:
        Total byte budget for the day.
        For downlink: CommsConfig.max_daily_downlink_bytes (default 1 GB = 1_073_741_824).
        For uplink:   CommsConfig.max_daily_uplink_bytes (default 100 MB = 104_857_600).

    Returns
    -------
    int
        Remaining bytes, clamped to [0, daily_limit]. Never negative.
    """
    return max(0, daily_limit - bytes_used)
