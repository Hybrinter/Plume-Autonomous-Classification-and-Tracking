"""PACT fault detection subsystem.

Monitors all processes via heartbeat watchdog, detects fault conditions, dispatches
fault handlers, and manages safe mode entry and exit.

Satisfies: REQ-SAFE-HIGH-002, REQ-GIMB-HIGH-003, GOAL-006.
"""
