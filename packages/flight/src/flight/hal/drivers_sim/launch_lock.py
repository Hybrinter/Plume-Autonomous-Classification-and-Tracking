"""Simulated launch lock: an in-memory engaged/released pin for SIL and tests.

Starts ENGAGED (flight configuration). release()/engage() flip the modeled microswitch state;
read_state() returns it. Satisfies LaunchLock structurally. There is no real LaunchLock driver
(the device is hardware-deferred), so this is the only implementation today.
"""

from flight.libs.types import FaultCode, LaunchLockState, Ok, Result


class SimLaunchLock:
    """In-memory launch-lock stand-in for SIL/tests (starts ENGAGED)."""

    def __init__(self, state: LaunchLockState = LaunchLockState.ENGAGED) -> None:
        """Initialize the modeled pin state (defaults to ENGAGED, the launch configuration)."""
        self._state = state

    def release(self) -> Result[None, FaultCode]:
        """Model a pin release: transition to RELEASED."""
        self._state = LaunchLockState.RELEASED
        return Ok(None)

    def engage(self) -> Result[None, FaultCode]:
        """Model a pin engage: transition to ENGAGED."""
        self._state = LaunchLockState.ENGAGED
        return Ok(None)

    def read_state(self) -> Result[LaunchLockState, FaultCode]:
        """Return the modeled microswitch state."""
        return Ok(self._state)
