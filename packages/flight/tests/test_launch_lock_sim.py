"""SimLaunchLock driver tests: starts ENGAGED, release/engage transitions."""

from flight.hal.drivers_sim import SimLaunchLock
from flight.libs.types import LaunchLockState, Ok


def test_starts_engaged() -> None:
    """The sim lock starts in the ENGAGED (launch) configuration."""
    lock = SimLaunchLock()
    state = lock.read_state()
    assert isinstance(state, Ok)
    assert state.value is LaunchLockState.ENGAGED


def test_release_then_engage_transitions() -> None:
    """release() -> RELEASED and engage() -> ENGAGED are reflected by read_state()."""
    lock = SimLaunchLock()
    assert isinstance(lock.release(), Ok)
    released = lock.read_state()
    assert isinstance(released, Ok) and released.value is LaunchLockState.RELEASED
    assert isinstance(lock.engage(), Ok)
    engaged = lock.read_state()
    assert isinstance(engaged, Ok) and engaged.value is LaunchLockState.ENGAGED
