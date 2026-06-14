"""Launch-lock hardware abstraction.

Defines the LaunchLock protocol: the payload's interface to the motorized launch-lock pin
(engaged/released microswitches). Release is a hazardous, ground-commanded operation; the
mechanical app owns this device. SAFE does not re-engage the lock (re-engagement is a
ground-commanded end-of-mission operation), so there is no autonomous engage path.

No real LaunchLock driver exists yet (the device is hardware-deferred -- a permanent VCRM gap,
spec Section 9.1); only SimLaunchLock implements this Protocol today. The mechanical app depends
only on this Protocol, so a real driver can be dropped in behind it later with no app change.

Satisfies: REQ-MECH-HIGH-001.
"""

from typing import Protocol, runtime_checkable

from flight.libs.types import FaultCode, LaunchLockState, Result


@runtime_checkable
class LaunchLock(Protocol):
    """Hardware abstraction for the motorized launch-lock pin (hazardous mechanism)."""

    def release(self) -> Result[None, FaultCode]:
        """Drive the pin to RELEASED. Hazardous; gated by the mechanical app's interlocks."""
        ...

    def engage(self) -> Result[None, FaultCode]:
        """Drive the pin to ENGAGED (ground-commanded end-of-mission operation)."""
        ...

    def read_state(self) -> Result[LaunchLockState, FaultCode]:
        """Read the current mechanism state from the engaged/released microswitches."""
        ...
