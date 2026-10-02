"""Payload records: compact observation and activation-context value types (pure).

IssSample and VisionSample are the shell-facing observations consumed by the
pure control core. ActivationKey, CaptureContext, CapturedVision, and
HealthSample are typed values for the graph contract; they carry data only.

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001.
"""

from __future__ import annotations

from dataclasses import dataclass

from flight.libs.messages import BlobMeta


@dataclass(frozen=True, slots=True)
class VisionSample:
    """One vision packet for the outer loop (shell queue payload).

    Attributes:
        t_s: Monotonic shutter time.
        frame_id: Stable frame identifier used to deduplicate delayed observations.
        z_v: Elevation boresight error in radians, or None when no blob.
        p_cog: Band-plane centroid, or None when no blob.
        exposure_us: Live frame exposure.
        blobs: Gated, matched blobs (empty on a miss).
        mode_flags: Inference mode_flags for SAFE latching.
        iss: ISS state at shutter, or None when ephemeris is dead.
        theta_g_rad: Encoder angle interpolated at shutter time, or None when
            the shared encoder stream does not bracket the shutter.
    """

    t_s: float
    frame_id: str
    z_v: float | None
    p_cog: tuple[float, float] | None
    exposure_us: float
    blobs: tuple[BlobMeta, ...]
    mode_flags: int
    iss: IssSample | None
    theta_g_rad: float | None = None


@dataclass(frozen=True, slots=True)
class IssSample:
    """ISS ECI state passed into the outer step (from the ephemeris HAL).

    Attributes:
        r_m: Position meters ECI.
        v_m_s: Inertial velocity m/s ECI.
        utc_s: UTC seconds for Earth rotation.
    """

    r_m: tuple[float, float, float]
    v_m_s: tuple[float, float, float]
    utc_s: float


@dataclass(frozen=True, slots=True)
class ActivationKey:
    """Authority-scoped activation identity: session epoch plus sequence.

    Attributes:
        epoch: Composition-root-provided session epoch.
        sequence: Authority-owned activation sequence within the epoch.
    """

    epoch: str
    sequence: int


@dataclass(frozen=True, slots=True)
class CaptureContext:
    """Identity of the capture work a vision sample or product belongs to.

    Attributes:
        activation_key: Activation under which the capture ran.
        policy_revision: Applied imaging/inference policy revision.
        model_version: Runtime model identity used for inference.
    """

    activation_key: ActivationKey
    policy_revision: int
    model_version: str


@dataclass(frozen=True, slots=True)
class CapturedVision:
    """A vision sample tagged with the capture context that produced it.

    Attributes:
        context: Capture context for staleness checks.
        sample: The vision observation.
    """

    context: CaptureContext
    sample: VisionSample


@dataclass(frozen=True, slots=True)
class HealthSample:
    """Compact actuator-health observation for one graph tick.

    Attributes:
        feedback_valid: Encoder feedback is present and fresh.
        inhibit_confirmed: Hardware confirms the actuator is inhibited.
        contained: Local containment (fault latch or interlock) is active.
    """

    feedback_valid: bool
    inhibit_confirmed: bool
    contained: bool
