"""Vision-latency rewind ring for the dual-axis Kalman filter (pure).

Stores outer-loop snapshots of (t, x, P, u, theta_enc). A delayed vision sample
rolls back to the last snapshot at or before shutter time, applies update_vis
once, then replays predict and encoder updates forward.

Satisfies: REQ-GIMB-HIGH-002.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from flight.libs.types import Ok
from flight.payload.tracking.kalman import (
    AxisKalmanState,
    DualKalmanState,
    KalmanFilter,
    predict,
    update_enc,
    update_vis,
)


@dataclass(frozen=True, slots=True)
class EstimatorSnapshot:
    """One outer-tick snapshot used to rewind a delayed vision update."""

    t: float
    az: AxisKalmanState
    el: AxisKalmanState
    u_az: float
    u_el: float
    theta_enc_az: float | None
    theta_enc_el: float | None


@dataclass(frozen=True, slots=True)
class EstimatorRing:
    """Bounded time-ordered snapshot buffer."""

    snapshots: tuple[EstimatorSnapshot, ...]
    max_len: int


def empty_ring(max_len: int) -> EstimatorRing:
    """Return an empty ring with the given capacity (at least 1)."""
    return EstimatorRing(snapshots=(), max_len=max(1, max_len))


def push_snapshot(ring: EstimatorRing, snap: EstimatorSnapshot) -> EstimatorRing:
    """Append a snapshot and drop the oldest when over capacity."""
    snaps = ring.snapshots + (snap,)
    if len(snaps) > ring.max_len:
        snaps = snaps[-ring.max_len :]
    return replace(ring, snapshots=snaps)


def apply_vision(
    kf: KalmanFilter,
    ring: EstimatorRing,
    z_az: float,
    z_el: float,
    t_shutter: float,
    now: float,
    u_az_now: float,
    u_el_now: float,
    current: DualKalmanState,
) -> DualKalmanState | None:
    """Roll back, apply vision once, replay to now. None if shutter is older than the ring.

    Inputs:
        kf: Shared filter matrices.
        ring: Prior outer-tick snapshots, oldest first.
        z_az, z_el: Vision measurements of e (deg).
        t_shutter: Monotonic capture time of the image.
        now: Current monotonic time.
        u_az_now, u_el_now: Held rate commands from now-back to the last snapshot.
        current: Filter state at `now` (unused except as a fallback type; replay rebuilds it).

    Outputs:
        Updated DualKalmanState at `now`, or None when the shutter predates the ring.
    """
    del current
    snaps = ring.snapshots
    if not snaps or t_shutter < snaps[0].t:
        return None
    idx = 0
    for i, snap in enumerate(snaps):
        if snap.t <= t_shutter:
            idx = i
        else:
            break
    origin = snaps[idx]
    vis_az = update_vis(kf, origin.az, z_az)
    vis_el = update_vis(kf, origin.el, z_el)
    if not isinstance(vis_az, Ok) or not isinstance(vis_el, Ok):
        return None
    az = vis_az.value
    el = vis_el.value
    t_prev = origin.t
    u_az = origin.u_az
    u_el = origin.u_el
    for snap in snaps[idx + 1 :]:
        dt = snap.t - t_prev
        az = predict(kf, az, u_az, dt)
        el = predict(kf, el, u_el, dt)
        if snap.theta_enc_az is not None:
            enc = update_enc(kf, az, snap.theta_enc_az)
            if isinstance(enc, Ok):
                az = enc.value
        if snap.theta_enc_el is not None:
            enc = update_enc(kf, el, snap.theta_enc_el)
            if isinstance(enc, Ok):
                el = enc.value
        t_prev = snap.t
        u_az = snap.u_az
        u_el = snap.u_el
    dt_now = now - t_prev
    az = predict(kf, az, u_az_now, dt_now)
    el = predict(kf, el, u_el_now, dt_now)
    return DualKalmanState(az=az, el=el)
