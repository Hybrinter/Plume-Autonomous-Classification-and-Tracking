"""Two-state residual Kalman filter on elevation error and residual rate (pure, SI).

State x = [e, omega_t_res]. omega_g (y_m) and omega_t_nom are known inputs.
Vision measures e at shutter time t_s. Predict runs every outer tick. A lagged
z_v rewinds through a snapshot ring, updates, and replays.

Satisfies: REQ-GIMB-HIGH-002.
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass, replace

# third-party
import numpy as np

# internal
from flight.libs.config import ResidualConfig


@dataclass(frozen=True, slots=True)
class ResidualState:
    """Immutable residual-filter state.

    Attributes:
        x: np.ndarray[float64, (2,)] -- [e_rad, omega_t_res_rad_s].
        P: np.ndarray[float64, (2, 2)] -- covariance.
        has_measurement: True after the first accepted z_v.
    """

    x: np.ndarray
    P: np.ndarray  # noqa: N815
    has_measurement: bool


@dataclass(frozen=True, slots=True)
class ResidualSnapshot:
    """One outer-tick snapshot for rewind.

    Attributes:
        t_s: Monotonic time of this predict.
        state: Filter state after the predict (before a vision update at this tick).
        dt_s: Period used for this predict.
        omega_t_nom: Predictor rate input, rad/s.
        y_m: Encoder-rate input, rad/s.
    """

    t_s: float
    state: ResidualState
    dt_s: float
    omega_t_nom: float
    y_m: float


@dataclass(frozen=True, slots=True)
class ResidualFilter:
    """Configured Q, R, and P0 for the 2-state residual filter."""

    q11: float
    q22: float
    r_v: float
    p0_11: float
    p0_22: float
    dt_outer_s: float

    @staticmethod
    def from_config(cfg: ResidualConfig, dt_outer_s: float) -> ResidualFilter:
        """Build filter scalars from residual config and the outer period.

        Inputs:
            cfg: ResidualConfig with Q_diag, R_v, P0_diag, and rewind fields.
            dt_outer_s: Outer-loop period used to scale discrete Q.

        Outputs:
            ResidualFilter: Frozen gain/noise holder.
        """
        return ResidualFilter(
            q11=float(cfg.Q_diag[0]),
            q22=float(cfg.Q_diag[1]),
            r_v=cfg.R_v,
            p0_11=float(cfg.P0_diag[0]),
            p0_22=float(cfg.P0_diag[1]),
            dt_outer_s=dt_outer_s,
        )

    def initial_state(self) -> ResidualState:
        """Cold filter: e=0, omega_res=0, has_measurement=False.

        Outputs:
            ResidualState: Zero state with P0 on the diagonal.
        """
        p = np.array([[self.p0_11, 0.0], [0.0, self.p0_22]], dtype=np.float64)
        return ResidualState(
            x=np.zeros(2, dtype=np.float64),
            P=p,
            has_measurement=False,
        )


def _q(filt: ResidualFilter, dt_s: float) -> np.ndarray:
    """Process-noise matrix for this outer step, rescaled if dt differs from T_out.

    Inputs:
        filt: Configured filter.
        dt_s: Actual step, seconds.

    Outputs:
        np.ndarray[float64, (2, 2)]: Diagonal Q.
    """
    if filt.dt_outer_s <= 0.0:
        scale11 = 1.0
        scale22 = 1.0
    else:
        ratio = dt_s / filt.dt_outer_s
        scale11 = ratio**3
        scale22 = ratio
    return np.array(
        [[filt.q11 * scale11, 0.0], [0.0, filt.q22 * scale22]],
        dtype=np.float64,
    )


def predict(
    filt: ResidualFilter,
    state: ResidualState,
    dt_s: float,
    omega_t_nom: float,
    y_m: float,
) -> ResidualState:
    """Kalman predict: x- = F x + u, P- = F P F' + Q.

    Inputs:
        filt: Configured filter.
        state: Prior state.
        dt_s: Outer period for this step.
        omega_t_nom: Predictor rate, rad/s.
        y_m: Encoder rate, rad/s.

    Outputs:
        ResidualState: Predicted state (has_measurement unchanged).
    """
    if dt_s <= 0.0:
        return state
    f = np.array([[1.0, dt_s], [0.0, 1.0]], dtype=np.float64)
    u = np.array([dt_s * (omega_t_nom - y_m), 0.0], dtype=np.float64)
    x_pred = f @ state.x + u
    p_pred = f @ state.P @ f.T + _q(filt, dt_s)
    return ResidualState(x=x_pred, P=p_pred, has_measurement=state.has_measurement)


def update(filt: ResidualFilter, state: ResidualState, z_v: float) -> ResidualState:
    """Vision update of elevation error. First update may snap e to z_v.

    Inputs:
        filt: Configured filter.
        state: Predicted state at t_s.
        z_v: Vision measurement of e, rad.

    Outputs:
        ResidualState: Posterior with has_measurement=True.
    """
    if not state.has_measurement:
        x = state.x.copy()
        x[0] = z_v
        p = state.P.copy()
        p[0, 0] = filt.r_v
        p[0, 1] = 0.0
        p[1, 0] = 0.0
        return ResidualState(x=x, P=p, has_measurement=True)

    p = state.P
    s = float(p[0, 0] + filt.r_v)
    if abs(s) < 1e-30:
        return replace(state, has_measurement=True)
    k = np.array([p[0, 0] / s, p[1, 0] / s], dtype=np.float64)
    innov = z_v - float(state.x[0])
    x_upd = state.x + k * innov
    kc = np.array([[k[0], 0.0], [k[1], 0.0]], dtype=np.float64)
    p_upd = (np.eye(2, dtype=np.float64) - kc) @ p
    return ResidualState(x=x_upd, P=p_upd, has_measurement=True)


def rewind_update(
    filt: ResidualFilter,
    snapshots: tuple[ResidualSnapshot, ...],
    current: ResidualState,
    now: float,
    t_s: float,
    z_v: float,
    horizon_s: float,
) -> ResidualState:
    """Restore a snapshot at t_s, apply z_v, replay predicts to now.

    Inputs:
        filt: Configured filter.
        snapshots: Outer snapshots, oldest to newest.
        current: Filter state at now (used when the ring is empty).
        now: Current monotonic seconds.
        t_s: Shutter time of z_v.
        z_v: Vision measurement, rad.
        horizon_s: Drop samples older than now - horizon_s.

    Outputs:
        ResidualState: Filter at now after the delayed update.
    """
    if now - t_s > horizon_s or t_s > now + 1e-12:
        return current
    if not snapshots:
        pred = current
        if t_s < now:
            # No history: update as if current is already at t_s.
            return update(filt, pred, z_v)
        return update(filt, pred, z_v)

    usable = tuple(s for s in snapshots if now - s.t_s <= horizon_s + 1e-12)
    if not usable:
        return update(filt, current, z_v)

    idx = 0
    for i, snap in enumerate(usable):
        if snap.t_s <= t_s + 1e-12:
            idx = i
    snap = usable[idx]
    dt_to_meas = t_s - snap.t_s
    pred = snap.state
    if dt_to_meas > 1e-12:
        pred = predict(filt, pred, dt_to_meas, snap.omega_t_nom, snap.y_m)
    posterior = update(filt, pred, z_v)
    replay_from = t_s
    for later in usable[idx + 1 :]:
        dt = later.t_s - replay_from
        if dt > 1e-12:
            posterior = predict(filt, posterior, dt, later.omega_t_nom, later.y_m)
        replay_from = later.t_s
    dt_tail = now - replay_from
    if dt_tail > 1e-12:
        last = usable[-1]
        posterior = predict(filt, posterior, dt_tail, last.omega_t_nom, last.y_m)
    return posterior


def push_snapshot(
    snapshots: tuple[ResidualSnapshot, ...],
    snap: ResidualSnapshot,
    max_count: int,
) -> tuple[ResidualSnapshot, ...]:
    """Append a snapshot and drop the oldest past max_count.

    Inputs:
        snapshots: Existing ring, oldest first.
        snap: New snapshot.
        max_count: Ring capacity.

    Outputs:
        tuple[ResidualSnapshot, ...]: Updated ring.
    """
    combined = snapshots + (snap,)
    if len(combined) <= max_count:
        return combined
    return combined[-max_count:]
