"""Pure two-state residual estimator and chronological event history.

The physical state is ``[e, omega_t_res]``. Encoder feedback is an uncertain
known input: a displacement changes ``e`` directly and its uncertainty is
charged to the position covariance. The history keeps raw timestamped events
separate from the posterior so delayed observations can be replayed without
inverse prediction.

``ResidualSnapshot`` and the old rewind helpers remain as compatibility shims
for the detailed-plant transition. New code should use ``ResidualHistory`` and
``estimate_at``.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np

from flight.libs.config import ResidualConfig

_EPS = 1.0e-12


@dataclass(frozen=True, slots=True)
class ResidualState:
    """Immutable residual-filter state ``x=[e_rad, omega_t_res_rad_s]``."""

    x: np.ndarray
    P: np.ndarray  # noqa: N815
    has_measurement: bool


@dataclass(frozen=True, slots=True)
class EncoderSample:
    """Timestamped unwrapped encoder angle and endpoint variance."""

    sample_id: str
    t_s: float
    angle_rad: float
    angle_variance_rad2: float = 0.0

    @property
    def event_id(self) -> str:
        return self.sample_id


@dataclass(frozen=True, slots=True)
class NominalRateSample:
    """Sampled nominal rate integrated with causal zero-order hold."""

    sample_id: str
    t_s: float
    rate_rad_s: float

    @property
    def event_id(self) -> str:
        return self.sample_id


@dataclass(frozen=True, slots=True)
class PredictorReferenceChange:
    """Explicit nominal-reference replacement at one monotonic timestamp."""

    change_id: str
    t_s: float
    old_rate_rad_s: float
    new_rate_rad_s: float

    @property
    def event_id(self) -> str:
        return self.change_id


@dataclass(frozen=True, slots=True)
class VisionObservation:
    """Timestamped relative elevation-error observation."""

    frame_id: str
    t_s: float
    error_rad: float
    measurement_variance_rad2: float

    @property
    def event_id(self) -> str:
        return self.frame_id

    @property
    def z_v(self) -> float:
        """Compatibility alias for the previous measurement spelling."""

        return self.error_rad


type ResidualEvent = (
    EncoderSample | NominalRateSample | PredictorReferenceChange | VisionObservation
)


class ObservationDisposition(StrEnum):
    """Disposition recorded for an event submission or replay."""

    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    FUTURE = "future"
    EXPIRED = "expired"
    NO_ENCODER_BRACKET = "no_encoder_bracket"
    INVALID_ENCODER = "invalid_encoder"
    STALE_TIMING = "stale_timing"


@dataclass(frozen=True, slots=True)
class EventDisposition:
    """Stable event ID paired with a disposition."""

    event_id: str
    disposition: ObservationDisposition


@dataclass(frozen=True, slots=True)
class ResidualCheckpoint:
    """Posterior anchor from which raw events are replayed."""

    t_s: float
    state: ResidualState
    encoder_angle_rad: float | None
    encoder_endpoint_variance_rad2: float
    nominal_reference_rate_rad_s: float


@dataclass(frozen=True, slots=True)
class ResidualHistory:
    """Immutable event history and a preserved posterior checkpoint."""

    checkpoint: ResidualCheckpoint
    events: tuple[ResidualEvent, ...] = ()
    horizon_s: float = 0.10
    reversal_threshold_rad: float = 0.0
    reversal_variance_rad2: float = 0.0
    interpolation_span_max_s: float = math.inf
    seen_event_ids: tuple[str, ...] = ()

    @staticmethod
    def initial(
        state: ResidualState,
        *,
        t_s: float = 0.0,
        horizon_s: float = 0.10,
        reversal_threshold_rad: float = 0.0,
        reversal_variance_rad2: float = 0.0,
        interpolation_span_max_s: float = math.inf,
        encoder_angle_rad: float | None = None,
        encoder_endpoint_variance_rad2: float = 0.0,
        nominal_reference_rate_rad_s: float = 0.0,
    ) -> ResidualHistory:
        """Construct a cold history around ``state``."""

        return ResidualHistory(
            checkpoint=ResidualCheckpoint(
                t_s=t_s,
                state=state,
                encoder_angle_rad=encoder_angle_rad,
                encoder_endpoint_variance_rad2=encoder_endpoint_variance_rad2,
                nominal_reference_rate_rad_s=nominal_reference_rate_rad_s,
            ),
            horizon_s=horizon_s,
            reversal_threshold_rad=reversal_threshold_rad,
            reversal_variance_rad2=reversal_variance_rad2,
            interpolation_span_max_s=interpolation_span_max_s,
        )

    @property
    def encoder_samples(self) -> tuple[EncoderSample, ...]:
        return tuple(e for e in self.events if isinstance(e, EncoderSample))

    @property
    def nominal_rate_samples(self) -> tuple[NominalRateSample, ...]:
        return tuple(e for e in self.events if isinstance(e, NominalRateSample))

    @property
    def reference_changes(self) -> tuple[PredictorReferenceChange, ...]:
        return tuple(e for e in self.events if isinstance(e, PredictorReferenceChange))

    @property
    def vision_observations(self) -> tuple[VisionObservation, ...]:
        return tuple(e for e in self.events if isinstance(e, VisionObservation))


@dataclass(frozen=True, slots=True)
class EstimateResult:
    """Pure replay result; the returned history may contain a trim checkpoint."""

    state: ResidualState
    history: ResidualHistory
    dispositions: tuple[EventDisposition, ...]
    t_s: float

    def __iter__(self) -> Iterator[object]:
        """Allow ``state, dispositions = estimate_at(...)``."""

        yield self.state
        yield self.dispositions


@dataclass(frozen=True, slots=True)
class ResidualFilter:
    """Configured two-state filter with continuous acceleration noise."""

    q_a: float
    r_v: float
    p0_11: float
    p0_22: float
    dt_outer_s: float = 0.0
    rewind_horizon_s: float = 0.10
    reversal_threshold_rad: float = 0.0
    reversal_variance_rad2: float = 0.0
    interpolation_span_max_s: float = math.inf

    @staticmethod
    def from_config(cfg: ResidualConfig, dt_outer_s: float) -> ResidualFilter:
        """Build the continuous-noise estimator from typed chronology settings."""

        return ResidualFilter(
            q_a=float(cfg.q_accel_rad2_s3),
            r_v=float(cfg.R_v),
            p0_11=float(cfg.P0_diag[0]),
            p0_22=float(cfg.P0_diag[1]),
            dt_outer_s=float(dt_outer_s),
            rewind_horizon_s=float(cfg.rewind_horizon_s),
            reversal_threshold_rad=float(cfg.reversal_threshold_rad),
            reversal_variance_rad2=float(cfg.reversal_variance_rad2),
            interpolation_span_max_s=float(cfg.interpolation_span_max_s),
        )

    @property
    def q11(self) -> float:
        """Legacy covariance value at the configured outer period."""

        return self.q_a * self.dt_outer_s**3 / 3.0

    @property
    def q22(self) -> float:
        """Legacy rate covariance value at the configured outer period."""

        return self.q_a * self.dt_outer_s

    def initial_state(self) -> ResidualState:
        """Cold state with configured diagonal covariance."""

        p = np.array([[self.p0_11, 0.0], [0.0, self.p0_22]], dtype=np.float64)
        return ResidualState(x=np.zeros(2, dtype=np.float64), P=p, has_measurement=False)

    def initial_history(
        self,
        *,
        t_s: float = 0.0,
        encoder_angle_rad: float | None = None,
        encoder_endpoint_variance_rad2: float = 0.0,
        nominal_reference_rate_rad_s: float = 0.0,
    ) -> ResidualHistory:
        """Construct a history using this filter's horizon/noise settings."""

        return ResidualHistory.initial(
            self.initial_state(),
            t_s=t_s,
            horizon_s=self.rewind_horizon_s,
            reversal_threshold_rad=self.reversal_threshold_rad,
            reversal_variance_rad2=self.reversal_variance_rad2,
            interpolation_span_max_s=self.interpolation_span_max_s,
            encoder_angle_rad=encoder_angle_rad,
            encoder_endpoint_variance_rad2=encoder_endpoint_variance_rad2,
            nominal_reference_rate_rad_s=nominal_reference_rate_rad_s,
        )


def _symmetrize(p: np.ndarray) -> np.ndarray:
    return np.asarray(0.5 * (p + p.T), dtype=np.float64)


def _q(filt: ResidualFilter, dt_s: float) -> np.ndarray:
    """Continuous white-acceleration covariance ``q_a[[dt^3/3,...]]``."""

    if dt_s < -_EPS:
        raise ValueError("negative propagation interval")
    dt = max(0.0, dt_s)
    return filt.q_a * np.array([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]], dtype=np.float64)


def propagate_displacement(
    filt: ResidualFilter,
    state: ResidualState,
    dt_s: float,
    nominal_angle_delta_rad: float,
    encoder_angle_delta_rad: float,
    encoder_variance_rad2: float = 0.0,
    reversal_variance_rad2: float = 0.0,
    *,
    add_q: bool = True,
) -> ResidualState:
    """Propagate with a measured encoder displacement and explicit variances."""

    if dt_s < -_EPS or not math.isfinite(dt_s):
        raise ValueError("propagation interval must be finite and non-negative")
    if encoder_variance_rad2 < -_EPS or reversal_variance_rad2 < -_EPS:
        raise ValueError("encoder variances must be non-negative")
    f = np.array([[1.0, dt_s], [0.0, 1.0]], dtype=np.float64)
    u = np.array([nominal_angle_delta_rad - encoder_angle_delta_rad, 0.0], dtype=np.float64)
    x_pred = f @ state.x + u
    p_pred = f @ state.P @ f.T
    if add_q:
        p_pred += _q(filt, dt_s)
    p_pred[0, 0] += max(0.0, encoder_variance_rad2) + max(0.0, reversal_variance_rad2)
    return ResidualState(x=x_pred, P=_symmetrize(p_pred), has_measurement=state.has_measurement)


def predict(
    filt: ResidualFilter,
    state: ResidualState,
    dt_s: float,
    omega_t_nom: float,
    y_m: float,
    add_q: bool = True,
) -> ResidualState:
    """Compatibility wrapper using rate inputs as noiseless displacement."""

    if dt_s >= -_EPS:
        return propagate_displacement(
            filt, state, max(0.0, dt_s), dt_s * omega_t_nom, dt_s * y_m, add_q=add_q
        )
    # Kept only for the old snapshot API; the new replay path never rewinds.
    f = np.array([[1.0, dt_s], [0.0, 1.0]], dtype=np.float64)
    u = np.array([dt_s * (omega_t_nom - y_m), 0.0], dtype=np.float64)
    return ResidualState(
        x=f @ state.x + u,
        P=_symmetrize(f @ state.P @ f.T),
        has_measurement=state.has_measurement,
    )


def update(
    filt: ResidualFilter,
    state: ResidualState,
    z_v: float,
    measurement_variance_rad2: float | None = None,
) -> ResidualState:
    """Joseph-form elevation update; first vision initializes ``e`` directly."""

    r = filt.r_v if measurement_variance_rad2 is None else measurement_variance_rad2
    if r < 0.0 or not math.isfinite(r):
        raise ValueError("measurement variance must be finite and non-negative")
    if not state.has_measurement:
        x = state.x.copy()
        x[0] = z_v
        p = state.P.copy()
        p[0, 0] = r
        p[0, 1] = 0.0
        p[1, 0] = 0.0
        return ResidualState(x=x, P=_symmetrize(p), has_measurement=True)
    h = np.array([[1.0, 0.0]], dtype=np.float64)
    p = _symmetrize(state.P)
    s = float((h @ p @ h.T)[0, 0] + r)
    if s <= 1.0e-30:
        return replace(state, has_measurement=True, P=p)
    k = (p @ h.T)[:, 0] / s
    x_upd = state.x + k * (z_v - float(state.x[0]))
    ikh = np.eye(2, dtype=np.float64) - np.outer(k, h[0])
    p_upd = ikh @ p @ ikh.T + np.outer(k, k) * r
    return ResidualState(x=x_upd, P=_symmetrize(p_upd), has_measurement=True)


def _event_time(event: ResidualEvent) -> float:
    return event.t_s


def _event_id(event: ResidualEvent) -> str:
    return event.event_id


def _event_priority(event: ResidualEvent) -> int:
    if isinstance(event, PredictorReferenceChange):
        return 1
    if isinstance(event, EncoderSample):
        return 2
    if isinstance(event, VisionObservation):
        return 3
    return 0  # NominalRateSample: update active ZOH after propagating to t.


def _ordered_events(events: tuple[ResidualEvent, ...]) -> tuple[ResidualEvent, ...]:
    return tuple(sorted(events, key=lambda e: (_event_time(e), _event_priority(e), _event_id(e))))


def _valid_encoder(sample: EncoderSample) -> bool:
    return (
        bool(sample.sample_id)
        and math.isfinite(sample.t_s)
        and math.isfinite(sample.angle_rad)
        and math.isfinite(sample.angle_variance_rad2)
        and sample.angle_variance_rad2 >= 0.0
    )


@dataclass(frozen=True, slots=True)
class _EncoderEndpoint:
    angle_rad: float
    variance_rad2: float
    t_s: float
    sample_ids: tuple[str, ...]


def _interpolate_encoder(history: ResidualHistory, t_s: float) -> _EncoderEndpoint | None:
    """Interpolate only from exact/bracketing valid samples."""

    samples = sorted(
        (s for s in history.encoder_samples if _valid_encoder(s)),
        key=lambda s: (s.t_s, s.sample_id),
    )
    cp = history.checkpoint
    if cp.encoder_angle_rad is not None and abs(t_s - cp.t_s) <= _EPS:
        return _EncoderEndpoint(
            cp.encoder_angle_rad, cp.encoder_endpoint_variance_rad2, t_s, ("checkpoint",)
        )
    exact = [s for s in samples if abs(s.t_s - t_s) <= _EPS]
    if exact:
        s = exact[0]
        return _EncoderEndpoint(s.angle_rad, s.angle_variance_rad2, t_s, (s.sample_id,))
    before = [s for s in samples if s.t_s < t_s - _EPS]
    after = [s for s in samples if s.t_s > t_s + _EPS]
    if cp.encoder_angle_rad is not None and cp.t_s < t_s - _EPS:
        before.append(
            EncoderSample(
                "checkpoint", cp.t_s, cp.encoder_angle_rad, cp.encoder_endpoint_variance_rad2
            )
        )
        before.sort(key=lambda s: (s.t_s, s.sample_id))
    if not before or not after:
        return None
    left, right = before[-1], after[0]
    span = right.t_s - left.t_s
    if span <= _EPS or span > history.interpolation_span_max_s + _EPS:
        return None
    alpha = (t_s - left.t_s) / span
    angle = (1.0 - alpha) * left.angle_rad + alpha * right.angle_rad
    variance = (1.0 - alpha) ** 2 * left.angle_variance_rad2 + alpha**2 * right.angle_variance_rad2
    return _EncoderEndpoint(angle, variance, t_s, (left.sample_id, right.sample_id))


def _reversal_count(
    history: ResidualHistory,
    start_t_s: float,
    end_t_s: float,
    previous_angle_rad: float | None,
) -> int:
    points: list[tuple[float, float]] = []
    if previous_angle_rad is not None:
        predecessors = [
            s for s in history.encoder_samples if _valid_encoder(s) and s.t_s < start_t_s - _EPS
        ]
        if predecessors:
            predecessor = max(predecessors, key=lambda s: (s.t_s, s.sample_id))
            points.append((predecessor.t_s, predecessor.angle_rad))
        points.append((start_t_s, previous_angle_rad))
    points.extend(
        (s.t_s, s.angle_rad)
        for s in sorted(history.encoder_samples, key=lambda s: (s.t_s, s.sample_id))
        if _valid_encoder(s) and start_t_s < s.t_s <= end_t_s + _EPS
    )
    direction = 0
    reversals = 0
    threshold = max(0.0, history.reversal_threshold_rad)
    for (_t0, a0), (_t1, a1) in zip(points, points[1:]):
        delta = a1 - a0
        if abs(delta) <= threshold:
            continue
        current = 1 if delta > 0.0 else -1
        if direction and current != direction:
            reversals += 1
        direction = current
    return reversals


def _replay(
    history: ResidualHistory,
    filt: ResidualFilter,
    endpoint_t_s: float,
    *,
    collect_dispositions: bool,
) -> tuple[ResidualState, float, _EncoderEndpoint | None, tuple[EventDisposition, ...]]:
    """Replay all events through an endpoint from the stable checkpoint."""

    cp = history.checkpoint
    state = cp.state
    cursor_t = cp.t_s
    active_rate = cp.nominal_reference_rate_rad_s
    previous_endpoint: _EncoderEndpoint | None = None
    charged_reversals = 0
    dispositions: list[EventDisposition] = []

    def advance(t_s: float, endpoint: _EncoderEndpoint | None) -> None:
        nonlocal state, cursor_t, previous_endpoint, charged_reversals
        if t_s < cursor_t - _EPS:
            return
        dt = max(0.0, t_s - cursor_t)
        if endpoint is None:
            state = propagate_displacement(filt, state, dt, active_rate * dt, 0.0)
        else:
            if previous_endpoint is None:
                anchor = (
                    cp.encoder_angle_rad if cp.encoder_angle_rad is not None else endpoint.angle_rad
                )
                delta = endpoint.angle_rad - anchor
                variance = endpoint.variance_rad2 + (
                    cp.encoder_endpoint_variance_rad2 if cp.encoder_angle_rad is not None else 0.0
                )
            elif abs(endpoint.t_s - previous_endpoint.t_s) <= _EPS:
                delta, variance = 0.0, 0.0
            else:
                delta = endpoint.angle_rad - previous_endpoint.angle_rad
                variance = endpoint.variance_rad2 + previous_endpoint.variance_rad2
            cumulative_reversals = _reversal_count(
                history,
                cp.t_s,
                t_s,
                cp.encoder_angle_rad,
            )
            reversals = max(0, cumulative_reversals - charged_reversals)
            charged_reversals = cumulative_reversals
            state = propagate_displacement(
                filt,
                state,
                dt,
                active_rate * dt,
                delta,
                variance,
                reversals * history.reversal_variance_rad2,
            )
            previous_endpoint = endpoint
        cursor_t = t_s

    for event in _ordered_events(history.events):
        event_t = _event_time(event)
        if event_t > endpoint_t_s + _EPS:
            if collect_dispositions:
                dispositions.append(
                    EventDisposition(_event_id(event), ObservationDisposition.FUTURE)
                )
            continue
        if event_t < cp.t_s - _EPS:
            if collect_dispositions:
                dispositions.append(
                    EventDisposition(_event_id(event), ObservationDisposition.EXPIRED)
                )
            continue
        if isinstance(event, VisionObservation):
            endpoint = _interpolate_encoder(history, event.t_s)
            if endpoint is None:
                if collect_dispositions:
                    dispositions.append(
                        EventDisposition(
                            _event_id(event), ObservationDisposition.NO_ENCODER_BRACKET
                        )
                    )
                continue
            advance(event.t_s, endpoint)
            state = update(filt, state, event.error_rad, event.measurement_variance_rad2)
            if collect_dispositions:
                dispositions.append(
                    EventDisposition(_event_id(event), ObservationDisposition.ACCEPTED)
                )
        elif isinstance(event, PredictorReferenceChange):
            advance(event_t, None)
            state = replace(
                state,
                x=state.x + np.array([0.0, event.old_rate_rad_s - event.new_rate_rad_s]),
            )
            active_rate = event.new_rate_rad_s
        elif isinstance(event, NominalRateSample):
            advance(event_t, None)
            active_rate = event.rate_rad_s
        # EncoderSample is retained raw and establishes endpoints only when a
        # vision/reporting timestamp is replayed.

    requested_endpoint = _interpolate_encoder(history, endpoint_t_s)
    if requested_endpoint is not None:
        advance(endpoint_t_s, requested_endpoint)
    return state, active_rate, requested_endpoint, tuple(dispositions)


def submit_event(
    history: ResidualHistory,
    event: ResidualEvent,
    *,
    now_s: float | None = None,
) -> tuple[ResidualHistory, ObservationDisposition]:
    """Purely submit one event, deduplicating by stable event identity."""

    event_id = _event_id(event)
    known = set(history.seen_event_ids)
    known.update(_event_id(e) for e in history.events)
    if event_id in known:
        return history, ObservationDisposition.DUPLICATE
    if isinstance(event, EncoderSample) and not _valid_encoder(event):
        return history, ObservationDisposition.INVALID_ENCODER
    if not math.isfinite(event.t_s):
        return history, ObservationDisposition.STALE_TIMING
    if isinstance(event, NominalRateSample) and not math.isfinite(event.rate_rad_s):
        return history, ObservationDisposition.STALE_TIMING
    if isinstance(event, PredictorReferenceChange) and (
        not math.isfinite(event.old_rate_rad_s) or not math.isfinite(event.new_rate_rad_s)
    ):
        return history, ObservationDisposition.STALE_TIMING
    if isinstance(event, VisionObservation) and (
        not math.isfinite(event.error_rad)
        or not math.isfinite(event.measurement_variance_rad2)
        or event.measurement_variance_rad2 < 0.0
    ):
        return history, ObservationDisposition.STALE_TIMING
    if now_s is not None and event.t_s > now_s + _EPS:
        return history, ObservationDisposition.FUTURE
    if event.t_s < history.checkpoint.t_s - _EPS:
        return history, ObservationDisposition.EXPIRED
    if now_s is not None and event.t_s < now_s - history.horizon_s - _EPS:
        return history, ObservationDisposition.EXPIRED
    events = history.events + (event,)
    if len(events) > 4096:
        events = events[-4096:]
    seen = (history.seen_event_ids + (event_id,))[-4096:]
    return replace(history, events=events, seen_event_ids=seen), ObservationDisposition.ACCEPTED


def estimate_at(history: ResidualHistory, filt: ResidualFilter, t_s: float) -> EstimateResult:
    """Replay deterministically to ``t_s`` and return estimate/dispositions."""

    if not math.isfinite(t_s):
        raise ValueError("estimate time must be finite")
    if t_s < history.checkpoint.t_s - _EPS:
        return EstimateResult(history.checkpoint.state, history, (), t_s)
    state, _rate, _endpoint, dispositions = _replay(history, filt, t_s, collect_dispositions=True)
    updated = history
    boundary = t_s - max(0.0, history.horizon_s)
    if boundary > history.checkpoint.t_s + _EPS:
        boundary_state, boundary_rate, boundary_endpoint, _ = _replay(
            history, filt, boundary, collect_dispositions=False
        )
        if boundary_endpoint is not None:
            cp = ResidualCheckpoint(
                boundary,
                boundary_state,
                boundary_endpoint.angle_rad,
                boundary_endpoint.variance_rad2,
                boundary_rate,
            )
            old_encoder = [
                e for e in history.encoder_samples if _valid_encoder(e) and e.t_s < boundary - _EPS
            ]
            predecessor: ResidualEvent | None = (
                max(old_encoder, key=lambda e: (e.t_s, e.sample_id)) if old_encoder else None
            )
            kept = tuple(e for e in history.events if e is predecessor or e.t_s > boundary + _EPS)
            updated = replace(updated, checkpoint=cp, events=kept)
    return EstimateResult(state, updated, dispositions, t_s)


# Compatibility snapshot API retained for the detailed plant transition.


@dataclass(frozen=True, slots=True)
class ResidualSnapshot:
    """Legacy outer-tick snapshot for old callers."""

    t_s: float
    state: ResidualState
    dt_s: float
    omega_t_nom: float
    y_m: float


def rewind_update(
    filt: ResidualFilter,
    snapshots: tuple[ResidualSnapshot, ...],
    current: ResidualState,
    now: float,
    t_s: float,
    z_v: float,
    horizon_s: float,
) -> ResidualState:
    """Legacy rewind implementation for old detailed-plant callers."""

    if now - t_s > horizon_s or t_s > now + _EPS:
        return current
    if not snapshots:
        pred = current
        if t_s < now - _EPS:
            pred = predict(filt, current, t_s - now, 0.0, 0.0, add_q=False)
            return predict(filt, update(filt, pred, z_v), now - t_s, 0.0, 0.0)
        return update(filt, pred, z_v)
    usable = tuple(s for s in snapshots if now - s.t_s <= horizon_s + _EPS)
    if not usable:
        pred = predict(filt, current, t_s - now, 0.0, 0.0, add_q=False)
        return predict(filt, update(filt, pred, z_v), now - t_s, 0.0, 0.0)
    idx = 0
    found = False
    for i, snap in enumerate(usable):
        if snap.t_s <= t_s + _EPS:
            idx, found = i, True
    if not found:
        idx = 0
    snap = usable[idx]
    pred = snap.state
    dt_to_meas = t_s - snap.t_s
    if abs(dt_to_meas) > _EPS:
        pred = predict(filt, pred, dt_to_meas, snap.omega_t_nom, snap.y_m, add_q=dt_to_meas > 0.0)
    posterior = update(filt, pred, z_v)
    replay_from = t_s
    for later in usable[idx + 1 :]:
        dt = later.t_s - replay_from
        if dt > _EPS:
            posterior = predict(filt, posterior, dt, later.omega_t_nom, later.y_m)
        replay_from = later.t_s
    dt_tail = now - replay_from
    if dt_tail > _EPS:
        last = usable[-1]
        posterior = predict(filt, posterior, dt_tail, last.omega_t_nom, last.y_m)
    return posterior


def push_snapshot(
    snapshots: tuple[ResidualSnapshot, ...], snap: ResidualSnapshot, max_count: int
) -> tuple[ResidualSnapshot, ...]:
    """Append a legacy snapshot and retain the newest entries."""

    combined = snapshots + (snap,)
    return combined if len(combined) <= max_count else combined[-max_count:]
