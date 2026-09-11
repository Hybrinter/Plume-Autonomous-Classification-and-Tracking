"""Independent chronological residual oracle for estimator qualification tests.

This module deliberately owns its state transition equations.  It consumes the
analysis timeline event records, but does not import or call the flight
estimator's history, propagation, update, interpolation, or trimming code.
The oracle is intentionally small and deterministic so production replay can be
compared against it after each history trim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis.studies.tracking_estimator.model import (
    EncoderEvent,
    EventKind,
    PredictorTransitionEvent,
    PropagationEvent,
    TimelineEvent,
    VisionEvent,
    continuous_kinematic_noise,
)


def _event_key(event: TimelineEvent) -> tuple[float, int, str]:
    """Return the deterministic physical-time ordering for an event."""
    priority = {
        EventKind.PROPAGATE: 0,
        EventKind.PREDICTOR_TRANSITION: 1,
        EventKind.ENCODER: 2,
        EventKind.VISION: 3,
    }
    return event.t_s, priority[event.kind], event.event_id


@dataclass(frozen=True, slots=True)
class OracleState:
    """Residual posterior at an exact physical-time endpoint."""

    t_s: float
    x: np.ndarray
    covariance: np.ndarray
    nominal_rate_rad_s: float


def replay_residual(
    events: tuple[TimelineEvent, ...],
    *,
    start: OracleState | None = None,
    accel_psd_rad2_s3: float = 3.0e-7,
    vision_variance_rad2: float = 2.5e-7,
) -> OracleState:
    """Replay residual equations from ``start`` or the zero prior.

    Propagation events use the nominal rate held by the prior event (a
    zero-order hold) unless they explicitly provide a new value.  A predictor
    transition is different: it rebases the residual rate so total target rate
    remains continuous.  Encoder events are retained in the timeline for
    causal ordering but do not become measurements in the two-state observer.
    """
    if start is None:
        t_s = 0.0
        x = np.zeros(2, dtype=np.float64)
        covariance = np.diag(np.array([1.0e-3, 1.0e-4], dtype=np.float64))
        nominal = 0.0
    else:
        t_s = start.t_s
        x = start.x.copy()
        covariance = start.covariance.copy()
        nominal = start.nominal_rate_rad_s

    first_propagation = True
    for event in sorted(events, key=_event_key):
        if event.t_s < t_s - 1.0e-12:
            raise ValueError("oracle events precede the checkpoint")
        if isinstance(event, PropagationEvent):
            dt_s = event.t_s - t_s
            next_nominal = (
                nominal
                if event.nominal_target_rate_rad_s is None
                else event.nominal_target_rate_rad_s
            )
            # The first sample establishes the reference.  Subsequent sampled
            # changes are represented in residual coordinates, not as physical
            # target-rate impulses.
            if not first_propagation:
                x[1] += nominal - (next_nominal or 0.0)
            nominal = next_nominal or 0.0
            f, q = continuous_kinematic_noise(dt_s, accel_psd_rad2_s3)
            x = f @ x + np.array([dt_s * (nominal - event.encoder_rate_rad_s), 0.0])
            covariance = f @ covariance @ f.T + q
            t_s = event.t_s
            first_propagation = False
        elif isinstance(event, PredictorTransitionEvent):
            old = nominal if event.previous_rate_rad_s is None else event.previous_rate_rad_s
            new = 0.0 if event.next_rate_rad_s is None else event.next_rate_rad_s
            x[1] += old - new
            nominal = new
            t_s = max(t_s, event.t_s)
        elif isinstance(event, VisionEvent):
            h = np.array([1.0, 0.0], dtype=np.float64)
            innovation_cov = float(h @ covariance @ h.T + vision_variance_rad2)
            gain = (covariance @ h) / innovation_cov
            identity = np.eye(2, dtype=np.float64)
            x = x + gain * (event.relative_angle_rad - float(h @ x))
            kh = np.outer(gain, h)
            covariance = (identity - kh) @ covariance @ (identity - kh).T
            covariance += np.outer(gain, gain) * vision_variance_rad2
            covariance = 0.5 * (covariance + covariance.T)
        elif isinstance(event, EncoderEvent):
            # Physical encoder displacement is a known input to this observer;
            # it is carried by the production event timeline but never updated
            # as a second residual measurement.
            continue
    return OracleState(t_s, x, covariance, nominal)


class CheckpointedOracle:
    """Bounded event journal with a stable posterior checkpoint.

    The class is test infrastructure, not a flight implementation.  Trimming
    replays through the new boundary and stores that posterior before deleting
    old events.  Tests can therefore exercise repeated trims without silently
    cold-starting from the original prior.
    """

    def __init__(self, horizon_s: float) -> None:
        if horizon_s <= 0.0:
            raise ValueError("horizon_s must be positive")
        self.horizon_s = horizon_s
        self._checkpoint = OracleState(
            0.0,
            np.zeros(2, dtype=np.float64),
            np.diag(np.array([1.0e-3, 1.0e-4], dtype=np.float64)),
            0.0,
        )
        self._events: list[TimelineEvent] = []
        self._ids: set[str] = set()
        self._now_s = 0.0
        self.duplicate_ids: list[str] = []
        self.expired_ids: list[str] = []

    @property
    def checkpoint(self) -> OracleState:
        """Return a defensive copy of the trim checkpoint."""
        return OracleState(
            self._checkpoint.t_s,
            self._checkpoint.x.copy(),
            self._checkpoint.covariance.copy(),
            self._checkpoint.nominal_rate_rad_s,
        )

    def submit(self, event: TimelineEvent, arrival_s: float) -> bool:
        """Record an event and trim at ``arrival_s - horizon``."""
        if event.event_id in self._ids:
            self.duplicate_ids.append(event.event_id)
            return False
        if arrival_s - event.t_s > self.horizon_s + 1.0e-12:
            self.expired_ids.append(event.event_id)
            return False
        self._ids.add(event.event_id)
        self._events.append(event)
        self._now_s = max(self._now_s, arrival_s)
        self._trim()
        return True

    def _trim(self) -> None:
        boundary = self._now_s - self.horizon_s
        old = tuple(event for event in self._events if event.t_s <= boundary + 1.0e-12)
        if old:
            self._checkpoint = replay_residual(old, start=self._checkpoint)
            self._events = [event for event in self._events if event.t_s > boundary + 1.0e-12]

    def estimate(self) -> OracleState:
        """Replay retained events from the stable checkpoint."""
        return replay_residual(tuple(self._events), start=self._checkpoint)


# ---------------------------------------------------------------------------
# Anchored encoder-increment oracle used by estimator qualification tests.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AnchoredOracleCheckpoint:
    """Independent posterior plus encoder/nominal anchors."""

    t_s: float
    x: np.ndarray
    covariance: np.ndarray
    encoder_angle_rad: float | None
    encoder_variance_rad2: float
    nominal_rate_rad_s: float


@dataclass(frozen=True, slots=True)
class AnchoredOracleEstimate:
    """Result of replaying an anchored oracle journal."""

    state: AnchoredOracleCheckpoint
    accepted_vision_ids: tuple[str, ...]


def _anchored_event_key(event: TimelineEvent) -> tuple[float, int, str]:
    """Sort physical events with explicit causal equal-time ordering."""

    priority = {
        EventKind.PROPAGATE: 0,
        EventKind.PREDICTOR_TRANSITION: 1,
        EventKind.ENCODER: 2,
        EventKind.VISION: 3,
    }
    return event.t_s, priority[event.kind], event.event_id


@dataclass(frozen=True, slots=True)
class _OracleEndpoint:
    angle_rad: float
    variance_rad2: float
    t_s: float


def _oracle_interpolate(
    events: tuple[TimelineEvent, ...], checkpoint: AnchoredOracleCheckpoint, t_s: float
) -> _OracleEndpoint | None:
    """Linearly interpolate only from exact or valid encoder brackets."""

    encoder = sorted(
        (item for item in events if isinstance(item, EncoderEvent)),
        key=lambda item: (item.t_s, item.event_id),
    )
    if checkpoint.encoder_angle_rad is not None and abs(t_s - checkpoint.t_s) <= 1e-12:
        return _OracleEndpoint(
            checkpoint.encoder_angle_rad,
            checkpoint.encoder_variance_rad2,
            t_s,
        )
    exact = [item for item in encoder if abs(item.t_s - t_s) <= 1e-12]
    if exact:
        return _OracleEndpoint(exact[0].angle_rad, 1.0e-7, t_s)
    before = [item for item in encoder if item.t_s < t_s - 1e-12]
    after = [item for item in encoder if item.t_s > t_s + 1e-12]
    if checkpoint.encoder_angle_rad is not None and checkpoint.t_s < t_s - 1e-12:
        before.append(EncoderEvent("checkpoint", checkpoint.t_s, checkpoint.encoder_angle_rad))
        before.sort(key=lambda item: (item.t_s, item.event_id))
    if not before or not after:
        return None
    left, right = before[-1], after[0]
    span = right.t_s - left.t_s
    if span <= 1e-12:
        return None
    alpha = (t_s - left.t_s) / span
    angle = (1.0 - alpha) * left.angle_rad + alpha * right.angle_rad
    left_variance = checkpoint.encoder_variance_rad2 if left.event_id == "checkpoint" else 1.0e-7
    variance = (1.0 - alpha) ** 2 * left_variance + alpha**2 * 1.0e-7
    return _OracleEndpoint(angle, variance, t_s)


def _oracle_reversals(
    events: tuple[TimelineEvent, ...],
    start_t_s: float,
    end_t_s: float,
    previous_angle: float | None,
) -> int:
    """Count nonzero direction reversals in one raw encoder span."""

    encoder_events = sorted(
        (item for item in events if isinstance(item, EncoderEvent)),
        key=lambda item: (item.t_s, item.event_id),
    )
    points: list[float] = []
    if previous_angle is not None:
        predecessors = [item for item in encoder_events if item.t_s < start_t_s - 1e-12]
        if predecessors:
            points.append(predecessors[-1].angle_rad)
        points.append(previous_angle)
    points.extend(
        item.angle_rad for item in encoder_events if start_t_s < item.t_s <= end_t_s + 1e-12
    )
    direction = 0
    count = 0
    for before, after in zip(points, points[1:]):
        delta = after - before
        if abs(delta) <= 1.0e-8:
            continue
        next_direction = 1 if delta > 0.0 else -1
        if direction and next_direction != direction:
            count += 1
        direction = next_direction
    return count


def replay_anchored(
    events: tuple[TimelineEvent, ...],
    *,
    endpoint_t_s: float,
    checkpoint: AnchoredOracleCheckpoint | None = None,
    accel_psd_rad2_s3: float = 3.0e-7,
    vision_variance_rad2: float = 2.5e-7,
    reversal_variance_rad2: float = 0.0,
) -> AnchoredOracleEstimate:
    """Replay an independent two-state anchored residual model.

    This function intentionally duplicates the equations instead of importing
    production ``ResidualFilter``, ``ResidualHistory``, interpolation,
    propagation, update, or trimming functions.
    """

    if checkpoint is None:
        checkpoint = AnchoredOracleCheckpoint(
            0.0,
            np.zeros(2, dtype=np.float64),
            np.diag(np.array([1.0e-3, 1.0e-4], dtype=np.float64)),
            0.0,
            0.0,
            0.0,
        )
    x = checkpoint.x.copy()
    covariance = checkpoint.covariance.copy()
    cursor = checkpoint.t_s
    nominal = checkpoint.nominal_rate_rad_s
    previous_endpoint: _OracleEndpoint | None = None
    charged_reversals = 0
    accepted: list[str] = []

    def advance(t_s: float, endpoint: _OracleEndpoint | None) -> None:
        nonlocal x, covariance, cursor, previous_endpoint, charged_reversals
        if t_s < cursor - 1.0e-12:
            return
        dt = max(0.0, t_s - cursor)
        f = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        q = accel_psd_rad2_s3 * np.array(
            [[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]], dtype=np.float64
        )
        delta = 0.0
        variance = 0.0
        if endpoint is not None:
            if previous_endpoint is None:
                anchor = (
                    checkpoint.encoder_angle_rad
                    if checkpoint.encoder_angle_rad is not None
                    else endpoint.angle_rad
                )
                delta = endpoint.angle_rad - anchor
                variance = endpoint.variance_rad2 + (
                    checkpoint.encoder_variance_rad2
                    if checkpoint.encoder_angle_rad is not None
                    else 0.0
                )
            elif abs(endpoint.t_s - previous_endpoint.t_s) > 1.0e-12:
                delta = endpoint.angle_rad - previous_endpoint.angle_rad
                variance = endpoint.variance_rad2 + previous_endpoint.variance_rad2
            cumulative_reversals = _oracle_reversals(
                events,
                checkpoint.t_s,
                t_s,
                checkpoint.encoder_angle_rad,
            )
            reversals = max(0, cumulative_reversals - charged_reversals)
            charged_reversals = cumulative_reversals
            variance += reversals * reversal_variance_rad2
            previous_endpoint = endpoint
        x = f @ x + np.array([nominal * dt - delta, 0.0], dtype=np.float64)
        covariance = f @ covariance @ f.T + q
        covariance[0, 0] += variance
        covariance = 0.5 * (covariance + covariance.T)
        cursor = t_s

    for event in sorted(events, key=_anchored_event_key):
        if event.t_s > endpoint_t_s + 1.0e-12 or event.t_s < checkpoint.t_s - 1.0e-12:
            continue
        if isinstance(event, PropagationEvent):
            advance(event.t_s, None)
            if event.nominal_target_rate_rad_s is not None:
                nominal = event.nominal_target_rate_rad_s
        elif isinstance(event, PredictorTransitionEvent):
            advance(event.t_s, None)
            old = nominal if event.previous_rate_rad_s is None else event.previous_rate_rad_s
            new = 0.0 if event.next_rate_rad_s is None else event.next_rate_rad_s
            x[1] += old - new
            nominal = new
        elif isinstance(event, VisionEvent):
            endpoint = _oracle_interpolate(events, checkpoint, event.t_s)
            if endpoint is None:
                continue
            advance(event.t_s, endpoint)
            h = np.array([1.0, 0.0], dtype=np.float64)
            s = float(h @ covariance @ h.T + vision_variance_rad2)
            gain = (covariance @ h) / s
            x = x + gain * (event.relative_angle_rad - float(h @ x))
            identity = np.eye(2, dtype=np.float64)
            kh = np.outer(gain, h)
            covariance = (identity - kh) @ covariance @ (identity - kh).T
            covariance += np.outer(gain, gain) * vision_variance_rad2
            covariance = 0.5 * (covariance + covariance.T)
            accepted.append(event.event_id)

    endpoint = _oracle_interpolate(events, checkpoint, endpoint_t_s)
    if endpoint is not None:
        advance(endpoint_t_s, endpoint)
    state = AnchoredOracleCheckpoint(
        cursor,
        x,
        covariance,
        endpoint.angle_rad if endpoint is not None else checkpoint.encoder_angle_rad,
        endpoint.variance_rad2 if endpoint is not None else checkpoint.encoder_variance_rad2,
        nominal,
    )
    return AnchoredOracleEstimate(state, tuple(accepted))


class AnchoredOracleJournal:
    """Independent mutable journal used only by qualification tests."""

    def __init__(
        self,
        horizon_s: float,
        checkpoint: AnchoredOracleCheckpoint,
        *,
        reversal_variance_rad2: float = 0.0,
    ) -> None:
        self.horizon_s = horizon_s
        self.checkpoint = checkpoint
        self.reversal_variance_rad2 = reversal_variance_rad2
        self.events: list[TimelineEvent] = []
        self.ids: set[str] = set()

    def submit(self, event: TimelineEvent, arrival_s: float) -> bool:
        """Accept stable IDs and reject arrivals beyond the horizon."""
        if event.event_id in self.ids or arrival_s - event.t_s > self.horizon_s + 1.0e-12:
            return False
        self.ids.add(event.event_id)
        self.events.append(event)
        return True

    def estimate(self, endpoint_t_s: float) -> AnchoredOracleEstimate:
        """Replay and trim the journal at ``endpoint_t_s``."""
        result = replay_anchored(
            tuple(self.events),
            endpoint_t_s=endpoint_t_s,
            checkpoint=self.checkpoint,
            reversal_variance_rad2=self.reversal_variance_rad2,
        )
        boundary = endpoint_t_s - self.horizon_s
        if boundary > self.checkpoint.t_s + 1.0e-12:
            boundary_result = replay_anchored(
                tuple(self.events),
                endpoint_t_s=boundary,
                checkpoint=self.checkpoint,
                reversal_variance_rad2=self.reversal_variance_rad2,
            )
            boundary_state = boundary_result.state
            old_encoders = [
                item
                for item in self.events
                if isinstance(item, EncoderEvent) and item.t_s < boundary - 1.0e-12
            ]
            predecessor = (
                max(old_encoders, key=lambda item: (item.t_s, item.event_id))
                if old_encoders
                else None
            )
            self.checkpoint = boundary_state
            self.events = [
                item for item in self.events if item is predecessor or item.t_s > boundary + 1.0e-12
            ]
        return result
