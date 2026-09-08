"""Chronological event replay and an independent residual-filter oracle."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from analysis.studies.tracking_estimator.model import (
    AngularEstimator,
    EncoderEvent,
    EventKind,
    PredictorTransitionEvent,
    PropagationEvent,
    TimelineEvent,
    VisionEvent,
    continuous_kinematic_noise,
)


def _sort_key(event: TimelineEvent) -> tuple[float, int, str]:
    """Sort equal-time events by causal order and then stable identity."""
    priority = {
        EventKind.PROPAGATE: 0,
        EventKind.PREDICTOR_TRANSITION: 1,
        EventKind.ENCODER: 2,
        EventKind.VISION: 3,
    }
    return event.t_s, priority[event.kind], event.event_id


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """Result and explicit disposition counts for delayed-event handling."""

    estimator: AngularEstimator
    accepted_event_ids: tuple[str, ...]
    duplicate_event_ids: tuple[str, ...]
    expired_event_ids: tuple[str, ...]


class EventHistory:
    """Bounded, deterministic history that recomputes from chronological evidence.

    This is an analysis reference implementation, not the flight rewind design.
    It makes duplicate, equal-time, delayed, and horizon-expired behavior
    explicit so candidates can be judged against the same history semantics.
    """

    def __init__(self, factory: Callable[[], AngularEstimator], horizon_s: float) -> None:
        self._factory = factory
        self._horizon_s = horizon_s
        self._events: list[TimelineEvent] = []
        self._event_ids: set[str] = set()
        self._duplicate_ids: list[str] = []
        self._expired_ids: list[str] = []
        self._now_s: float | None = None

    def submit(self, event: TimelineEvent, now_s: float) -> None:
        """Accept a nonduplicate event unless it lies beyond the replay horizon."""
        if event.event_id in self._event_ids:
            self._duplicate_ids.append(event.event_id)
            return
        if now_s - event.t_s > self._horizon_s + 1.0e-12:
            self._expired_ids.append(event.event_id)
            return
        self._event_ids.add(event.event_id)
        self._events.append(event)
        self._now_s = max(now_s, self._now_s if self._now_s is not None else now_s)
        lower = self._now_s - self._horizon_s
        self._events = [item for item in self._events if item.t_s >= lower - 1.0e-12]

    def replay(self) -> ReplayResult:
        """Replay retained events in causal order from a fresh estimator."""
        estimator = self._factory()
        ordered = sorted(self._events, key=_sort_key)
        for event in ordered:
            estimator.apply(event)
        return ReplayResult(
            estimator=estimator,
            accepted_event_ids=tuple(event.event_id for event in ordered),
            duplicate_event_ids=tuple(self._duplicate_ids),
            expired_event_ids=tuple(self._expired_ids),
        )


@dataclass(frozen=True, slots=True)
class ResidualOracleState:
    """State emitted by the independent chronological residual oracle."""

    x: np.ndarray
    p: np.ndarray
    predictor_rate_rad_s: float


def residual_chronological_oracle(
    events: Iterable[TimelineEvent],
    accel_psd_rad2_s3: float,
    vision_variance_rad2: float,
) -> ResidualOracleState:
    """Independently execute the residual equations in chronological order.

    The oracle intentionally does not invoke an estimator candidate or its
    update helper.  Tests use it to check mean and covariance after delayed
    arrival replay against a chronological reference.
    """
    x = np.zeros(2, dtype=np.float64)
    p = np.diag(np.array([1.0e-3, 1.0e-4], dtype=np.float64))
    predictor_rate = 0.0
    last_t: float | None = None
    for event in sorted(events, key=_sort_key):
        if isinstance(event, PropagationEvent):
            if last_t is None:
                last_t = event.t_s
                predictor_rate = event.nominal_target_rate_rad_s or 0.0
                continue
            dt_s = event.t_s - last_t
            if dt_s < -1.0e-12:
                raise ValueError("oracle events are not chronological")
            nominal = (
                predictor_rate
                if event.nominal_target_rate_rad_s is None
                else event.nominal_target_rate_rad_s
            )
            x[1] += predictor_rate - nominal
            predictor_rate = nominal
            f, q = continuous_kinematic_noise(max(dt_s, 0.0), accel_psd_rad2_s3)
            x = f @ x + np.array([dt_s * (predictor_rate - event.encoder_rate_rad_s), 0.0])
            p = f @ p @ f.T + q
            last_t = event.t_s
        elif isinstance(event, PredictorTransitionEvent):
            old = predictor_rate if event.previous_rate_rad_s is None else event.previous_rate_rad_s
            new = 0.0 if event.next_rate_rad_s is None else event.next_rate_rad_s
            x[1] += old - new
            predictor_rate = new
        elif isinstance(event, VisionEvent):
            h = np.array([1.0, 0.0], dtype=np.float64)
            s = float(h @ p @ h.T + vision_variance_rad2)
            gain = (p @ h) / s
            identity = np.eye(2, dtype=np.float64)
            x = x + gain * (event.relative_angle_rad - float(h @ x))
            kh = np.outer(gain, h)
            p = (identity - kh) @ p @ (identity - kh).T + np.outer(
                gain, gain
            ) * vision_variance_rad2
            p = 0.5 * (p + p.T)
        elif isinstance(event, EncoderEvent):
            continue
    return ResidualOracleState(x=x, p=p, predictor_rate_rad_s=predictor_rate)
