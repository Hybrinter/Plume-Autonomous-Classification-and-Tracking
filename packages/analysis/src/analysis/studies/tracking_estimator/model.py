"""Timestamped, analysis-only angular estimator candidates.

The residual candidate estimates boresight error and rate relative to an
optional orbital predictor.  The joint candidate estimates target and gimbal
angle/rate directly.  Both consume the same chronological event stream and
use a continuous white-acceleration process model, so a fractional interval
has the same covariance as a partition of that interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np


class EventKind(Enum):
    """Kinds of timestamped evidence in the comparison timeline."""

    PROPAGATE = "propagate"
    PREDICTOR_TRANSITION = "predictor_transition"
    ENCODER = "encoder"
    VISION = "vision"


@dataclass(frozen=True, slots=True)
class PropagationEvent:
    """Known rate inputs that apply from the preceding event to ``t_s``."""

    event_id: str
    t_s: float
    nominal_target_rate_rad_s: float | None
    encoder_rate_rad_s: float
    kind: EventKind = EventKind.PROPAGATE


@dataclass(frozen=True, slots=True)
class PredictorTransitionEvent:
    """Change an optional predictor reference while preserving total target rate."""

    event_id: str
    t_s: float
    previous_rate_rad_s: float | None
    next_rate_rad_s: float | None
    kind: EventKind = EventKind.PREDICTOR_TRANSITION


@dataclass(frozen=True, slots=True)
class EncoderEvent:
    """Encoder-angle observation at its acquisition time."""

    event_id: str
    t_s: float
    angle_rad: float
    kind: EventKind = EventKind.ENCODER


@dataclass(frozen=True, slots=True)
class VisionEvent:
    """Relative target-to-boresight angle measured over a completed exposure."""

    event_id: str
    t_s: float
    relative_angle_rad: float
    kind: EventKind = EventKind.VISION


type TimelineEvent = PropagationEvent | PredictorTransitionEvent | EncoderEvent | VisionEvent


def continuous_kinematic_noise(
    dt_s: float, accel_psd_rad2_s3: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return F and Q for a constant-rate state driven by white acceleration.

    ``accel_psd_rad2_s3`` is the spectral density of angular acceleration.
    The returned Q satisfies the time-partition identity required by delayed
    replay: Q(a+b) = F(b) Q(a) F(b)' + Q(b).
    """
    if dt_s < 0.0:
        raise ValueError("continuous process noise is defined only for non-negative time")
    f = np.array([[1.0, dt_s], [0.0, 1.0]], dtype=np.float64)
    q = accel_psd_rad2_s3 * np.array(
        [[dt_s**3 / 3.0, dt_s**2 / 2.0], [dt_s**2 / 2.0, dt_s]], dtype=np.float64
    )
    return f, q


def _joseph_update(
    x: np.ndarray, p: np.ndarray, h: np.ndarray, z: float, variance: float
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a scalar measurement update with Joseph-form covariance."""
    innovation_covariance = float(h @ p @ h.T + variance)
    if innovation_covariance <= 0.0:
        raise ValueError("measurement covariance must be positive")
    gain = (p @ h) / innovation_covariance
    identity = np.eye(p.shape[0], dtype=np.float64)
    kh = np.outer(gain, h)
    posterior_x = x + gain * (z - float(h @ x))
    posterior_p = (identity - kh) @ p @ (identity - kh).T + np.outer(gain, gain) * variance
    return posterior_x, 0.5 * (posterior_p + posterior_p.T)


@dataclass(frozen=True, slots=True)
class Estimate:
    """Common estimator result at one chronology endpoint."""

    target_angle_rad: float | None
    target_rate_rad_s: float
    relative_angle_rad: float
    covariance: np.ndarray


class AngularEstimator(Protocol):
    """Shared interface used by the comparison runner."""

    @property
    def estimate(self) -> Estimate:
        """Return the current target-rate and relative-angle estimate."""

    def apply(self, event: TimelineEvent) -> None:
        """Consume one event in chronological order."""


@dataclass(slots=True)
class CorrectedResidualEstimator:
    """Residual observer with explicit predictor-reference correction.

    State is [relative angle, target-rate residual].  A reference transition
    transforms the second state so total target rate is continuous.  This is
    the correction missing when a new nominal orbital estimate is adopted.
    """

    residual_accel_psd_rad2_s3: float = 3.0e-7
    vision_variance_rad2: float = 2.5e-7
    initial_angle_variance_rad2: float = 1.0e-3
    initial_rate_variance_rad2_s2: float = 1.0e-4
    x: np.ndarray | None = None
    p: np.ndarray | None = None
    predictor_rate_rad_s: float = 0.0
    last_t_s: float | None = None

    def __post_init__(self) -> None:
        if self.x is None:
            self.x = np.zeros(2, dtype=np.float64)
        if self.p is None:
            self.p = np.diag(
                np.array(
                    [self.initial_angle_variance_rad2, self.initial_rate_variance_rad2_s2],
                    dtype=np.float64,
                )
            )

    @property
    def estimate(self) -> Estimate:
        assert self.x is not None and self.p is not None
        return Estimate(
            target_angle_rad=None,
            target_rate_rad_s=self.predictor_rate_rad_s + float(self.x[1]),
            relative_angle_rad=float(self.x[0]),
            covariance=self.p.copy(),
        )

    def _propagate(self, event: PropagationEvent) -> None:
        assert self.x is not None and self.p is not None
        if self.last_t_s is None:
            self.last_t_s = event.t_s
            self.predictor_rate_rad_s = event.nominal_target_rate_rad_s or 0.0
            return
        dt_s = event.t_s - self.last_t_s
        if dt_s < -1.0e-12:
            raise ValueError("chronology is not ordered")
        f, q = continuous_kinematic_noise(max(dt_s, 0.0), self.residual_accel_psd_rad2_s3)
        nominal = (
            self.predictor_rate_rad_s
            if event.nominal_target_rate_rad_s is None
            else event.nominal_target_rate_rad_s
        )
        # A continuously supplied nominal rate is a new reference, not a jump in target rate.
        self.x[1] += self.predictor_rate_rad_s - nominal
        self.predictor_rate_rad_s = nominal
        self.x = f @ self.x + np.array(
            [dt_s * (self.predictor_rate_rad_s - event.encoder_rate_rad_s), 0.0], dtype=np.float64
        )
        self.p = f @ self.p @ f.T + q
        self.last_t_s = event.t_s

    def _transition(self, event: PredictorTransitionEvent) -> None:
        assert self.x is not None
        old_rate = (
            self.predictor_rate_rad_s
            if event.previous_rate_rad_s is None
            else event.previous_rate_rad_s
        )
        new_rate = 0.0 if event.next_rate_rad_s is None else event.next_rate_rad_s
        self.x[1] += old_rate - new_rate
        self.predictor_rate_rad_s = new_rate
        self.last_t_s = event.t_s if self.last_t_s is None else self.last_t_s

    def _vision(self, event: VisionEvent) -> None:
        assert self.x is not None and self.p is not None
        self.x, self.p = _joseph_update(
            self.x,
            self.p,
            np.array([1.0, 0.0], dtype=np.float64),
            event.relative_angle_rad,
            self.vision_variance_rad2,
        )

    def apply(self, event: TimelineEvent) -> None:
        """Consume propagation, reference, and visual events; encoders are rate inputs here."""
        if isinstance(event, PropagationEvent):
            self._propagate(event)
        elif isinstance(event, PredictorTransitionEvent):
            self._transition(event)
        elif isinstance(event, VisionEvent):
            self._vision(event)


@dataclass(slots=True)
class JointAngularEstimator:
    """Joint [target angle/rate, gimbal angle/rate] angular observer.

    It has two physical measurement updates: encoder angle and relative visual
    angle.  The optional orbital predictor is deliberately not a measurement;
    it can be used later as a command aid without creating a discontinuity in
    this total-target-rate state.
    """

    target_accel_psd_rad2_s3: float = 3.0e-7
    gimbal_accel_psd_rad2_s3: float = 1.0e-6
    vision_variance_rad2: float = 2.5e-7
    encoder_variance_rad2: float = 1.0e-7
    x: np.ndarray | None = None
    p: np.ndarray | None = None
    last_t_s: float | None = None
    predictor_rate_rad_s: float | None = None

    def __post_init__(self) -> None:
        if self.x is None:
            self.x = np.zeros(4, dtype=np.float64)
        if self.p is None:
            self.p = np.diag(np.array([1.0e-3, 1.0e-4, 1.0e-3, 1.0e-4], dtype=np.float64))

    @property
    def estimate(self) -> Estimate:
        assert self.x is not None and self.p is not None
        return Estimate(
            target_angle_rad=float(self.x[0]),
            target_rate_rad_s=float(self.x[1]),
            relative_angle_rad=float(self.x[0] - self.x[2]),
            covariance=self.p.copy(),
        )

    def _propagate(self, event: PropagationEvent) -> None:
        assert self.x is not None and self.p is not None
        if self.last_t_s is None:
            self.last_t_s = event.t_s
            self.predictor_rate_rad_s = event.nominal_target_rate_rad_s
            return
        dt_s = event.t_s - self.last_t_s
        if dt_s < -1.0e-12:
            raise ValueError("chronology is not ordered")
        f_pair, q_target = continuous_kinematic_noise(max(dt_s, 0.0), self.target_accel_psd_rad2_s3)
        _unused, q_gimbal = continuous_kinematic_noise(
            max(dt_s, 0.0), self.gimbal_accel_psd_rad2_s3
        )
        f = np.zeros((4, 4), dtype=np.float64)
        q = np.zeros((4, 4), dtype=np.float64)
        f[:2, :2] = f_pair
        f[2:, 2:] = f_pair
        q[:2, :2] = q_target
        q[2:, 2:] = q_gimbal
        self.x = f @ self.x
        self.p = f @ self.p @ f.T + q
        self.predictor_rate_rad_s = event.nominal_target_rate_rad_s
        self.last_t_s = event.t_s

    def _transition(self, event: PredictorTransitionEvent) -> None:
        # The joint state represents total target rate, hence no state transform is needed.
        self.predictor_rate_rad_s = event.next_rate_rad_s

    def _encoder(self, event: EncoderEvent) -> None:
        assert self.x is not None and self.p is not None
        self.x, self.p = _joseph_update(
            self.x,
            self.p,
            np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64),
            event.angle_rad,
            self.encoder_variance_rad2,
        )

    def _vision(self, event: VisionEvent) -> None:
        assert self.x is not None and self.p is not None
        self.x, self.p = _joseph_update(
            self.x,
            self.p,
            np.array([1.0, 0.0, -1.0, 0.0], dtype=np.float64),
            event.relative_angle_rad,
            self.vision_variance_rad2,
        )

    def apply(self, event: TimelineEvent) -> None:
        """Consume events in time order."""
        if isinstance(event, PropagationEvent):
            self._propagate(event)
        elif isinstance(event, PredictorTransitionEvent):
            self._transition(event)
        elif isinstance(event, EncoderEvent):
            self._encoder(event)
        else:
            self._vision(event)
