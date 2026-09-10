"""Paired estimator and optical closed-loop scenarios for the selection checkpoint."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from analysis.studies.tracking_estimator.chronology import EventHistory
from analysis.studies.tracking_estimator.model import (
    AngularEstimator,
    CorrectedResidualEstimator,
    EncoderEvent,
    JointAngularEstimator,
    PredictorTransitionEvent,
    PropagationEvent,
    TimelineEvent,
    VisionEvent,
)
from analysis.studies.tracking_estimator.optical import PlumeRegion, integrate_exposure


@dataclass(frozen=True, slots=True)
class EstimatorMetrics:
    """Comparable estimates over a synthetic delayed-observation timeline."""

    target_rate_rmse_rad_s: float
    relative_angle_rmse_rad: float
    final_target_rate_rad_s: float
    final_relative_angle_rad: float
    accepted_events: int


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Results for both candidates; selection deliberately remains unset."""

    residual: EstimatorMetrics
    joint: EstimatorMetrics
    selection_status: str


def _target_angle(t_s: float) -> float:
    """Return a deterministic target trajectory with changing apparent rate."""
    return float(0.011 * t_s + 0.0015 * np.sin(0.8 * t_s))


def _target_rate(t_s: float) -> float:
    """Return the derivative of the synthetic target trajectory."""
    return float(0.011 + 0.0012 * np.cos(0.8 * t_s))


def _gimbal_angle(t_s: float) -> float:
    """Return a deterministic encoder trajectory used by both candidates."""
    return float(0.008 * t_s + 0.0003 * np.sin(0.5 * t_s))


def _gimbal_rate(t_s: float) -> float:
    """Return the derivative of the synthetic gimbal trajectory."""
    return float(0.008 + 0.00015 * np.cos(0.5 * t_s))


def delayed_navigation_scenario() -> tuple[TimelineEvent, ...]:
    """Build navigation-free/start-loss/recovery and delayed-vision evidence.

    Frames are delivered in deliberately nonchronological arrival order by the
    runner.  The chronology layer restores physical time order.  The rate
    predictor starts absent, appears, then changes reference at 2.5 s.
    """
    events: list[TimelineEvent] = []
    times = np.cumsum(np.array([0.0, 0.17, 0.11, 0.23, 0.19, 0.13] * 8, dtype=float))
    current_nominal: float | None = None
    for index, t_s in enumerate(times):
        requested_nominal: float | None
        if t_s < 0.6:
            requested_nominal = None
        elif t_s < 2.5:
            requested_nominal = 0.010
        elif t_s < 4.0:
            requested_nominal = 0.012
        else:
            requested_nominal = None
        events.append(
            PropagationEvent(
                event_id=f"p-{index}",
                t_s=float(t_s),
                nominal_target_rate_rad_s=current_nominal,
                encoder_rate_rad_s=_gimbal_rate(float(t_s)),
            )
        )
        events.append(EncoderEvent(f"e-{index}", float(t_s), _gimbal_angle(float(t_s))))
        if requested_nominal != current_nominal:
            events.append(
                PredictorTransitionEvent(
                    event_id=f"reference-{index}",
                    t_s=float(t_s),
                    previous_rate_rad_s=current_nominal,
                    next_rate_rad_s=requested_nominal,
                )
            )
            current_nominal = requested_nominal
        if index % 2 == 0:
            events.append(
                VisionEvent(
                    event_id=f"v-{index}",
                    t_s=float(t_s),
                    relative_angle_rad=_target_angle(float(t_s)) - _gimbal_angle(float(t_s)),
                )
            )
    return tuple(events)


def _metric_for(
    factory: Callable[[], AngularEstimator], events: tuple[TimelineEvent, ...]
) -> EstimatorMetrics:
    """Replay shuffled arrivals and compare posterior estimates at vision instants."""
    history = EventHistory(factory, horizon_s=20.0)
    # Receive every third vision after newer evidence, modelling inference latency.
    arrivals = [
        (
            item,
            item.t_s
            + (0.31 if item.event_id.startswith("v-") and int(item.event_id[2:]) % 3 == 0 else 0.0),
        )
        for item in events
    ]
    delayed = sorted(arrivals, key=lambda item: (item[1], item[0].event_id))
    errors_rate: list[float] = []
    errors_relative: list[float] = []
    for item, arrival_t_s in delayed:
        history.submit(item, now_s=arrival_t_s)
        result = history.replay()
        estimate = result.estimator.estimate
        t_s = item.t_s
        errors_rate.append(estimate.target_rate_rad_s - _target_rate(t_s))
        errors_relative.append(
            estimate.relative_angle_rad - (_target_angle(t_s) - _gimbal_angle(t_s))
        )
    final = history.replay().estimator.estimate
    return EstimatorMetrics(
        target_rate_rmse_rad_s=float(np.sqrt(np.mean(np.square(errors_rate)))),
        relative_angle_rmse_rad=float(np.sqrt(np.mean(np.square(errors_relative)))),
        final_target_rate_rad_s=final.target_rate_rad_s,
        final_relative_angle_rad=final.relative_angle_rad,
        accepted_events=len(history.replay().accepted_event_ids),
    )


def run_estimator_comparison() -> ComparisonResult:
    """Run a fair candidate comparison without integrating either into flight."""
    events = delayed_navigation_scenario()
    return ComparisonResult(
        residual=_metric_for(CorrectedResidualEstimator, events),
        joint=_metric_for(JointAngularEstimator, events),
        selection_status="UNSELECTED: review paired scenario evidence before flight integration",
    )


@dataclass(frozen=True, slots=True)
class OpticalMetrics:
    """Image-forming closed-loop metrics, separate from estimator truth inputs."""

    exposure_us: float
    estimator_name: str
    pointing_rmse_rad: float
    retained_sharp_detected_area_px_s: float
    area_weighted_p90_blur_px: float
    worst_local_blur_px: float
    frames: int


def optical_closed_loop(
    factory: Callable[[], AngularEstimator], exposure_us: float, duration_s: float = 3.0
) -> OpticalMetrics:
    """Run a simple rate-servo loop driven only by rendered image centroids.

    The plant is intentionally minimal: it proves timing/image integration and
    exposes tradeoffs, but it is not a hardware qualification model.
    """
    dt_s = 0.02
    ifov_rad_per_px = float(np.deg2rad(0.002636))
    exposure_s = exposure_us * 1.0e-6
    estimator = factory()
    theta_g = 0.0
    target_rate = 0.011
    target_theta = 0.0
    rate_command = 0.0
    pointing_errors: list[float] = []
    sharp_area_px_s = 0.0
    blur_samples: list[tuple[float, float]] = []
    worst_blur = 0.0
    steps = int(duration_s / dt_s)
    for step in range(steps):
        t_s = step * dt_s
        target_theta += target_rate * dt_s
        theta_g += rate_command * dt_s
        midpoint_error_px = (target_theta - theta_g) / ifov_rad_per_px
        # The two regions move differently to reveal local blur missed by one centroid.
        regions = (
            PlumeRegion(
                "core",
                32.0,
                32.0 + midpoint_error_px,
                0.0,
                target_rate / ifov_rad_per_px,
                3.5,
                95.0,
            ),
            PlumeRegion(
                "shear",
                39.0,
                38.0 + midpoint_error_px,
                0.35,
                1.35 * target_rate / ifov_rad_per_px,
                2.4,
                45.0,
            ),
        )
        image = integrate_exposure(
            regions,
            (72, 72),
            exposure_s,
            rate_command,
            ifov_rad_per_px,
            smear_budget_px=1.0,
        )
        estimator.apply(PropagationEvent(f"p-{step}", t_s, None, rate_command))
        estimator.apply(EncoderEvent(f"e-{step}", t_s, theta_g))
        if image.centroid_y_px is not None:
            estimator.apply(
                VisionEvent(f"v-{step}", t_s, (image.centroid_y_px - 32.0) * ifov_rad_per_px)
            )
        estimate = estimator.estimate
        rate_command = float(
            np.clip(estimate.target_rate_rad_s + 2.0 * estimate.relative_angle_rad, -0.18, 0.18)
        )
        pointing_errors.append(target_theta - theta_g)
        sharp_area_px_s += image.sharp_detected_area_px * dt_s
        blur_samples.extend(
            (item.displacement_px, region.apparent_area_px * dt_s)
            for item, region in zip(image.local_blur, regions, strict=True)
            if item.detected
        )
        worst_blur = max(worst_blur, *(item.displacement_px for item in image.local_blur))
    blur_samples.sort(key=lambda item: item[0])
    total_weight = sum(weight for _blur, weight in blur_samples)
    threshold = 0.9 * total_weight
    cumulative = 0.0
    p90_blur = 0.0
    for blur, weight in blur_samples:
        cumulative += weight
        p90_blur = blur
        if cumulative >= threshold:
            break
    return OpticalMetrics(
        exposure_us=exposure_us,
        estimator_name=factory.__name__,
        pointing_rmse_rad=float(np.sqrt(np.mean(np.square(pointing_errors)))),
        retained_sharp_detected_area_px_s=sharp_area_px_s,
        area_weighted_p90_blur_px=p90_blur,
        worst_local_blur_px=worst_blur,
        frames=steps,
    )


def run_optical_exposure_sweep() -> tuple[OpticalMetrics, ...]:
    """Evaluate hardware-minimum and one-millisecond exposure anchors for both candidates."""
    metrics: list[OpticalMetrics] = []
    for factory in (CorrectedResidualEstimator, JointAngularEstimator):
        for exposure_us in (13.0, 1_000.0):
            metrics.append(optical_closed_loop(factory, exposure_us))
    return tuple(metrics)
