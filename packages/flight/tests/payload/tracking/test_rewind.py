"""Tests for vision-latency rewind of the dual-axis Kalman filter."""

from flight.libs.config import ControllerConfig
from flight.payload.tracking import (
    DualKalmanState,
    KalmanFilter,
    apply_vision,
    empty_ring,
    predict,
    push_snapshot,
    update_vis,
)
from flight.payload.tracking.rewind import EstimatorSnapshot


def test_apply_vision_none_when_shutter_older_than_ring() -> None:
    """A shutter time before the oldest snapshot is skipped."""
    kf = KalmanFilter.from_config(ControllerConfig())
    dual = KalmanFilter.initial_state()
    ring = empty_ring(4)
    snap = EstimatorSnapshot(
        t=1.0,
        az=dual.az,
        el=dual.el,
        u_az=0.0,
        u_el=0.0,
        theta_enc_az=0.0,
        theta_enc_el=0.0,
    )
    ring = push_snapshot(ring, snap)
    assert apply_vision(kf, ring, 1.0, 1.0, 0.0, 1.2, 0.0, 0.0, dual) is None


def test_apply_vision_once_does_not_collapse_like_a_repeat() -> None:
    """Rewind applies the vision sample once; repeating update_vis at now shrinks P more."""
    kf = KalmanFilter.from_config(ControllerConfig())
    dual = KalmanFilter.initial_state()
    ring = empty_ring(8)
    t0 = 0.0
    az = dual.az
    el = dual.el
    for i in range(4):
        t = t0 + 0.05 * (i + 1)
        az = predict(kf, az, 0.0, 0.05)
        el = predict(kf, el, 0.0, 0.05)
        ring = push_snapshot(
            ring,
            EstimatorSnapshot(
                t=t,
                az=az,
                el=el,
                u_az=0.0,
                u_el=0.0,
                theta_enc_az=float(az.x[1]),
                theta_enc_el=float(el.x[1]),
            ),
        )
    current = DualKalmanState(az=az, el=el)
    rewound = apply_vision(kf, ring, 1.5, -0.5, 0.05, 0.20, 0.0, 0.0, current)
    assert rewound is not None
    assert rewound.az.x[0] != 0.0
    from flight.libs.types import Ok

    twice = update_vis(kf, az, 1.5)
    assert isinstance(twice, Ok)
    twice2 = update_vis(kf, twice.value, 1.5)
    assert isinstance(twice2, Ok)
    assert float(rewound.az.p[0, 0]) > float(twice2.value.p[0, 0])
