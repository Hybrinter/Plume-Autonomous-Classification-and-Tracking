"""Analysis block tests for the cascaded elevation controller (design brief §18)."""

from __future__ import annotations

import math

import numpy as np
from analysis.studies.elevation_controller.plant import ElevationPlant
from flight.hal.drivers_sim import SimIssEphemeris
from flight.libs.config import ControllerConfig, EphemerisConfig, GimbalConfig, SensorConfig
from flight.libs.messages import BlobMeta, InferenceResultMsg
from flight.libs.time import ManualClock
from flight.libs.types import GimbalCommandMode, GimbalState, MessageType, Ok
from flight.payload.control import PayloadController, VisionSample
from flight.payload.gimbal.inner import inner_step
from flight.payload.gimbal.outer import outer_rate, smear_cap_rad_s
from flight.payload.gimbal.predictor import predict_los
from flight.payload.gimbal.rate_fit import fit_rate
from flight.payload.tracking.residual import (
    ResidualFilter,
    ResidualSnapshot,
    predict,
    push_snapshot,
    rewind_update,
    update,
)


def test_inner_tracks_constant_r() -> None:
    """A constant r is tracked with bounded rate error on the rigid-body plant."""
    dt = 0.001
    r = math.radians(1.0)
    plant = ElevationPlant()
    integrator = 0.0
    ring: tuple[float, ...] = ()
    for _ in range(3000):
        ring = (ring + (plant.theta_rad,))[-7:]
        y_m = fit_rate(ring, dt, 7, 2)
        result = inner_step(r, y_m, integrator, dt, 0.008, 0.04, 200.0, 10_000.0, 1.0, False)
        integrator = result.integrator
        plant.step(result.tau_nm, dt)
    assert abs(plant.omega_rad_s - r) < math.radians(0.2)


def test_inner_clip_has_no_windup() -> None:
    """Torque clip freezes the integrator while unsaturated error remains large."""
    dt = 0.001
    integrator = 0.0
    for _ in range(200):
        result = inner_step(20.0, 0.0, integrator, dt, 0.008, 0.04, 200.0, 10_000.0, 1.0, False)
        integrator = result.integrator
        assert result.clipped is True
        assert result.tau_nm == 1.0
    assert integrator == 0.0


def test_ym_is_polynomial_not_two_point() -> None:
    """y_m on a quadratic ring matches instantaneous rate, not the chord slope."""
    dt = 0.001
    accel = 3.0
    n = 7
    ring = tuple(0.5 * accel * (dt * i) ** 2 for i in range(n))
    y_m = fit_rate(ring, dt, n, 2)
    true_rate = accel * dt * (n - 1)
    two_point = (ring[-1] - ring[0]) / ((n - 1) * dt)
    assert abs(y_m - true_rate) < 1e-6
    assert abs(y_m - two_point) > 5.0 * abs(y_m - true_rate)


def test_predictor_matches_theta_finite_difference() -> None:
    """Frozen-ECEF omega_t_nom matches a central difference of theta_los."""
    eph = EphemerisConfig()
    sim = SimIssEphemeris(ManualClock(utc_s=eph.epoch_utc_s), eph)
    t0 = eph.epoch_utc_s
    s0 = sim.read_state(t0)
    assert isinstance(s0, Ok)
    r_iss = s0.value.r_m
    r_norm = math.hypot(*r_iss)
    scale = eph.wgs84_a_m / r_norm
    r_cog = (r_iss[0] * scale, r_iss[1] * scale, r_iss[2] * scale)
    _theta, omega = predict_los(
        t0, r_iss, s0.value.v_m_s, r_cog, eph.omega_earth_rad_s, eph.epoch_utc_s
    )
    dt = 0.05
    sp = sim.read_state(t0 + dt)
    sm = sim.read_state(t0 - dt)
    assert isinstance(sp, Ok) and isinstance(sm, Ok)
    tp, _ = predict_los(
        t0 + dt,
        sp.value.r_m,
        sp.value.v_m_s,
        r_cog,
        eph.omega_earth_rad_s,
        eph.epoch_utc_s,
    )
    tm, _ = predict_los(
        t0 - dt,
        sm.value.r_m,
        sm.value.v_m_s,
        r_cog,
        eph.omega_earth_rad_s,
        eph.epoch_utc_s,
    )
    fd = (tp - tm) / (2.0 * dt)
    assert abs(omega - fd) / max(abs(fd), 1e-9) < 0.05


def test_walking_cog_does_not_set_nominal_rate() -> None:
    """omega_t_nom is the co-rotating rate at the current point, not the walk slope."""
    eph = EphemerisConfig()
    sim = SimIssEphemeris(ManualClock(utc_s=eph.epoch_utc_s), eph)
    t0 = eph.epoch_utc_s
    s0 = sim.read_state(t0)
    assert isinstance(s0, Ok)
    r_iss = s0.value.r_m
    r_norm = math.hypot(*r_iss)
    scale = eph.wgs84_a_m / r_norm
    p1 = (r_iss[0] * scale, r_iss[1] * scale, r_iss[2] * scale)
    p2 = (p1[0], p1[1] + 20_000.0, p1[2])
    dt = 1.0
    s1 = sim.read_state(t0 + dt)
    assert isinstance(s1, Ok)
    th1, _w1 = predict_los(t0, r_iss, s0.value.v_m_s, p1, eph.omega_earth_rad_s, eph.epoch_utc_s)
    th2, w2 = predict_los(
        t0 + dt,
        s1.value.r_m,
        s1.value.v_m_s,
        p2,
        eph.omega_earth_rad_s,
        eph.epoch_utc_s,
    )
    walk_slope = (th2 - th1) / dt
    _th2_frozen, w2_frozen = predict_los(
        t0 + dt,
        s1.value.r_m,
        s1.value.v_m_s,
        p2,
        eph.omega_earth_rad_s,
        eph.epoch_utc_s,
    )
    assert w2 == w2_frozen
    assert abs(w2 - walk_slope) > abs(w2) * 0.05 + 1e-6


def test_residual_recovers_extra_rate() -> None:
    """An extra 0.1 deg/s residual rate is recovered after vision updates."""
    cfg = ControllerConfig()
    filt = ResidualFilter.from_config(cfg.residual, cfg.outer.dt_s)
    dt = cfg.outer.dt_s
    extra = math.radians(0.1)
    state = filt.initial_state()
    e_true = 0.0
    omega_nom = 0.0
    y_m = 0.0
    for _ in range(120):
        e_true += dt * (omega_nom + extra - y_m)
        state = predict(filt, state, dt, omega_nom, y_m)
        state = update(filt, state, e_true)
    assert abs(float(state.x[1]) - extra) < math.radians(0.03)


def test_lagged_zv_rewind_differs_from_current_update() -> None:
    """Rewind does not apply a stale z_v as if it were a measurement at now."""
    cfg = ControllerConfig()
    filt = ResidualFilter.from_config(cfg.residual, cfg.outer.dt_s)
    dt = cfg.outer.dt_s
    state = filt.initial_state()
    snaps: tuple[ResidualSnapshot, ...] = ()
    now = 0.0
    y_m = 0.05
    for _ in range(5):
        now += dt
        state = predict(filt, state, dt, 0.0, y_m)
        snaps = push_snapshot(
            snaps,
            ResidualSnapshot(t_s=now, state=state, dt_s=dt, omega_t_nom=0.0, y_m=y_m),
            cfg.residual.rewind_snapshots,
        )
    z_v = 0.02
    t_s = dt
    rewound = rewind_update(filt, snaps, state, now, t_s, z_v, cfg.residual.rewind_horizon_s)
    naive = update(filt, state, z_v)
    assert abs(float(rewound.x[0]) - float(naive.x[0])) > 1e-8


def test_smear_from_live_exposure() -> None:
    """TRACKING and REWIND |r| respect the live exposure smear cap."""
    ifov = 0.002636
    hw = math.radians(10.0)
    limb = math.radians(45.0)
    long_exp = outer_rate(
        0.0,
        0.0,
        math.radians(-4.0),
        8.0,
        GimbalState.TRACKING,
        True,
        0.0,
        limb,
        hw,
        2000.0,
        1.0,
        ifov,
    )
    short_exp = outer_rate(
        0.0,
        0.0,
        math.radians(-4.0),
        8.0,
        GimbalState.TRACKING,
        True,
        0.0,
        limb,
        hw,
        500.0,
        1.0,
        ifov,
    )
    assert abs(abs(long_exp) - smear_cap_rad_s(2000.0, 1.0, ifov)) < 1e-12
    assert abs(abs(short_exp) - smear_cap_rad_s(500.0, 1.0, ifov)) < 1e-12
    rewind = outer_rate(
        0.0,
        0.0,
        0.0,
        8.0,
        GimbalState.REWIND,
        False,
        0.0,
        limb,
        hw,
        2000.0,
        1.0,
        ifov,
    )
    assert abs(rewind - smear_cap_rad_s(2000.0, 1.0, ifov)) < 1e-12


def test_safe_position_loop_and_cold_start() -> None:
    """SAFE stows via the position loop; cold TRACKING holds r=0; no azimuth field."""
    controller = PayloadController.from_config(
        ControllerConfig(), SensorConfig(), GimbalConfig(), EphemerisConfig()
    )
    cold = controller.initial_state()
    coast = controller.outer_step(cold, 0.02, 0.0, None, None, False, False)
    assert coast.state.r_rad_s == 0.0
    assert coast.state.arbiter.gimbal_state is GimbalState.TRACKING

    safe = controller.outer_step(cold, 0.02, 0.0, None, None, True, False)
    assert safe.state.arbiter.gimbal_state is GimbalState.SAFE
    assert safe.request is not None
    assert safe.request.mode is GimbalCommandMode.STOW
    assert not hasattr(safe.request, "az_deg")
    assert safe.state.r_rad_s < 0.0

    blob = BlobMeta(
        blob_id=1,
        bbox=(0, 0, 20, 20),
        centroid_raw=(612.0, 900.0),
        pixel_area=200,
        mean_confidence=0.9,
        persistence_count=1,
    )
    vision = VisionSample(
        t_s=0.0,
        z_v=-0.01,
        p_cog=(612.0, 900.0),
        exposure_us=1000.0,
        blobs=(blob,),
        mode_flags=0,
    )
    ignored = controller.outer_step(safe.state, 0.04, 0.0, vision, None, False, False)
    assert ignored.state.arbiter.gimbal_state is GimbalState.SAFE
    assert ignored.state.pose_mode is GimbalCommandMode.STOW


def test_no_azimuth_command_on_controller() -> None:
    """GimbalRequest and ControlState expose no azimuth command."""
    controller = PayloadController.from_config(ControllerConfig(), SensorConfig(), GimbalConfig())
    state = controller.initial_state()
    result = InferenceResultMsg(
        msg_type=MessageType.INFERENCE_RESULT,
        timestamp_utc="t",
        frame_id=1,
        mask=np.zeros((8, 8), dtype=np.float32),
        blobs=(
            BlobMeta(
                blob_id=1,
                bbox=(0, 0, 10, 10),
                centroid_raw=(682.0, 512.0),
                pixel_area=80,
                mean_confidence=0.9,
                persistence_count=1,
            ),
        ),
        model_version="t",
        inference_ms=0.0,
        mode_flags=0,
    )
    state, sample = controller.ingest_inference(state, result, 0.0, 1000.0)
    tick = controller.outer_step(state, 0.02, 0.0, sample, None, False, False)
    assert tick.request is None
    assert not hasattr(tick.state, "commanded_az_rate_deg_per_s")
