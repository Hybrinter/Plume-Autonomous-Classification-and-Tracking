"""Analysis block tests for the cascaded elevation controller (design brief §18)."""

from __future__ import annotations

import math
from dataclasses import fields

import numpy as np
from analysis.studies.elevation_controller.plant import ElevationPlant
from flight.hal.drivers_sim import SimIssEphemeris
from flight.libs.config import ControllerConfig, EphemerisConfig, GimbalConfig, SensorConfig
from flight.libs.messages import BlobMeta, GimbalCommandMsg, InferenceResultMsg
from flight.libs.time import ManualClock
from flight.libs.types import GimbalCommandMode, GimbalState, MessageType, Ok
from flight.payload.control import PayloadController, VisionSample
from flight.payload.gimbal.inner import inner_step
from flight.payload.gimbal.outer import outer_rate
from flight.payload.gimbal.predictor import predict_los
from flight.payload.gimbal.rate_fit import fit_rate
from flight.payload.gimbal.request import GimbalRequest
from flight.payload.tracking.residual import (
    EncoderSample,
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


def test_inner_tracks_with_mismatched_plant_copies() -> None:
    """Mismatched J-hat and B-hat still track a constant r."""
    dt = 0.001
    r = math.radians(1.0)
    plant = ElevationPlant(j_kg_m2=0.008, b_nms_per_rad=0.04)
    integrator = 0.0
    ring: tuple[float, ...] = ()
    for _ in range(4000):
        ring = (ring + (plant.theta_rad,))[-7:]
        y_m = fit_rate(ring, dt, 7, 2)
        result = inner_step(r, y_m, integrator, dt, 0.012, 0.02, 200.0, 10_000.0, 1.0, False)
        integrator = result.integrator
        plant.step(result.tau_nm, dt)
    assert abs(plant.omega_rad_s - r) < math.radians(0.3)


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
    _theta, omega, _omega_az = predict_los(
        t0, r_iss, s0.value.v_m_s, r_cog, eph.omega_earth_rad_s, eph.epoch_utc_s
    )
    dt = 0.05
    sp = sim.read_state(t0 + dt)
    sm = sim.read_state(t0 - dt)
    assert isinstance(sp, Ok) and isinstance(sm, Ok)
    tp, _, _ = predict_los(
        t0 + dt,
        sp.value.r_m,
        sp.value.v_m_s,
        r_cog,
        eph.omega_earth_rad_s,
        eph.epoch_utc_s,
    )
    tm, _, _ = predict_los(
        t0 - dt,
        sm.value.r_m,
        sm.value.v_m_s,
        r_cog,
        eph.omega_earth_rad_s,
        eph.epoch_utc_s,
    )
    fd = (tp - tm) / (2.0 * dt)
    assert abs(omega - fd) / max(abs(fd), 1e-9) < 0.05


def test_walking_cog_at_same_iss_time() -> None:
    """omega_t_nom at one ISS epoch differs for two ECEF points; it is not the walk slope."""
    eph = EphemerisConfig()
    sim = SimIssEphemeris(ManualClock(utc_s=eph.epoch_utc_s), eph)
    t0 = eph.epoch_utc_s
    s0 = sim.read_state(t0)
    assert isinstance(s0, Ok)
    r_iss = s0.value.r_m
    v_iss = s0.value.v_m_s
    r_norm = math.hypot(*r_iss)
    scale = eph.wgs84_a_m / r_norm
    p1 = (r_iss[0] * scale, r_iss[1] * scale, r_iss[2] * scale)
    p2 = (p1[0], p1[1] + 20_000.0, p1[2])
    _th1, w1, _az1 = predict_los(t0, r_iss, v_iss, p1, eph.omega_earth_rad_s, eph.epoch_utc_s)
    _th2, w2, _az2 = predict_los(t0, r_iss, v_iss, p2, eph.omega_earth_rad_s, eph.epoch_utc_s)
    assert abs(w1 - w2) > 1e-8
    s1 = sim.read_state(t0 + 1.0)
    assert isinstance(s1, Ok)
    th1, _, _ = predict_los(t0, r_iss, v_iss, p1, eph.omega_earth_rad_s, eph.epoch_utc_s)
    th2, w2_later, _az_later = predict_los(
        t0 + 1.0,
        s1.value.r_m,
        s1.value.v_m_s,
        p2,
        eph.omega_earth_rad_s,
        eph.epoch_utc_s,
    )
    walk_slope = (th2 - th1) / 1.0
    assert abs(w2_later - walk_slope) > abs(w2_later) * 0.05 + 1e-6


def test_earth_rate_changes_nominal_rate() -> None:
    """omega_t_nom with Earth rotation differs from the Omega_E = 0 case."""
    eph = EphemerisConfig()
    sim = SimIssEphemeris(ManualClock(utc_s=eph.epoch_utc_s), eph)
    t0 = eph.epoch_utc_s
    s0 = sim.read_state(t0)
    assert isinstance(s0, Ok)
    r_iss = s0.value.r_m
    v_iss = s0.value.v_m_s
    r_norm = math.hypot(*r_iss)
    scale = eph.wgs84_a_m / r_norm
    r_cog = (r_iss[0] * scale, r_iss[1] * scale, r_iss[2] * scale)
    _th, w_on, _az_on = predict_los(t0, r_iss, v_iss, r_cog, eph.omega_earth_rad_s, eph.epoch_utc_s)
    _th0, w_off, _az_off = predict_los(t0, r_iss, v_iss, r_cog, 0.0, eph.epoch_utc_s)
    assert abs(w_on - w_off) > 1e-8


def test_residual_recovers_extra_rate_through_rewind() -> None:
    """Closed-loop rewind updates stop e ramping and recover extra rate."""
    cfg = ControllerConfig()
    filt = ResidualFilter.from_config(cfg.residual, cfg.outer.dt_s)
    dt = cfg.outer.dt_s
    extra = math.radians(0.1)
    kp = cfg.outer.Kp
    state = filt.initial_state()
    snaps: tuple[ResidualSnapshot, ...] = ()
    e_true = 0.0
    now = 0.0
    y_m = 0.0
    for k in range(160):
        e_shutter = e_true
        now += dt
        e_true += dt * (extra - y_m)
        state = predict(filt, state, dt, 0.0, y_m)
        snaps = push_snapshot(
            snaps,
            ResidualSnapshot(t_s=now, state=state, dt_s=dt, omega_t_nom=0.0, y_m=y_m),
            cfg.residual.rewind_snapshots,
        )
        if k > 0:
            state = rewind_update(
                filt, snaps, state, now, now - dt, e_shutter, cfg.residual.rewind_horizon_s
            )
        y_m = kp * float(state.x[0]) + float(state.x[1])
    assert abs(float(state.x[1]) - extra) < math.radians(0.05)
    assert abs(float(state.x[0])) < math.radians(0.05)
    assert abs(e_true) < math.radians(0.05)


def test_rewind_posterior_matches_discrete_oracle() -> None:
    """Rewind posterior matches an explicit F, u replay; a no-op fails that oracle."""
    cfg = ControllerConfig()
    filt = ResidualFilter.from_config(cfg.residual, cfg.outer.dt_s)
    dt = cfg.outer.dt_s
    state = update(filt, filt.initial_state(), 0.0)
    snaps: tuple[ResidualSnapshot, ...] = ()
    now = 0.0
    omega = 0.08
    y_m = 0.02
    for _ in range(4):
        now += dt
        state = predict(filt, state, dt, omega, y_m)
        snaps = push_snapshot(
            snaps,
            ResidualSnapshot(t_s=now, state=state, dt_s=dt, omega_t_nom=omega, y_m=y_m),
            cfg.residual.rewind_snapshots,
        )
    t_s = dt * 2
    z_v = 0.015
    rewound = rewind_update(filt, snaps, state, now, t_s, z_v, cfg.residual.rewind_horizon_s)
    snap = next(s for s in snaps if abs(s.t_s - t_s) < 1e-12)
    posterior = update(filt, snap.state, z_v)
    replay_from = t_s
    for later in snaps:
        if later.t_s <= t_s + 1e-12:
            continue
        step = later.t_s - replay_from
        f = np.array([[1.0, step], [0.0, 1.0]], dtype=np.float64)
        u = np.array([step * (later.omega_t_nom - later.y_m), 0.0], dtype=np.float64)
        q_ratio = step / filt.dt_outer_s if filt.dt_outer_s > 0.0 else 1.0
        q = np.array(
            [[filt.q11 * q_ratio**3, 0.0], [0.0, filt.q22 * q_ratio]],
            dtype=np.float64,
        )
        posterior_x = f @ posterior.x + u
        posterior_p = f @ posterior.P @ f.T + q
        posterior = type(posterior)(x=posterior_x, P=posterior_p, has_measurement=True)
        replay_from = later.t_s
    tail = now - replay_from
    if tail > 1e-12:
        last = snaps[-1]
        f = np.array([[1.0, tail], [0.0, 1.0]], dtype=np.float64)
        u = np.array([tail * (last.omega_t_nom - last.y_m), 0.0], dtype=np.float64)
        q_ratio = tail / filt.dt_outer_s if filt.dt_outer_s > 0.0 else 1.0
        q = np.array(
            [[filt.q11 * q_ratio**3, 0.0], [0.0, filt.q22 * q_ratio]],
            dtype=np.float64,
        )
        posterior = type(posterior)(
            x=f @ posterior.x + u,
            P=f @ posterior.P @ f.T + q,
            has_measurement=True,
        )
    assert np.allclose(rewound.x, posterior.x)
    assert not np.allclose(rewound.x, state.x)


def test_smear_oracle_is_separate_from_control_rate() -> None:
    """Optical smear scales with exposure without reducing control authority."""
    ifov = 0.002636
    ifov_rad = math.radians(ifov)
    hw = math.radians(10.0)
    oracle_13us = 1.0 * ifov_rad / 13e-6
    assert oracle_13us > hw
    r_13 = outer_rate(
        0.0,
        0.0,
        math.radians(4.0),
        8.0,
        GimbalState.TRACKING,
        True,
        math.radians(10.0),
        math.radians(45.0),
        hw,
        13.0,
        1.0,
        ifov,
    )
    assert abs(abs(r_13) - hw) < 1e-12
    t_exp = 2000e-6
    oracle = 1.0 * ifov_rad / t_exp
    r_long = outer_rate(
        0.0,
        0.0,
        math.radians(4.0),
        8.0,
        GimbalState.TRACKING,
        True,
        math.radians(10.0),
        math.radians(45.0),
        hw,
        2000.0,
        1.0,
        ifov,
    )
    assert oracle < hw
    assert abs(abs(r_long) - hw) < 1e-12


def test_safe_position_loop_and_cold_start() -> None:
    """SAFE stows via the position loop; cold TRACKING holds r=0."""
    controller = PayloadController.from_config(
        ControllerConfig(), SensorConfig(), GimbalConfig(), EphemerisConfig()
    )
    cold = controller.initial_state()
    encoder = EncoderSample(sample_id="encoder:0", t_s=0.02, angle_rad=0.0)
    coast = controller.outer_step(cold, 0.02, encoder, None, None, False, False)
    assert coast.state.r_rad_s == 0.0
    assert coast.state.arbiter.gimbal_state is GimbalState.TRACKING

    safe = controller.outer_step(cold, 0.02, encoder, None, None, True, False)
    assert safe.state.arbiter.gimbal_state is GimbalState.SAFE
    assert safe.request is not None
    assert safe.request.mode is GimbalCommandMode.STOW
    assert safe.state.r_rad_s < 0.0

    blob = BlobMeta(
        blob_id=1,
        bbox=(0, 0, 20, 20),
        centroid_raw=(612.0, 124.0),
        pixel_area=200,
        mean_confidence=0.9,
        persistence_count=1,
    )
    vision = VisionSample(
        t_s=0.0,
        frame_id="frame:0",
        z_v=0.01,
        p_cog=(612.0, 124.0),
        exposure_us=1000.0,
        blobs=(blob,),
        mode_flags=0,
        iss=None,
    )
    ignored = controller.outer_step(
        safe.state,
        0.04,
        EncoderSample(sample_id="encoder:1", t_s=0.04, angle_rad=0.0),
        vision,
        None,
        False,
        False,
    )
    assert ignored.state.arbiter.gimbal_state is GimbalState.SAFE
    assert ignored.state.pose_mode is GimbalCommandMode.STOW


def test_no_azimuth_on_request_or_command() -> None:
    """GimbalRequest and GimbalCommandMsg expose no azimuth field."""
    request_names = {item.name for item in fields(GimbalRequest)}
    command_names = {item.name for item in fields(GimbalCommandMsg)}
    assert not any("az" in name for name in request_names)
    assert not any("az" in name for name in command_names)
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
    tick = controller.outer_step(
        state,
        0.02,
        EncoderSample(sample_id="encoder:0", t_s=0.02, angle_rad=0.0),
        sample,
        None,
        False,
        False,
    )
    assert tick.request is None
    assert not hasattr(tick.state, "commanded_az_rate_deg_per_s")
