"""OPERATE TRACKING node: CoG tracking with the residual estimator (pure).

Reproduces the control outer-loop TRACKING composition: fresh CoG intersect on
an accepted sample, acquire residual reset at the shutter, encoder/nominal/
reference/vision events into the residual history, estimate replay at the
encoder time, and the scene-plus-residual rate law. Unknown navigation keeps
the scene nominal at zero as a computational fallback only.

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001, REQ-AIML-GIMB-008.
"""

from __future__ import annotations

import math
from dataclasses import replace

from flight.payload.gimbal.intersect import intersect_cog
from flight.payload.gimbal.outer import (
    RateDecision,
    clip_rate,
    rate_decision,
    smear_cap_rad_s,
)
from flight.payload.gimbal.pointing import pinhole_error_rad
from flight.payload.gimbal.predictor import predict_los
from flight.payload.gimbal.request import InhibitReference, RateReference
from flight.payload.gimbal.scene import cog_scene
from flight.payload.graphs.base import NodeOutcome, TickInputs
from flight.payload.graphs.operate.state import (
    OperateNode,
    State,
    TargetState,
    acquire_resets_residual,
    bookkeep_vision,
)
from flight.payload.graphs.parameters import GraphParameters, encoder_fresh
from flight.payload.tracking import (
    NominalRateSample,
    PredictorReferenceChange,
    VisionObservation,
    estimate_at,
    submit_event,
)


def step(
    state: State,
    inputs: TickInputs,
    params: GraphParameters,
) -> tuple[State, NodeOutcome[OperateNode]]:
    """One TRACKING tick: feed the residual history and emit the rate reference.

    Inputs:
        state: OPERATE state; `node` still holds the previous node so the
            acquire-reset policy can see where this tick came from.
        inputs: Tick observations with the resolved effective vision sample.
        params: Graph parameters.

    Outputs:
        tuple[State, NodeOutcome[OperateNode]]: Updated state and the
            RateReference outcome under the science envelope.
    """
    policy = params.default_policy(enabled=True)
    encoder = inputs.encoder
    if encoder is None or not encoder_fresh(inputs, params):
        return state, NodeOutcome(
            reference=InhibitReference(reason="tracking_stale_feedback"), policy=policy
        )
    theta_g_rad = encoder.angle_rad
    vision = inputs.vision
    sample = vision.sample if vision is not None else None
    iss = inputs.navigation
    eph = params.config.ephemeris
    height_m = params.config.controller.predictor.cog_height_m

    previous_node = state.node
    new_blob_ids = frozenset(blob.blob_id for blob in sample.blobs) if sample else frozenset()
    reset = acquire_resets_residual(
        previous_node,
        state.aggregate_live,
        frozenset(blob.blob_id for blob in state.tracked_blobs),
        new_blob_ids,
    )

    r_cog = state.target.r_cog_ecef_m
    fresh_cog = False
    if (
        sample is not None
        and sample.p_cog is not None
        and sample.iss is not None
        and sample.theta_g_rad is not None
    ):
        inter = intersect_cog(
            sample.p_cog,
            sample.theta_g_rad,
            sample.iss.r_m,
            sample.iss.v_m_s,
            sample.iss.utc_s,
            eph.epoch_utc_s,
            eph.omega_earth_rad_s,
            eph.wgs84_a_m,
            eph.wgs84_f,
            params.camera,
            height_m,
        )
        if inter is not None:
            r_cog = inter.point_ecef_m
            fresh_cog = True

    residual = state.residual
    history = state.residual_history
    filt = params.residual_filter
    if reset:
        if not fresh_cog:
            r_cog = None
        seed_t_s = encoder.t_s
        seed_angle_rad: float | None = encoder.angle_rad
        seed_angle_var = encoder.angle_variance_rad2
        if sample is not None and sample.z_v is not None and sample.t_s <= inputs.now_s + 1e-12:
            seed_t_s = sample.t_s
            seed_angle_rad = sample.theta_g_rad
            if seed_angle_rad is None:
                seed_angle_var = 0.0
        residual = filt.initial_state()
        history = filt.initial_history(
            t_s=seed_t_s,
            encoder_angle_rad=seed_angle_rad,
            encoder_endpoint_variance_rad2=seed_angle_var,
        )

    iss_r = None if iss is None else iss.r_m
    iss_v = None if iss is None else iss.v_m_s
    iss_utc = None if iss is None else iss.utc_s
    scene = cog_scene(r_cog, iss_r, iss_v, iss_utc, eph.omega_earth_rad_s, eph.epoch_utc_s)
    omega_az = state.target.last_omega_az_nom
    if scene.los is None:
        omega_t_nom = 0.0
        theta_los = 0.0
        omega_az = 0.0
    else:
        omega_t_nom = scene.los.elevation_rate_rad_s
        theta_los = scene.los.elevation_rad
        omega_az = scene.los.azimuth_rate_rad_s

    reference_change = inputs.reference_change
    if (
        not reset
        and reference_change is None
        and r_cog is not None
        and state.target.r_cog_ecef_m is not None
        and r_cog != state.target.r_cog_ecef_m
        and iss is not None
    ):
        old = predict_los(
            iss.utc_s,
            iss.r_m,
            iss.v_m_s,
            state.target.r_cog_ecef_m,
            eph.omega_earth_rad_s,
            eph.epoch_utc_s,
        )
        reference_change = PredictorReferenceChange(
            change_id=f"cog:{encoder.sample_id}",
            t_s=encoder.t_s,
            old_rate_rad_s=old.elevation_rate_rad_s,
            new_rate_rad_s=omega_t_nom,
        )
    history, _ = submit_event(history, encoder, now_s=inputs.now_s)
    if scene.los is not None:
        nominal = NominalRateSample(
            sample_id=f"nominal:{encoder.sample_id}",
            t_s=encoder.t_s,
            rate_rad_s=omega_t_nom,
        )
        history, _ = submit_event(history, nominal, now_s=inputs.now_s)
    if reference_change is not None:
        history, _ = submit_event(history, reference_change, now_s=inputs.now_s)
    if sample is not None and sample.z_v is not None:
        observation = VisionObservation(
            frame_id=sample.frame_id,
            t_s=sample.t_s,
            error_rad=sample.z_v,
            measurement_variance_rad2=filt.r_v,
        )
        history, _ = submit_event(history, observation, now_s=inputs.now_s)
    estimate_t_s = max(encoder.t_s, history.checkpoint.t_s)
    estimate = estimate_at(history, filt, estimate_t_s)
    residual = estimate.state
    history = estimate.history
    vision_disposition = None
    if sample is not None:
        matching = [
            item.disposition for item in estimate.dispositions if item.event_id == sample.frame_id
        ]
        if matching:
            vision_disposition = matching[-1]

    state = bookkeep_vision(state, vision, inputs, params)
    live = state.aggregate_live
    e_hat = float(residual.x[0])
    omega_res = float(residual.x[1])
    if live:
        omega_scene_el = omega_t_nom + omega_res
    else:
        omega_scene_el = 0.0

    exposure_us = (
        sample.exposure_us
        if sample is not None and sample.exposure_us > 0.0
        else state.target.last_exposure_us
    )
    gimbal = params.config.gimbal
    omega_hw = math.radians(gimbal.max_hw_slew_rate_deg_per_s)
    omega_sharp = smear_cap_rad_s(
        exposure_us,
        params.config.preprocessing.max_motion_smear_px,
        params.config.sensor.optics.ifov_band_deg_per_px,
    )
    if live:
        omega_scene = omega_t_nom + omega_res
        omega_rel = clip_rate(params.config.controller.outer.Kp * e_hat, omega_sharp)
        decision = rate_decision(
            omega_scene,
            omega_rel,
            omega_scene + omega_rel,
            omega_sharp,
            theta_g_rad,
            math.radians(gimbal.el_science_min_deg),
            math.radians(gimbal.el_science_max_deg),
            omega_hw,
            params.max_decel_rad_s2,
            params.rate_loop_bandwidth_rad_s,
        )
    else:
        decision = RateDecision(
            scene_rate_rad_s=0.0,
            requested_relative_rate_rad_s=0.0,
            requested_rate_rad_s=0.0,
            commanded_rate_rad_s=0.0,
            smear_limit_rad_s=omega_sharp,
            hardware_limited=False,
            science_limited=False,
        )

    state = replace(
        state,
        node=OperateNode.TRACKING,
        residual=residual,
        residual_history=history,
        last_rate_decision=decision,
        last_e_az=(
            pinhole_error_rad(
                sample.p_cog,
                params.camera.width_px,
                params.camera.height_px,
                params.camera.pixel_pitch_m,
                params.camera.focal_length_m,
            )[0]
            if sample is not None and sample.p_cog is not None
            else state.last_e_az
        ),
        vision_disposition=vision_disposition,
        target=TargetState(
            r_cog_ecef_m=r_cog,
            last_exposure_us=exposure_us,
            last_theta_los=theta_los,
            last_omega_t_nom=omega_t_nom,
            last_omega_az_nom=omega_az,
            last_omega_scene_el=omega_scene_el,
        ),
        hold=state.hold,
    )
    return state, NodeOutcome(
        reference=RateReference(
            rate_rad_s=decision.commanded_rate_rad_s,
            envelope=params.science_envelope,
        ),
        policy=policy,
    )
