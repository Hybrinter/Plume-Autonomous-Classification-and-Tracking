"""OPERATE FAST_REWIND node: hardware-rate hunt toward the science limb (pure).

After the REWIND timer expires the hunt runs at the hardware slew rate; the
nominal scene rate is not added. The node stores no target CoG and leaves the
residual history frozen.

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001, REQ-AIML-GIMB-008.
"""

from __future__ import annotations

import math
from dataclasses import replace

from flight.libs.types import Err, FaultCode
from flight.payload.gimbal.outer import RateDecision, rate_decision, smear_cap_rad_s
from flight.payload.gimbal.request import InhibitReference, RateReference
from flight.payload.gimbal.scene import boresight_scene
from flight.payload.graphs.base import NodeOutcome, SystemRequestIntent, TickInputs
from flight.payload.graphs.operate.state import OperateNode, State, bookkeep_vision
from flight.payload.graphs.parameters import GraphParameters, encoder_fresh


def step(
    state: State,
    inputs: TickInputs,
    params: GraphParameters,
) -> tuple[State, NodeOutcome[OperateNode]]:
    """One FAST_REWIND tick: hardware-rate hunt, nominal not added.

    Inputs:
        state: OPERATE state.
        inputs: Tick observations with the resolved effective vision sample.
        params: Graph parameters.

    Outputs:
        tuple[State, NodeOutcome[OperateNode]]: Updated state and the
            RateReference outcome under the science envelope.
    """
    resolved = params.operating_policy(params.config.payload_policy.fast_rewind)
    if isinstance(resolved, Err):
        return state, NodeOutcome(
            reference=InhibitReference(reason="invalid_policy"),
            policy=params.default_policy(enabled=False),
            system_request=SystemRequestIntent.SAFE,
            faults=(FaultCode.COMMAND_INVALID,),
        )
    policy = resolved.value
    encoder = inputs.encoder
    if encoder is None or not encoder_fresh(inputs, params):
        return state, NodeOutcome(
            reference=InhibitReference(reason="fast_rewind_stale_feedback"), policy=policy
        )
    state = bookkeep_vision(state, inputs.vision, inputs, params)
    theta_g_rad = encoder.angle_rad
    iss = inputs.navigation
    eph = params.config.ephemeris
    height_m = params.config.controller.predictor.cog_height_m
    iss_r = None if iss is None else iss.r_m
    iss_v = None if iss is None else iss.v_m_s
    iss_utc = None if iss is None else iss.utc_s
    scene = boresight_scene(
        iss_r,
        iss_v,
        iss_utc,
        theta_g_rad,
        height_m,
        eph.omega_earth_rad_s,
        eph.epoch_utc_s,
        eph.wgs84_a_m,
        eph.wgs84_f,
    )
    if scene.los is None:
        omega_t_nom = 0.0
        theta_los = 0.0
        omega_az = 0.0
    else:
        omega_t_nom = scene.los.elevation_rate_rad_s
        theta_los = scene.los.elevation_rad
        omega_az = scene.los.azimuth_rate_rad_s

    sample = inputs.vision.sample if inputs.vision is not None else None
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
    theta_sci_max = math.radians(gimbal.el_science_max_deg)
    if theta_g_rad >= theta_sci_max - 1e-9:
        decision = RateDecision(
            scene_rate_rad_s=omega_t_nom,
            requested_relative_rate_rad_s=0.0,
            requested_rate_rad_s=0.0,
            commanded_rate_rad_s=0.0,
            smear_limit_rad_s=omega_sharp,
            hardware_limited=False,
            science_limited=True,
        )
    else:
        decision = rate_decision(
            omega_t_nom,
            omega_hw,
            omega_hw,
            omega_sharp,
            theta_g_rad,
            math.radians(gimbal.el_science_min_deg),
            theta_sci_max,
            omega_hw,
            params.max_decel_rad_s2,
            params.rate_loop_bandwidth_rad_s,
        )

    state = replace(
        state,
        node=OperateNode.FAST_REWIND,
        last_rate_decision=decision,
        target=replace(
            state.target,
            r_cog_ecef_m=None,
            last_exposure_us=exposure_us,
            last_theta_los=theta_los,
            last_omega_t_nom=omega_t_nom,
            last_omega_az_nom=omega_az,
            last_omega_scene_el=omega_t_nom,
        ),
    )
    return state, NodeOutcome(
        reference=RateReference(
            rate_rad_s=decision.commanded_rate_rad_s,
            envelope=params.science_envelope,
        ),
        policy=policy,
    )
