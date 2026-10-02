"""Graph parameters: typed config projection for the pure payload graphs.

GraphParameters wraps PactConfig once at the shell boundary so node and graph
functions receive one typed argument. Envelopes, camera geometry, plant terms,
policy defaults, and sensor limits derive only from existing config values;
nothing here tunes constants or reads hardware.

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from flight.libs.config import PactConfig
from flight.payload.gimbal.intersect import CameraGeometry
from flight.payload.gimbal.request import TravelEnvelope
from flight.payload.graphs.base import (
    EffectivePolicy,
    ImagingPolicy,
    InferencePolicy,
    ModelSelection,
    PolicyLimits,
    TickInputs,
)
from flight.payload.tracking.residual import ResidualFilter


@dataclass(frozen=True, slots=True)
class GraphParameters:
    """Typed config projection for pure graph and node functions.

    Attributes:
        config: Full flight configuration; graphs read their slices only.
        detailed_plant: True selects the simulation plant decel/bandwidth caps;
            False selects the production unbounded-law path.
        policy_revision: Capture-policy revision checked against CapturedVision
            context.
    """

    config: PactConfig
    detailed_plant: bool = True
    policy_revision: int = 0

    @property
    def camera(self) -> CameraGeometry:
        """Band-plane pinhole geometry for centroid intersection."""
        sensor = self.config.sensor
        return CameraGeometry(
            width_px=sensor.width_px,
            height_px=sensor.height_px,
            pixel_pitch_m=sensor.pixel_um * 1.0e-6,
            focal_length_m=sensor.optics.focal_length_mm * 1.0e-3,
        )

    @property
    def science_envelope(self) -> TravelEnvelope:
        """Science-window travel bounds with the hardware slew cap."""
        gimbal = self.config.gimbal
        return TravelEnvelope(
            theta_min_rad=math.radians(gimbal.el_science_min_deg),
            theta_max_rad=math.radians(gimbal.el_science_max_deg),
            omega_max_rad_s=math.radians(gimbal.max_hw_slew_rate_deg_per_s),
        )

    @property
    def hardware_envelope(self) -> TravelEnvelope:
        """Hardware travel bounds with the hardware slew cap."""
        gimbal = self.config.gimbal
        return TravelEnvelope(
            theta_min_rad=math.radians(gimbal.el_hw_min_deg),
            theta_max_rad=math.radians(gimbal.el_hw_max_deg),
            omega_max_rad_s=math.radians(gimbal.max_hw_slew_rate_deg_per_s),
        )

    @property
    def pose_envelope(self) -> TravelEnvelope:
        """Hardware travel bounds capped at the pose-loop rate limit."""
        gimbal = self.config.gimbal
        pose_cap = math.radians(self.config.controller.position.r_max_deg_per_s)
        hardware_cap = math.radians(gimbal.max_hw_slew_rate_deg_per_s)
        return TravelEnvelope(
            theta_min_rad=math.radians(gimbal.el_hw_min_deg),
            theta_max_rad=math.radians(gimbal.el_hw_max_deg),
            omega_max_rad_s=min(pose_cap, hardware_cap),
        )

    @property
    def stow_envelope(self) -> TravelEnvelope:
        """Hardware travel bounds capped at the bounded stow reference rate."""
        gimbal = self.config.gimbal
        return TravelEnvelope(
            theta_min_rad=math.radians(gimbal.el_hw_min_deg),
            theta_max_rad=math.radians(gimbal.el_hw_max_deg),
            omega_max_rad_s=math.radians(gimbal.xeryon.stow_reference_rate_deg_per_s),
        )

    @property
    def max_decel_rad_s2(self) -> float:
        """Plant deceleration bound; unbounded without a detailed plant."""
        if not self.detailed_plant:
            return math.inf
        gimbal = self.config.gimbal
        return gimbal.tau_max_nm / gimbal.J_kg_m2

    @property
    def rate_loop_bandwidth_rad_s(self) -> float:
        """Inner loop bandwidth bound; unbounded without a detailed plant."""
        if not self.detailed_plant:
            return math.inf
        return self.config.controller.inner.kp

    @property
    def policy_limits(self) -> PolicyLimits:
        """Validated sensor bounds for policy validation."""
        capture = self.config.sensor.capture
        return PolicyLimits(
            exposure_min_us=capture.exposure_min_us,
            exposure_max_us=capture.exposure_max_us,
            gain_min_db=capture.gain_min_db,
            gain_max_db=capture.gain_max_db,
            max_frame_rate_hz=capture.max_frame_rate_hz,
        )

    @property
    def residual_filter(self) -> ResidualFilter:
        """The configured residual estimator for OPERATE."""
        controller = self.config.controller
        return ResidualFilter.from_config(controller.residual, controller.outer.dt_s)

    @property
    def feedback_max_age_s(self) -> float:
        """Maximum encoder feedback age before motion must inhibit."""
        return self.config.controller.integrity.feedback_max_age_s

    def default_policy(self, enabled: bool) -> EffectivePolicy:
        """Graph default policy: acquisition and inference on or fully off.

        The capture interval is the slowest of the outer cadence, the camera
        frame-rate period, and the configured initial exposure. Exposure, gain,
        and duty cycle keep their configured values. Inference runs only when
        the graph enables it and the duty cycle is nonzero; products publish
        only when enabled.
        """
        capture = self.config.sensor.capture
        interval = max(
            self.config.controller.outer.dt_s,
            1.0 / capture.max_frame_rate_hz,
            capture.initial_exposure_us * 1.0e-6,
        )
        imaging = ImagingPolicy(
            acquisition_enabled=enabled,
            capture_interval_s=interval,
            duty_cycle=capture.duty_cycle,
            exposure_us=capture.initial_exposure_us,
            gain_db=capture.initial_gain_db,
            publish_products=enabled,
        )
        inference = InferencePolicy(
            enabled=enabled and capture.duty_cycle > 0.0,
            every_n_frames=1,
            model=ModelSelection.CONFIGURED,
        )
        return EffectivePolicy(imaging=imaging, inference=inference)


def encoder_fresh(inputs: TickInputs, params: GraphParameters) -> bool:
    """True when feedback is valid and the encoder sample is finite and timely.

    Inputs:
        inputs: Tick observations carrying the encoder sample.
        params: Parameters supplying the feedback age bound.

    Outputs:
        bool: False for missing, nonfinite, future, or stale feedback.
    """
    encoder = inputs.encoder
    if not inputs.health.feedback_valid or encoder is None:
        return False
    if not (
        math.isfinite(encoder.t_s)
        and math.isfinite(encoder.angle_rad)
        and math.isfinite(encoder.angle_variance_rad2)
    ):
        return False
    if encoder.angle_variance_rad2 < 0.0:
        return False
    gimbal = params.config.gimbal
    theta_min = math.radians(gimbal.el_hw_min_deg)
    theta_max = math.radians(gimbal.el_hw_max_deg)
    if not (theta_min - 1.0e-12 <= encoder.angle_rad <= theta_max + 1.0e-12):
        return False
    age = inputs.now_s - encoder.t_s
    return age >= -1.0e-12 and age <= params.feedback_max_age_s + 1.0e-12
