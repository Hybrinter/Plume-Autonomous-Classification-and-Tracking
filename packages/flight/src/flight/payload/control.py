"""Payload servo control: the mode-free inner elevation loop (pure core).

The ServoController maps a typed ControlReference produced by the active
payload graph into a bounded rate reference, then advances the encoder ring,
polynomial rate estimate, PI + computed torque. It holds no graph, mode, or
vision knowledge; SAFE/inhibit reach it only as references.

Pure: no I/O, no bus, no clock reads. Time, encoder samples, and the control
reference are arguments.

Satisfies: REQ-AIML-GIMB-002, REQ-GIMB-HIGH-001, REQ-GIMB-HIGH-003.
"""

from __future__ import annotations

# stdlib
import math
from dataclasses import dataclass, replace

# internal
from flight.libs.config import ControllerConfig, GimbalConfig
from flight.libs.types import Err, FaultCode, Ok, Result
from flight.payload.gimbal import (
    fit_rate_timed,
    inner_step,
    position_rate,
)
from flight.payload.gimbal.outer import stopping_cap
from flight.payload.gimbal.request import (
    ControlReference,
    InhibitReference,
    PoseReference,
    StowReference,
    validate_reference,
)
from flight.payload.tracking import EncoderSample


@dataclass(frozen=True, slots=True)
class EncoderState:
    """Inner-loop encoder ring and polynomial rate estimate.

    Attributes:
        samples: Timestamped encoder samples, oldest to newest. One sequence.
        last_theta_enc_rad: Last encoder elevation, radians, or None.
        measured_rate_rad_s: Polynomial rate at the newest sample (inner y_m).
    """

    samples: tuple[EncoderSample, ...]
    last_theta_enc_rad: float | None
    measured_rate_rad_s: float


@dataclass(frozen=True, slots=True)
class InnerControlState:
    """Inner PI memory owned by inner_step.

    Attributes:
        integrator: Inner PI integrator.
        last_inner_s: Monotonic time of the last inner step, or None.
        last_tau_nm: Last inner torque, N·m.
    """

    integrator: float
    last_inner_s: float | None
    last_tau_nm: float


@dataclass(frozen=True, slots=True)
class IntegrityState:
    """Light integrity detector strike counter.

    Attributes:
        freeze_strikes: Consecutive encoder-freeze inner ticks.
    """

    freeze_strikes: int


@dataclass(frozen=True, slots=True)
class ServoState:
    """Servo state threaded across inner ticks: the physical-plant memory only.

    Attributes:
        encoder: Inner encoder ring and measured rate.
        inner: Inner PI integrator, time, and torque.
        integrity: Encoder-freeze strike counter.
        commanded_rate_rad_s: Last applied rate reference.
    """

    encoder: EncoderState
    inner: InnerControlState
    integrity: IntegrityState
    commanded_rate_rad_s: float


@dataclass(frozen=True, slots=True)
class InnerTick:
    """Outputs of one inner_step on the servo controller.

    Attributes:
        state: Updated ServoState.
        tau_nm: Torque command, N·m.
    """

    state: ServoState
    tau_nm: float


@dataclass(frozen=True)
class ServoController:
    """Pure mode-free servo: typed reference in, bounded torque out.

    Attributes:
        cfg: ControllerConfig (inner/position/integrity slices only).
        gimbal: GimbalConfig (plant, envelopes, encoder).
    """

    cfg: ControllerConfig
    gimbal: GimbalConfig

    @staticmethod
    def from_config(cfg: ControllerConfig, gimbal: GimbalConfig) -> ServoController:
        """Build the servo controller from its config slices."""
        return ServoController(cfg=cfg, gimbal=gimbal)

    def initial_state(self) -> ServoState:
        """Empty encoder ring, zero integrator, zero commanded rate."""
        return ServoState(
            encoder=EncoderState(
                samples=(),
                last_theta_enc_rad=None,
                measured_rate_rad_s=0.0,
            ),
            inner=InnerControlState(integrator=0.0, last_inner_s=None, last_tau_nm=0.0),
            integrity=IntegrityState(freeze_strikes=0),
            commanded_rate_rad_s=0.0,
        )

    def reference_rate(
        self,
        reference: ControlReference,
        theta_rad: float,
        detailed_plant: bool = True,
    ) -> Result[float, FaultCode]:
        """Map a typed reference onto a bounded absolute rate, rad/s.

        Inputs:
            reference: The committed ControlReference.
            theta_rad: Current encoder elevation, rad.
            detailed_plant: Use finite plant deceleration and loop bandwidth in
                the stopping guard; False uses unbounded (production) terms.

        Outputs:
            Result[float, FaultCode]: The bounded rate, or Err(COMMAND_INVALID)
            on an invalid reference. InhibitReference yields 0.0.
        """
        check = validate_reference(reference)
        if isinstance(check, Err):
            return check
        if not math.isfinite(theta_rad):
            return Err(FaultCode.COMMAND_INVALID)
        if isinstance(reference, InhibitReference):
            return Ok(0.0)
        envelope = reference.envelope
        if isinstance(reference, (PoseReference, StowReference)):
            r = position_rate(
                reference.target_rad,
                theta_rad,
                self.cfg.position.K_pos,
                min(
                    math.radians(self.cfg.position.r_max_deg_per_s),
                    envelope.omega_max_rad_s,
                ),
            )
            return Ok(max(-envelope.omega_max_rad_s, min(envelope.omega_max_rad_s, r)))
        max_decel = self.gimbal.tau_max_nm / self.gimbal.J_kg_m2 if detailed_plant else math.inf
        rate_cap = self.cfg.inner.kp if detailed_plant else math.inf
        guard = math.radians(self.cfg.integrity.science_boundary_guard_deg)
        r = reference.rate_rad_s
        if r > 0.0:
            remaining = envelope.theta_max_rad - guard - theta_rad
            r = min(r, stopping_cap(remaining, max_decel, rate_cap))
        elif r < 0.0:
            remaining = theta_rad - envelope.theta_min_rad - guard
            r = max(r, -stopping_cap(remaining, max_decel, rate_cap))
        return Ok(max(-envelope.omega_max_rad_s, min(envelope.omega_max_rad_s, r)))

    def inner_step(
        self,
        state: ServoState,
        now: float,
        encoder: EncoderSample,
        reference: ControlReference,
        dt_s: float | None = None,
    ) -> InnerTick:
        """One inner tick: push encoder, fit y_m, map the reference, PI torque.

        Inputs:
            state: Current servo state.
            now: Monotonic seconds of this tick.
            encoder: Timestamped encoder sample for this tick.
            reference: The committed ControlReference; InhibitReference zeroes
                the rate and resets dynamic PI outputs.
            dt_s: Inner period; defaults to cfg.inner.dt_s.

        Outputs:
            InnerTick: Updated state and torque.
        """
        dt = self.cfg.inner.dt_s if dt_s is None else dt_s
        theta_enc_rad = encoder.angle_rad
        samples = state.encoder.samples + (encoder,)
        max_n = self.cfg.inner.rate_fit_n
        if len(samples) > max_n:
            samples = samples[-max_n:]
        y_m = fit_rate_timed(
            tuple(item.angle_rad for item in samples),
            tuple(item.t_s for item in samples),
            self.cfg.inner.rate_fit_n,
            self.cfg.inner.rate_fit_degree,
        )
        el_deg = math.degrees(theta_enc_rad)
        stopped = (
            el_deg <= self.gimbal.el_hw_min_deg + 1e-9 or el_deg >= self.gimbal.el_hw_max_deg - 1e-9
        )
        quantum_deg = (
            self.gimbal.xeryon.effective_encoder_resolution_urad * 1.0e-6 * (180.0 / math.pi)
        )
        if isinstance(reference, InhibitReference):
            return self._zero_tick(state, samples, theta_enc_rad, y_m, now)
        rate = self.reference_rate(reference, theta_enc_rad)
        if isinstance(rate, Err):
            return self._zero_tick(state, samples, theta_enc_rad, y_m, now)
        r = rate.value
        theta_min = reference.envelope.theta_min_rad
        theta_max = reference.envelope.theta_max_rad
        # One encoder count differentiated at the inner rate looks like several deg/s.
        # Against a hard stop that phantom rate commands torque into the stop, and
        # the stage cannot move the other way to bleed it. Drop the inbound estimate
        # while the rate command points off the stop.
        if el_deg <= self.gimbal.el_hw_min_deg + 3.0 * quantum_deg and r > 0.0:
            y_m = min(y_m, 0.0)
        elif el_deg >= self.gimbal.el_hw_max_deg - 3.0 * quantum_deg and r < 0.0:
            y_m = max(y_m, 0.0)
        at_bound = (theta_enc_rad <= theta_min + 1e-9 and r < 0.0) or (
            theta_enc_rad >= theta_max - 1e-9 and r > 0.0
        )
        integrator_in = state.inner.integrator
        result = inner_step(
            r,
            y_m,
            integrator_in,
            dt,
            self.gimbal.J_kg_m2,
            self.gimbal.B_nms_per_rad,
            self.cfg.inner.kp,
            self.cfg.inner.ki,
            self.gimbal.tau_max_nm,
            stopped or at_bound,
        )
        new_state = replace(
            state,
            encoder=EncoderState(
                samples=samples,
                last_theta_enc_rad=theta_enc_rad,
                measured_rate_rad_s=y_m,
            ),
            inner=InnerControlState(
                integrator=result.integrator,
                last_inner_s=now,
                last_tau_nm=result.tau_nm,
            ),
            commanded_rate_rad_s=r,
        )
        return InnerTick(state=new_state, tau_nm=result.tau_nm)

    @staticmethod
    def _zero_tick(
        state: ServoState,
        samples: tuple[EncoderSample, ...],
        theta_enc_rad: float,
        y_m: float,
        now: float,
    ) -> InnerTick:
        """Encoder-frame update with zeroed dynamic PI outputs and no torque."""
        new_state = replace(
            state,
            encoder=EncoderState(
                samples=samples,
                last_theta_enc_rad=theta_enc_rad,
                measured_rate_rad_s=y_m,
            ),
            inner=InnerControlState(
                integrator=0.0,
                last_inner_s=now,
                last_tau_nm=0.0,
            ),
            commanded_rate_rad_s=0.0,
        )
        return InnerTick(state=new_state, tau_nm=0.0)
