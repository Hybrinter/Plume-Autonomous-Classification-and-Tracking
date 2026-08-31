# ADR-REPO-0014: Cascaded gimbal LQG

**Status:** Accepted
**Date:** 2026-08-31
**Topic:** feature-add
**Supersedes:** ADR-REPO-0008
**Superseded-by:** none
**Related:** ADR-REPO-0005, ADR-REPO-0007

## Context

ADR-REPO-0008 closed the loop with a frame-synchronous Kalman+LQR placeholder, ROI crop in
TRACKING, and RATE commands into a first-order SimGimbal. The payload computer must run a
software inner rate loop that commands torque, because the custom gimbal has no vendor PTU
rate servo. Imaging must feed the full demosaiced 512x512 band plane. Vision latency needs a
rewind update. Inner PI ticks must not wait on inference.

## Decision

Replace the per-frame placeholder with a cascaded LQG:

- **Outer:** two uncoupled 4-state Kalman filters, `x = [e, θ_g, ω_t, ω_g]`. Continuous LQR
  on the stabilizable pair `[e, ω_g]`. Vision is an area-weighted CoM, EMA-smoothed, applied
  once through a rewind ring at shutter time. `r = 0` until the first accepted `z_vis`.
- **Inner:** two SISO rate PIs plus computed torque `τ = Ĵ v + B̂ y_m`. HAL `set_torque`.
  SIL plant `J ω̇ + B ω = τ`.
- **Imaging:** mosaic 1024 → demosaic → full 512x512 plane. No ROI crop. No search-mode
  decimation. `InferenceConfig` input size equals the band plane.
- **Threads:** `PayloadApp.process_frame` ingests only. A control thread inside `PayloadApp`
  waits `dt_inner_min`, reads the encoder, steps the PI onto `set_torque`, and periodically
  steps the outer loop. SIL calls the same pure ticks with no extra thread.
- **Safety:** clip travel, `r`, and `τ`. PI anti-windup at stops. Drop live deadband and
  encoder-runaway gates. SAFE still STOWs and resets the PI integrator.
- **Units:** pointing, LQR, and telemetry stay in degrees. Servo and SimGimbal plant use
  rad and N·m.

Tools training and ONNX export stay at 256 until a follow-up retraining. CI uses
`ScriptedDetector`. `OnnxDetector` follows flight config (512); shipped 256 artifacts do not
match until retrained.

## Consequences

- `GimbalActuator.set_torque` is the inner command. `RealGimbal.set_torque` is a stub until
  hardware is selected.
- SCAN writes the same `r` register with a stub P-on-angle law. A path generator is out of
  scope.
- Coupled 8×8 outer filter, encoder-rate Kalman update, `J(θ)`, and Coriolis are out of
  scope.

## Notes

Descriptive pages and code do not cite this identifier.
