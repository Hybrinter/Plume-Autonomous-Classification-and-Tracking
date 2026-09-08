# Estimator structure follow-up

Analysis date: 2026-09-08. Selection remains open; no flight estimator is changed
by this investigation.

## What the initial checkpoint does not establish

The marginal residual-filter advantage in `tracking-estimator-comparison.md`
does not establish a preferred structure:

- The residual candidate receives analytic gimbal rate. The joint candidate
  ignores that input and estimates gimbal motion from encoder-angle updates.
  Receiving the same event objects does not mean using equivalent information.
- The delayed-arrival metric compares the replayed endpoint against truth at
  the arriving event's acquisition time, which may be an older time.
- The history runner discards old events without retaining a posterior at the
  window boundary. It therefore restarts from the initial prior after trimming.
  Measurement events also need propagation to their actual timestamps; sorting
  events alone does not provide that propagation.
- Both initial candidates use noise covariances but the original trajectory
  observations contain no sampled measurement noise.
- The residual candidate rebases its rate on every nominal-rate change.
  After initialization, this cancels navigation's effect on total-rate
  propagation. A smooth physical prediction and a replacement of a reference
  must have different semantics.
- The optical sweep saturates the sharp-area score for both candidates. Its
  intensity-weighted rendered centroid also differs from the specified centroid
  of the union of accepted pixels. It is an image-integration demonstration,
  not a discriminating test of the intended tracking objective.

## Separate the design choices

State coordinates, encoder uncertainty, and navigation trust are separate
decisions. A two-state filter can estimate either a residual rate or a total
apparent target rate. A joint angular filter can also use a navigation model;
its current implementation simply does not do so.

For residual coordinates, a discrete reference replacement should transform
the rate state to preserve the estimated total rate. Smooth evolution of the
same physical predictor should remain in propagation. In total-rate
coordinates, the equivalent model supplies predicted angular acceleration.
Neither formulation requires estimating plume range, altitude, or motor inertia.

Navigation derived partly from the same image geometry must not be added as an
independent measurement without accounting for the shared uncertainty.

## Candidate structures

| Structure | Benefit | Cost or limitation |
| --- | --- | --- |
| Two-state residual with uncertain encoder input | Preserves the intended predictor-plus-visual-correction structure | Encoder rate-fit noise and its temporal correlation need treatment; reference semantics must be repaired |
| Two-state target angle/rate from synchronized encoder plus vision | Avoids differentiating encoder angle in the outer estimator | Requires shutter-time alignment and combined measurement uncertainty; interpolated/reused encoder errors are correlated |
| Joint target and gimbal angle/rate | Handles encoder and relative vision as distinct updates with state cross-covariance | Adds a gimbal motion model and tuning; model mismatch can contaminate inferred target motion |

The inner encoder rate loop remains separate in all three options. An outer
estimator replacement does not imply replacing the inner PI loop.

## Evidence needed for selection

Compare candidates with the same noisy encoder samples and accepted-pixel
centroids, at matching estimate timestamps. Separate startup, established
tracking, bounded visual gaps, and reacquisition. Sweep navigation bias and
acceleration error as well as encoder timing/noise. Report covariance consistency
alongside angle/rate error. Validate delayed replay against an independent
chronological solution across history-window boundaries.

## Executed sensitivity experiments

Three GPT-5.6 Luna agents at xhigh effort generated analysis scripts. These
experiments modify no flight code. Scripts and full output are currently in
`/tmp`; the principal results and limitations are recorded here for persistence.

### Noisy measurements and process mismatch

`/tmp/pact_estimator_sweep.py` compared 12 paired seeds (100 through 111),
10-second trajectories, and a 2-second startup exclusion. The residual filter
used the existing timestamped polynomial rate fit on the same noisy encoder
angles supplied to the joint filter. A third, two-state total-target candidate
used synchronized encoder-plus-vision angle observations.

Mean established-tracking RMSE, expressed as rate / relative angle:

| Case | Residual with fitted encoder rate | Joint angular | Two-state total target |
| --- | --- | --- | --- |
| Baseline | 0.628 mrad/s / 0.327 mrad | 0.652 mrad/s / 0.313 mrad | 0.656 mrad/s / 0.345 mrad |
| Encoder noise standard deviation 0.349 mrad | 0.685 mrad/s / 0.411 mrad | 0.665 mrad/s / 0.321 mrad | 0.672 mrad/s / 0.443 mrad |
| Eightfold target acceleration amplitude | 4.514 mrad/s / 1.644 mrad | 4.868 mrad/s / 1.455 mrad | 4.913 mrad/s / 1.920 mrad |
| 60% random visual dropout | 0.857 mrad/s / 0.650 mrad | 0.860 mrad/s / 0.647 mrad | 0.884 mrad/s / 0.684 mrad |

These are synthetic sensitivity results with fixed tuning, not a hardware noise
requirement or an optimized comparison. Encoder sampling intervals were
0.11--0.23 seconds, much slower than the intended inner-loop sampling. The
joint filter can improve relative-angle error while worsening target-rate
error. The total-target alternative is viable but did not dominate this sweep.
Full output: `/tmp/pact_estimator_sweep_total_12.txt`.

### Smooth prediction versus reference replacement

`/tmp/predictor_structure_study.py` isolates navigation with a stationary
gimbal, matched initial target-rate priors, 20 ms propagation, 10 Hz noisy
vision, and true target acceleration of 0.0025 rad/s². It uses one seed
(20260908), an explicit reference replacement, and a four-second visual gap.
The gap is a diagnostic stress test beyond the bounded flight coast.

During that gap, the current residual candidate's angle RMSE was 0.014969 rad
regardless of the predictor's acceleration. A variant that rebases only at
explicit reference replacements produced:

| Predictor acceleration | Gap angle RMSE | Gap rate RMSE |
| --- | ---: | ---: |
| Correct | 0.000838 rad | 0.000331 rad/s |
| Half the true acceleration | 0.007896 rad | 0.004088 rad/s |
| Opposite sign | 0.029117 rad | 0.015426 rad/s |

Thus smooth prediction materially helps when correct and can materially hurt
when wrong. This demonstrates prediction-model semantics, not inherent
superiority of residual state coordinates: a total-rate or joint estimator
could receive the same predicted acceleration. Final verified output:
`/tmp/predictor_structure_final.txt`.

### Chronology audit

`/tmp/estimator_audit.py` reproduced the original comparison. Scoring at the
replay endpoint corrected relative-angle RMSE from 0.000630 to 0.000592 rad
for the residual candidate and from 0.000683 to 0.000648 rad for the joint
candidate. It did not reverse this particular ranking. Short history windows
also exposed prior loss, and a propagation carrying a new nominal reference
followed by an explicit transition could apply the reference correction twice.
These are study-harness defects to resolve before relying on broader delayed
or asynchronous comparisons.

## Current recommendation

Keep the two-state residual architecture as the provisional baseline, repair
its distinction between continuous prediction and discrete reference changes,
and represent encoder-input uncertainty. Retain the joint filter as a serious
challenger for noisy or asynchronous encoder data. Do not promote either
existing implementation on the basis of the initial checkpoint numbers.
