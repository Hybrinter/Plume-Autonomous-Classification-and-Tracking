# Two-state residual estimator with encoder angle increments

This plan replaces encoder-rate input in the flight residual estimator with
timestamped encoder angle displacement. It also repairs delayed-observation
chronology and prepares the production control path for the Xeryon
XRT-U-40-109-HV with an XD-C controller. The physical estimator state remains

\[
x = [e,\; \omega_{t,res}]^T,
\]

where `e` is target-to-boresight elevation error and `omega_t,res` is apparent
target rate relative to the nominal orbital predictor.

## Selected behavior

For an interval from `a` to `b`, propagate the state with

\[
e_b = e_a + \Delta\theta_{t,nom} + \Delta t\,\omega_{t,res}
      - (\theta_{g,b}-\theta_{g,a}).
\]

The estimator consumes encoder positions and sample times. It never estimates
the outer-loop gimbal displacement by differentiating encoder angle. Smooth
changes in the nominal predictor contribute to `delta_theta_t_nom`. An explicit
replacement of the predictor reference rebases `omega_t,res` by
`old_nominal_rate - new_nominal_rate` so the estimated total target rate stays
continuous.

Encoder uncertainty is an uncertain-input covariance, not a second measurement
update. A displacement carries the uncertainty of its anchor and endpoint plus
a separate reversal term. Repeated estimates from the same anchor must not add
the shared anchor uncertainty on every outer tick. The implementation therefore
replays from a stable posterior anchor and computes one net encoder displacement
to each vision or reporting endpoint. This is the selected two-state anchored
correlation approximation. It must be assessed with innovation and Monte Carlo
coverage tests; it is not advertised as an exact model of every interpolated
encoder-error correlation.

## 1. Introduce explicit estimator-domain types

Replace `ResidualSnapshot` and its `y_m` field in
`packages/flight/src/flight/payload/tracking/residual.py` with immutable types
whose units and timestamp meanings are explicit:

- `EncoderSample`: stable sample ID, encoder sample time, unwrapped angle, and
  angle variance.
- `NominalRateSample`: sample time and nominal apparent target rate. Between
  samples, integrate the rate with the documented causal interpolation rule.
- `PredictorReferenceChange`: time, old reference rate, and replacement rate.
- `VisionObservation`: frame ID, shutter time, relative elevation error, and
  measurement variance.
- `ResidualCheckpoint`: posterior time, `ResidualState`, encoder angle at the
  checkpoint, encoder endpoint variance, and active nominal reference.
- `ResidualHistory`: one preserved checkpoint plus bounded raw encoder,
  predictor, reference-change, and accepted-vision events.
- `ObservationDisposition`: accepted, duplicate, future, expired, no encoder
  bracket, invalid encoder, or stale timing.

Keep these types in the pure tracking package. Do not import HAL classes, read a
clock, log, or perform I/O in the estimator.

Add a low-level `propagate_displacement()` operation with inputs
`dt_s`, `nominal_angle_delta_rad`, `encoder_angle_delta_rad`, and explicit
encoder/reversal variances. Replace the current diagonal, outer-period-scaled
process noise with a continuous white-acceleration model:

\[
Q(\Delta t)=q_a
\begin{bmatrix}
\Delta t^3/3 & \Delta t^2/2\\
\Delta t^2/2 & \Delta t
\end{bmatrix}.
\]

Use Joseph-form measurement updates and symmetrize covariance after each
update. Reject negative propagation intervals rather than using inverse
prediction as a rewind mechanism.

## 2. Replace rewind snapshots with deterministic event replay

Add a pure `submit_event()` and `estimate_at()` history API. `submit_event()`
deduplicates by stable event ID and records an explicit disposition.
`estimate_at()` sorts by `(event time, causal priority, event ID)` and replays
from `ResidualCheckpoint` through the requested endpoint.

The causal order at an equal timestamp is:

1. Finish physical propagation to the timestamp.
2. Apply an explicit predictor-reference replacement.
3. Establish the encoder endpoint used by later displacement.
4. Apply the vision observation.

Interpolate encoder angle at a shutter time only from valid bracketing samples.
Do not extrapolate or substitute a cached angle. Propagate between posterior
anchors with the net unwrapped encoder displacement; add endpoint quantization
uncertainty once for that span and one reversal variance for each detected
direction reversal. Keep reversal detection thresholds in configuration so
encoder chatter does not create false reversals.

When history exceeds its horizon, first replay all events through the new
boundary and save that posterior and interpolated encoder anchor as the new
checkpoint. Then discard older events while retaining the predecessor encoder
sample needed to bracket the boundary. Trimming must never cold-start from
`P0` or discard an accepted vision correction.

The history API returns the estimate and event dispositions. It must be
deterministic for repeated calls and must not mutate the committed posterior
merely because the caller asks for a newer reporting time.

## 3. Make outer-loop encoder sampling independent of the inner model

Update `ControlState` in `packages/flight/src/flight/payload/control.py` to own a
`ResidualHistory` and the latest residual estimate. Remove `snapshots` from the
state. Keep `y_m`, the polynomial-rate ring, PI integrator, and torque fields
only for the detailed plant simulation and its integrity checks during the
transition.

Change `PayloadController.outer_step()` to receive a valid timestamped encoder
sample rather than an unqualified `theta_g_rad`. At each outer step it:

1. Submits new encoder and nominal-predictor samples to the history.
2. Submits any explicit predictor-reference change.
3. Submits a vision event at its shutter time when the encoder history brackets
   that time.
4. Requests the replayed estimate at the current encoder sample time.
5. Computes `r = omega_t_nom + omega_t_res + Kp * e`, then applies the existing
   science, smear, and slew limits.

Treat smooth predictor evolution as sampled physics. Add a separate controller
method or input variant for an explicit reference replacement; do not infer a
replacement from every change in the numeric nominal rate.

In `packages/flight/src/flight/payload/app.py`, maintain one timestamped encoder
stream for both frame association and outer control. Update the outer baseline
only after a valid sample. Encoder failure invalidates motion authority and the
baseline; recovery starts a new checkpoint and cannot fabricate zero
displacement from the cached angle. Remove the production 1 kHz software torque
thread when the Xeryon rate-command driver is enabled. Retain the current inner
PI and polynomial rate fit behind a simulation-only detailed-plant mode.

Fix SIL catch-up before enabling the new estimator: historical outer ticks may
consume only encoder samples whose sample times belong to those ticks. They may
not reread the current encoder and label that value with an older control time.

## 4. Promote sample-time and Xeryon contracts through HAL and configuration

Keep `GimbalPosition` as the basic feedback record but strengthen its contract:
`timestamp_s` is mapped encoder sample time in the application monotonic domain,
not host receipt time. Add raw controller time, sequence, status bits, and the
time-mapping uncertainty if the real driver needs them to establish that claim.

Replace the production torque command with a signed rate command carrying a
valid-until deadline. The Xeryon adapter owns the vendor Python library and one
serial connection. It maps absolute rate to the nearest 0.01 deg/s `SSPD`, uses
a half-step deadband, and sequences direction changes as stop, confirm, set
speed, and scan. Configure UART at 76,800 baud for the intended Jetson path
after electrical validation and retain USB at 115,200 baud for bench work.
Configure the feedback stream initially as `INFO=4`, `POLI=2 ms`, then measure
the achieved complete-frame cadence and serial utilization before freezing the
setting.

Add typed production hardware configuration for:

- XRT-U-40-109-HV model and XD-C transport.
- 86,400 controller counts per revolution and 109 microradian effective encoder
  resolution as separate values.
- Minimum incremental motion, uni/bidirectional repeatability, and wobble as
  characterization bounds rather than independent sample noise.
- 0.01 deg/s command quantum, rated speed limits, and the existing 10 deg/s
  software tracking limit.
- Encoder/reversal covariance, feedback age, interpolation span, and timing
  uncertainty limits.
- The HV 120-second maximum-on interval and 50 percent duty constraint.
- Bounded stow-reference rate and timeout.

Keep simulation inertia, damping, torque, and encoder-noise fields in a
simulation-specific config section. Do not use the stage's maximum supported
payload inertia as the simulated payload inertia.

The real driver remains motion-disabled until the vendor source/settings pass a
license and Python 3.14 audit, an independent hardware watchdog is selected and
tested, and the bounded stow-reference procedure passes bench tests. Referencing
uses the constrained stow switch; it does not command a full-turn index search.

## 5. Add observability and explicit failure behavior

Publish enough estimator telemetry to reconstruct every accepted result:

- `e`, residual/nominal/total target rate, and the 2-by-2 covariance.
- Encoder anchor and endpoint IDs, net displacement, sample ages, interpolation
  span, and timing uncertainty.
- Process, endpoint, and reversal covariance contributions.
- Vision event ID, shutter/arrival times, and disposition.
- Requested and quantized gimbal rate, Xeryon status, duty credit, and watchdog
  state.

Map encoder-invalid, controller-error, thermal, safety-timeout, loss-of-closed-
loop, stale feedback, and time-mapping failures to immediate local stop and
drive inhibition. Serial acknowledgement alone does not establish physical
inhibition; the external watchdog gate must confirm removal of motion authority.

Implement HV duty as a conservative credit bucket. Motor-on status drains one
second per second and motor-off status replenishes at the same rate. At zero
credit, command stop, inhibit, and hold motion disabled until the full
120 seconds of credit has recovered. Configure the XD-C motor-on safety timeout
as a backup, not as the independent communication watchdog.

## 6. Migrate tests and analysis before deleting the old residual path

Build an independent chronological oracle under analysis tests. It may share
event dataclasses and the continuous-noise formula, but it must not call the
production history, propagation, update, interpolation, or trimming functions.

Add focused flight tests for:

- Constant target and gimbal kinematics using nonuniform angle increments.
- Encoder displacement affecting the residual independently of `y_m`.
- Process covariance time-partition identity.
- No encoder covariance inflation when the same anchored span is partitioned
  into more reporting ticks.
- One additional reversal covariance charge per genuine reversal.
- Smooth nominal evolution versus explicit reference replacement.
- First vision acquisition without navigation.
- Delayed and out-of-order vision replay matching the oracle in mean and
  covariance, including after multiple history trims.
- Equal timestamps, duplicates, future events, expired events, missing encoder
  brackets, stale feedback, and clock reset.
- Recovery starting a new encoder anchor without a synthetic zero increment.

Use a fake vendor library to test rate rounding, deadband, sign-change command
order, feedback pairing, controller-time mapping, status faults, duty lockout,
watchdog gating, and shutdown. Verify that shutdown never calls a vendor helper
that homes the stage as a side effect.

Extend SIL with paired deterministic seeds and the XRT model: 109 microradian
effective resolution, 86,400-count conversion, 0.01 deg/s command steps,
measured feedback cadence, timestamp jitter, vision delay/dropout, reversals,
navigation outage, 250 ms visual coast, reacquisition, serial loss, watchdog
expiry, and duty exhaustion. Keep the joint and total-target estimators as
analysis-only benchmarks; do not add a runtime estimator selector.

## 7. Acceptance and delivery sequence

Deliver the change as reviewable stages:

1. **Estimator math and oracle:** new domain types, continuous process noise,
   displacement propagation, reference semantics, and unit tests. No controller
   integration.
2. **History and chronology:** event replay, shutter interpolation,
   dispositions, trimming, and randomized oracle equivalence tests.
3. **Flight integration:** controller/app state migration, timing-safe SIL
   stepping, telemetry, and deletion of outer-estimator `y_m` use.
4. **Xeryon interface:** rate-command HAL, fake vendor adapter, XRT config,
   duty logic, and fail-closed real-driver scaffolding.
5. **Qualification:** hardware-in-the-loop timing/containment tests, tuned noise
   parameters, updated comparison report, requirements evidence, and production
   motion enablement.

The estimator stages pass only when all chronological cases match the
independent oracle, covariance stays symmetric positive semidefinite, and
Monte Carlo innovations are consistent with the configured uncertainty.

The nominal science SIL target is at most one band pixel steady-state relative
angle RMSE and at most two band pixels through a configured 250 ms vision coast.
One band pixel is approximately 46 microradians for the configured optics. Run
at least 32 paired deterministic seeds and require every run to remain within
the gimbal safety envelope. A miss blocks promotion; it does not trigger a
runtime switch to a competing estimator.

Hardware promotion additionally requires measured controller/sample timing,
time-mapping uncertainty compatible with the one-pixel target at the maximum
science tracking rate, physical watchdog cutoff on Jetson hang and serial
disconnect, bounded stow referencing, safe inhibit/shutdown behavior, and
thermal-duty verification.

After each stage, run the affected unit and SIL tests. Before merge, run every
repository gate listed in `AGENTS.md`, regenerate the estimator comparison and
VCRM artifacts, and update the tracking, controller, app, HAL, configuration,
simulation, and design documentation to describe the final as-built behavior.
