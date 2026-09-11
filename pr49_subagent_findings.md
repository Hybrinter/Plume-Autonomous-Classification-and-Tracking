# PR #49: Controller Review and Agent Handoff

## Project and study briefing

**Project:** PACT (Plume Autonomous Classification and Tracking) is an ISS-attached payload project
for imaging, classifying, and tracking Earth-viewed plumes. Its Python workspace contains flight
software, simulated hardware and software-in-the-loop (SIL) execution, ground-support equipment
(GSE), ML/tooling, and analysis studies. Flight software uses typed messages between subsystem apps,
pure decision cores, and injected hardware interfaces. Simulation supports development without
flight hardware; passing those simulations is not automatically evidence of physical accuracy.

**Why this study exists:** The user is replacing the former dual-axis/LQR pointing approach with a
single elevation gimbal. Their rationale is that ISS orbital motion and the Earth-viewing window
favor tracking within a single slice. That is the motivation, not a demonstrated guarantee that
cross-track motion or field-of-view loss is negligible. PR #49 is a candidate implementation to
critically examine, not a design the study must defend or preserve.

**Intended behavior:** Use orbital/time information and Earth rotation to predict nominal apparent
plume motion, then use vision to estimate residual motion and request gimbal angular velocity.
An encoder-informed inner PI loop commands motor torque at a sufficiently faster control bandwidth;
exact 1 kHz execution is not required. Point at the area-weighted union centroid of all accepted
visible plume pixels, while pursuing sharp plume imagery rather than merely low gimbal speed.
If encoder feedback becomes untrustworthy, inhibit drive first; do not blindly stow on failed feedback.

**What the user wants from the study:** Establish whether the physics, estimation, control, timing,
safety, architecture, and validation actually support that intent. Separate confirmed defects from
model assumptions, missing evidence, and unresolved mission requirements. Challenge earlier review
claims as rigorously as the PR itself. Compare better formulations, including joint angular
estimators, when justified; do not add unobservable plume range/altitude states or restore azimuth.
Hardware, timing, attitude/frame contracts, and science-quality limits remain partly unresolved.

**Handoff boundary:** This is a grounding review leading to informed design decisions, not an
implementation assignment. The user wants ranked improvements, worthwhile further investigations,
and focused questions before committing to changes. Five research leads and their SWE delegates
have completed source-based clarification checkpoints, not experimental qualification or final
architecture selection. The remaining sections preserve their evidence and the user's clarifications.

## Handoff scope and evidence status

This is a research handoff, not an implementation plan or an approved controller specification.
It consolidates the five Astra research leads' Stage 1 reports into recommended improvements,
open design questions, and independent validation criteria. The recommendations below are the
parent agent's synthesis. All five leads stopped at clarification checkpoints; they did not
complete a second-stage comparison or approve a final architecture.

- Repository: `/home/aiden/Plume-Autonomous-Classification-and-Tracking`.
- PR: https://github.com/Hybrinter/Plume-Autonomous-Classification-and-Tracking/pull/49
- Reviewed branch: `cursor/elevation-controller-3eeb`.
- Reviewed HEAD: `829364338a58b0f9368f0fb1091ee8d2e6eabd1d`.
- Scope: the single-axis cascaded elevation controller, not the old dual-axis code on `main`.
- Verification: source inspection, test inspection, and mathematical reasoning. The leads ran
  no tests or executable experiments. "Confirmed" below means confirmed by that evidence,
  not demonstrated on hardware or by a newly executed regression test.
- At handoff preparation, Git confirmed the same HEAD with no tracked modifications. This
  already-existing report was the only untracked file. Only this report is edited for the handoff.
- No flight, simulation, test, or configuration changes are authorized by this document.
  PR lifecycle state has not been refreshed and is not inferred from completion of this review.

The original report bodies remain below under **Evidence appendix**. They include narrower
qualifications and unresolved questions that must not be lost during later planning.

### How the next agent should use this document

Continue the grounding/design review unless the user explicitly authorizes implementation.
Recheck the checkout before relying on line numbers. Read the evidence for any finding you intend
to change. Distinguish defects that have clear required outcomes from architecture choices that
still need mission or hardware input. Do not silently turn provisional assumptions into requirements.

A useful next deliverable is a comparison of viable designs, their operating assumptions, and the
independent evidence needed to select one. Preserve unresolved requirements explicitly. Do not
replace the existing controller merely because a more elaborate estimator is possible.

## Established user intent

These points are already settled; do not ask the user to choose them again.

1. There is one actuated elevation axis. Do not restore azimuth actuation or dual-axis control.
2. The intended cascade combines orbital/Earth-rotation prediction and vision feedback in an outer
   loop, with encoder-informed rate PI and motor torque in an inner loop.
3. Exact 1 kHz execution is not mandatory. Bandwidth separation and feasible, measured timing matter.
   Motors and amplifiers are not selected, and formal stability qualification is not complete.
4. The target is the center of geometry of the union of accepted visible plume pixels, including
   multiple disjoint accepted components. It is not the first blob or an equally weighted blob mean.
5. Plume sharpness is the imaging objective. Low absolute gimbal speed is not itself that objective.
6. Unavailable or untrustworthy encoder feedback requires drive inhibition first, not blind stowing
   with the same failed feedback.
7. The user is open to different estimators and control formulations. The PR is not final.
8. The user intended "dual-update" to mean two measurement updates. A valid joint formulation may
   use encoder angle and relative visual angle; an encoder does not directly measure plume rate.
9. No independently measured plume altitude or ground-source position is assumed. A 3-D plume
   state is not established as observable from mask centroids and encoders alone.
10. Orbital information is expected through an external-data abstraction. Its placement under HAL
    is not itself an error. Frame, attitude, freshness, and mount alignment remain unresolved.

The earlier preference to maintain target continuity must not be used to reintroduce single-blob
selection. Continuity now concerns the aggregate and its membership/lifecycle policy.

## Executive assessment and priorities

The cascade is a defensible starting point, but current safety containment, measurement chronology,
and target semantics do not support a claim of physically validated autonomous plume tracking.
Correctness of a residual model on paper is separate from correctness of its delayed-data integration.

Priority denotes consequence and value of investigation, not implementation order or measured risk.
Confidence is high in the reported source behavior; deployed performance and physical consequences
remain unquantified where stated.

| Priority | Area | Classification | Recommended outcome |
|---|---|---|---|
| P0 | Failed encoder and integrity-trip torque path | Confirmed defect against inhibit-first intent | No continued feedback-controlled torque using invalid feedback; separate inhibition from stowing |
| P0 | Torque-write failure, inner-thread death, shutdown | Confirmed containment gap; missing hardware contract | Local containment, independently bounded command validity, and observable inhibit status |
| P0 | Limits, mechanical interlocks, SAFE clearance | Observed logic gaps; design mismatch and validation gap | Fault-specific recovery evidence and verified physical transition conditions |
| P1 | Delayed residual replay | Confirmed chronology defects | Preserve all accepted corrections and use inputs from the correct time intervals |
| P1 | Fractional-time covariance | Confirmed mathematical discrepancy; noise-model question | Consistent stochastic propagation over arbitrary interval partitions |
| P1 | Exposure, encoder, and controller time | Confirmed association/timing defects | Explicit event-time alignment and actual sampling semantics |
| P1 | Aggregate centroid and estimator lifecycle | Confirmed design-intent mismatch; policy gap | Union-pixel CoG and explicit handling of membership changes and missing observations |
| P1 | Smear objective and optical SIL | Design-intent mismatch and validation gap | Target-relative exposure metrics and an image-forming closed-loop oracle |
| P1 | HAL ownership and production/SIL execution | Contract and validation gap | Defined actuator ownership and comparable execution semantics |
| P2 | Navigation frames and feedforward uncertainty | Modeling assumptions and validation gaps | Explicit frame/time/attitude contracts and an error budget |
| P2 | Estimator alternatives and cascade tuning | Unresolved design choice and validation gap | Evidence-based residual-versus-joint comparison and robust bandwidth assessment |

## Recommended improvements and continued investigations

Evidence paths below use these prefixes:

- `F/` = `packages/flight/src/flight/`
- `S/` = `packages/sim/src/sim/`
- `G/` = `packages/gse/src/gse/`
- Test paths and configuration paths are repository-relative.

### R1. Make invalid-feedback containment immediate and distinct from SAFE stow

**Finding:** Both inner execution paths can reuse `last_theta_enc_rad` after an encoder `Err` and
continue writing torque. This cached-angle fallback does not itself publish an encoder fault.
Failure without a previous sample instead attempts zero torque. An integrity trip publishes its
fault and latches SAFE before writing the torque calculated for that tick; the fault publication
does not suppress that unsafe write. Later SAFE ticks use a stow position loop with the same encoder input.

**Recommended outcome:** Invalid feedback must revoke permission to command feedback-controlled
motion before another normal torque command is issued. Treat drive inhibition, controlled stopping,
stow, and recovery as distinct conditions. Stow is conditional on trustworthy feedback and a safe
actuation path; it is not the universal meaning of SAFE.

**Investigate:** Error, stale timestamp, frozen angle with fresh timestamps, implausible motion, and
sensor recovery need distinct validity evidence. The freeze detector's approximately 1 deg/s command
threshold can exclude slow failures. Fifty consecutive strikes are not a wall-clock deadline.
The lock-motion test averages displacement since lock engagement, which can dilute motion that
starts much later. Determine detection coverage without confusing freshness with motion consistency.

**Evidence:** `F/payload/app.py:486-514,620-650`; `F/payload/control.py:366-380`;
`F/payload/gimbal/integrity.py:60-67,108-125`; `F/hal/drivers_sim/gimbal.py:198-219`;
`config/default.toml:39-45`. Lead: `732dbdc1`.

### R2. Define containment across software, communication, and actuator failures

**Finding:** Torque-write errors publish faults without a local inhibit path. Fault processing uses
an approximately five-second loop, with additional payload-consumption delay possible; five seconds
is not a guaranteed shutdown bound. The scheduler supervises the parent app, not its nested inner
thread. The parent can remain apparently healthy after inner-thread failure. Shutdown has no explicit
final inhibit operation. HAL has no disable acknowledgement or command-expiry contract.

**Recommended outcome:** Define a bounded command-validity contract, local fault response, inner-loop
liveness supervision, and safe shutdown behavior. Compare independent amplifier command timeout and
an independently available disable mechanism. A zero-torque request over a failed communication path
cannot establish inhibition. A software stop cannot establish that the mechanism physically stopped.

**Investigate:** Whether containment must survive process death, bus/serial loss, and a failed-on
amplifier; whether unpowered motion is acceptable or braking/holding is required; how inhibition is
acknowledged. These are hardware-selection and hazard-analysis inputs, not facts that software can
invent. The real gimbal is a no-op stub, so retained torque or coasting on real hardware is unknown.

**Evidence:** `F/payload/app.py:648-689,733-736`; `F/fault/app.py:141-148,224-237`;
`F/core/scheduler.py:94-120`; `F/hal/interfaces/gimbal.py:44-66`;
`F/hal/drivers_real/gimbal.py:21-56`; `F/hal/drivers_sim/gimbal.py:78-95,124-146`.
Leads: `732dbdc1`, `42afa1b2`.

### R3. Require physical transition and recovery evidence

**Finding:** Integrator freezing at a travel bound does not prevent existing proportional/integral
terms from producing outward torque. The simulator's ideal clamps can hide this. Mechanical release
checks use commands or telemetry drained in the current cycle rather than confirmed current drive
and motion state. The safety lead found no flight-app consumer of `read_stow_switch()`. `EXIT_SAFE`
checks for new SAFE-triggering events in that tick rather than affirmative health recovery.

**Recommended outcome:** Separate motion permission, directional limit protection, confirmed stow,
and recovered health. Define fault-specific latching and clearance conditions; silence from a failed
producer must not count as recovery. Retain the useful existing UNKNOWN/ENGAGED launch-lock
zero-torque/integrator-freeze safeguard; it does not establish independent hardware inhibition.

**Investigate:** Hard versus science limits, allowed inward recovery motion, physical stopping distance,
contact loads, stow-switch evidence, interlock freshness, and whether re-enable requires operator
permission or bounded autonomous recovery. Do not assume every boundary requires identical behavior.

**Evidence:** `F/payload/control.py:361-398`; `F/payload/gimbal/inner.py:65-80`;
`F/hal/drivers_sim/gimbal.py:90-105`; `F/mechanical/app.py:101-145`;
`F/fault/app.py:132-169,176-198`; `F/fault/policy.py:82-96`. Lead: `732dbdc1`.

### R4. Make delayed filtering equivalent to a correctly ordered measurement history

**Finding:** Snapshots are saved before the current vision correction. Rewind restores an old state
and replays predictions, not all intervening vision corrections. A later delayed image can therefore
discard an earlier correction, even when image timestamps increase monotonically. This does not
happen on every update. For an off-grid image, the first partial propagation uses the selected
snapshot's preceding-interval input, although it is propagating into the next interval.

**Recommended outcome:** Define the meaning of each history record and preserve accepted measurement
corrections across rewind. Consider corrected posterior histories or an event-history replay design;
compare their complexity and storage needs rather than prescribing one prematurely. Partial intervals
must use the input valid over those intervals.

**Investigate:** Multiple delayed images, changing predictor and encoder inputs, equal timestamps,
out-of-order arrival, horizon boundaries, dropped observations, and image-dependent anchor updates.
History contains nominal/encoder inputs already; the problem is their chronology and consistency,
not the complete absence of history.

**Independent criterion:** Compare state mean and covariance at a common endpoint against an offline
chronological filter with the same accepted observations and input history. The oracle must not call
or reproduce the production rewind algorithm.

**Evidence:** `F/payload/control.py:523-542`; `F/payload/tracking/residual.py:190-253`;
`packages/flight/tests/payload/tracking/test_residual.py:51-124`. Lead: `627dab11`.

### R5. Specify a process-noise model valid for variable and fractional intervals

**Finding:** The separately scaled diagonal process covariance does not preserve covariance when one
propagation interval is split. For the same continuous stochastic model, consistent subdivision
requires `Q(a+b) = F(b) Q(a) F(b)^T + Q(b)`. Current scaling does not satisfy that identity.

**Recommended outcome:** Define the physical interpretation and units of process noise before choosing
its discretization. Derive variable-interval covariance from that model, including required
angle/rate cross terms. If a fixed-grid discrete noise model is retained, explicitly define what
fractional replay means under it. Do not copy a different Q formula without resolving this model.

**Investigate:** Positive semidefiniteness, covariance symmetry, time-partition consistency, and
calibration against innovation statistics. Quantify uncertain encoder-rate input and the correlation
between the image-derived anchor and the vision measurement. Those omissions are modeling assumptions,
not evidence that the current filter performs two independent measurement updates.

**Evidence:** `F/payload/tracking/residual.py:101-186`; `F/payload/control.py:491-542`.
Leads: `627dab11`, `a4669d3b`.

### R6. Establish event-time alignment and honest execution timing

**Finding:** The real sensor timestamps after retrieval/copy/release, not from physical exposure
metadata. The encoder is read after acquisition; its timestamp is not used to reconstruct shutter-time
angle. The inner thread performs work and then waits the full nominal period while rate fitting and
PI use that nominal period. Angle history has no sample timestamps. With actual uniform spacing `h`
and assumed spacing `d`, a constant-rate estimate scales by `h/d`.

Production also captures outer `now` before acquisition and uses it after processing/inference/storage.
Historical outer catch-up ticks read current encoder/state values under old tick labels. Mapping a
historical monotonic time to UTC uses the current offset, with no documented treatment of clock steps.

**Recommended outcome:** Distinguish exposure time, sample time, arrival time, execution time, and
command application time. Compare timestamp-aware estimation and control against a genuinely
bounded fixed-period acquisition/control architecture. Reconstruct angle at exposure time with
explicit interpolation uncertainty. Handle missed deadlines without relabeling current samples as
historical ones. Preserve preprocessing/inference co-location while considering decoupling imaging
latency from control scheduling.

**Investigate:** Camera exposure metadata and shutter model, timestamped raw encoder availability,
clock synchronization, latency bounds, jitter, rate-fit delay/noise trade-offs, and historical UTC
mapping. Do not assume hardware triggering exists or that raising the nominal frequency solves timing.

**Evidence:** `F/hal/drivers_real/sensor.py:107-128`; `F/payload/app.py:325-344,393-423,551-558,604-732`;
`F/payload/control.py:289-297,354-359,523-530`; `F/payload/gimbal/rate_fit.py:37-51`.
Lead: `42afa1b2`.

### R7. Define one actuator owner and comparable production/SIL execution semantics

**Finding:** The inner and outer threads call the same gimbal outside the control-state lock.
Thread safety and transaction ordering are unspecified, not demonstrated hardware corruption.
The parent also directly checked the execution-path distinction during handoff verification:
production defines a threaded `inner_loop()` in `F/payload/app.py:604-675`, while SIL invokes
`advance_inner()` synchronously in `S/sil/stepping.py:91-107`. SIL plant catch-up debt prevents the
blanket cumulative double integration alleged in earlier reviews, but frozen-clock substeps move the
plant while encoder timestamps remain at `_last_t`.

**Recommended outcome:** Compare a single software actuator owner with an external real-time owner
of encoder acquisition, PI execution, and inhibit authority. Specify command arbitration and telemetry
ownership for either option. Prefer shared control-step semantics between production and SIL; test
real concurrency separately rather than treating deterministic stepping as proof of threaded behavior.

**Investigate:** HAL call ordering, error and lock handling across both paths, nested-thread failure,
command ages, and timing under imaging load. Validate SimGimbal event chronology against an independent
piecewise-torque integrator for regular, irregular, and frozen-clock stepping. Debt accounting alone
does not establish correct sample event times.

**Evidence:** `F/payload/app.py:486-514,620-716`; `F/hal/drivers_sim/gimbal.py:109-145,204-219`;
`S/sil/runner.py:168-172`; `S/sil/stepping.py:78-90`; `G/harness.py:257-272`.
Lead: `42afa1b2`; safety overlap: `732dbdc1`.

### R8. Implement the aggregate observation contract, not a first-component shortcut

**Finding:** Component extraction already produces area and centroid, but control uses
`matched[0].centroid_raw`, and the arbiter assigns the first component's ID. For disjoint accepted
components, the required union CoG is `sum(A_i * c_i) / sum(A_i)`. An equal-weight component mean is
incorrect. Existing single-component fixtures do not expose this mismatch.

**Recommended outcome:** Define an aggregate observation with acceptance/visibility semantics and a
lifecycle separate from any single component's ID. Distinguish an observed empty mask from a zero
centroid and a predicted recovery target from an observed aggregate. Separate apparent translation
from membership-induced centroid changes where evidence supports that distinction.

**Investigate:** Confidence/area gates, temporal acceptance hysteresis, clipped fragments, false
positives, split/merge events, component reordering, and total dropout. Decide when to preserve,
inflate uncertainty in, or invalidate the residual/anchor history. Current TRACKING/REWIND transitions
lack an explicit estimator lifecycle boundary; that is not proof that every transition needs a reset.
Do not reject legitimate aggregate changes merely to preserve a previous single-plume identity.

**Evidence:** `F/payload/blobs.py:29-53`; `F/payload/control.py:303-329,470-542`;
`F/payload/tracking/tracker.py:127-156`; `F/payload/gimbal/arbiter.py:144-169`;
`packages/flight/tests/payload/test_payload_controller.py:22-35`.
Leads: `627dab11`, `63e8cb15`, `a4669d3b`.

### R9. Separate centroid pointing, exposure sharpness, and hardware motion limits

**Finding:** The outer "smear" cap limits the full absolute commanded gimbal rate. Preprocessing also
uses gimbal speed alone; production supplies an inter-frame angle difference, not exposure-resolved
relative motion. Near boresight for rigid constant-rate motion, elevation smear is approximately
`abs(omega_target - omega_gimbal_actual) * exposure / IFOV`, using consistent angular units.
A moving gimbal can freeze a moving target; a stationary gimbal can record a blurred target.

**Recommended outcome:** Keep hardware rate/torque/travel constraints separate from science-quality
criteria. Evaluate actual target-relative motion during exposure, not only commanded motion or an
inter-frame average. Report uncontrollable cross-track degradation separately from elevation control
performance and also include it in total image-quality assessment.

**Investigate:** An area-weighted, percentile, or worst-region sharpness objective; exposure-time
trajectory blur; deformation; differential motion of plume regions; and recentering versus science
continuity. Two components can move oppositely with a stationary union centroid. Membership changes
can move the centroid without material motion. Neither case is resolved by centroid stabilization
alone. Exposure adjustment or temporary non-science frames requires mission authorization.

This is a mismatch with the user's sharpness objective, not proof that every absolute rate cap is
wrong. An independent mechanical cap may be necessary. A direct image-sharpness optimizer is not
required if a simpler validated motion controller meets the science criterion.

**Evidence:** `F/payload/gimbal/outer.py:94-110`; `F/payload/preprocess/quality.py:96-99`;
`F/payload/app.py:695-706`; `F/payload/gimbal/pointing.py:3-4,34-46`.
Leads: `63e8cb15`, `a4669d3b`.

### R10. Preserve useful navigation structure, but bound its assumptions

**Finding:** WGS-84 ray intersection creates a surface proxy, then predicts a frozen Earth-fixed
point. It does not measure the plume's physical range or 3-D aggregate position. The mount is equated
with an LVLH triad, while the ephemeris contract supplies generic "ECI" position/velocity without
an attitude/alignment or uncertainty contract. One image supplies both the new anchor and the KF
measurement, creating correlated errors that the current known-input model does not explicitly carry.

**Recommended outcome:** Define frame, time scale, sample epoch, freshness, orientation, and mount
calibration contracts. Treat the surface anchor as nominal angular-motion feedforward with uncertainty.
Compare validated nominal feedforward against image/encoder-based fallback where mission policy
permits it; do not assume fallback currently exists or must always be allowed.

**Investigate:** Sensitivity to range/height, attitude offsets and rates, frame conventions, ephemeris
age, Earth orientation approximation, FOV geometry, anchor misses, and changing aggregate membership.
An error budget should determine which geometric refinements materially affect pointing and smear.

**Evidence:** `F/payload/gimbal/intersect.py:75-100`; `F/payload/gimbal/predictor.py:42-68`;
`F/payload/gimbal/geo.py:108-139,197-213`; `F/hal/interfaces/ephemeris.py:20-42`;
`F/payload/control.py:491-542`. Lead: `a4669d3b`.

### R11. Compare estimator and control alternatives on observable quantities

**Current structure:** Outer proportional angular-error feedback plus nominal/residual rate
feedforward; inner encoder-rate PI with computed-torque scaling and saturation. The residual filter
uses `x = [e, omega_res]`, with `e_dot = omega_nom + omega_res - y_m`, and vision observes `e`.
This is not LQR or a full LQG regulator.

**Comparison candidates, not approved designs:**

| Candidate | Potential value | Required scrutiny |
|---|---|---|
| Small residual KF with corrected chronology and uncertain inputs | Low state dimension; retains useful nominal predictor | Encoder input uncertainty, anchor correlation, delayed updates, membership changes |
| Joint angular target/gimbal estimator | Distinct encoder-angle and relative-vision measurement updates | Observability, model mismatch, timestamp alignment, computational cost, correlated derivatives |
| Image/encoder angular tracking with optional validated feedforward | Reduced dependence on uncertain navigation geometry | Dropout capability, motion observability, delay, and permission to operate without external data |

For a joint candidate, a possible starting state is `[theta_target, omega_target, theta_g, omega_g]`
in explicitly defined compatible angular coordinates. Encoder angle observes `theta_g`; vision
observes relative angle through the calibrated projection. This is only a comparison model, not an
observability proof or a prescription. Do not count encoder angle and its derived rate as independent
measurements. Do not count an image-derived anchor as independent of the same image's relative angle.

Assess actual inner/outer bandwidth separation, estimator and rate-fit phase lag, imaging delay,
jitter, saturation, parameter uncertainty, and stopping transients together. Selected loop rates or
nominal gains alone do not establish stability. Do not invent motor parameters or formal margins.

**Evidence:** `F/payload/gimbal/outer.py:103-105`; `F/payload/gimbal/inner.py:63-80`;
`F/payload/tracking/residual.py:124-186`; `F/payload/gimbal/rate_fit.py:37-51`.
Leads: `627dab11`, `a4669d3b`, with timing evidence from `42afa1b2`.

### R12. Add independent physical and chronological validation, not formula restatements

**Finding:** The inspected SIL/GSE scene pre-renders a fixed Gaussian and supplies a fixed off-center
square mask. The inspected scripted segmentor ignores its input tensor, so actuation cannot move or
clip that mask. SIL passes zero slew to preprocessing. The closed-loop pointing scenario checks
movement and inference count, not plume centering or sharpness.

**Recommended outcome:** Preserve those useful integration checks, but distinguish them from physical
pointing validation. Compare controllers using an image-forming model where geometry and actual
gimbal motion affect projection, clipping, exposure integration, and measured masks. Isolate controller
validation with known masks when useful, and test segmentation effects separately as well.

**Evidence:** `S/scene/plume.py:45-125`; `F/payload/inference/segmentor.py:36-61`;
`S/sil/stepping.py:78-90`; `scenarios/closed_loop_pointing.toml:10-20`.
Existing useful block studies: `packages/analysis/tests/studies/elevation_controller/test_block_checks.py:31-60,174-302`.
Analysis FOV utilities: `packages/analysis/src/analysis/lib/tracking.py:246-280`; their half-disk
visibility assumption is not an aggregate-mask requirement. Lead: `63e8cb15`.

## Independent validation questions and acceptance evidence

Numerical tolerances, safety deadlines, and science thresholds remain to be specified. The criteria
below describe what evidence must show, not tests that have already passed.

| Investigation | Independent reference or adversarial case | Evidence sought |
|---|---|---|
| Delayed KF | Offline time-ordered filter; multiple off-grid images; changing inputs | Same endpoint mean/covariance, with no lost accepted corrections |
| Noise discretization | Analytical or independently integrated stochastic model; random interval partitions | Partition-consistent covariance, symmetry, PSD, calibrated uncertainty |
| Exposure alignment | Known motion with separate shutter, retrieval, and encoder sample times | Reconstructed exposure-time geometry within a specified uncertainty budget |
| Production timing | Nonuniform encoder samples, processing stalls, live threaded fake HAL | Honest sample ages/dt, bounded command age, supervision and no synthetic historical samples |
| Sim chronology | Independent piecewise-torque plant integration | Consistent physical event times under regular, irregular, and frozen-clock advances |
| Encoder containment | Err after a valid sample, stale timestamp, fresh-timestamp freeze, slow commands | Inhibit-first behavior and separately measured fault-to-inhibit latency |
| Actuator containment | Write failure, inner-thread death, process/communication loss, active-torque shutdown | Bounded output validity and acknowledged containment under the stated fault model |
| Recovery/interlocks | Silent failed producer, late lock motion, outward boundary torque, false stow indication | No unsupported re-enable, release, or stow confirmation |
| Aggregate CoG | Direct union-mask pixel sum; unequal areas and reordered components | Exact accepted-pixel centroid and explicit missing-observation semantics |
| Aggregate lifecycle | Split/merge, clipping, dropout, new false component, appearance/disappearance | Documented continuity policy without treating every centroid jump as physical velocity |
| Sharpness | Exposure-integrated projected image; matched tracking, stationary gimbal, opposite region motion | Correct relative-motion quality and reported 2-D degradation, independent of absolute speed |
| Navigation | Independent ray/projection geometry; frame, attitude, range and ephemeris-age perturbations | Bounded angular/rate error over the proposed operating envelope |
| Cascade robustness | Independent plant/model sweep with delay, inertia/damping uncertainty and saturation | Demonstrated stability/performance margins and feasible bandwidth separation |

Existing tests are not absent or worthless. Plant-based inner checks, residual recovery tests, and
predictor derivative checks provide useful evidence. Replay tests largely repeat the discrete replay
formulas, and the inspected smear oracle repeats the absolute-rate formula. The inspected run-loop
shutdown test starts with an already-set stop event; it does not prove shutdown from active torque.

## Consolidated clarification questions

Ask focused subsets when needed. "Unknown; carry this as a hardware/mission requirement" is a valid
answer. The defaults below are review assumptions only, not user-approved behavior.

| ID | Question | Why it changes the recommendation | Provisional treatment |
|---|---|---|---|
| Q1 | What defines accepted pixels beyond confidence/area thresholds? Do clipped visible fragments count, and may acceptance use temporal hysteresis? | Defines the target and membership-change semantics | All accepted visible pixels have equal weight; no inferred off-chip pixels |
| Q2 | What continuity is allowed after empty masks, large membership changes, or late vision/navigation data? Is bounded coasting permitted, and for how long? | Sets estimator lifecycle, stale-data limits and fallback authority | Empty mask is missing data; predictions are labeled; no indefinite stale-command validity |
| Q3 | Is sharpness area-weighted, percentile-based, worst-region, or another science metric? What 2-D limits apply? | Centroid stationarity does not guarantee local sharpness | Assess pointing and exposure degradation separately; report cross-track effects |
| Q4 | May recentering produce temporarily non-science-quality frames? May acquisition defer or shorten exposures within SNR constraints? | Determines centering versus sharpness trade-offs | Do not assume exposure-control authority; qualify recovery frames explicitly |
| Q5 | Can camera exposure metadata, timestamped raw encoder angles, synchronization, and bounded monotonic/UTC mapping be required? | Determines defensible delayed fusion and available estimator models | Retrieval time is not exposure time; synchronization is unvalidated |
| Q6 | Must inner control remain on the Jetson, or may an external real-time device own sensing, PI, and inhibit authority? | Changes timing, ownership and containment alternatives | Compare both; no hardware selection is made |
| Q7 | What ephemeris frame/time/freshness and time-tagged body/mount attitude will exist? Must tracking operate without valid attitude/navigation? | Determines feedforward authority and fallback operating capability | LVLH-to-mount identity is not established flight truth |
| Q8 | After inhibition, is passive motion safe at every pose, or is brake/hold required? What excursion, stop-time and contact-load limits apply? | Separates drive removal from physical safety | Neither zero torque nor unpowered motion is presumed mechanically safe |
| Q9 | Must containment survive process death, communication loss, and a failed-on amplifier? | Sets independent timeout/disable and hardware protection requirements | No indefinite drive after process/communication loss; amplifier fault scope unresolved |
| Q10 | Does re-enable require operator authorization plus health evidence, or may recovery be autonomous? Must stow be verified before science resumes? | Defines latches and recovery transitions | No clearance based only on absence of new fault events |

## Corrections and boundaries to preserve

- Reject the blanket SIL cumulative double-integration claim: catch-up debt cancels a subsequent
  matching clock advance. Event-time consistency remains an open validation question.
- A successful same-ray surface intersection preserves immediate observed angular direction.
  Unknown altitude/range affects later angular prediction, not an automatic immediate pointing offset.
- `y_dot = 0` for the orbital plane normal is valid for central-force elliptical as well as circular
  motion: angular momentum is constant. Plane-changing dynamics require separate treatment.
- A constant Earth-rotation phase offset can cancel under consistent transforms and an axisymmetric
  Earth. That does not establish compatibility with an external frame or eliminate attitude errors.
- The current residual model is conceptually valid with encoder rate treated as a known input.
  Its uncertainty assumptions and implementation chronology are separate review concerns.
- Ephemeris under HAL, an unused documented design-statement parameter such as `tau_cl_s`, and the
  absence of LQR are not by themselves defects. Do not restore the obsolete dual-axis stand-in.
- A generic real-driver stub is a flight-readiness limitation, not evidence of deployed actuator
  behavior. Do not claim a real amplifier coasts, retains torque, or supports inhibit without evidence.
- A frozen integrator is not necessarily zero torque. Zero torque is not necessarily stopped motion.
  A parent heartbeat is not necessarily inner-loop health. No new fault event is not recovered health.
- The earlier alleged REWIND overshoot issue was not adjudicated by these five Stage 1 reports.
  If pursued, independently inspect the mode semantics and boundary cases before calling it confirmed.
- Green CI, successful SIL movement, and completed research agents do not prove physical correctness
  or establish PR merge readiness.

## Repository constraints and verification context

Read `AGENTS.md`, `CLAUDE.md`, and relevant subsystem `CONTEXT.md` before further work. Preserve pure
cores with injected time, typed `Result` errors, HAL protocols, composition-root ownership, and typed
`PactConfig`. Keep real/sim drivers independent. Do not introduce app cross-imports or publish large
image artifacts on the bus. Preserve preprocessing/inference co-location when exploring scheduling.

Use the Python 3.14+ `uv` workspace. Standard documented checks include:

```bash
uv run ruff check packages scripts
uv run ruff format --check packages scripts
uv run mypy packages scripts
uv run lint-imports
uv run python scripts/check_vcrm.py
uv run pytest -m "not e2e"
```

These commands were not run for the five research reports or this document-only synthesis. Select
relevant checks when implementation is authorized; tests need independent oracles as described above.
Use SIL/GSE or analysis studies, not `flight.core.main` or real hardware SDKs in this environment.
The design document `docs/design/single-axis-elevation-controller.md` is context, not authority to
override the user's clarified objectives.

## Research provenance

| Lead | Focus | Successful SWE evidence-gathering children |
|---|---|---|
| `627dab11` | Estimator and aggregate-centroid behavior | 1 |
| `42afa1b2` | Time and execution | 1 |
| `732dbdc1` | Fault containment and recovery | 1 |
| `63e8cb15` | Image objective and independent validation | 1 |
| `a4669d3b` | Navigation and control formulation | 2 |

All five leads independently checked decisive child findings and corrected overclaims where noted.
Their reports overlap and share source evidence; agreement is not five independent experiments.
The retained report text below is the handoff's evidence appendix.

---

# Evidence appendix: Stage 1 Astra Research Lead Outputs

The following report bodies are preserved from the pre-existing report file. The synthesized
recommendations above do not alter the leads' original clarification checkpoints or imply that
those leads completed a final design recommendation.

---

## Table of Contents

1. [Agent 627dab11 — Estimator and Aggregate-Centroid Behavior](#agent-627dab11)
2. [Agent 42afa1b2 — Time and Execution](#agent-42afa1b2)
3. [Agent 732dbdc1 — Fault Containment and Recovery](#agent-732dbdc1)
4. [Agent 63e8cb15 — Image Objective and Independent Validation](#agent-63e8cb15)
5. [Agent a4669d3b — Navigation and Control Formulation](#agent-a4669d3b)

---

## Agent 627dab11

### Scope / Revision

Inspected the five primary files and relevant tests in `/home/aiden/Plume-Autonomous-Classification-and-Tracking`. Direct reads of `.git/HEAD` and its branch ref confirm `829364338a58b0f9368f0fb1091ee8d2e6eabd1d`. Working-tree cleanliness is supplied, not independently verified.

Citations below use paths relative to `packages/flight/src/flight/payload/`.

### Preliminary Verified Evidence

| Classification | Finding and evidence |
|---|---|
| **Design-intent mismatch — high confidence** | **The commanded centroid is the first component, not the aggregate.** `blobs.py:29–53` computes each accepted connected component’s pixel count and centroid. `control.py:303–319` gates/matches components, then uses `matched[0].centroid_raw`. `tracking/tracker.py:127–156` preserves incoming component order; association does not produce an aggregate. `gimbal/arbiter.py:144–165` also assigns the first component’s ID as the target. |
| **Confirmed defect — high confidence** | **Rewinding can discard an earlier vision correction.** `control.py:523–542` saves the pre-update snapshot and updates only the current residual. `tracking/residual.py:229–253` restores a historical snapshot and replays predictions, not intervening vision updates. Example: a correction at tick `t_1`, followed at `t_2` by an image timestamped between those ticks, restores the uncorrected `t_1` snapshot. This can happen even with monotonically increasing image timestamps; it does **not** happen on every update. |
| **Confirmed defect — high confidence** | **Off-grid replay uses the wrong input for its first partial interval.** Snapshot inputs describe the prediction interval **ending** at that snapshot (`control.py:523–530`). Forward propagation from that snapshot to an intervening shutter time instead uses those same, preceding-interval inputs (`tracking/residual.py:238–247`). Constant-input tests conceal this mismatch. |
| **Confirmed mathematical discrepancy — high confidence** | **Covariance propagation depends on interval partitioning.** The diagonal, separately scaled `Q` in `tracking/residual.py:101–121` does not satisfy `Q(a+b)=F(b)Q(a)F(b)^T+Q(b)`, required for consistent subdivision of the same stochastic dynamics. At two half-period steps, the accumulated process contribution has off-diagonal `h q_{22}/2`, versus zero for one full step. The intended physical noise model remains unresolved. |
| **Validation gap — high confidence** | **Ordinary loss/reacquisition has no explicit estimator lifecycle boundary.** SAFE clear resets residual/history (`control.py:481–483`), but TRACKING↔REWIND does not. Residual prediction continues and reacquisition updates the existing state (`control.py:470–542`; `gimbal/arbiter.py:144–169`). Whether continuity is desirable depends on dropout and aggregate-change policy. |

### Important Qualifications and Sound Behavior

- **Aggregate geometry is recoverable from existing component metadata:** for disjoint accepted masks, union centroid is `sum A_i c_i / sum A_i`, not an equal-weight component mean. Component extraction itself is not the aggregation defect.
- **Modeling assumption — high confidence:** the residual model treats encoder-derived gimbal rate as a known input, while vision measures elevation error (`tracking/residual.py:124–185`). It does **not** directly measure residual target rate with the encoder. An uncertain-input residual model and a joint angular target/gimbal estimator remain open alternatives—not conclusions at this checkpoint.
- **Modeling assumption — high confidence:** the Earth-intersection/nominal-prediction path (`control.py:491–521`) supplies a geometric proxy; it does not establish an observed physical 3D plume centroid.
- **Validation gap — high confidence:** inspected residual tests reproduce replay formulas and check state means, but do not establish chronological multi-image equivalence or covariance partition consistency (`tests/payload/tracking/test_residual.py:51–124`). Controller centroid fixtures contain only one component (`tests/payload/test_payload_controller.py:22–35`).

**Child-result corrections:** its metadata search missed accessible Git files; the revision was independently confirmed. Its covariance example omitted propagation of rate noise into angle variance. The correct half-step angle contribution is `q_{11}/4 + h^2 q_{22}/2`; the discrepancy still holds.

### Questions for the User

1. **What determines “accepted” pixels beyond segmentation thresholding—component area/confidence gates, temporal persistence, or border clipping?**
   - **Why it matters:** these rules change the aggregate target itself.
   - **Provisional assumption:** retain current confidence/area acceptance provisionally; include visible portions of border-clipped components; use equal weight per accepted pixel, without persistence-based exclusion.

2. **After total dropout or a large aggregate shift caused by appearance/disappearance, should tracking preserve short-term continuity or promptly reacquire the newly visible aggregate? Is any coast duration already intended?**
   - **Why it matters:** this determines estimator validity/reset policy and the permitted transient between centering and sharp imaging.
   - **Provisional assumption:** every valid aggregate remains the commanded target, but membership-induced jumps are not automatically physical translation; prolonged dropout invalidates confident extrapolation. Duration remains unspecified.

3. **Are timestamped raw encoder angles and shutter-time synchronization expected to be available, with buffered history, or only controller-time/derived-rate samples?**
   - **Why it matters:** independent encoder/vision updates require defensible timing and uncertainty handling; this materially affects the residual-versus-joint-estimator comparison.
   - **Provisional assumption:** raw angles are available, but synchronization, latency bounds, and noise characteristics are unvalidated.

### Verification / Stopping Point

One `swe-worker` child performed read/search-only dependency and test investigation under explicit no-edit, no-command, no-side-effect constraints. The Astra lead independently checked decisive source passages and the covariance argument. No tests or executable experiments were run; no repository changes were made.

Stopping at this checkpoint for answers; no final architecture recommendation or implementation plan yet.

---

## Agent 42afa1b2

### Scope / Revision

TIME AND EXECUTION for PR49, repository `/home/aiden/Plume-Autonomous-Classification-and-Tracking`. Read-only Git metadata confirms HEAD `829364338a58b0f9368f0fb1091ee8d2e6eabd1d`. Working-tree cleanliness is parent-reported, not independently verified.

Below, `F`, `S`, and `G` denote `packages/flight/src/flight`, `packages/sim/src/sim`, and `packages/gse/src/gse`.

### Evidence Summary

1. **Exposure/encoder association — confirmed defect, high confidence.**
   `RealSensor` timestamps **after** image retrieval, copying, and release, not from physical exposure metadata (`F/hal/drivers_real/sensor.py:107–128`). Production then reads the encoder after acquisition (`F/payload/app.py:693–706`). `process_frame` uses its angle without its timestamp; the controller describes that input as shutter-time elevation (`app.py:333–344`; `control.py:289–297`). Thus shutter-time geometry is not established. This is a software association defect, not evidence that a future camera cannot supply suitable timing.

2. **Nominal versus actual inner period — confirmed defect, high confidence.**
   The inner thread performs work and then waits a full configured period. Its derivative and PI calculations nevertheless receive the configured `dt`, not elapsed sample time (`F/payload/app.py:604–672`). The encoder ring contains angles only, and `fit_rate` assigns uniformly spaced synthetic times (`F/payload/control.py:354–359`; `F/payload/gimbal/rate_fit.py:37–51`). For constant physical rate and actual uniform spacing `h`, fitting with nominal spacing `d` scales the estimated rate by `h/d`. The magnitude under deployed scheduling remains unmeasured; this does **not** establish any required loop frequency.

3. **Acquisition blocks outer execution, but not directly the inner thread — confirmed defect, high confidence.**
   Production captures `now` before acquisition, then performs acquisition, processing/inference and storage before calling `advance_outer` with that old `now` (`F/payload/app.py:677–732`, `325–331`). Catch-up outer ticks read the currently available encoder and use current state rate estimates at historical tick labels, rather than reconstructing historical encoder inputs (`app.py:393–423`; `control.py:523–530`). The separate inner thread is a useful separation already present; it does not make outer execution independent of imaging latency.

4. **Device ownership and supervision — validation gap, high confidence in structure.**
   Both payload threads access the same gimbal outside the state-copy lock (`F/payload/app.py:620–648, 693–716`). The scheduler supervises registered app threads, not the payload’s nested inner thread (`F/core/scheduler.py:94–120`; `app.py:674–675`). Actual hardware transaction safety and inner-loop failure response are unvalidated. Importantly, `RealGimbal` is a no-op torque stub, not an implemented serial motor controller (`F/hal/drivers_real/gimbal.py:21–56`); claims of demonstrated hardware command corruption would overreach.

5. **Historical inputs and clock mapping — modeling assumption, high confidence.**
   Historical residual snapshots **do exist**, including nominal target rate and measured gimbal rate; rewind reuses them (`F/payload/control.py:523–531`; `F/payload/tracking/residual.py:223–252`). Their existence does not establish physically correct historical sampling. Monotonic-to-UTC conversion uses the **current** clock offset for the requested historical time (`F/payload/app.py:551–558`). Stability/uncertainty of that mapping across synchronization adjustments is unspecified.

6. **SIL chronology correction — blanket cumulative double-count claim rejected.**
   `run_steps` advances `ManualClock` **after** stepping; GSE does likewise (`S/sil/runner.py:168–172`; `G/harness.py:257–272`). Frozen-clock torque calls advance the plant and accumulate debt; a subsequent equal clock advance subtracts that debt, avoiding another integration of the same interval (`F/hal/drivers_sim/gimbal.py:109–145`). However, encoder timestamps remain `_last_t` while frozen-clock substeps move the plant (`:204–219`). **Validation gap, high confidence:** physical event-time consistency across substeps, irregular advances and historical inputs is not established by this debt mechanism.

### Questions for the User

1. **What timing capabilities or interface commitments may we require from the eventual camera, encoder and ISS time source?**
   - Specifically: exposure start/end or midpoint metadata, timestamped encoder samples, synchronization/trigger facilities, and a bounded mapping to common monotonic/UTC time. “Unknown” is sufficient.
   - **Consequence:** determines whether exposure-aligned reconstruction can rely on synchronized timestamps or must explicitly accommodate uncertain timing.
   - **Provisional assumption:** no hardware synchronization guarantee; retrieval time is not exposure time.

2. **Are there constraints requiring encoder acquisition, PI execution and drive-inhibit authority to remain on the Jetson, or may an external real-time controller own them?**
   - This asks permission to compare architectures, not to select hardware.
   - **Consequence:** changes the viable ownership, command-expiry and failure-containment alternatives.
   - **Provisional assumption:** retain both Jetson-owned and external-real-time-owned options for later comparison.

3. **What science-continuity policy applies when vision or external data becomes late/unavailable while encoder feedback remains trustworthy?**
   - Is bounded predictive coasting acceptable, and are exposure-relative smear, maximum data age or interruption-duration budgets available?
   - **Consequence:** determines deadline handling and stale-data acceptance criteria rather than treating every missed image as an immediate stop or indefinitely replaying history.
   - **Provisional assumption:** numerical limits remain unresolved; no indefinite stale-command validity. Untrustworthy encoder feedback separately requires drive inhibition first.

### Verification / Stopping Point

Used one `swe-worker` child for HAL timing, rate-fit dependencies and test inventory. Delegation succeeded; no nesting/profile failure or substitution. The Astra lead independently checked decisive source passages and the catch-up chronology above.

Tests were **inspected, not executed**: notably the frozen-debt test, SIL stepping characterization, residual rewind tests, and payload run-loop test. The latter starts with an already-set stop event; it does not validate live concurrent timing. Residual tests largely reproduce replay formulas rather than provide independent chronological oracles. No commands, edits, configuration changes or executable experiments were performed.

Stopping here for clarification; no final architecture recommendation or implementation plan.

---

## Agent 732dbdc1

### Scope / Revision

PR49 actuator safety/fault containment, read-only inspection of `/home/aiden/Plume-Autonomous-Classification-and-Tracking`. Readable Git HEAD/ref metadata resolves to `829364338a58b0f9368f0fb1091ee8d2e6eabd1d`. Findings describe the files currently present; working-tree cleanliness was not verified.

Paths below are relative to `packages/flight/src/flight/` unless stated otherwise.

### Verified Evidence

| Finding and classification | Evidence and consequence |
|---|---|
| **Confirmed defect — high confidence:** failed encoder reuse violates inhibit-first. | Both inner-loop paths reuse `last_theta_enc_rad` after `read_position()` returns `Err`, then continue computing/writing torque. That branch does **not** itself publish an encoder fault. Only failure without a previous sample attempts zero torque. `payload/app.py:486–514, 620–650`. |
| **Confirmed defect — high confidence:** integrity-triggered SAFE does not inhibit drive first. | An integrity trip publishes a fault and sets the SAFE latch, but still sends that tick’s computed torque. Subsequent SAFE ticks run the stow position loop using the same encoder input. `payload/app.py:505–514, 638–650`; `payload/control.py:366–380`. |
| **Modeling assumption / validation gap — high confidence:** encoder freshness and motion consistency are not equivalent. | Inner paths consume angle but do not validate its timestamp. Simulated encoder freezing preserves the angle while clock timestamps continue advancing: `hal/drivers_sim/gimbal.py:198–219`. Freeze detection requires consecutive ticks with `abs(r) > r_min`; commands at/below the threshold reset strikes even if nonzero torque persists. Default threshold is `0.01745 rad/s`; 50 strikes are **not a validated wall-clock deadline**. `payload/gimbal/integrity.py:108–125`; root `config/default.toml:39–45`. The lock-motion detector also averages displacement over the entire time since engagement, potentially diluting late motion: `integrity.py:60–67`. |
| **Confirmed containment gap — high confidence:** torque-write faults have no local inhibit; the inner thread is not supervised independently. | Torque `Err` only publishes a fault. FDIR processes events on its watchdog-period loop, currently configured at 5 seconds; payload consumption can add further delay. This is not a guaranteed five-second response bound. The nested `payload-inner` thread can die while its parent continues heartbeating and appears healthy to the scheduler. `payload/app.py:648–689`; `fault/app.py:141–148, 224–237`; `core/scheduler.py:100–120`. |
| **Validation gap / missing hardware contract — high confidence:** shutdown and command expiry do not establish a safe physical output. | Payload cleanup sets the stop event, joins the inner thread and stops acquisition, without a final actuator inhibit operation. The HAL exposes torque/pose commands but no disable, command lease or inhibit acknowledgement. Simulated torque remains held until replaced. `payload/app.py:733–736`; `hal/interfaces/gimbal.py:44–66`; `hal/drivers_sim/gimbal.py:78–95, 124–146`. Neither zero torque nor loss of software execution proves physical stopping. |
| **Design-intent mismatch / validation gap — high confidence:** limits, interlock and stow are incomplete safety evidence. | At a travel bound, the PI freezes integration but can still command outward torque: `payload/control.py:361–398`; `payload/gimbal/inner.py:65–80`. Simulation instead imposes ideal position/rate clamps: `hal/drivers_sim/gimbal.py:90–105`. Mechanical release protection uses commands/pointing telemetry drained that cycle, not confirmed current motion or drive inhibition: `mechanical/app.py:101–145`. No flight-app consumer of `read_stow_switch()` was found; existing SAFE-stow testing reads the switch itself. |
| **Design-intent mismatch — high confidence:** absence of a new fault event is treated as clearance. | `EXIT_SAFE` checks whether a SAFE-triggering event occurred **this tick**, not affirmative encoder/actuator recovery evidence. A silent failed producer can therefore leave the clearance gate satisfied. `fault/app.py:132–169, 176–198`; `fault/policy.py:82–96`. |

### Corrections and Sound Behavior

- UNKNOWN/ENGAGED launch-lock state inhibits payload motion; locked inner ticks command zero and freeze integration. This is a useful existing safeguard, although it does not establish independent hardware inhibition.
- The real gimbal is an **explicit no-op stub**, not evidence of a functioning amplifier safety path: `hal/drivers_real/gimbal.py:21–26, 46–56`.
- The Astra lead corrected child-report chronology: integrity faults are published **before** the unsafe same-tick torque write; cached encoder fallback may publish **no** fault. It also rejected its assertion that real hardware would coast after thread death—the amplifier’s retained-output behavior is unknown.

### Questions for the User

1. **After drive inhibition, is passive motion acceptable at every allowed pose, or must the mechanism brake/hold to protect the payload or ISS? Are allowable excursion, stop time or contact-load limits already available?**
   - **Why it matters:** distinguishes removal of motor drive from achievement of a mechanically safe state.
   - **Provisional assumption:** inhibit first remains mandatory; neither zero torque nor unpowered motion is presumed safe. Numerical limits remain mission/hardware inputs.

2. **Must containment survive failure of the flight process or actuator communication path—and a failed-on amplifier—or is a narrower fault-containment scope intended?**
   - **Why it matters:** determines whether independent command timeout, drive-disable/power interruption and separate limit protection must be hardware-selection requirements rather than software-only safeguards.
   - **Provisional assumption:** process/communication loss must not leave indefinite commanded drive; failed-on-amplifier protection remains unresolved.

3. **After encoder or actuator faults, should re-enabling motion require operator authorization plus demonstrated health, or may bounded autonomous recovery be attempted? Is verified stow required before science resumes?**
   - **Why it matters:** determines fault-specific latching, recovery evidence and whether “recovered,” “drive permitted” and “stow confirmed” are separate conditions.
   - **Provisional assumption:** no automatic re-enable merely because fault events stop; stow requires trustworthy feedback and explicit completion evidence.

### Verification / Stopping Point

Used one `swe-worker` for scheduler, fault routing, command handling and test inventories, with explicit prohibitions on edits, commands, configuration changes and side effects. The Astra lead independently checked decisive source passages and detector mathematics. Tests were **inspected, not executed**, including integrity, SAFE-stow and shutdown tests; the latter uses an already-set stop event and does not demonstrate shutdown from active torque.

Stopping here for clarification; no patches or implementation plan produced.

---

## Agent 63e8cb15

### Scope / Revision

Read-only inspection of PR49 in `/home/aiden/Plume-Autonomous-Classification-and-Tracking`; `.git/HEAD` and its referenced branch file resolve to `829364338a58b0f9368f0fb1091ee8d2e6eabd1d`. Working-tree cleanliness is supplied, not independently checked.

### Preliminary Evidence

1. **Design-intent mismatch — high confidence: flight currently points at one component, not the aggregate.**
   `packages/flight/src/flight/payload/blobs.py:29–52` computes each accepted connected component’s pixel area and centroid. `packages/flight/src/flight/payload/control.py:303–329` then uses `matched[0].centroid_raw`. Two accepted unequal-area components therefore do not produce the required union-pixel CoG. The component area/centroid records already contain sufficient information to recover the exact union centroid for these disjoint components; an equally weighted component mean would be incorrect.

2. **Design-intent mismatch — high confidence: the “smear” constraint limits absolute commanded gimbal rate, not plume-relative image motion.**
   `packages/flight/src/flight/payload/gimbal/outer.py:94–110` clips the complete nominal + residual + centering rate using exposure and IFOV. Preprocessing similarly estimates smear from gimbal speed alone (`preprocess/quality.py:96–99`); the production caller supplies an inter-frame encoder-angle difference, not exposure-resolved motion (`app.py:695–706`).

   **Physical counterexamples:** matching a moving plume can yield negligible relative smear at substantial gimbal speed; a stationary gimbal can still image a moving plume with substantial smear. Near boresight, for rigid constant-rate motion, along-axis smear scales with
   ```
   |ω_target − ω_gimbal,actual| · T_exp / IFOV,
   ```
   not absolute commanded rate. This approximation does not cover deformation, cross-track motion, or within-exposure reversals.

3. **Modeling limitation — high confidence: centroid stationarity is insufficient to establish plume sharpness.**
   Two equal-area plume regions moving oppositely can retain a stationary aggregate centroid while both smear. Conversely, appearance, disappearance, threshold crossings and FOV clipping can move the aggregate centroid without translating existing plume material. These must not automatically become “physical plume velocity.” The single-axis limitation is already correctly explicit in `gimbal/pointing.py:3–4, 34–46`; cross-track image motion remains a visibility/quality limitation, not an invitation to command azimuth.

4. **Validation gap — high confidence: the inspected SIL/GSE scene is not an optical feedback loop.**
   `packages/sim/src/sim/scene/plume.py:45–106` pre-renders a fixed-position Gaussian; `:109–125` supplies a fixed off-center square mask. `packages/flight/src/flight/payload/inference/segmentor.py:36–61` explicitly ignores the frame tensor. Gimbal motion therefore cannot move or clip that mask. Moreover, `packages/sim/src/sim/sil/stepping.py:78–90` passes **zero slew** into preprocessing.
   `scenarios/closed_loop_pointing.toml:10–20` checks movement and inference count—not centering, visibility or sharpness. These are useful integration checks, but cannot establish physical image stabilization.

5. **Validation gap, not “no tests” — high confidence.**
   `packages/analysis/tests/studies/elevation_controller/test_block_checks.py` includes useful plant-based inner-loop checks (`31–60`), closed-loop residual-rate recovery with delayed updates (`174–203`), and explicit rewind replay (`206–261`). However, the replay largely restates the discrete model, and the smear “oracle” (`264–302`) repeats the absolute-rate formula; neither independently establishes plume sharpness. The analysis tracking library also provides covering-disk/FOV geometry (`packages/analysis/src/analysis/lib/tracking.py:1–6, 246–280`), but its half-disk visibility assumption is not an aggregate-mask requirement.

**Correction to child findings:** absence of a direct sharpness optimizer is **not itself a defect**—sound relative-motion control could meet the imaging objective without one. Likewise, existing block tests should not be described as fully independent physical validation.

### Questions for the User

1. **What defines “accepted,” particularly at image edges and during short dropouts?**
   - Should current confidence/minimum-component-area gates define inclusion, with every surviving visible fragment—including clipped plumes—immediately contributing? May acceptance use temporal hysteresis? When nothing survives, is a predicted target allowed only as explicitly marked recovery state?
   - **Why it matters:** this defines the target itself and distinguishes legitimate membership jumps from measurement errors.
   - **Provisional assumption:** all currently accepted visible pixels contribute equally; no reconstruction of off-chip pixels or preferential plume selection. Predictions are not reported as observed CoG.

2. **How should sharpness be judged across the accepted plume area when different regions move differently?**
   - Is success an area-weighted/percentile motion-blur criterion, a worst-region bound, or another science metric? Must total two-dimensional degradation be reported even when elevation control cannot remove it?
   - **Why it matters:** holding the aggregate centroid cannot guarantee every plume region is sharp; cross-track motion and deformation can make a universal bound unattainable.
   - **Provisional assumption:** centroid pointing and exposure-time local image degradation are separate metrics; total 2-D degradation is reported alongside the controllable elevation contribution. Numerical thresholds remain mission-dependent.

3. **When the aggregate is far from boresight, what trade-off is permitted between recovery and usable imaging?**
   - May bounded recentering temporarily produce non-science-quality frames? Is exposure adjustment within controller authority, subject to signal/noise constraints?
   - **Why it matters:** preserving sharpness during every exposure can conflict with rapid recentering before FOV exit. A zero-relative-motion policy can leave a persistent pointing offset.
   - **Provisional assumption:** hardware-bounded recovery may temporarily lose science-quality imaging while retaining explicitly qualified tracking observations; exposure remains fixed unless authority is granted.

### Verification Performed

Used one `swe-worker` child, successfully, for bounded read-only source and coverage discovery. The Astra lead independently checked the decisive centroid, smear-law, caller, scripted-scene, SIL stepping, scenario and block-test passages, and evaluated the physical counterexamples above. No commands, tests, edits, configuration changes or side-effecting experiments were performed.

Stopping at this checkpoint; no final architecture recommendation or implementation plan yet.

---

## Agent a4669d3b

### Scope / Revision

Read-only inspection of PR49 in `/home/aiden/Plume-Autonomous-Classification-and-Tracking`: navigation geometry, predictor, outer/inner control, rate fit, estimator integration, configuration, ephemeris HAL, and associated studies/tests. `.git/HEAD` and its branch ref resolve to `829364338a58b0f9368f0fb1091ee8d2e6eabd1d`. This verifies HEAD, not that working-tree files have no uncommitted differences.

Paths below are relative to the repository; `flight/` abbreviates `packages/flight/src/flight/`.

### Verified Evidence

1. **Design-intent mismatch — high confidence: the commanded measurement is not the aggregate centroid.**
   After confidence/area gating and matching, `flight/payload/control.py:303–319` takes `matched[0].centroid_raw`. Other accepted components do not contribute to that measurement. This contradicts the already-resolved requirement to target the union of all accepted foreground pixels; no further question about single-target versus aggregate selection is needed.

2. **Modeling assumption — high confidence: the surface intersection is a nominal-motion surrogate, not a measured physical plume location.**
   `flight/payload/gimbal/intersect.py:75–100` intersects the centroid ray with WGS-84 and retains the previous anchor on a miss. `predictor.py:42–68` propagates that anchor as a frozen Earth-fixed point.
   An aggregate image centroid can move when masks appear, disappear, split, merge, or clip at the FOV—even with no material translation. Consequently, its nominal surface-anchor motion and estimated residual cannot automatically be interpreted as physical plume motion.

3. **Modeling assumption / validation gap — high confidence in the data dependence; unquantified effect on performance.**
   The same image supplies both the new surface anchor and the KF measurement (`control.py:491–542`). The KF then treats nominal target rate and encoder-derived rate as known inputs:
   ```
   e_{k+1} = e_k + Δt(ω_nom + ω_res − y_m)
   ```
   Its covariance propagation has no explicit anchor/image cross-covariance or encoder-input uncertainty (`flight/payload/tracking/residual.py:124–186`). This is **not two explicit independent measurement updates**, but it does create correlated input/measurement errors whose effect needs examination.

   Rewind replays stored nominal/encoder inputs, rather than a complete reconstructed history of image-dependent anchors (`residual.py:190–253`). The Astra lead is **not** claiming the newest anchor is used throughout the entire shutter-to-current interval; that would overstate the implementation.

4. **Design-intent mismatch — high confidence: the “smear cap” bounds absolute commanded gimbal rate, not exposure-time image motion.**
   `flight/payload/gimbal/outer.py:94–109` clips the full feedforward-plus-feedback command using pixels × IFOV / exposure. Counterexamples:
   - Matching a fast target’s angular rate can produce little elevation smear despite high gimbal speed.
   - A zero gimbal command can leave substantial target-relative smear.

   Actual inner-loop response and unactuated cross-track motion also matter. Aggregate-centroid stabilization alone does not establish sharpness of every visible plume.

5. **Modeling assumptions / validation gaps — high confidence.**
   - The as-built outer controller is proportional angular-error feedback plus nominal/residual rate feedforward—not LQR (`outer.py:103–105`).
   - The inner loop is encoder-rate PI with computed-torque scaling, saturation, and integrator freezing (`inner.py:63–80`).
   - Rate fitting assumes uniformly spaced encoder samples (`rate_fit.py:37–51`); timing sensitivity remains a validation question, not proof that a particular execution frequency is required.
   - Geometry equates the mount with an orbital LVLH triad (`geo.py:108–139,197–213`). The ephemeris interface carries position, velocity, epoch, and a generic `"ECI"` tag, but no attitude/alignment or uncertainty contract (`flight/hal/interfaces/ephemeris.py:20–42`). **HAL placement itself is not a defect.**

### Corrections to Inherited Claims

These are **verified modeling boundaries**, not reasons to discard the predictor:

- **No immediate altitude-induced angular error from a successful same-ray surface anchor.** The intersection preserves the observed direction at anchoring. Unknown range affects subsequent angular prediction through relative-motion geometry; its magnitude is not established here.
- **`y_dot = 0` is not limited to circular orbits.** For any central-force orbit, including elliptical motion, `dot{h} = r × a = 0`. Plane-changing dynamics require further treatment; merely saying “perturbed orbit” is insufficient to establish a nonzero plane-normal derivative.
- **A constant Earth-rotation phase offset can cancel** when intersection and propagation use it consistently with an axisymmetric Earth. This does not settle external frame compatibility, but it does invalidate a blanket demand for full Earth-orientation modeling without an error budget.

### Questions for the User

1. **What is the acceptance/dropout policy at image boundaries?**
   - Should edge-clipped accepted components contribute their visible pixels normally, and should one empty accepted mask immediately mean “no current aggregate,” or permit a defined grace interval?
   - **Why it matters:** Determines target continuity, coast/reacquisition behavior, and which centroid changes are valid observations rather than segmentation faults.
   - **Provisional assumption:** Use all currently accepted visible pixels, without reconstructing hidden plume area; an empty union is a missing measurement, not a zero centroid.

2. **When aggregate recentering conflicts with sharp exposure, which has priority?**
   - After a genuine aggregate-centroid jump, may acquisition temporarily defer/shorten exposures while the elevation axis recenters, or must imaging continue uninterrupted?
   - **Why it matters:** Changes the control/exposure coordination trade and the acceptable transient response.
   - **Provisional assumption:** Sharpness is evaluated from actual target-relative, two-dimensional exposure motion; centroid jumps are not automatically physical velocity. Numerical limits remain mission inputs.

3. **Can a time-tagged ISS-body/mount orientation contract be assumed eventually, or must tracking remain useful without it?**
   - This asks for the required operating capability—not selection of an API now.
   - **Why it matters:** Determines how much authority orbital feedforward can have versus image/encoder-based tracking when attitude, alignment, or external-data quality is unavailable.
   - **Provisional assumption:** Orbital/time access is expected, but LVLH-to-mount identity is not established flight truth.

### Verification and Stopping Point

Used two `swe-worker` children for bounded evidence gathering; no nesting failure occurred. The Astra lead independently checked decisive geometry, control, covariance, configuration, and study-test passages and the mathematical corrections above.

**Tests inspected, not executed.** Existing block studies include predictor finite differences and residual recovery. The inspected rewind oracle reproduces the replay formulas, and the smear oracle reproduces the cap formula (`packages/analysis/tests/studies/elevation_controller/test_block_checks.py:206–269`); neither independently establishes chronological estimator consistency or physical exposure sharpness.

No commands, edits, configuration changes, installations, or executable experiments were performed. **Stopping here for clarification before controller/estimator alternatives and final recommendations.**
