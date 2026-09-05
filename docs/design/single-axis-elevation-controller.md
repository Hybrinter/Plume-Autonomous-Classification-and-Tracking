# Single-axis elevation gimbal controller

**Audience:** an implementation agent. This brief is the conceptual specification
for the pointing controller. It is not as-built documentation and not a code sketch.

**First implementation scope:** flight pure cores, HAL Protocols, sim drivers,
composition-root wiring, the payload-app loops that bind them, and analysis-package
block tests with scripted inputs. Do not redesign the SIL harness. Do not implement
the station TLE HTTP API. Do not run a formal gain or plant-identification study.

---

## 1. How to use this document

1. Treat this file as the source of truth for *what to build*.
2. The modules under `flight.payload.control`, `flight.payload.gimbal.lqr`, and
   `flight.payload.tracking.kalman` are a stand-in. Do not preserve that stand-in.
3. There is **no azimuth motor**. Elevation is the only actuated axis. Remove
   gimbal azimuth from control state, HAL readback, commands, and tools datapoints.
   Optical azimuth in the FOV still enters the CoG ray. The dual-axis analysis
   study under `packages/analysis/.../single_axis_vs_dual_axis_gimbal/` keeps
   azimuth; nowhere else does.
4. `REWIND` is the hunt mode. Azimuth `SCAN` is gone. TRACKING, REWIND, and SAFE
   are the only arbiter states.
5. Keep the repo invariants: pure cores (no I/O, no bus, no clock reads; `now` is
   an argument), `Result` in library code, HAL Protocol injection, no app-to-app
   imports, config from `PactConfig` / `config/default.toml` only.
6. After the code exists, update STE-mirrored pages to match behavior. Do not cite
   this brief from those pages as design rationale.

---

## 2. Settled choices

| Topic | Choice |
| --- | --- |
| Actuator command | Torque \(\tau\) (N·m) on the HAL. Motor current is a later linear map in the real driver, not in the control law. |
| Axes | Elevation motor only. No azimuth hardware, no azimuth tracking law. |
| Hunt | `REWIND` toward the science limb. Not an azimuth raster. |
| Arbiter | TRACKING / REWIND / SAFE. No IDLE. No ACQUIRING. Limb wait is TRACKING with \(r=0\). |
| Ephemeris | New HAL Protocol. ISS state is ECI, SI meters. Sim driver is a circular Keplerian orbit. Station TLE API later behind the same Protocol. |
| Plume target | Recompute Earth intersect of the CoG LOS every accepted vision frame. The CoG is not a fixed ground origin. |
| Inner rate | Cheap causal polynomial differentiator on an encoder ring. Not a Kalman filter. |
| Outer estimator | Two-state residual Kalman filter on boresight elevation error and residual rate. |
| Vision to outer | In-process shell queue of \((t_s, z_v, \mathbf{p}_{\mathrm{cog}})\). Not the MessageBus. |
| Smear cap | Live camera `exposure_us` in (19), also clipped to \(\omega_{\mathrm{hw}}\). |
| Runaway monitor | Removed. Travel, torque, slew, and smear clips plus SAFE stow are the envelope. |
| Hardware | Raw motor + amp. The FLIR PTU rate/position ASCII driver is dead. Real `set_torque` is a stub until the amp interface exists. |
| Validation | Analysis-package block tests with simulated I/O. SIL harness architecture is out of this pass. |
| Placeholders | Numeric \(J,B,\tau_{\max},k_p,k_i,K_p,Q,R_v\) in config until later studies retune them. |

---

## 3. What to build

A cascaded **elevation** controller for a **raw motor** (no vendor rate servo):

| Layer | Period | Job |
| --- | --- | --- |
| Inner | \(T_{\mathrm{in}}\ll T_{\mathrm{out}}\) | Track a scalar rate reference \(r\) by commanding torque |
| Outer | own loop, \(T_{\mathrm{out}}\lesssim T_v\) | Hold boresight elevation error \(e\approx 0\) using kinematic feedforward plus a residual Kalman filter |
| Vision | irregular \(T_v\) | Measure \(e\) from the CoG pinhole ray, tagged at shutter time \(t_s\); enqueue for the outer loop |
| Predictor | outer ticks | From ISS ECI state + current CoG Earth point + Earth rotation (elevation component only), compute nominal target rate \(\omega_{t,\mathrm{nom}}\) |

Inference can run near camera frame rate (detect wall on the order of 4 ms on the
current Orin mix). That does **not** let the outer loop command torque. The plant
\(J\dot\omega_g+B\omega_g=\tau\) still needs a fast inner loop. The hierarchy is
timescale separation. Outer bandwidth is \(K_p\), not \(1/T_{\mathrm{out}}\).

This is **not** an LQG regulator. There is no Riccati solve on gimbal inertial
state. The only Kalman filter in the pointing law is the 2-state residual filter.
The inner loop is PI + computed torque.

STOW/HOME/`goto_angle` use a proportional position loop that writes \(r\) into the
same inner PI. That loop is not smear-capped.

---

## 4. Non-goals (first implementation)

- Station TLE HTTP client, authentication, or CCSDS encoding of ephemeris.
- SIL harness redesign or 1 kHz SIL stepping architecture. `step_once` may call
  payload catch-up methods so ManualClock jumps still move the plant.
- Formal gain scheduling, online inertia identification, or a RESULTS.md campaign
  beyond the elevation-controller block tests.
- Auto-exposure or other camera-control loops. The smear cap **reads** the live
  exposure; it does not set it.
- Slant, incidence, or GSD as controller or predictor **outputs**. Slant may exist
  as an intermediate of the ray-Earth intersect; it must not enter \(r\) or the
  residual state.
- Putting \(J,B\) in an outer regulator or estimating them online.
- Estimating process noise \(w\). \(w\) exists only in the filter covariance \(Q\).
- An inner-loop Kalman filter or any inner state besides the PI integrator and the
  encoder-rate ring.
- Encoder runaway / commanded-vs-measured rate FDIR.
- Changing FDIR SAFE latching: SAFE still stows and latches until ground clears it.
- Mount misalignment relative to ISS. Identity mount until a later placement map.
- \(K_t\) current mapping in the real driver.

---

## 5. Notation

Elevation \(\theta\) is **signed off-nadir**: \(0\) at geocentric nadir, \(+\)
along-track (ISS velocity), \(-\) look-back. One actuated axis.

### 5.1 Mount and camera frames

Right-handed mount frame:

- \(\hat z\): geocentric nadir (boresight at \(\theta_g=0\))
- \(\hat y\): along-track (ISS velocity)
- \(\hat x\): starboard; elevation **rotation axis** (\(\hat x\times\hat y=\hat z\))

Right-hand \(R_x(+\alpha)\) sends the nadir boresight toward look-back. Signed
elevation is still \(+\) along-track, so the boresight in mount coordinates is

\[
\hat{\mathbf{u}}_b(\theta_g)=\big(0,\ \sin\theta_g,\ \cos\theta_g\big),
\]

i.e. \(R_x(-\theta_g)\) applied to \(\hat z\). A unit test must show that
\(+\theta_g\) increases \(\hat{\mathbf{u}}_b\cdot\hat y\).

Camera at nadir (identity mount vs ISS):

- \(+\mathrm{Z}\): boresight
- \(+\mathrm{X}\): image right = mount \(\hat x\) (unactuated optical azimuth)
- \(+\mathrm{Y}\): image down = mount \(-\hat y\) (look-back; image \(+y\) is \(-e\))

Pinhole, not \(\mathrm{px}\times\mathrm{IFOV}\). Band-plane pitch
\(p=2\times 3.45\,\mu\mathrm{m}\), \(f=150\,\mathrm{mm}\), principal point at the
plane center:

\[
\mathbf{d}_{\mathrm{cam}}
=\mathrm{normalize}\big((u-u_0)p/f,\ (v-v_0)p/f,\ 1\big).
\]

Rotate into mount with \(R_x(-\theta_g)\). Distortion stays off until a map exists.

### 5.2 Gimbal and motor

| Symbol | Meaning |
| --- | --- |
| \(\theta_g\) | True gimbal elevation (rad in the inner loop; convert at the core boundary) |
| \(\omega_g=\dot\theta_g\) | True gimbal rate |
| \(\tau\) | Motor torque command (N·m) |
| \(\tau_{\max}\) | Torque clip |
| \(J\), \(B\) | True inertia and viscous damping (\(B\) is a linearized friction stand-in) |
| \(\hat J\), \(\hat B\) | Config copies used in the inner law |
| \(K_t\) | Torque-to-current slope (real driver later): \(i \approx \tau / K_t\) |
| \(\theta_{\mathrm{hw,min}},\theta_{\mathrm{hw,max}}\) | Hardware travel |
| \(\theta_{\mathrm{sci,min}},\theta_{\mathrm{sci,max}}\) | Science imaging window (config: 0 to +45 deg) |
| \(\omega_{\mathrm{hw}}\) | Hardware slew cap |

### 5.3 Encoder rate estimator

| Symbol | Meaning |
| --- | --- |
| \(\theta_{\mathrm{enc}}[k]\) | Encoder elevation at inner tick \(k\) (includes quantization and noise) |
| \(N_\omega\) | Ring length (number of encoder samples in the fit) |
| \(d_\omega\) | Polynomial degree (\(d_\omega \ge 2\), \(N_\omega > d_\omega\)) |
| \(y_m=\hat\omega_g\) | Estimated gimbal rate from the polynomial fit, used by the PI |
| \(\delta_\omega\) | Group delay of the stencil; the \((N_\omega-1)T_{\mathrm{in}}/2\) formula is a centered-SG figure. The causal endpoint quadratic spends noise gain, not that delay. Do not grow \(N_\omega\) to buy margin. |

### 5.4 Inner PI

| Symbol | Meaning |
| --- | --- |
| \(r\) | Rate reference from the outer law or the position loop |
| \(\varepsilon=r-y_m\) | Rate error |
| \(I\) | PI integrator state |
| \(k_p,k_i\) | PI gains |
| \(v\) | Commanded angular acceleration (PI output, rad/s\(^2\)) |

### 5.5 Vision and CoG geometry

| Symbol | Meaning |
| --- | --- |
| \(\mathbf{p}_{\mathrm{cog}}=(u,v)\) | Blob center of geometry in band-plane pixels |
| \(e_{\mathrm{az}}, e\) | Optical boresight error (az unactuated, not a gimbal command; \(e\) is elevation, the KF measurement) |
| \(z_v\) | Vision measurement of \(e\) at shutter time \(t_s\) |
| \(\hat{\mathbf{u}}_{\mathrm{cog}}\) | Unit line of sight through the CoG, mount frame |
| \(\mathbf{r}_{\mathrm{cog}}\) | Earth-ellipsoid intersect of \(\hat{\mathbf{u}}_{\mathrm{cog}}\) (ECEF, meters) |
| \(\rho\) | Slant range along that ray (intersect intermediate; discarded afterward) |
| \(t_s\) | Shutter time of the frame that produced \(z_v\) (monotonic vehicle clock) |
| \(t_v\) | Time the vision packet arrives (after inference) |
| \(T_v\) | Vision / inference period (irregular) |

The CoG is **not** the stack origin. Wind and unmodeled plume physics move it.
The CoG still sits in the rotating Earth frame at each instant, so its inertial
motion has a large co-rotating-Earth term plus a smaller walk.

### 5.6 Predictor and residual filter

| Symbol | Meaning |
| --- | --- |
| \(\mathbf{r}_s(t),\mathbf{v}_s(t)\) | ISS position (m) and inertial velocity (m/s) in ECI from the ephemeris HAL |
| \(\Omega_E\) | Earth rotation rate |
| \(R_z(\phi)\) | Rotation about Earth polar axis (ECEF CoG into ECI) |
| \(\theta_{\mathrm{los}}\) | Elevation of the LOS to \(\mathbf{r}_{\mathrm{cog}}\) (signed off-nadir) |
| \(\omega_t=\dot\theta_{\mathrm{los}}\) | True CoG elevation rate (unknown) |
| \(\omega_{t,\mathrm{nom}}\) | Elevation rate of a **co-rotating** point that is currently at \(\mathbf{r}_{\mathrm{cog}}\) |
| \(\omega_{t,\mathrm{res}}=\omega_t-\omega_{t,\mathrm{nom}}\) | Residual (wind / CoG walk / TLE and mount error) |
| \(e=\theta_{\mathrm{los}}-\theta_g\) | Elevation boresight error (same geometric meaning as \(z_v\)) |
| \(\mathbf{x}=[e,\ \omega_{t,\mathrm{res}}]^\top\) | Outer filter state |
| \(w\) | Process noise driving the residual random walk (not estimated) |
| \(v_v\) | Vision measurement noise |
| \(\nu\) | Innovation \(z_v-\hat e^-(t_s)\) |
| \(K_f\) | Kalman gain (2×1) |
| \(P,Q,R_v\) | Covariance, process-noise, vision-noise matrices |
| \(K_p\) | Outer proportional gain on \(\hat e\) |
| \(K_{\mathrm{pos}}\) | Position-loop gain for STOW/HOME/GOTO |

### 5.7 Timing and vehicle clock

| Symbol | Meaning |
| --- | --- |
| \(T_{\mathrm{in}},T_{\mathrm{out}},T_v\) | Inner, outer, vision periods |
| \(\tau_{cl}\) | Closed-loop time constant of the inner rate servo (design statement) |
| \(r_{\max,\mathrm{img}}\) | Smear-limited rate cap from live exposure |
| \(r_{\max}(\mathrm{mode})\) | Mode-dependent rate clip |

One injected `Clock` is the vehicle clock:

- `monotonic_s()` for inner, outer, rewind, encoder stamps, and shutter \(t_s\).
- `utc_s()` for ephemeris and Earth rotation. `ManualClock.advance` steps both.
- `wall_clock_iso()` for bus telemetry only.

Do not mix ISO wall time into the residual rewind. `MosaicFrame` carries monotonic
`timestamp_s` at shutter.

---

## 6. Plant (raw motor)

True physics. No sensors in this equation:

\[
J\dot\omega_g + B\omega_g = \tau, \qquad \dot\theta_g=\omega_g. \tag{1}
\]

Limits:

\[
\theta_g\in[\theta_{\mathrm{hw,min}},\theta_{\mathrm{hw,max}}],
\quad |\tau|\le\tau_{\max},
\quad |\omega_g|\le\omega_{\mathrm{hw}}. \tag{2}
\]

Science imaging uses the tighter window \([\theta_{\mathrm{sci,min}},\theta_{\mathrm{sci,max}}]\).
The arbiter enforces the science window in TRACKING/REWIND. The driver enforces (2).

Sim must integrate (1) in SI, then convert pose to degrees for encoder readback.
Do not keep the first-order `sim_time_constant_s` rate plant as the tracking truth
model. \(B\) in the placeholder plant is a linearized friction stand-in, not true
viscous damping. Coulomb friction is a disturbance the PI must reject.

**Current mapping (real driver, later):** \(\,i = \tau / K_t\,\) (or a calibrated
affine line \(i = c_1\tau + c_0\)). The inner law always outputs \(\tau\). Do not
fold \(K_t\) into \(k_p,k_i\).

---

## 7. Timescale split

\[
T_{\mathrm{in}}\ll\tau_{cl},\qquad K_p\ll 1/\tau_{cl},\qquad T_{\mathrm{out}}\lesssim T_v. \tag{3}
\]

- Inner: stabilize (1), reject torque disturbance, enforce \(\tau_{\max}\). Period
  1 ms (config). Must not wait on inference.
- Outer: own loop at 20 ms (config). Update \(r\) from predictor + residual filter.
  Coast on encoder and the last \(\mathbf{r}_{\mathrm{cog}}\) between blobs.
- Vision: irregular. Extra frames update \(z_v\) through the shell queue. They do
  not raise the torque loop rate.

If the inner loop is tight, the outer *design model* of the gimbal is a first-order
rate servo, not (1):

\[
\dot\omega_g \approx \frac{1}{\tau_{cl}}(r-\omega_g). \tag{4}
\]

Use (4) only as a statement of timescale separation. Do not put \(J,B\) in the
outer filter. \(T_{\mathrm{out}}\gg\tau_{cl}\) is **not** required; enforce
separation with \(K_p\).

**Threading:** the app shell owns threads, HAL, and the bus. Pure cores stay
side-effect free. The inner loop has its own time base (`stop_event.wait` style),
encoder reads, and torque writes. The outer loop is independent of vision. Vision
enqueues measurements. `PayloadApp.advance_inner(now)` and `advance_outer(now)`
catch up in \(T_{\mathrm{in}}\) / \(T_{\mathrm{out}}\) steps so a ManualClock jump
still integrates the plant. SIL `step_once` calls those methods. It does not become
a 1 kHz harness.

---

## 8. Inner loop

Keep **plant**, **measurement**, and **control law** as three different objects.
There is no inner Kalman state.

### 8.1 Inner plant and encoder measurement

\[
\mathbf{x}_{\mathrm{in}}=\begin{bmatrix}\theta_g\\ \omega_g\end{bmatrix},
\quad
\dot{\mathbf{x}}_{\mathrm{in}}
=
\begin{bmatrix}0&1\\ 0&-B/J\end{bmatrix}\mathbf{x}_{\mathrm{in}}
+
\begin{bmatrix}0\\ 1/J\end{bmatrix}\tau,
\quad
\theta_{\mathrm{enc}}=\begin{bmatrix}1&0\end{bmatrix}\mathbf{x}_{\mathrm{in}}+v_{\mathrm{enc}}.
\tag{5}
\]

Rate is not a sensor. Vision is not an inner measurement. Encoder quantization
uses `encoder_counts_per_rev` (placeholder 18-bit), not the dead PTU
`counts_per_deg`.

### 8.2 Causal polynomial rate estimator (not two-point finite difference)

A first-order backward difference

\[
\omega_{\mathrm{raw}}[k]=\frac{\theta_{\mathrm{enc}}[k]-\theta_{\mathrm{enc}}[k-1]}{T_{\mathrm{in}}}
\]

is **not** the inner-loop rate. Encoder quantization and the short \(T_{\mathrm{in}}\)
make that two-point slope noisy, and the PI would amplify it into \(\tau\).

The inner loop is much faster than the outer loop, so it can hold a short ring of
encoder samples and fit a local polynomial. That is a **causal Savitzky-Golay /
least-squares differentiator**: a higher-order finite-difference family with a
smoothing window. It is not a Kalman filter (no \(P\), no \(Q\), no process model).

**Ring.** At inner tick \(k\), store the last \(N_\omega\) pairs
\((t_j,\theta_{\mathrm{enc}}[j])\) for \(j=k-N_\omega+1,\ldots,k\). Times may be
taken as uniform \(t_j = t_k - (k-j)T_{\mathrm{in}}\) if the inner period is
constant.

**Fit.** Degree \(d_\omega\ge 2\), \(N_\omega > d_\omega\). Let \(\tau_j = t_j-t_k\)
(so the newest sample is at \(\tau=0\)):

\[
\theta_{\mathrm{enc}}(t_k+\tau)\approx
\sum_{m=0}^{d_\omega} a_m\,\tau^m. \tag{6}
\]

Solve the linear least-squares problem \(Va\approx\theta_{\mathrm{enc}}\) on the
ring (\(V\) is the Vandermonde matrix of the \(\tau_j\)). The rate used by the PI
is the first derivative **at the newest sample**:

\[
y_m[k]=\hat\omega_g(t_k)=a_1. \tag{7}
\]

Default: \(N_\omega=7\), \(d_\omega=2\).

**Constraints.**

- Causal only: never use future encoder samples (no centered stencil that waits).
- Do not grow \(N_\omega\) to chase the centered-SG delay formula. The causal
  endpoint quadratic is not that stencil.
- While the ring is short at startup, drop \(d_\omega\) to \(\min(d_\omega, n-1)\)
  for \(n\) available samples. Prefer the reduced-order fit over a two-point slope.
  Hold \(y_m=0\) until at least two samples exist.
- Do not add a second low-pass on \(y_m\) unless a later study shows the polynomial
  fit is not enough. One smoother is enough.
- Uniform \(T_{\mathrm{in}}\) may compile to a fixed convolution
  \(y_m[k]=(1/T_{\mathrm{in}})\sum_{j=0}^{N_\omega-1} c_j\,\theta_{\mathrm{enc}}[k-j]\).
  That is an implementation of (6)-(7), not a different estimator.

### 8.3 Control law (computed torque + PI)

This is **not** the plant. \(y_m\) appears because the controller cancels \(B\omega\).

\[
\varepsilon = r - y_m,
\qquad
v = k_p\varepsilon + k_i I,
\qquad
\dot I = \varepsilon \text{ (freeze on clip or travel stop)},
\qquad
\tau_{\mathrm{cmd}} = \hat J\, v + \hat B\, y_m,
\qquad
\tau = \mathrm{clip}(\tau_{\mathrm{cmd}},\tau_{\max}).
\tag{8}
\]

Substitute (8) into (1) when \(\hat J=J\), \(\hat B=B\), \(y_m=\omega_g\):

\[
J\dot\omega_g + B\omega_g = J v + B\omega_g
\quad\Rightarrow\quad
\dot\omega_g = v. \tag{9}
\]

Then \(v\) is commanded acceleration and the PI is an ordinary rate servo on an
integrator. If \(\hat J,\hat B\) are wrong, the PI still has to stabilize (1).
That is why those scalars stay in the inner loop.

**Units:** inner loop SI (rad, rad/s, N·m). Convert encoder degrees at the shell or
at the core boundary. Do not mix deg and rad inside the PI.

**Anti-windup:** freeze \(I\) when \(\tau\) is clipped or \(\theta_{\mathrm{enc}}\)
is on a hardware stop.

### 8.4 Position loop (STOW / HOME / GOTO)

Not the tracking law. Same inner PI:

\[
r=\mathrm{sat}\big(K_{\mathrm{pos}}(\theta_{\mathrm{cmd}}-\theta_g);\; r_{\max,\mathrm{stow}}\big).
\]

Not smear-capped. Arrival is \(|\theta_g-\theta_{\mathrm{cmd}}|\) within a small
config band. Driver `stow()` / `home()` / `goto_angle(el)` set that setpoint.

---

## 9. Ephemeris HAL

ISS position and velocity are **not** read inside a pure core and are **not**
parsed from CCSDS on the inner/outer loop. Inject them through `IssEphemeris`.

Conceptual surface:

- `read_state(now) -> Result[IssState, FaultCode]`
- `IssState` holds \(\mathbf{r}_s\) (m, ECI), \(\mathbf{v}_s\) (m/s, ECI), frame
  tag `ECI`, and the source epoch as UTC seconds.

If a later vehicle source is ECEF or LVLH, convert **in the driver**, not in the
core. Body/LVLH attitude alone is not enough for the predictor.

**Sim driver:** circular Keplerian ECI from TLE mean elements (inclination, mean
motion), SI meters. No `sgp4` in flight.

**Real driver:** stub `Err` until the station TLE/state API exists.

If `read_state` returns `Err` or is cold: \(\omega_{t,\mathrm{nom}}=0\). Do not
slew on a dead ephemeris.

This Protocol is separate from `StationLink` (CCSDS byte transport).

---

## 10. Kinematics predictor

The plume CoG is **not** a fixed ECEF origin. Each accepted vision frame
recomputes where the CoG sits on Earth. The predictor then answers: if that
point **co-rotates with Earth from this instant**, what elevation rate should
the gimbal use?

Do **not** finite-difference successive intersects to get \(\omega_{t,\mathrm{nom}}\).
That folds wind walk into the nominal rate (residual contamination). Wind belongs
in \(\omega_{t,\mathrm{res}}\). The jump in \(\mathbf{r}_{\mathrm{cog}}\) is a
position error \(e\) for the vision update. The extra rate versus the co-rotating
model is \(\omega_{t,\mathrm{res}}\).

### 10.1 CoG ray and Earth intersect (every accepted vision frame)

1. Take \(\mathbf{p}_{\mathrm{cog}}\) from the matched blob.
2. Build \(\mathbf{d}_{\mathrm{cam}}\) with the pinhole in §5.1. Include the
   unactuated optical azimuth so \(\mathbf{r}_{\mathrm{cog}}\) is the ground point
   under the CoG, not under the optical axis.
3. Rotate into mount with \(R_x(-\theta_g)\). Transform into ECI using ISS nadir
   and along-track from \(\mathbf{r}_s,\mathbf{v}_s\).
4. Intersect \(\mathbf{r}_s + \rho\hat{\mathbf{u}}_{\mathrm{cog}}\) with the WGS-84
   ellipsoid (\(a=6378137\,\mathrm{m}\), \(f=1/298.257223563\)). Store
   \(\mathbf{r}_{\mathrm{cog}}\) in ECEF meters. Discard \(\rho\).
5. If the intersect fails (no hit, behind the camera, below the limb): keep the
   last good \(\mathbf{r}_{\mathrm{cog}}\) and treat the frame as a miss. Do not
   NaN the rate.

### 10.2 Nominal elevation rate (co-rotating point)

At time \(t\), with current \(\mathbf{r}_{\mathrm{cog,ECEF}}\) and ISS ECI state,
rotate the CoG into ECI with \(R_z(\Omega_E(t-t_0))\) (epoch of the ECEF vector):

\[
\mathbf{r}_t(t)=R_z(\Omega_E (t-t_0))\,\mathbf{r}_{\mathrm{cog,ECEF}}, \tag{10}
\]

\[
\mathbf{l}=\mathbf{r}_t-\mathbf{r}_s,
\qquad
\theta_{\mathrm{los}}=\operatorname{atan2}(\mathbf{l}\cdot\hat y,\ \mathbf{l}\cdot\hat z).
\tag{11}
\]

\(\hat y\) is along-track and \(\hat z\) is nadir, matching §5.1. Do not use the
analysis-package 90°-at-nadir elevation convention.

\[
\omega_{t,\mathrm{nom}}(t)=\dot\theta_{\mathrm{los}}(t). \tag{12}
\]

Compute (12) by an analytic Jacobian of (11) with \(\mathbf{r}_{\mathrm{cog,ECEF}}\)
**held fixed** and Earth rotation plus ISS motion allowed to run, or by a small
central difference of (11) with that same freeze. Include the **elevation**
component of Earth rotation. Do **not** command an azimuth rate.

### 10.3 Between vision frames (coast)

Hold the last \(\mathbf{r}_{\mathrm{cog,ECEF}}\) and let it co-rotate with Earth
only. Do not invent wind. Recompute (12) with the new ISS state each outer tick.

On the next accepted vision frame, replace \(\mathbf{r}_{\mathrm{cog}}\) with the
new intersect. A jump in ECEF is expected (CoG walk).

Predictor signature (pure):

`(now, iss_state, r_cog_ecef) -> (theta_los, omega_t_nom)`.

---

## 11. Residual Kalman filter

Exact error kinematics:

\[
\dot e=\omega_t-\omega_g=\omega_{t,\mathrm{nom}}+\omega_{t,\mathrm{res}}-\omega_g. \tag{13}
\]

Model the unknown residual as a random walk driven by **unestimated** process
noise \(w\):

\[
\dot\omega_{t,\mathrm{res}}=w(t). \tag{14}
\]

Outer state. Not gimbal inertia. Not \(\omega_g\):

\[
\mathbf{x}=\begin{bmatrix}e\\ \omega_{t,\mathrm{res}}\end{bmatrix},
\quad
\dot{\mathbf{x}}
=
\begin{bmatrix}0&1\\ 0&0\end{bmatrix}\mathbf{x}
+
\begin{bmatrix}\omega_{t,\mathrm{nom}}-\omega_g\\ 0\end{bmatrix}
+
\begin{bmatrix}0\\ 1\end{bmatrix}w.
\tag{15}
\]

\(\omega_{t,\mathrm{nom}}\) and \(\omega_g\) (use \(y_m\)) are **known inputs**.
The encoder is not a measurement of \(\mathbf{x}\).

Vision measures elevation error only:

\[
z_v(t_s)=e(t_s)+v_v=\begin{bmatrix}1&0\end{bmatrix}\mathbf{x}(t_s)+v_v. \tag{16}
\]

\(z_v\) is the elevation component of the pinhole boresight error. Unactuated
optical \(e_{\mathrm{az}}\) may be logged; it is not in \(\mathbf{x}\) and it is
not a gimbal command.

Drop the EMA as a pointing estimator. Blob gates and IoU match remain as vision
preprocessing; they are not a second filter in series with (15).

Drop the 4-state dual-axis constant-velocity Kalman and the dual-axis LQR.

### 11.1 Discrete practice

Let \(T=T_{\mathrm{out}}\).

\[
F=\begin{bmatrix}1&T\\ 0&1\end{bmatrix},
\quad
\mathbf{u}[k]=\begin{bmatrix}T(\omega_{t,\mathrm{nom}}[k]-\omega_g[k])\\ 0\end{bmatrix}.
\tag{17}
\]

**Every outer tick (predict), even with no blob:**

1. \(\omega_{t,\mathrm{nom}}[k]\) from section 10 (coasted or freshly intersected
   \(\mathbf{r}_{\mathrm{cog}}\)).
2. \(\omega_g[k]\leftarrow y_m\) from the inner loop.
3. \(\hat{\mathbf{x}}^-=F\hat{\mathbf{x}}+\mathbf{u}\), \(P^-=FPF^\top+Q\).

**When a vision packet is dequeued (update at shutter time):**

1. Drop samples older than the rewind horizon (100 ms).
2. Rewind: restore the filter snapshot at \(t_s\), or roll back through a ring of
   \((\hat{\mathbf{x}},P,\mathbf{u})\) (~8 outer snapshots).
3. Innovation \(\nu=z_v-\hat e^-(t_s)\).
4. \(K_f=P^-C^\top(CP^-C^\top+R_v)^{-1}\), \(\hat{\mathbf{x}}\leftarrow\hat{\mathbf{x}}^-+K_f\nu\).
5. Replay predict to now using recorded \(\mathbf{u}(\cdot)\).

On the first accepted \(z_v\), snap \(\hat e\leftarrow z_v\) and shrink \(P_{11}\)
toward \(R_v\); keep \(P_{22}\) large.

Until that first accepted \(z_v\): \(\hat e=0\), \(\hat\omega_{t,\mathrm{res}}=0\),
\(r=0\) except REWIND/SAFE. Do not track on a cold filter.

The vision queue (depth 4, drop oldest) is **not** the encoder ring and **not**
the rewind snapshot ring.

**Never estimate \(w\).** If \(T_{\mathrm{out}}\) changes, rescale
\(Q_{22}\propto T_{\mathrm{out}}\), \(Q_{11}\propto T_{\mathrm{out}}^3\).

---

## 12. Outer law

\[
r=\mathrm{sat}\big(\omega_{t,\mathrm{nom}}+\hat\omega_{t,\mathrm{res}}+K_p\hat e;\;
r_{\max}(\mathrm{mode})\big). \tag{18}
\]

Scalar \(K_p\) is enough. \(\hat\omega_{t,\mathrm{res}}\) is already in the
feedforward; do not add a second copy.

If \(\hat e\to 0\) and \(\hat\omega_{t,\mathrm{res}}\approx\omega_{t,\mathrm{res}}\),
then \(r\to\omega_t\) and (13) stays at zero: the gimbal leads the orbit instead
of chasing pixels.

**Smear cap (imaging).** During TRACKING (once live) and REWIND:

\[
r_{\max,\mathrm{img}}
=\frac{\sigma_{\mathrm{smear}}\,\mathrm{IFOV}_{\mathrm{band}}}{\Delta t_{\mathrm{exp}}},
\qquad
|r|\le \min(r_{\max,\mathrm{img}},\omega_{\mathrm{hw}}). \tag{19}
\]

\(\sigma_{\mathrm{smear}}\) is `max_motion_smear_px`. \(\Delta t_{\mathrm{exp}}\)
is the **live** frame exposure (and the last known exposure during REWIND). Do
not use `initial_exposure_us` as a frozen science exposure. Also clip to
\(\omega_{\mathrm{hw}}\): a 13 µs exposure makes (19) larger than the hardware cap.

Do not suppress \(\omega_{t,\mathrm{nom}}+\hat\omega_{t,\mathrm{res}}\) while
TRACKING. Orbit feedforward must continue even when \(e\) is small.

SAFE/STOW/HOME use the position loop and are not smear-capped.

---

## 13. Modes

Keep a pure arbiter. It selects the \(r\) **policy**. It does not emit axis rates
or call `set_rate`. The inner loop turns \(r\) into \(\tau\).

There is no RATE command mode. `GimbalCommandMode` is ABSOLUTE / STOW / HOME.

| State | Rate reference \(r\) | Notes |
| --- | --- | --- |
| TRACKING (cold / limb wait) | \(0\) | No first accepted \(z_v\), or arrived at the science limb with no plume. Wait on orbital motion. |
| TRACKING (live) | (18) with live smear cap | Blob → TRACKING immediately (no ACQUIRING). Re-intersect CoG every accepted frame. |
| Miss coast | keep last \(\hat\omega_{t,\mathrm{res}}\), predict-only, still (18) on the coasted \(\mathbf{r}_{\mathrm{cog}}\) | Until `release_persistence_frames`. |
| REWIND | \(r=\mathrm{sign}(\theta_{\mathrm{sci,max}}-\theta_g)\,r_{\max,\mathrm{img}}\) toward the science limb | Hunt after loss below the limb. Not an azimuth raster. |
| REWIND at limb | → TRACKING with \(r=0\) | Wait. Blob → TRACKING live. |
| SAFE | position loop to stow | Latch until ground clear. Blobs ignored. |

REWIND uses the inner rate loop at the smear cap, not an open-loop absolute
goto. Arrival at the limb is \(|\theta_g-\theta_{\mathrm{sci,max}}|\) within a
small config tolerance.

---

## 14. HAL surface

### 14.1 Gimbal

Elevation torque plant.

- `set_torque(tau_nm) -> Result[None, FaultCode]` -- tracking and pose-loop path
- `read_position() -> Result[GimbalPosition, FaultCode]` -- elevation + monotonic timestamp
- `read_stow_switch() -> Result[bool, FaultCode]`
- `stow()` / `home()` / `goto_angle(el_deg)` -- set the position-loop target
- Hardware envelope clamp on \(\tau\), \(\omega\), \(\theta\)

`GimbalPosition` is elevation plus timestamp. No azimuth field. No `set_rate`.

Sim driver: integrate (1), encoder quantization and noise, travel/torque/slew
clamps, stow switch.

Real driver: stub `set_torque`. Delete the PTU ASCII path.

### 14.2 Ephemeris

Section 9. Sim circular ECI required. Real stub `Err`.

---

## 15. Module split (purity)

All of these are **pure** (config in, `now` in, new state + values out):

1. **Kinematics predictor** -- geometry only.
2. **Residual KF** -- predict / update / rewind on (15)-(16).
3. **Outer law** -- (18)-(19).
4. **Inner rate PI** -- (6)-(8), SI.
5. **Polynomial rate fit** -- encoder ring to \(y_m\).
6. **Position loop** -- STOW/HOME/GOTO \(r\).
7. **Arbiter** -- discrete mode, no torque math.
8. **CoG intersect** -- pixel \(\to\) \(\mathbf{r}_{\mathrm{cog}}\) (pure if ISS
   state and \(\theta_g\) are arguments).

**Payload app shell:** threads, encoder read, torque write, vision queue,
ephemeris HAL read, bus, heartbeats, `advance_inner` / `advance_outer`.

**Composition:** only `flight.core` / `sim.sil` construct drivers and the bus.

Do not publish tensors or covariance dumps on the bus. Compact telemetry: mode,
\(e\), \(r\), \(\tau\), \(\omega_{t,\mathrm{nom}}\), \(\hat\omega_{t,\mathrm{res}}\),
\(y_m\).

---

## 16. Config (placeholders)

Defaults must match `config/default.toml`. Do not hide numbers in source.

| Key | Placeholder | Notes |
| --- | ---: | --- |
| `J_kg_m2` / `B_nms_per_rad` | 0.008 / 0.04 | Conservative plant; \(B\) is a friction stand-in |
| `tau_max_nm` | 1.0 | |
| `max_hw_slew_rate_deg_per_s` | 10.0 | keep |
| `encoder_counts_per_rev` | 262144 | 18-bit; drop PTU 77.6 counts/deg |
| `controller.inner.dt_s` | 0.001 | |
| `controller.inner.rate_fit_n` / `rate_fit_degree` | 7 / 2 | |
| `controller.inner.kp` / `ki` | 200 / 10000 | 1/s and 1/s² on rad/s |
| `controller.inner.tau_cl_s` | 0.010 | design statement |
| `controller.outer.dt_s` | 0.020 | |
| `controller.outer.Kp` | 8.0 | 1/s on rad |
| `controller.residual.Q_diag` | `[1e-11, 3e-8]` | per outer step |
| `controller.residual.R_v` | 2.12e-9 | rad² (~1 band px) |
| `controller.residual.P0_diag` | `[1e-3, 1.2e-5]` | |
| `controller.residual.rewind_horizon_s` | 0.10 | |
| `controller.residual.rewind_snapshots` | 8 | |
| `controller.vision.queue_depth` | 4 | |
| `controller.position.K_pos` | 4.0 | 1/s |
| `controller.position.r_max_deg_per_s` | 8.0 | not smear-capped |
| WGS-84 `a_m`, `f` | 6378137, 1/298.257223563 | |
| `omega_earth_rad_s` | 7.2921159e-5 | |

Keep elevation envelopes: hardware `[-45, +90]` deg, science `[0, +45]`,
stow `-45`, home / wait `+45`.

Remove `lqr_*`, dual-axis `kalman_*`, `ema_alpha`, `retarget_rate_limit_hz`,
`runaway_*`, `sim_time_constant_s`, `counts_per_deg`.

---

## 17. What to stop using

- Dual-axis constant-velocity Kalman `[pan, tilt, pan_rate, tilt_rate]`
- Dual-axis LQR / DARE as the tracking law
- EMA as a pointing estimator
- `set_rate` as the TRACKING actuator command
- RATE command mode
- ACQUIRING and IDLE arbiter states
- Encoder-runaway monitor
- First-order sim rate plant as tracking truth
- Gimbal azimuth in flight, sim, gse, and tools
- Two-point first-order encoder difference as \(y_m\)
- Azimuth SCAN raster (already gone; do not revive it)
- PTU ASCII real driver

Keep: pinhole / boresight geometry, blob gates, IoU match, SAFE stow latch,
encoder timestamp, `REWIND` as the hunt mode, quality smear flag as an imaging
gate (the controller respects the same cap via (19)).

---

## 18. Conceptual checks (analysis block tests)

These proofs live under `packages/analysis/` as tests with simulated I/O. They
are not a substitute for later plant-ID or gain studies. They are not a SIL
harness.

1. **Inner:** simulated (1) at \(T_{\mathrm{in}}\); a constant \(r\) is tracked with
   bounded \(\varepsilon\); torque clips without integrator windup; travel stop does
   not chatter unbounded \(\tau\); \(y_m\) comes from the polynomial fit, not a
   two-point slope.
2. **Predictor:** with a fixed ECEF point, \(\omega_{t,\mathrm{nom}}\) matches a
   finite-difference of \(\theta_{\mathrm{los}}\); Earth rotation changes elevation
   rate; no azimuth command exists.
3. **CoG update:** two successive intersects that walk along-track change
   \(\mathbf{r}_{\mathrm{cog}}\) but \(\omega_{t,\mathrm{nom}}\) is still the
   co-rotating rate at the *current* point, not the walk slope between intersects.
4. **Residual:** true rate \(=\omega_{t,\mathrm{nom}}+0.1^\circ/\mathrm{s}\); after
   vision updates, \(\hat\omega_{t,\mathrm{res}}\to 0.1\); \(e\) stops ramping.
5. **Delay:** \(z_v\) lagged by inference time; rewind does not apply a stale
   error as if it were current.
6. **Smear:** TRACKING and REWIND \(|r|\) respect (19) from live \(\Delta t_{\mathrm{exp}}\).
7. **SAFE:** still stows via the position loop and ignores blobs until cleared.
8. **Cold start:** no tracking torque before the first accepted blob except
   REWIND/SAFE.
9. **Single axis:** no azimuth tracking command.

---

## 19. Invariants

1. Inner law is (8). Inner physics is (1). Do not put \(y_m\) in the plant equation.
2. Inner rate is (6)-(7). Not a two-point FD. Not a Kalman filter.
3. Outer state is only \((e,\omega_{t,\mathrm{res}})\). \(\omega_g\) is an input.
4. \(w\) is process noise. Estimate \(\omega_{t,\mathrm{res}}\) from vision
   innovation, not \(w\).
5. \(\omega_{t,\mathrm{nom}}\) is the co-rotating rate at the current CoG Earth
   point. Re-intersect every accepted frame. Do not FD successive intersects for
   the nominal rate.
6. Feedforward is the predictor. The KF is a residual observer, not an orbit
   propagator.
7. Hierarchy exists even if vision is fast.
8. No azimuth motor, no azimuth law, no SCAN raster, no RATE command.
9. Station TLE API, \(K_t\), mount misalignment, and SIL harness architecture
   come later.
