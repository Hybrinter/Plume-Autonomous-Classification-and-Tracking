# Single-axis elevation gimbal controller

**Audience:** an implementation agent. This brief is the conceptual specification
for the pointing controller. It is not as-built documentation and not a code sketch.

**First implementation scope:** flight pure cores, HAL Protocols, sim drivers, and
the composition-root wiring that binds them. Stop at the HAL. Do not implement the
station TLE HTTP API. Do not add analysis studies in this pass; later studies will
validate each block (plant, inner loop, predictor, residual filter, smear cap).

---

## 1. How to use this document

1. Treat this file as the source of truth for *what to build*.
2. The modules under `flight.payload.control`, `flight.payload.gimbal.lqr`, and
   `flight.payload.tracking.kalman` are a stand-in. Config already marks the two-axis
   LQR arrays as retained until this redesign. Do not preserve that stand-in.
3. There is **no azimuth motor**. Elevation is the only actuated axis. Drivers
   already pin azimuth at 0. Remove azimuth from the control state and from the
   tracking command surface.
4. `REWIND` already exists as the hunt mode. Azimuth `SCAN` is gone. This brief
   specifies how TRACKING and REWIND *move* (rate loop, smear cap, predictor).
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
| Ephemeris | New HAL Protocol. Sim driver now; station TLE API later behind the same Protocol. |
| Plume target | Recompute Earth intersect of the CoG LOS every accepted vision frame. The CoG is not a fixed ground origin. |
| Inner rate | Cheap causal polynomial differentiator on an encoder ring. Not a Kalman filter. |
| Outer estimator | Two-state residual Kalman filter on boresight elevation error and residual rate. |
| Validation studies | Out of scope for the first implementation. |

---

## 3. What to build

A cascaded **elevation** controller for a **raw motor** (no vendor rate servo):

| Layer | Period | Job |
| --- | --- | --- |
| Inner | \(T_{\mathrm{in}}\ll T_{\mathrm{out}}\) | Track a scalar rate reference \(r\) by commanding torque |
| Outer | \(T_{\mathrm{out}}\lesssim T_v\) | Hold boresight elevation error \(e\approx 0\) using kinematic feedforward plus a residual Kalman filter |
| Vision | \(T_v\) | Measure \(e\) from blob CoG \(\times\) IFOV, tagged at shutter time \(t_s\) |
| Predictor | outer ticks | From ISS state + current CoG Earth point + Earth rotation (elevation component only), compute nominal target rate \(\omega_{t,\mathrm{nom}}\) |

Inference can run near camera frame rate (detect wall on the order of 4 ms on the
current Orin mix). That does **not** let the outer loop command torque. The plant
\(J\dot\omega_g+B\omega_g=\tau\) still needs a fast inner loop. The hierarchy is
timescale separation.

This is **not** an LQG regulator. There is no Riccati solve on gimbal inertial
state. The only Kalman filter in the pointing law is the 2-state residual filter.
The inner loop is PI + computed torque.

---

## 4. Non-goals (first implementation)

- Station TLE HTTP client, authentication, or CCSDS encoding of ephemeris.
- Analysis studies, gain scheduling, or online inertia identification.
- Auto-exposure or other camera-control loops.
- Slant, incidence, or GSD as controller or predictor **outputs**. Slant may exist
  as an intermediate of the ray-Earth intersect; it must not enter \(r\) or the
  residual state.
- Putting \(J,B\) in an outer regulator or estimating them online.
- Estimating process noise \(w\). \(w\) exists only in the filter covariance \(Q\).
- An inner-loop Kalman filter or any inner state besides the PI integrator and the
  encoder-rate ring.
- Changing FDIR SAFE latching: SAFE still stows and latches until ground clears it.

---

## 5. Notation

Elevation \(\theta\) is **signed off-nadir**: \(0\) at geocentric nadir, \(+\)
along-track (ISS velocity), \(-\) look-back. One actuated axis.

### 5.1 Gimbal and motor

| Symbol | Meaning |
| --- | --- |
| \(\theta_g\) | True gimbal elevation (rad in the inner loop; deg at the outer/vision boundary is fine if conversions are explicit) |
| \(\omega_g=\dot\theta_g\) | True gimbal rate |
| \(\tau\) | Motor torque command (N·m) |
| \(\tau_{\max}\) | Torque clip |
| \(J\), \(B\) | True inertia and viscous damping |
| \(\hat J\), \(\hat B\) | Config copies used in the inner law |
| \(K_t\) | Torque-to-current slope (real driver only): \(i \approx \tau / K_t\) |
| \(i\) | Motor current (HAL mapping, not a controller state) |
| \(\theta_{\mathrm{hw,min}},\theta_{\mathrm{hw,max}}\) | Hardware travel |
| \(\theta_{\mathrm{sci,min}},\theta_{\mathrm{sci,max}}\) | Science imaging window (config: 0 to +45 deg) |
| \(\omega_{\mathrm{hw}}\) | Hardware slew cap |

### 5.2 Encoder rate estimator

| Symbol | Meaning |
| --- | --- |
| \(\theta_{\mathrm{enc}}[k]\) | Encoder elevation at inner tick \(k\) (includes quantization and noise) |
| \(N_\omega\) | Ring length (number of encoder samples in the fit) |
| \(d_\omega\) | Polynomial degree (\(d_\omega \ge 2\), \(N_\omega > d_\omega\)) |
| \(y_m=\hat\omega_g\) | Estimated gimbal rate from the polynomial fit, used by the PI |
| \(\delta_\omega\) | Group delay of the stencil, \(\approx (N_\omega-1)T_{\mathrm{in}}/2\) |

### 5.3 Inner PI

| Symbol | Meaning |
| --- | --- |
| \(r\) | Rate reference from the outer law |
| \(\varepsilon=r-y_m\) | Rate error |
| \(I\) | PI integrator state |
| \(k_p,k_i\) | PI gains |
| \(v\) | Commanded angular acceleration (PI output, rad/s\(^2\)) |

### 5.4 Vision and CoG geometry

| Symbol | Meaning |
| --- | --- |
| \(\mathbf{p}_{\mathrm{cog}}=(u,v)\) | Blob center of geometry in band-plane pixels |
| \(e_{\mathrm{az}}, e\) | Boresight error (az unactuated; \(e\) is elevation, the KF measurement) |
| \(z_v\) | Vision measurement of \(e\) at shutter time \(t_s\) |
| \(\hat{\mathbf{u}}_{\mathrm{cog}}\) | Unit line of sight through the CoG, mount frame |
| \(\mathbf{r}_{\mathrm{cog}}\) | Earth-ellipsoid intersect of \(\hat{\mathbf{u}}_{\mathrm{cog}}\) (ECEF) |
| \(\rho\) | Slant range along that ray (intersect intermediate; discarded afterward) |
| \(t_s\) | Shutter time of the frame that produced \(z_v\) |
| \(t_v\) | Time the vision packet arrives (after inference) |
| \(T_v\) | Vision / inference period |

The CoG is **not** the stack origin. Wind and unmodeled plume physics move it.
The CoG still sits in the rotating Earth frame at each instant, so its inertial
motion has a large co-rotating-Earth term plus a smaller walk.

### 5.5 Predictor and residual filter

| Symbol | Meaning |
| --- | --- |
| \(\mathbf{r}_s(t),\mathbf{v}_s(t)\) | ISS position and inertial velocity from the ephemeris HAL |
| \(\Omega_E\) | Earth rotation rate |
| \(R_z(\phi)\) | Rotation about Earth polar axis |
| \(\theta_{\mathrm{los}}\) | Elevation of the LOS to \(\mathbf{r}_{\mathrm{cog}}\) |
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

### 5.6 Timing

| Symbol | Meaning |
| --- | --- |
| \(T_{\mathrm{in}},T_{\mathrm{out}},T_v\) | Inner, outer, vision periods |
| \(\tau_{cl}\) | Closed-loop time constant of a tight inner rate servo (design statement only) |
| \(r_{\max,\mathrm{img}}\) | Smear-limited rate cap during science frames |
| \(r_{\max}(\mathrm{mode})\) | Mode-dependent rate clip |

Mount frame for the LOS: \(\hat z\) toward geocentric nadir, \(\hat x\) along-track
(velocity). Match the existing boresight sign convention: image \(+x\) is unactuated
azimuth error, image \(+y\) (down) is \(-e\).

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
model.

**Current mapping (real driver, later):** \(\,i = \tau / K_t\,\) (or a calibrated
affine line \(i = c_1\tau + c_0\)). The inner law always outputs \(\tau\). Do not
fold \(K_t\) into \(k_p,k_i\).

---

## 7. Timescale split

\[
T_{\mathrm{in}}\ll T_{\mathrm{out}}\lesssim T_v. \tag{3}
\]

- Inner: stabilize (1), reject torque disturbance, enforce \(\tau_{\max}\). Period
  on the order of 1 ms (config). Must not wait on inference.
- Outer: update \(r\) from predictor + residual filter. May run at vision rate or
  slightly faster (coast on encoder between blobs).
- Vision: may be near frame rate. Extra frames update \(z_v\). They do not raise
  the torque loop rate.

If the inner loop is tight, the outer *design model* of the gimbal is a first-order
rate servo, not (1):

\[
\dot\omega_g \approx \frac{1}{\tau_{cl}}(r-\omega_g). \tag{4}
\]

Use (4) only as a statement of timescale separation. Do not put \(J,B\) in the
outer filter.

**Threading:** the app shell owns threads, HAL, and the bus. Pure cores stay
side-effect free. The inner loop needs its own time base (`stop_event.wait` style),
encoder reads, and torque writes. The outer/vision path may live in the payload
frame loop.

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

Rate is not a sensor. Vision is not an inner measurement.

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

Default starting point (config, pending a later study): \(N_\omega=7\),
\(d_\omega=2\). Odd \(N_\omega\) is conventional, not required.

**Constraints.**

- Causal only: never use future encoder samples (no centered stencil that waits).
- Group delay \(\delta_\omega \approx (N_\omega-1)T_{\mathrm{in}}/2\) must stay
  \(\ll T_{\mathrm{out}}\) and \(\ll \tau_{cl}\). Do not grow \(N_\omega\) until
  the PI is lagging the plant.
- While the ring is short at startup, drop \(d_\omega\) to \(\min(d_\omega, n-1)\)
  for \(n\) available samples, or hold \(y_m=0\) until \(n>d_\omega\). Prefer the
  reduced-order fit over a two-point slope.
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

---

## 9. Ephemeris HAL

ISS position and velocity are **not** read inside a pure core and are **not**
parsed from CCSDS on the inner/outer loop. Inject them through a new HAL Protocol
(name at implementer discretion, e.g. `IssEphemeris`).

Conceptual surface:

- `read_state(now) -> Result[IssState, FaultCode]`
- `IssState` holds \(\mathbf{r}_s\), \(\mathbf{v}_s\), a frame tag (document ECEF vs
  ECI and stick to one), and the source epoch.

**Sim driver:** propagate a TLE or a recorded trajectory. This unblocks SIL and
unit tests.

**Real driver (later, same Protocol):** station TLE/state API. First
implementation may omit the real driver or stub it. Do not block the controller
on the API contract.

Lazy-import SDKs if any. Composition root selects sim vs real, same as other HAL
axes. The predictor is a pure function of `IssState` plus \(\mathbf{r}_{\mathrm{cog}}\).

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
That would fold wind walk into the "nominal" rate. Wind belongs in
\(\omega_{t,\mathrm{res}}\).

### 10.1 CoG ray and Earth intersect (every accepted vision frame)

1. Take \(\mathbf{p}_{\mathrm{cog}}\) from the matched blob.
2. Convert to a unit LOS \(\hat{\mathbf{u}}_{\mathrm{cog}}\) in mount frame using
   band IFOV and the existing boresight geometry. Include the small unactuated
   azimuth offset in the FOV so \(\mathbf{r}_{\mathrm{cog}}\) is the ground point
   under the CoG, not under the optical axis.
3. Intersect the ray \(\mathbf{r}_s + \rho\hat{\mathbf{u}}_{\mathrm{cog}}\) with
   the Earth ellipsoid (document WGS-84 vs spherical in config). Store
   \(\mathbf{r}_{\mathrm{cog}}\) in ECEF. Discard \(\rho\).
4. If the intersect fails (no hit, behind the camera, below the limb): keep the
   last good \(\mathbf{r}_{\mathrm{cog}}\) and treat the frame as a miss for the
   predictor update, or fall through to miss-coast. Do not NaN the rate.

### 10.2 Nominal elevation rate (co-rotating point)

At time \(t\), with current \(\mathbf{r}_{\mathrm{cog,ECEF}}\) and ISS state:

\[
\mathbf{r}_t(t)=R_z(\Omega_E t)\,\mathbf{r}_{\mathrm{cog,ECEF}}, \tag{10}
\]

\[
\mathbf{l}=\mathbf{r}_t-\mathbf{r}_s,
\qquad
\theta_{\mathrm{los}}=\operatorname{atan2}(\mathbf{l}\cdot\hat x,\ \mathbf{l}\cdot\hat z).
\tag{11}
\]

\[
\omega_{t,\mathrm{nom}}(t)=\dot\theta_{\mathrm{los}}(t). \tag{12}
\]

Compute (12) by an analytic Jacobian of (11) with \(\mathbf{r}_{\mathrm{cog,ECEF}}\)
**held fixed** and Earth rotation plus ISS motion allowed to run, or by a small
central difference of (11) with that same freeze. Include the **elevation**
component of Earth rotation. Do **not** command an azimuth rate. The unactuated
az component of Earth rotation is dropped.

### 10.3 Between vision frames (coast)

Hold the last \(\mathbf{r}_{\mathrm{cog,ECEF}}\) and let it co-rotate with Earth
only (apply \(R_z(\Omega_E\Delta t)\) as needed for the representation). Do not
invent wind. Recompute (12) with the new ISS state each outer tick.

On the next accepted vision frame, replace \(\mathbf{r}_{\mathrm{cog}}\) with the
new intersect. A jump in ECEF is expected (CoG walk). The residual filter absorbs
the rate that the co-rotating model missed; the vision update absorbs the
position error \(e\).

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

\(z_v\) is the elevation component of `boresight_error_deg` from the CoG.
Unactuated \(e_{\mathrm{az}}\) may be logged; it is not in \(\mathbf{x}\).

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

No new information about the residual yet. The filter is integrating (13) with
the last \(\hat\omega_{t,\mathrm{res}}\).

**When a vision packet arrives (update at shutter time):**

1. Rewind: restore the filter snapshot at \(t_s\), or roll back through a ring of
   \((\hat{\mathbf{x}},P,\mathbf{u})\).
2. Innovation \(\nu=z_v-\hat e^-(t_s)\).
3. \(K_f=P^-C^\top(CP^-C^\top+R_v)^{-1}\), \(\hat{\mathbf{x}}\leftarrow\hat{\mathbf{x}}^-+K_f\nu\).
4. Replay predict to now using recorded \(\mathbf{u}(\cdot)\).

\((A,C)\) is observable: if \(e\) grows faster than \(\omega_{t,\mathrm{nom}}-\omega_g\)
predicts, the extra \(\dot e\) is \(\omega_{t,\mathrm{res}}\). \(Q_{22}\) and \(R_v\)
set how fast \(\hat\omega_{t,\mathrm{res}}\) may move. **Never estimate \(w\).**

Until the first accepted \(z_v\): \(\hat e=0\), \(\hat\omega_{t,\mathrm{res}}=0\),
\(r=0\) except REWIND/SAFE policies below. Do not track on a cold filter.

Vision rewind is required. Inference delay is not zero even when it is a few
milliseconds.

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

**Smear cap (imaging).** During science frames, including REWIND:

\[
r_{\max,\mathrm{img}}
=\frac{\sigma_{\mathrm{smear}}\,\mathrm{IFOV}_{\mathrm{band}}}{\Delta t_{\mathrm{exp}}},
\qquad
|r|\le r_{\max,\mathrm{img}}. \tag{19}
\]

\(\sigma_{\mathrm{smear}}\) is `max_motion_smear_px`. Hardware cap
\(\omega_{\mathrm{hw}}\) is separate and larger. Default: always apply (19) in
TRACKING, ACQUIRING (once live), and REWIND. SAFE/STOW may use driver pose
profiles and are not smear-capped the same way.

Do not suppress \(\omega_{t,\mathrm{nom}}+\hat\omega_{t,\mathrm{res}}\) while
TRACKING. Orbit feedforward must continue even when \(e\) is small.

---

## 13. Modes

Keep a pure arbiter. Change what each state commands through the **rate loop**.
Do not use `set_rate` on the HAL; the inner loop turns \(r\) into \(\tau\).

| State | Rate reference \(r\) | Notes |
| --- | --- | --- |
| IDLE | \(0\) | No residual updates from blobs. Predictor may still run. |
| ACQUIRING | (18) after first accepted \(z_v\); else \(0\) | Persistence gate unchanged in spirit. |
| TRACKING | (18) with \(r_{\max,\mathrm{img}}\) | Re-intersect CoG every accepted frame. |
| Miss coast | keep last \(\hat\omega_{t,\mathrm{res}}\), predict-only, still (18) on the coasted \(\mathbf{r}_{\mathrm{cog}}\) | Until `release_persistence_frames`. |
| REWIND | \(r=\mathrm{sign}(\theta_{\mathrm{sci,max}}-\theta_g)\,r_{\max,\mathrm{img}}\) toward the science limb | Hunt after loss below the limb. Not an azimuth raster. |
| REWIND at limb | \(0\) | Wait on orbital motion through the FOV. Blob \(\to\) ACQUIRING. |
| SAFE | do not track | Stow via the existing SAFE path; latch until ground clear. Driver `stow()` is allowed. |

REWIND uses the inner rate loop at the smear cap, not an open-loop absolute
goto. Arrival at the limb is \(|\theta_g-\theta_{\mathrm{sci,max}}|\) within a
small config tolerance (today's arbiter already has a limb-arrival band).

Runaway compares commanded \(r\) to measured \(y_m\), elevation only.

---

## 14. HAL surface (first implementation stops here)

### 14.1 Gimbal

Replace the two-axis rate/angle tracking surface with an elevation torque plant.

Conceptual Protocol:

- `set_torque(tau_nm) -> Result[None, FaultCode]` -- tracking path
- `read_position() -> Result[GimbalPosition, FaultCode]` -- elevation + timestamp
- `read_stow_switch() -> Result[bool, FaultCode]`
- `stow()` / `home()` -- SAFE and pose primitives; driver may profile these on
  the same plant
- Hardware envelope clamp on \(\tau\), \(\omega\), \(\theta\)

Drop azimuth from commands and from control state. `GimbalPosition` is elevation
plus timestamp. Do not keep `set_rate` as the TRACKING actuator command.

Sim driver: integrate (1), encoder noise, travel/torque/slew clamps, stow switch,
azimuth absent (or a constant 0 only if a message field still needs it for one
release; prefer dropping it).

Real driver: Protocol with `set_torque`. Current mapping \(i=\tau/K_t\) lives
here when the amp interface is known. First implementation may leave the real
driver as a thin stub if the wire protocol is unknown.

### 14.2 Ephemeris

Section 9. Sim driver required. Real/station API out of scope for this pass.

---

## 15. Module split (purity)

All of these are **pure** (config in, `now` in, new state + values out):

1. **Kinematics predictor** -- geometry only.
2. **Residual KF** -- predict / update / rewind on (15)-(16).
3. **Outer law** -- (18)-(19).
4. **Inner rate PI** -- (6)-(8), SI.
5. **Polynomial rate fit** -- encoder ring to \(y_m\).
6. **Arbiter** -- discrete mode, no torque math.
7. **CoG intersect** -- pixel \(\to\) \(\mathbf{r}_{\mathrm{cog}}\) (pure if ISS
   state and mount attitude are arguments).

**Payload app shell:** threads, encoder read, torque write, inference \(\to z_v\)
with shutter timestamp, ephemeris HAL read, bus, heartbeats.

**Composition:** only `flight.core` / `sim.sil` construct drivers and the bus.

Do not publish tensors or covariance dumps on the bus. Compact telemetry: mode,
\(e\), \(r\), \(\tau\), \(\omega_{t,\mathrm{nom}}\), \(\hat\omega_{t,\mathrm{res}}\),
\(y_m\).

---

## 16. Config (conceptual keys)

Implementation chooses names. Defaults must match `config/default.toml`. At least:

- Plant: `J_kg_m2`, `B_nms_per_rad`, `tau_max_nm`
- Inner: `kp`, `ki`, `dt_inner_*`, `rate_fit_n`, `rate_fit_degree`
- Outer: `Kp`, `dt_outer_*`, residual `Q` diag, vision `R_v`
- Rewind ring length, smear cap inputs (or derive (19) from sensor exposure +
  `max_motion_smear_px` + band IFOV)
- Predictor: Earth rate, ellipsoid choice
- Ephemeris sim: TLE path or trajectory seed
- Real driver later: `K_t` (A per N·m) or affine current map

Numeric \(J,B,\tau_{\max},k_p,k_i,K_p,Q,R_v,N_\omega\) may start as documented
placeholders. Do not hide them in source. A later analysis study retunes them.

Remove `lqr_Q_diag` / `lqr_R_diag` and dual-axis Kalman fields once unused.
`kalman_dt_s` must not silently mean both filter dt and camera dt.

Keep existing elevation envelopes: hardware `[-45, +90]` deg, science `[0, +45]`,
stow `-45`, home / wait `+45`.

---

## 17. What to stop using

- Dual-axis constant-velocity Kalman `[pan, tilt, pan_rate, tilt_rate]`
- Dual-axis LQR / DARE as the tracking law
- EMA as a pointing estimator
- `set_rate` as the TRACKING actuator command
- First-order sim rate plant as tracking truth
- Azimuth command in the pointing law
- Two-point first-order encoder difference as \(y_m\)
- Azimuth SCAN raster (already gone; do not revive it)

Keep: boresight-error geometry (IFOV, CoG vs plane center), blob gates, IoU match,
SAFE stow latch, encoder timestamp, `REWIND` as the hunt mode, quality smear flag
as an imaging gate (the controller respects the same cap via (19)).

---

## 18. Conceptual checks (not a study)

These are the minimum proofs that the architecture is wired, not a substitute for
later analysis studies.

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
6. **Smear:** TRACKING and REWIND \(|r|\) respect (19).
7. **SAFE:** still stows and ignores blobs until cleared.
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
8. No azimuth motor, no azimuth law, no SCAN raster.
9. First implementation stops at the HAL. Station TLE API and component studies
   come later.
