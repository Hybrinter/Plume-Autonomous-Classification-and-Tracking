# Mix-and-match world twin

**Audience:** an implementation agent. This brief is the conceptual specification
for the simulation world twin. It is not as-built documentation and not a code
sketch.

**First implementation scope:** `sim.twin` composition, named-model Protocols,
frozen sample records, a `TwinConfig` registry, wrap the physics that already
exists, and an optional SIL closed-loop render path. Do not add SGP4, 3-D
advected plumes, Monte Carlo runners, or a new flight `EnvironmentConfig` axis.

---

## 1. How to use this document

1. Treat this file as the source of truth for *what to build* in `packages/sim`.
2. Keep the two selection spaces distinct. HAL axes stay in `EnvironmentConfig`.
   World axes live in `TwinConfig`. Do not merge them.
3. Keep the repo invariants: flight never imports `sim`; apps never know whether
   a driver is real or sim; pure cores take `now` as an argument; large artifacts
   stay off the bus.
4. After the code exists, update STE-mirrored pages under `docs/sim` to match
   behavior. Do not cite this brief from those pages as design rationale.
5. Analysis studies under `packages/analysis` become *consumers* of the twin.
   They must not grow a second copy of Earth, orbit, look, or pinhole geometry.

---

## 2. Why this exists

PACT already selects **payload devices** per axis (`sensor`, `gimbal`,
`ephemeris`, `compute`, `link`, `clock`) through `EnvironmentConfig` and
`select_drivers`. That matrix answers "which driver talks to the flight app?"

Selection-committee performance work needs a second question: "what world does
that driver look at?" Closed-loop pointing, smear, detection, and track time
all change when Earth, orbit kinematics, wind, plume geometry, and the camera
projection change. Those objects are not payload devices. They must be named
physics models with selectable fidelity, mixable independently of HAL axes.

Today those pieces are split and static:

| Location | What it does | Gap |
| --- | --- | --- |
| `sim.scene.plume` | Pre-renders a fixed Gaussian blob in band-plane pixels | Scene does not move with the gimbal |
| `sim.twin/` | Empty scaffold | No composed world |
| `SimSensor` | Replays a frame list | No live query of a world |
| `SimGimbal` | Rigid-body plant with `true_el_deg()` | Plant is not coupled to a scene |
| `SimIssEphemeris` | Circular Keplerian HAL stand-in | Treated as both truth and observation |
| `analysis.lib.{orbit,look,optics}` | Study-only geometry in kilometres | Duplicate of flight `geo` / `intersect`; SIL cannot use it |
| `tools.analysis` | Passive SIL capture | Scores telemetry, not truth-versus-estimate |

The twin is the shared substrate that closes those gaps. GSE scenarios, SIL
closed-loop tests, and analysis studies all compose the same models.

---

## 3. Settled choices

| Topic | Choice |
| --- | --- |
| Two spaces | HAL axes (`sim` / `real` devices) stay in `EnvironmentConfig`. World axes (named physics models) live in `TwinConfig`, sim-only. |
| World vs device | A world model is anything the payload is not: Earth, truth orbit, wind, plume, projection, silhouette. Gimbal plant, encoder, camera electronics, ONNX, and the station link stay HAL. |
| Names, not rungs | Models have identity names (`circular_kepler`, `wgs84_ellipsoid`, `pinhole`). Do not use `low` / `med` / `high`. |
| Truth vs observation | The twin emits **truth**. HAL sim drivers emit **observations** (encoder, frames, optionally biased ephemeris). Flight apps see only observations. |
| Units | SI meters, radians, UTC seconds inside the twin. Convert at analysis report boundaries. |
| Frames | ECI, ECEF, mount, camera, band-plane pixels. Reuse flight `geo` / `intersect` conventions. Do not invent a parallel look-angle frame. |
| Time | One `TwinTime` per sample: `monotonic_s`, `utc_s`, `wall_clock_iso`. The SIL `ManualClock` is the source. Models do not read a clock. |
| Import direction | `flight` never imports `sim`. `sim` may import flight types, `geo`, and `intersect`. `analysis` and `gse` import `sim.twin`. |
| Coupling | The SIL composition root binds twin to drivers. `SimSensor` does not import the twin. |
| Scene feedback | Each step, the twin samples at the **true** gimbal elevation (`SimGimbal.true_el_deg()`), not the quantized encoder. |
| Replay is a model | "Real world input" means a named `replay_*` model (recorded mosaics, recorded ISS state). It is not a HAL `real` axis. |
| Monte Carlo | An outer trial harness in `analysis` / `tools`. Not a method on the twin. |
| Existing SIL | Default GSE / CI scenes keep `static_gaussian_bandplane` + pre-rendered frames. Closed-loop twin rendering is opt-in. |

---

## 4. Two selection spaces

```
                    TwinConfig (world axes, named models)
                    -------------------------------------
                    earth | orbit | wind | plume | projection | silhouette
                                      |
                                      v
                               Twin.sample(...)
                                      |
                         truth: pose, CoG, mosaic, mask, look
                                      |
            +-------------------------+--------------------------+
            |                         |                          |
            v                         v                          v
     SimSensor (frame)      optional ScriptedDetector      analysis oracle
     SimIssEphemeris            (mask / ONNX)              (never on the bus)
     SimGimbal encoder
            |
            v
     EnvironmentConfig (HAL axes, sim | real)
            |
            v
     select_drivers -> build_apps -> step_once
```

A run is a **point in both spaces**. Examples:

- CI closed-loop command-direction test: all HAL `sim`, world
  `static_gaussian_bandplane` (today). No twin sampling required.
- Pointing-performance study: HAL `sim` devices, world
  `wgs84_ellipsoid` + `circular_kepler` + `constant_ecef` wind +
  `gaussian_column` + `pinhole` + `geometric_silhouette`. Twin renders
  each frame from true elevation.
- Detector-in-the-loop study: same world, HAL `compute=real` (ONNX) and
  `silhouette=radiometric_mosaic`.
- Ephemeris-error study: twin orbit is truth `circular_kepler`; HAL
  ephemeris is the same model plus a bias, or a coarser model.
- Replay study: `orbit=replay_iss_state`, `plume=replay_mosaic`; HAL
  sensor still `sim` because the camera is not on the bench.

SIL, PIL, and HIL remain corners of the **HAL** matrix. They do not name
world fidelity.

Do not add world fields to `PactConfig` or `config/default.toml`. The flight
image has no Earth model. `TwinConfig` is loaded by `sim.sil`, GSE, or an
analysis study.

---

## 5. Truth, observation, and scoring

Three records leave each step. They must not collapse into one object.

### 5.1 `TwinTruth` (oracle, sim-only)

Frozen. Never published on the bus. Never passed into a flight app.

- `time: TwinTime`
- `iss_eci: IssState` (reuse the flight dataclass; SI meters)
- `plume_cog_ecef_m: tuple[float, float, float]`
- `look: LookSample` (az/el/eta/slant/incidence/visible, same meaning as
  flight / analysis look angles)
- `centroid_band_px: tuple[float, float] | None` (pinhole projection of CoG)
- `true_el_rad: float` (optical elevation that formed the image)

Analysis scores flight telemetry against this record.

### 5.2 `TwinObservation` (what HAL may replay)

- `mosaic: MosaicFrame` (or `None` if the sensor axis is `real`)
- `mask: object | None` (probability mask for `ScriptedDetector`)
- `iss_observed: IssState | None` (HAL ephemeris, possibly degraded)

The SIL root pushes `mosaic` into the sensor driver and, when the compute
axis is `sim` and the silhouette model is not ONNX, pushes `mask` into the
scripted detector.

### 5.3 Flight telemetry (already exists)

Control state, gimbal encoder, residual filter, arbiter mode, detections.
`tools.analysis` already captures these. The twin does not duplicate them.

The scoring identity is: **flight estimate minus twin truth**, never flight
minus HAL observation, unless the study is about sensor noise itself.

---

## 6. Named world axes and first models

Each axis is a `@runtime_checkable Protocol`. Each implementation is a frozen
dataclass plus pure methods. Construction is a registry: a string name plus a
frozen param block. Unknown names fail at composition, not at `sample()`.

### 6.1 Earth

`EarthModel`: sphere or ellipsoid queries used by intersect and look.

| Name | Behavior |
| --- | --- |
| `wgs84_ellipsoid` | Flight `geo.wgs84_intersect_at_height`. Default. |
| `sphere` | Mean WGS-84 radius; cheaper limb tests. |

Do not add DEM, refraction, or terrain in the first pass.

### 6.2 Orbit (truth)

`OrbitModel.state_eci(utc_s) -> IssState`.

| Name | Behavior |
| --- | --- |
| `circular_kepler` | Same kinematics as `SimIssEphemeris` (ascending node at epoch). Default. |
| `replay_iss_state` | Tabulated `(utc_s, IssState)` interpolator. Later. |

SGP4 is a later name on the same Protocol. The HAL `SimIssEphemeris` may
share the circular implementation **as a library function in `sim.twin`**,
but the HAL driver remains the observation channel. Flight code keeps its
own copy of the circular propagator until a later extraction; do not make
`flight.hal.drivers_sim.ephemeris` import `sim`.

Epoch convention: twin truth and HAL observation must document whether
`t = epoch` is the ascending node (HAL today) or the northern sub-satellite
point (`analysis.lib.orbit` today). The twin uses the HAL convention so a
zero-error ephemeris study is identical. Analysis studies that need the
sub-satellite convention pass an explicit `u0` in the param block.

### 6.3 Wind

`WindModel.velocity_ecef_m_s(utc_s, pos_ecef_m) -> tuple[float, float, float]`.

| Name | Behavior |
| --- | --- |
| `still` | Zero. Default. |
| `constant_ecef` | Constant ECEF vector from params. |

Wind advects the **plume column**, not the ISS and not the gimbal. Do not
put aerodynamic torque on `SimGimbal`.

### 6.4 Plume

`PlumeModel.sample(utc_s) -> PlumeState`.

`PlumeState` is frozen: CoG ECEF, principal axes (optional), along-track and
cross-track sigma, peak contrast per band, height-proxy meters (default
2000 m, matching the flight tracking ellipsoid).

| Name | Behavior |
| --- | --- |
| `static_gaussian_bandplane` | Today's `sim.scene.plume`: CoG fixed in **band-plane pixels**, not ECEF. Exists so CI does not change. |
| `fixed_ecef_column` | Gaussian column standing on a frozen ECEF CoG at the height proxy. First closed-loop world. |
| `advected_gaussian` | `fixed_ecef_column` whose CoG integrates wind. Later. |

`static_gaussian_bandplane` is a compatibility model. New studies must not
use it when they claim geometric tracking performance.

### 6.5 Projection

`ProjectionModel`: camera from mount elevation + ISS state to band-plane
pixels and, optionally, a mosaic.

| Name | Behavior |
| --- | --- |
| `pinhole` | Flight `pinhole_cam_ray` inverted: ECEF point -> band-plane `(x, y)`. Optics from `CameraGeometry`. |
| `pinhole_smear` | Integrate the pinhole CoG over `exposure_us` along true `omega`. Later. |

The twin **must** use `flight.payload.gimbal.geo` and `intersect` for the
forward and inverse rays. Do not re-derive a thin-lens FOV in kilometres
inside `analysis.lib.optics` for closed-loop work. That module may keep
FOV bookkeeping for paper figures; the twin is the geometric authority.

### 6.6 Silhouette

`SilhouetteModel`: truth plume + camera pose -> detection mask and/or mosaic
radiometry.

| Name | Behavior |
| --- | --- |
| `oracle_mask` | Paint a square/Gaussian at the projected CoG. Matches today's `plume_detector()`. |
| `geometric_silhouette` | Project the column ellipsoid, fill the silhouette, add read noise into a mosaic. First performance-grade observation. |
| `radiometric_mosaic` | Later; feeds HAL `compute=real` (ONNX). |

When HAL `compute=real`, the silhouette model must produce a mosaic the
flight preprocessor will actually demosaic. When HAL `compute=sim`, an
oracle mask is legal and is the CI default.

---

## 7. Twin composition

The twin is a pure stepper. It holds model instances and a seed. It does not
hold a clock, a bus, or a gimbal.

```
TwinConfig  --registry-->  Twin
                               |
Twin.sample(time, vehicle) -> TwinSample(truth, observation)
```

`VehicleSample` is the only vehicle input:

- `true_el_rad: float`
- `true_el_rate_rad_s: float` (for later smear models; zero is allowed)
- `exposure_us: float`
- `gain_db: float`

`sample` does not mutate the twin except RNG consumption for noise. Prefer
an explicit `rng` argument (`np.random.Generator`) over hidden instance
state. If a model needs state (advected CoG), that state lives in a
`PlumeState` returned and fed back as `prior: PlumeState | None`. Do not
store ISS position or gimbal angle inside the twin: those are inputs.

Function composition inside `sample`:

1. `iss = orbit.state_eci(time.utc_s)`
2. `plume = plume_model.sample(time.utc_s, prior, wind, earth)`
3. `look = look_from(iss, plume.cog, vehicle.true_el_rad, earth)`
4. `centroid = projection.project(iss, vehicle, plume.cog)`
5. `observation = silhouette.observe(plume, vehicle, projection, rng)`
6. return `TwinSample(TwinTruth(...), TwinObservation(...))`

Each step is a free function or a Protocol method. Do not build a god-object
with twenty fields of cached pose.

Registry: `build_twin(config: TwinConfig, rng: Generator) -> Twin`. String
names resolve through an explicit dict in `sim.twin.registry`. No callable
dispatch beyond that dict; no plugin discovery.

---

## 8. SIL coupling (scene feedback)

`flight.hal.drivers_sim` cannot import `sim`. The bind lives in `sim.sil`.

**Push, not pull.** Keep `SimSensor` as a replay driver. Each SIL step:

1. Read `true_el_deg` / rate from `SimGimbal` (inspection accessors; not the
   HAL Protocol).
2. `sample = twin.sample(TwinTime.from_clock(clock), vehicle)`.
3. Append or replace the next `SimSensor` frame with `sample.observation.mosaic`.
4. If compute is scripted, refresh the detector mask.
5. Call existing `step_once`.
6. Record `sample.truth` into the analysis sink. Not onto the bus.

This needs a small, documented mutator on `SimSensor` (for example
`push_frame(frame)`) or a one-frame buffer the SIL root owns and passes into
a new `SimSensor` each step. Prefer a **single-slot buffer** on `SimSensor`
(`load_next(frame)`) over rebuilding the driver. That is a flight-package
change, composition-root visible, Protocol-free: `ImagingSensor` stays
`acquire_frame` only.

Do **not** add a `FrameSource` Protocol to `flight.hal.interfaces`. That
would put world-sampling into the HAL surface that real hardware implements.

Clock: `SilHarness.run_steps` already advances `ManualClock` after
`step_once` so `SimGimbal` integrates. Twin sampling must use the same
`now` / `utc_s` as that step. Sample the twin **before** `acquire_frame`
using the gimbal pose at the start of the step (shutter time). Document that
choice; do not silently sample after `advance_inner`.

Existing GSE `SceneSpec` (frame count, seed, scalar scripts) remains valid.
A later GSE field may name a `TwinConfig` TOML. Do not require it for CI.

---

## 9. Tools that are not world models

These sit *around* the twin.

| Tool | Where it lives | Role |
| --- | --- | --- |
| Simulated clock | `flight.libs.time.ManualClock` (exists) | Sole time source for SIL + twin |
| Orbit propagator | `OrbitModel` | Not a separate runner |
| Trial runner | `analysis` or `tools.analysis` | Nested loops: trial seed × TwinConfig × duration |
| Monte Carlo | same trial runner | Independent RNG streams per trial; write Parquet |
| Parameter sweep | same | Cartesian product of named models and numeric params |
| Capture / plots | `tools.analysis` (exists) | Join flight datapoints to `TwinTruth` |

Trial record (frozen, parquet-friendly scalars plus paths to arrays):

- `trial_id`, `seed`, `TwinConfig` as a JSON-stable dict
- HAL `EnvironmentConfig` axes
- time series of truth vs flight residuals
- summary: RMS boresight error, in-window time, smear p90, detect rate

Determinism: one `np.random.Generator` per trial, seeded from
`master_seed + trial_id`. Do not use global `numpy.random`. `SimGimbal`
already has `sim_seed`; the trial runner must set it.

---

## 10. Physics notes (authority and pitfalls)

**Earth.** Flight already intersects a WGS-84 ellipsoid at a 2 km height
proxy. The twin Earth model must hit that same ellipsoid for a zero-error
geometry study. `analysis.lib.orbit.wgs84_geocentric_radius_km` is a
sphere-equivalent radius for look-angle papers; it is not the tracking
surface.

**Look angles.** Elevation is 90 deg at geocentric nadir and decreases
toward the limb along-track. Azimuth is optical (unactuated). Reuse flight
`geo` / analysis `look_at` conventions after a unit conversion. Do not mix
the analysis kilometre `Look` dataclass with the twin SI records.

**Pinhole.** Principal point is band-plane center. Band-plane pitch is
`2 * mosaic pitch`. Inverse projection of a CoG must be consistent with
`pinhole_cam_ray` used in flight `intersect`. A one-pixel convention error
here silently wrecks residual-filter studies.

**Plume height.** 2 km is a tracking proxy, not stereo. Wind advection
moves the CoG on that ellipsoid. The silhouette has vertical extent; the
oracle centroid is the CoG, not the brightest pixel, unless a named
silhouette model says otherwise.

**Smear.** First pass may omit `pinhole_smear`. When added, integrate true
elevation rate over `exposure_us` only. Do not smear azimuth with a motor
that does not exist.

**Ephemeris error.** A useful study is twin truth `circular_kepler` versus
HAL `SimIssEphemeris` with a wrong inclination or epoch. That is
configuration, not a new model. Do not secretly feed flight the twin truth
ISS state.

**Units.** Flight geometry is meters. Analysis lib is kilometres. The twin
is meters. Report writers convert once.

---

## 11. What not to build

- World fields on `EnvironmentConfig` or `PactConfig`.
- A `FrameSource` HAL Protocol implemented by real cameras.
- Twin access from any flight app or pure core.
- `ProcessedFrameMsg` or truth messages on the bus.
- A second gimbal plant inside the twin (`SimGimbal` is the plant).
- Aerodynamic or ISS-attitude models in the first pass.
- Full radiative transfer, MODTRAN, or DEM.
- SGP4, EGM gravity, or two-body + J2 until a named model is added.
- Monte Carlo inside `Twin.sample`.
- Plugin / entry-point discovery for models.
- Mutable "current pose" caches on the twin.
- Replacing CI `build_frames` / `plume_detector` in the first pass.

---

## 12. First implementation (ordered)

1. **Records.** `TwinTime`, `VehicleSample`, `PlumeState`, `LookSample`,
   `TwinTruth`, `TwinObservation`, `TwinSample`, `TwinConfig` — all frozen
   dataclasses with `slots=True`. Prefer tuples of floats over `np.ndarray`
   in the truth record so hashing and parquet stay simple.
2. **Protocols.** One Protocol per world axis, methods as in §6. No extra
   hooks.
3. **Wrap existing physics** as named models: `wgs84_ellipsoid`,
   `circular_kepler`, `still`, `static_gaussian_bandplane`, `pinhole`,
   `oracle_mask`. `fixed_ecef_column` + `geometric_silhouette` are in
   scope if they reuse `geo` / `intersect` without new numerics.
4. **Registry + `build_twin`.** Fail on unknown names. Param blocks are
   nested frozen dataclasses on `TwinConfig`, not loose `dict[str, Any]`.
5. **`SimSensor.load_next`.** One-frame push. Tests for stall-when-empty
   remain.
6. **Optional SIL path.** `SilHarness` (or a thin wrapper) samples the
   twin when constructed with one. Default CI stays pre-rendered frames.
7. **Truth sink.** In-memory list of `TwinTruth` per step, enough for an
   analysis study to compute RMS error. Do not add matplotlib here.
8. **Tests.** Pure `sample()` tests with frozen seeds; a SIL test that
   true elevation motion moves the projected centroid; a test that flight
   apps still cannot import `sim.twin`.
9. **Docs.** STE pages for new modules; keep this brief until behavior
   matches. Do not cite architecture-decision identifiers from STE pages.

Later, in other briefs or studies: `advected_gaussian`, `pinhole_smear`,
`replay_*`, SGP4, ONNX-in-the-loop radiometry, GSE `[[twin]]` TOML, and
the Monte Carlo trial runner.

---

## 13. Package layout (target)

```
packages/sim/src/sim/twin/
  __init__.py          # re-export Twin, TwinConfig, build_twin, records
  config.py            # TwinConfig + per-axis param blocks
  records.py           # TwinTime, TwinSample, ...
  protocols.py         # EarthModel, OrbitModel, ...
  registry.py          # name -> builder
  twin.py              # Twin.sample composition
  models/
    earth.py
    orbit.py
    wind.py
    plume.py
    projection.py
    silhouette.py
```

`sim.scene.plume` stays as the implementation behind
`static_gaussian_bandplane` until a later cleanup. Do not delete it in the
first pass.

`packages/analysis/src/analysis/lib/{orbit,look,optics}.py` keep serving
existing studies. New closed-loop studies import `sim.twin`. A later
migration can point analysis helpers at twin models; that is out of this
pass.

---

## 14. Naming

| Use | Do not use |
| --- | --- |
| twin | digital twin, world engine, simulator core |
| world axis | fidelity rung, physics HAL, environment axis |
| named model | fidelity level, plugin, backend (that word is the detector) |
| truth / observation | ground truth vs "sim" (sim already means HAL stand-in) |
| sample | tick, propagate (orbit may propagate internally) |
| trial | run, case (GSE already has scenario cases) |
| `load_next` | `set_frame`, `inject` (inject is GSE telecommand) |

Config field names match axis names: `earth`, `orbit`, `wind`, `plume`,
`projection`, `silhouette`. Model names are `snake_case` strings stored as
`Literal` unions on `TwinConfig` so mypy rejects typos.

---

## 15. Dataclass and state rules

- Every record and config object is `@dataclass(frozen=True, slots=True)`.
- Protocols are structural; implementations do not subclass a base Model.
- `Twin` may be a frozen dataclass of model instances plus the `Generator`
  if the generator is treated as an opaque seedable handle. If frozen +
  Generator fights mutation, keep `Twin` as a concrete class with private
  `_models` and `_rng` and no other fields.
- No parallel "state dict". If a model is stateful, the state is a frozen
  dataclass returned from `sample` and passed back as `prior`.
- Do not cache rotation matrices across steps unless a test proves it
  matters; ISS state is cheap.
- Do not store `PactConfig` on the twin. Optics come from a
  `CameraGeometry` (flight) passed in `TwinConfig`.
- `LookSample` should be a twin dataclass, not `analysis.lib.look.Look`,
  so `sim` does not import `analysis`.

---

## 16. Risks the first pass must not ignore

1. **Convention drift.** HAL orbit epoch vs analysis sub-satellite epoch vs
   flight ECI/ECEF alignment at `epoch_utc_s`. One comment and one test
   that zero-error HAL ephemeris equals twin truth.
2. **Pixel convention.** Integer vs half-pixel principal point; mosaic vs
   band-plane; y-down image vs along-track elevation. One test that a
   nadir CoG projects to band-plane center at nadir elevation.
3. **Double integration.** Gimbal plant integrates on the clock; a smear
   model must not integrate the same rate a second time against a
   different dt.
4. **Cost.** Geometric silhouette at 1024×1224 inside a Monte Carlo is
   later. First pass may project CoG + paint a Gaussian; full ellipse fill
   is optional.
5. **CI determinism.** Closed-loop rendering must be seed-stable. Noise
   in `sim.scene.plume` already uses a local Generator; keep that.
6. **Leakage.** A study that feeds twin truth ISS state into the payload
   predictor is not an ephemeris-error study. The trial runner must say
   which ISS state HAL received.
