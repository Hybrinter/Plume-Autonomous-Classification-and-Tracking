# Mix-and-match world twin

**Audience:** an implementation agent. This brief is the conceptual specification
for the simulation world twin. It is not as-built documentation and not a code
sketch.

**First implementation scope:** `sim.twin` composition, named-model Protocols,
frozen records, a `TwinConfig` registry, wrap the physics that already exists,
and an optional SIL closed-loop path bound through **both** `SilHarness` and
`ValidationHarness`. Do not add SGP4, 3-D advected plumes, Monte Carlo runners,
or a new flight `EnvironmentConfig` axis.

**What this is:** a flight-consistent **geometry oracle** (same `geo` /
`intersect` conventions as the payload). It is not a full-physics Earth/plume
simulator. Selection-committee pointing studies may quote **relative** error
(flight estimate minus twin truth) from the v1 stack below. They must not quote
absolute smear, detection, or pass-timing performance until later named models
exist.

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
| Truth vs driver feed | `evaluate_twin` emits **truth**. The SIL root pushes a **driver feed** into sim HAL drivers. Flight apps see only HAL observations. |
| Units | SI meters, radians, UTC seconds inside the twin. Convert at analysis report boundaries. |
| Look convention | Flight: elevation 0 rad at geocentric nadir, positive along-track toward the limb. Not analysis `Look` (degrees, 90 at nadir, kilometres). |
| Frames | ECI, ECEF, mount, camera, band-plane pixels. Reuse flight `geo` / `intersect`. |
| Time | `TwinTime` is `monotonic_s` + `utc_s`. Map from the step's `now` the same way `PayloadApp._read_iss_at` maps monotonic to UTC. Models do not read a clock. |
| Import direction | `flight` never imports `sim`. `sim` may import flight types, `geo`, and `intersect`. `analysis` and `gse` import `sim.twin`. Do not import `analysis.lib` from `sim.twin`. |
| Coupling | `SilTwinBind.pre_step` in `sim.sil` binds twin to drivers. Used by **both** `SilHarness` and `ValidationHarness`. |
| Scene feedback | True gimbal elevation and rate (`true_el_deg`, `true_omega_rad_s`) after plant integrate. Not the quantized encoder. |
| Shutter contract | Zero-order hold: evaluate at **start-of-step** true pose, **before** `acquire_frame`. Inner/outer catch-up in `step_once` runs after ingest. Do not evaluate after `advance_inner`. |
| Replay is a model | Recorded mosaics / ISS state are named `replay_*` models. They are not a HAL `real` axis. |
| Monte Carlo | Outer trial harness (`TrialSpec`). Not a method on the twin. |
| Existing SIL | Default GSE / CI keeps `build_frames` + `plume_detector()`. Closed-loop twin rendering is opt-in. CI does not call `evaluate_twin`. |
| v1 physics | Geometry oracle aligned with flight. Not SGP4, smear integration, radiometry, or ISS attitude. |

---

## 4. Two selection spaces

```
                    TwinConfig (world axes, named models)
                    -------------------------------------
                    earth | orbit | wind | plume | projection | silhouette
                                      |
                                      v
                          evaluate_twin(...) -> TwinSample
                                      |
                    +-----------------+------------------+
                    |                                    |
                    v                                    v
              TwinTruth                            TwinDriverFeed
              (oracle; never on bus)               (SIL root only)
                    |                                    |
                    v                                    v
              analysis sink                    SimSensor.load_next
                                               ScriptedDetector.load_mask
                                               (HAL ephemeris stays its own driver)
                    |
                    v
     EnvironmentConfig (HAL axes, sim | real)
                    |
                    v
     select_drivers -> build_apps -> step_once
```

A run is a **pair**: `EnvironmentConfig` plus `TwinConfig`. They are separate
arguments to the SIL builder. Examples:

- CI command-direction test: all HAL `sim`, **no twin**. Pre-rendered
  `static_gaussian_bandplane` frames. Today's path.
- Pointing-performance study: HAL `sim` devices, world `wgs84_ellipsoid` +
  `circular_kepler` + `still` + `fixed_ecef_column` + `pinhole` +
  `geometric_silhouette` (or `oracle_mask` for controller-only checks).
- Detector-in-the-loop: same world, HAL `compute=real` (ONNX) and
  `silhouette=radiometric_mosaic` (later).
- Ephemeris-error: twin orbit is truth `circular_kepler`; HAL
  `SimIssEphemeris` uses a **different** `EphemerisConfig` (bias). Flight
  never reads twin ISS state.
- Replay: `orbit=replay_iss_state`, `plume=replay_mosaic`; HAL sensor still
  `sim`.

SIL, PIL, and HIL remain corners of the **HAL** matrix. They do not name
world fidelity.

Do not add world fields to `PactConfig` or `config/default.toml`. `TwinConfig`
is loaded by `sim.sil`, GSE, or an analysis study.

---

## 5. Records

All new records are `@dataclass(frozen=True, slots=True)`. Prefer tuples of
floats in truth records. Reuse flight `IssState`, `CameraGeometry`, and
`MosaicFrame`.

### 5.1 `TwinTime`

- `monotonic_s: float` — the SIL step's `now`
- `utc_s: float` — `clock.utc_s() + (now - clock.monotonic_s())`

Factory: `TwinTime.from_step(clock, now)`. Do **not** use `clock.monotonic_s()`
or `clock.utc_s()` alone. `SilHarness.run_steps` and GSE `InProcessBackend`
call `step_once(..., now=T)` while the clock still reads `T - dt` until after
the step. Wall-clock ISO is derived at the report boundary, not stored here.

### 5.2 `ShutterPose`

Gimbal optical state at shutter. Not "vehicle" (ISS is also a vehicle).

- `true_el_rad: float`
- `true_el_rate_rad_s: float` (`SimGimbal.true_omega_rad_s`; zero allowed)
- `exposure_us: float` — from `PactConfig.sensor` initial (or last commanded)
- `gain_db: float` — same source

The accessor that fills this record must integrate the plant first (same path
as `read_position`).

### 5.3 `PlumeState` (ECEF, v1)

- `frame: Literal["ecef", "bandplane"]`
- `cog_ecef_m: tuple[float, float, float] | None` — required when `frame="ecef"`
- `centroid_band_px: tuple[float, float] | None` — required when
  `frame="bandplane"`
- `along_sigma_m: float`
- `cross_sigma_m: float`
- `height_proxy_m: float` — default 2000, must equal `cog_height_m`

No principal axes and no per-band contrast in v1. Radiometry lives on
silhouette params. `static_gaussian_bandplane` uses `frame="bandplane"` and
**does not** go through the ECEF pipeline.

### 5.4 `LookAngles`

Flight convention: radians; elevation 0 at geocentric nadir; azimuth optical
(unactuated); slant in meters; incidence in radians; `visible: bool`.

Do not import `analysis.lib.look.Look`.

### 5.5 `SceneGeometry`

Handoff between truth assembly and observation rendering:

- `iss: IssState`
- `plume: PlumeState`
- `look: LookAngles`
- `centroid_band_px: tuple[float, float] | None`

### 5.6 `TwinTruth` (oracle, sim-only)

Never published on the bus. Never passed into a flight app.

- `time: TwinTime`
- `iss: IssState`
- `plume: PlumeState`
- `look: LookAngles`
- `centroid_band_px: tuple[float, float] | None`

Do not copy `ShutterPose.true_el_rad` onto truth. The harness recorder pairs
`(shutter, truth)` per step.

Scoring identity: **flight estimate minus twin truth**. Not flight minus HAL
observation, unless the study is about sensor noise itself.

### 5.7 `TwinDriverFeed` (what the SIL root may push)

- `mosaic: MosaicFrame | None`
- `mask: object | None` — probability mask, `np.ndarray[float32, (H, W)]` at
  band-plane size; type in code as a numpy array, not `object`, except at the
  flight.libs boundary if needed
- `iss_ephemeris: IssState | None` — **logging / scoring only**, populated
  from HAL `SimIssEphemeris.read_state` at the same UTC. **Not** pushed into
  the ephemeris driver. Payload already reads that driver in
  `PayloadApp._read_iss_at`.

Pushed mosaic fields: `timestamp_s=now` (the step monotonic), matching
`timestamp_utc` from the clock mapping, plus the shutter exposure and gain.
Pre-rendered CI frames keep `timestamp_s=float(frame_id)` because GSE steps
with `now=frame_id`.

### 5.8 Bundles

- `TwinSample(truth, feed)`
- `TwinModels` — frozen refs to the six Protocol instances
- `Twin` — small **non-frozen** class holding `TwinModels` + `CameraGeometry`.
  No RNG, no prior plume, no clock, no bus, no gimbal.

---

## 6. Named world axes and first models

Each axis is a `@runtime_checkable Protocol`. Each implementation is a frozen
dataclass plus pure methods. Construction is a **per-axis**
`dict[str, Builder]` in `sim.twin.registry`. Unknown names fail in
`build_twin`, not in `evaluate_twin`. No plugin discovery. No single
`WorldModel` Protocol.

### 6.1 Earth

`EarthModel`: sphere or ellipsoid queries used by intersect and look.

| Name | Behavior |
| --- | --- |
| `wgs84_ellipsoid` | Flight `geo.wgs84_intersect_at_height`. Default. |
| `sphere` | Mean WGS-84 radius. Limb tests only. **Do not mix** with flight ellipsoid intersect in the same study. |

Do not add DEM, refraction, or terrain in the first pass. The 2 km height
proxy is uniform semiaxis inflation (`a+h`, `b+h`), matching flight — not
geodetic height.

### 6.2 Orbit (truth)

`OrbitModel.state_eci(utc_s) -> IssState`.

| Name | Behavior |
| --- | --- |
| `circular_kepler` | Same kinematics as `SimIssEphemeris` (ascending node at epoch). Default. |
| `replay_iss_state` | Tabulated `(utc_s, IssState)`. Later. |

SGP4 / J2 / eccentricity are later names on the same Protocol.

`flight.hal.drivers_sim.ephemeris` must not import `sim`. First pass may copy
the circular formulae into `sim.twin`. A **shared numerical test vector**
must prove twin truth equals `SimIssEphemeris.read_state` at zero bias and
matched UTC. A later extraction into a flight-safe library module is allowed;
do not leave two untested copies.

Epoch: twin uses the HAL convention (ascending node at `epoch_utc_s`).
`analysis.lib.orbit` uses a northern sub-satellite `u0`. Studies that need
that convention pass an explicit `u0` in `CircularKeplerParams`. `epoch_utc_s`
and Earth-rate constants come from `EphemerisConfig` at bind time, not from a
second copy of those numbers.

### 6.3 Wind

`WindModel.velocity_ecef_m_s(utc_s, pos_ecef_m) -> tuple[float, float, float]`.

| Name | Behavior |
| --- | --- |
| `still` | Zero. Default. |
| `constant_ecef` | Constant ECEF vector from params. |

Wind advects the **plume column** on the height-proxy ellipsoid (project the
velocity onto the local tangent plane; do not integrate off the ellipsoid).
It does not torque the ISS or the gimbal. For `fixed_ecef_column`, wind is
inert until `advected_gaussian`.

### 6.4 Plume

`PlumeModel.evaluate(time, prior, wind, earth) -> PlumeState`.

One signature. No `sample(utc_s)` variant.

| Name | Behavior |
| --- | --- |
| `static_gaussian_bandplane` | Today's `sim.scene.plume`: CoG fixed in band-plane pixels. `frame="bandplane"`. Ignores Earth, orbit, wind, and `ShutterPose.true_el_rad`. CI only. |
| `fixed_ecef_column` | Gaussian column on a frozen ECEF CoG at the height proxy. First closed-loop world. |
| `advected_gaussian` | `fixed_ecef_column` whose CoG integrates wind on the ellipsoid. Later. |

New geometric studies must not use `static_gaussian_bandplane`.

### 6.5 Projection

| Name | Behavior |
| --- | --- |
| `pinhole` | Inverse of flight `pinhole_cam_ray`: ECEF CoG -> band-plane `(x, y)`. Optics from `CameraGeometry` resolved at bind time from `PactConfig` (`2 * mosaic pitch`, principal point at band-plane center). |
| `pinhole_smear` | Integrate the pinhole CoG over `exposure_us` along true `omega`. Later. Do not double-count plant integration. |

Do not use `analysis.lib.optics` for closed-loop projection.

### 6.6 Silhouette

| Name | Behavior |
| --- | --- |
| `oracle_mask` | Paint a square or Gaussian at the projected CoG. Matches today's `plume_detector()`. |
| `geometric_silhouette` | Project the column, fill the silhouette, add read noise into a mosaic. Document whether the painted centroid equals the CoG oracle. |
| `radiometric_mosaic` | Later; feeds HAL `compute=real`. |

When HAL `compute=real`, the silhouette must produce a mosaic the preprocessor
will demosaic. `oracle_mask` plus `compute=real` is not a detection-performance
study.

---

## 7. Twin composition

`evaluate_twin` is a thin orchestrator over named pure functions. Physics does
not live in the orchestrator.

```
TwinConfig --build_twin--> Twin (TwinModels + CameraGeometry)
                                |
Twin.evaluate(time, shutter, rng, prior_plume) -> TwinSample
                                |
                         evaluate_twin(...)   # same pipeline
```

Pipeline (`sim/twin/pipeline.py`):

1. `orbit_state_eci(models.orbit, time.utc_s) -> IssState`
2. `evaluate_plume(models.plume, time, prior, wind, earth) -> PlumeState`
3. `build_scene_geometry(...) -> SceneGeometry` — if `plume.frame == "bandplane"`,
   skip ECEF look/project and copy `centroid_band_px` from the plume; else
   `compute_look` + `project_centroid` via flight `geo` / `intersect`
4. `compose_truth(time, geom) -> TwinTruth`
5. `render_feed(models.silhouette, geom, shutter, rng) -> TwinDriverFeed`
6. return `TwinSample(truth, feed)`

After a successful `build_twin`, `evaluate_twin` is infallible (geometry miss
is `visible=False` / `centroid_band_px=None`, not `Err`). Replay-table misses
are out of v1.

**State threading:** the **caller** (SIL bind / trial runner) holds
`prior_plume` and the truth log. The twin does not.

**RNG:** `evaluate_twin(..., rng: Generator)` on every call. No `Twin._rng`.
Trial seed is `master_seed + trial_id`. `SimGimbal.sim_seed` is an independent
stream. `build_frames` keeps its local Generator for CI.

**`CameraGeometry`:** built at bind time from `PactConfig` sensor / inference
fields. Not a world axis. Optional scalar overrides belong under
`PinholeParams` only.

---

## 8. SIL coupling (scene feedback)

`flight.hal.drivers_sim` cannot import `sim`. Binding lives in `sim.sil` as
`SilTwinBind.pre_step`, invoked from **both** `SilHarness.step` and
`ValidationHarness.step` (GSE `InProcessBackend` uses the latter).

When a twin is bound, each step:

1. Integrate the gimbal plant; read `true_el_deg` / `true_omega_rad_s`.
2. `time = TwinTime.from_step(clock, now)`.
3. `shutter = ShutterPose(...)` from those rates plus configured exposure/gain.
4. `sample = twin.evaluate(time, shutter, rng, prior_plume)`.
5. `sensor.load_next(sample.feed.mosaic)` with `timestamp_s=now`.
6. If compute is scripted, `detector.load_mask(sample.feed.mask)`.
7. Call existing `step_once`.
8. Append `(shutter, sample.truth)` to the harness truth log. Optionally log
   HAL `iss_ephemeris` from `SimIssEphemeris.read_state` at the same UTC.
9. `prior_plume = sample.truth.plume`.

This is a **ZOH**: the image is the start-of-step pose; controller catch-up to
`now` runs after ingest. Do not move twin evaluation to after `advance_inner`.

### 8.1 `SimSensor.load_next`

Not on `ImagingSensor`. Concrete `SimSensor` only, like `true_el_deg`.

Single-slot buffer:

| State | `acquire_frame` |
| --- | --- |
| no pending frame and replay exhausted | `Err(CAMERA_STALL)` |
| pending frame | consume it, slot empty |

A second `load_next` before `acquire_frame` **overwrites** (tested). First
closed-loop step must `load_next` before the first `step_once` (or pre-seed
one frame at bind). `select_drivers` may construct `SimSensor` with
`frames=[]` when the twin will feed every step.

### 8.2 `ScriptedDetector.load_mask`

Closed-loop scripted compute needs a sim-only mutator (mask is immutable
today; GSE builds `plume_detector()` once). Mirror `load_next`. Without it,
a moving centroid cannot drive TRACKING.

### 8.3 GSE `SceneSpec`

`SceneSpec` stays a harness recipe (frame count, seed, scalar scripts). It is
not a world model and not a profile. A later optional
`twin_config_path: str | None` may name a twin TOML. CI must not require it.

---

## 9. Tools that are not world models

| Tool | Where | Role |
| --- | --- | --- |
| Simulated clock | `ManualClock` (exists) | Sole SIL time source |
| Orbit propagator | `OrbitModel` | Not a separate runner |
| `TrialSpec` | `analysis` / `tools` | `trial_id`, `master_seed`, `TwinConfig`, `EnvironmentConfig` snapshot, `PactConfig` (or profile hash), `steps`, `dt_s`, `record_truth` |
| `StepRecord` | same | `step_index`, `ShutterPose`, `TwinTruth` (flight telemetry joined later by `tools.analysis`) |
| `TrialRecord` | same | spec + summaries: RMS boresight, in-window time, detect rate |
| Monte Carlo | trial runner | Nested loops; independent Generator per trial |
| Capture / plots | `tools.analysis` (exists) | Join flight datapoints to `TwinTruth` |

---

## 10. Physics notes

**Authority.** Closed-loop geometry uses `flight.payload.gimbal.geo` and
`intersect` only. `analysis.lib` stays for existing paper studies (kilometres,
sub-satellite epoch, 90 deg nadir). New SIL studies import `sim.twin`.

**Height proxy.** Twin Earth and plume CoG use `wgs84_intersect_at_height`
with the same `height_m` as `cog_height_m`. Do not place plumes with
`wgs84_geocentric_radius_km` and then intersect the inflated ellipsoid.

**Pinhole.** Principal point is band-plane center. Band-plane pitch is
`2 * mosaic pitch`. Image +Y down is **minus** elevation. Inverse `project`
must round-trip `pinhole_cam_ray`: a nadir CoG at `true_el_rad=0` lands on
`(width/2, height/2)`.

**Wind.** Tangent-plane advection on the proxy ellipsoid. v1 `still` /
`fixed_ecef_column` means wind does not change pointing RMS yet.

**Ephemeris error.** Configuration: twin `circular_kepler` versus HAL
`SimIssEphemeris` with a wrong inclination or epoch. Do not feed twin ISS
state into the payload predictor. Default shared `EphemerisConfig` means
observation equals truth — that is a zero-error study, not leakage.

**Smear.** v1 omits `pinhole_smear`. Exposure-limited smear p90 claims wait
for that model.

**ISS attitude.** Mount equals LVLH. No flex, no residual attitude in v1.

### 10.1 Dangerous mixes (reject or flag at `build_twin` / study setup)

| Mix | Failure |
| --- | --- |
| `sphere` Earth + flight `wgs84` intersect | Limb and slant disagree |
| `analysis.lib.orbit` epoch + twin `circular_kepler` without `u0` | ~quarter-orbit along-track shift |
| `analysis.lib.look` / `optics` + twin truth | Units, nadir convention, IFOV |
| `static_gaussian_bandplane` + closed-loop gimbal | Image does not move with elevation |
| Surface intersect (`height_m=0`) + flight `cog_height_m=2000` | Systematic CoG offset |
| `oracle_mask` centroid vs `geometric_silhouette` as if they were one truth | Incompatible scoring |
| Twin evaluated after catch-up | Wrong shutter / smear phase |
| Twin truth ISS pushed into flight | Fake zero ephemeris error |

**Coherent v1 stack:** `wgs84_ellipsoid` + `circular_kepler` (HAL epoch) +
`still` + `fixed_ecef_column` + `pinhole` + `geometric_silhouette` or
`oracle_mask`, sampled at start-of-step true elevation.

---

## 11. What not to build

- World fields on `EnvironmentConfig` or `PactConfig`.
- A `FrameSource` HAL Protocol or `load_next` on `ImagingSensor`.
- Twin access from any flight app or pure core.
- Truth messages on the bus.
- A second gimbal plant inside the twin.
- Aerodynamic or ISS-attitude models in the first pass.
- Full radiative transfer, MODTRAN, or DEM.
- SGP4, EGM, or J2 until a named model is added.
- Monte Carlo inside `evaluate_twin`.
- Plugin / entry-point discovery.
- Mutable pose caches or an RNG field on `Twin`.
- Importing `analysis.lib` from `sim.twin`.
- Replacing CI `build_frames` / `plume_detector` in the first pass.
- Pydantic tagged unions for *runtime* model instances.

---

## 12. Config loading and `Result`

`TwinConfig` is a pydantic frozen dataclass (`extra="forbid"`), **not** a
member of `PactConfig`.

- Axis names are `Literal[...]` unions.
- Per-axis param records are nested (`CircularKeplerParams`, …). A
  `@model_validator` rejects a param block that does not match the selected
  name. Unselected blocks are unused, not junk drawers. No `trial_id`, seeds,
  or HAL axes on `TwinConfig`.
- `sim.twin.config_loader.load_twin_config(path) -> Result[TwinConfig, str]`
  and `load_twin_config_dict`.
- `TwinConfig()` Python defaults for unit tests. Optional reference file
  `packages/sim/config/twin_defaults.toml` guarded like
  `test_config_defaults`.
- `build_twin(config, camera) -> Result[Twin, str]` — unknown name or param
  mismatch is `Err`.
- `evaluate_twin` after a successful build does not return `Result`.
- GSE `load_scenario` continues to raise `ValidationError`.
- `SimSensor.load_next` returns `None` (composition-root push). Stall remains
  `acquire_frame -> Err(CAMERA_STALL)`.

---

## 13. First implementation (ordered)

1. Records in §5, `TwinModels`, `TwinConfig` + param blocks, Protocols.
2. Wrap existing physics: `wgs84_ellipsoid`, `circular_kepler`, `still`,
   `static_gaussian_bandplane`, `pinhole`, `oracle_mask`. `fixed_ecef_column`
   + `geometric_silhouette` if they reuse `geo` / `intersect` without new
   numerics.
3. `registry.py` per-axis dicts; `build_twin`; `evaluate_twin` pipeline.
4. `SimSensor.load_next` and `ScriptedDetector.load_mask`.
5. `SilTwinBind.pre_step` on **both** harnesses; default CI unbound.
6. Truth log in the bind object. No matplotlib here.
7. Tests: frozen-seed `evaluate_twin`; true elevation moves the ECEF
   centroid; bandplane compat skips ECEF; `TwinTime.from_step` vs clock lag;
   mosaic `timestamp_s=now`; load_next stall/overwrite/bootstrap; zero-error
   HAL ephemeris equals twin truth; nadir CoG at band-plane center;
   `flight` still cannot import `sim.twin`.
8. STE pages for new modules. Keep this brief until behavior matches.

Later: `advected_gaussian` (tangent-plane), `pinhole_smear`, `replay_*`,
SGP4/J2, `radiometric_mosaic`, `iss_attitude_*`, `atmospheric_refraction`,
`lens_distortion_map`, GSE `twin_config_path`, Monte Carlo trial runner.

---

## 14. Package layout (target)

```
packages/sim/src/sim/twin/
  __init__.py          # Twin, TwinConfig, build_twin, records (not every model)
  config.py            # TwinConfig + param blocks
  config_loader.py     # load_twin_config -> Result
  records.py           # TwinTime, ShutterPose, TwinSample, ...
  protocols.py
  registry.py          # one dict per axis
  pipeline.py          # evaluate_twin and the pure steps
  twin.py              # Twin.evaluate delegates to pipeline
  models/
    earth.py
    orbit.py           # not "ephemeris" (that word is the HAL axis)
    wind.py
    plume.py           # ECEF column models; bandplane calls sim.scene
    projection.py
    silhouette.py
```

`sim.scene.plume` stays behind `static_gaussian_bandplane`. Do not grow
`sim.scene`. Do not add `sim.twin.scene`.

`packages/analysis/src/analysis/lib/{orbit,look,optics}.py` keep serving
existing studies.

---

## 15. Naming

| Use | Do not use |
| --- | --- |
| twin | digital twin, world engine, simulator core |
| world axis | fidelity rung, physics HAL, environment axis |
| named model | fidelity level, plugin, backend |
| truth / driver feed | "observation" on the SIL push path (overloaded) |
| `evaluate` / `evaluate_twin` | `sample()` (collides with housekeeping `sample`) |
| `ShutterPose` | `VehicleSample` |
| `LookAngles` | `LookSample` (redundant suffix; also not analysis `Look`) |
| `TwinDriverFeed` | `TwinObservation` |
| trial | run, case, profile |
| `load_next` / `load_mask` | `inject` (GSE telecommands) |
| `SceneSpec` | world model (it is a harness recipe) |
| orbit (twin axis) | ephemeris (HAL axis) |

Config field names match axis names. Model names are `snake_case` `Literal`
unions.

---

## 16. Dataclass and state rules

- Records and config: `@dataclass(frozen=True, slots=True)`.
- Protocols are structural; no `Model` base class.
- `Twin` is a small concrete class: `_models`, `_camera`. Nothing else.
- No parallel state dict. Stateful plume is `prior_plume` held by the caller.
- Do not cache rotation matrices across steps.
- Do not store `PactConfig` on the twin.
- `LookAngles` is twin-local. `IssState` / `MosaicFrame` / `CameraGeometry`
  are reused.
- Mask arrays are numpy, not `object`, except where flight.libs already uses
  `object` to avoid a numpy import.

---

## 17. Risks the first pass must not ignore

1. **Clock lag.** Twin UTC must use `from_step`, not raw `clock.utc_s()`.
2. **Epoch convention.** HAL ascending node vs analysis sub-satellite `u0`.
   Shared test vector: zero-error HAL read equals twin truth.
3. **Pixel convention.** Half-pixel principal point; mosaic vs band-plane;
   image-y vs elevation sign. Nadir CoG test.
4. **Double integration.** Smear models must not integrate the plant twice.
5. **Compat short-circuit.** `static_gaussian_bandplane` must not pretend
   Earth/orbit/wind ran.
6. **Harness coverage.** GSE will never sample a twin bound only on
   `SilHarness`.
7. **Cost.** Full 1024x1224 ellipse fill is optional in v1; CoG + Gaussian
   paint is enough.
8. **Leakage.** A study that feeds twin ISS into the predictor is not an
   ephemeris-error study. The trial record must say which ISS state HAL used.
