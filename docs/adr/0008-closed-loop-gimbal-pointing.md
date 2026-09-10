# ADR 0008: Closed-loop gimbal pointing

**Status:** Accepted (2026-06-11)

**Implements:** spec Section 5 (Pointing and gimbal control) and the SAFE-actuation parts of
Section 6 (FDIR) of `docs/superpowers/specs/2026-06-09-pact-flight-final-state-design.md`.

## Context

The 2026-06-06 baseline (`docs/superpowers/baseline/2026-06-06-pact-flight-parity-baseline.md`,
Sections 4.2 and 4.4) found the pointing chain wired-but-wrong in four distinct ways:

1. **Open-loop delta commands.** `GimbalCommandMsg` carried `az_delta_deg`/`el_delta_deg` and the
   HAL `send_command(GimbalCommandMsg)` applied them with no feedback. There was no absolute
   positioning, no rate command, and no encoder read in the control path.
2. **Absolute-centroid pointing error.** The arbiter multiplied a detected blob's *absolute*
   pixel centroid by a hardcoded `PIXEL_TO_DEG = 0.04` -- so a target dead-center produced a large
   spurious slew instead of zero. The error was never measured relative to the boresight.
3. **Unwired safety gates.** `check_deadband` and `check_rate_limit` existed and were unit-tested
   but were never called in the live control path. A runaway target could not be caught.
4. **SAFE was a no-op for the gimbal.** The FDIR app published `ModeChangeMsg(SAFE)`, but no
   subsystem consumed it: nothing stowed the gimbal. The single largest payload hazard (an
   uncommanded slew into a keep-out zone) had no mechanical response.

The ingest phase (ADR 0007) additionally deferred the **ROI crop**: the payload ran every frame at
full band-plane resolution with `crop_origin_px=(0, 0)`, `scale_factor=1.0`, leaving
`crop_to_roi`/`backproject_pixel` exported but unused. Closing that deferral is coupled to the
pointing math (the crop transform must be inverted to compute boresight error), so it landed here.

## Decision

**`GimbalRequest` pure-core command value.** The decision cores (arbiter, controller) emit a typed
`GimbalRequest(mode, elevation_deg, reason)` -- a pure value, not a bus message. `GimbalCommandMode`
is `RATE` / `ABSOLUTE` / `STOW` / `HOME`. The app shell maps the request onto the HAL
(`set_velocity` / `set_position` / `stow` / `home`) and publishes a `GimbalCommandMsg` telemetry
record of what it issued (`mode`, `elevation_value_deg`, `state`, `reason`). Image association and
ROI estimation remain two-dimensional, but only vertical displacement drives the actuator.

**Closed-loop HAL surface.** `GimbalActuator` is `initialize` / `shutdown` / `find_index` /
`set_position` / `set_velocity` / `stop` / `home` / `stow` / `reset_faults` / `read_state`.
`read_state` returns a timestamped `GimbalAxisState` with position, derived velocity, target,
controller status, and local safety indicators. `SimGimbal`
implements first-order dynamics with lazy clock integration (every call advances the pose by the
elapsed clock time, so the one driver is honest under both the threaded `RealClock` flight loop and
the stepped `ManualClock` SIL), travel/slew clamps, seeded encoder noise, and a stow switch.
`RealGimbal` is a lazy Xeryon v1.88 adapter for one physical elevation axis on an XD-C
controller (`Stage.XRTU_40_109`, axis `X`).  It imports the vendor module only during
composition-root initialization, requires the deployment-managed `settings_default.txt`,
sets degree units plus `INFO=4` and `POLI=2 ms`, indexes before declaring the axis ready, and
uses queued DPOS/startScan/stopScan operations.  Its typed facade maps vendor failures to
`GIMBAL_FAULT` and never exposes vendor objects to application code.

**Boresight-relative pointing error via IFOV.** `boresight_error_deg` inverts the preprocess
crop/decimation transform (tensor pixel -> full-plane pixel via `crop_origin_px` and
`scale_factor`), measures the offset *from the plane center*, and scales by
`SensorConfig.ifov_deg_per_px`. Sign convention: image +x -> +azimuth, while image +y
(downward) -> +elevation to match the physical 0°..45° imaging sweep. `PIXEL_TO_DEG` is deleted.

**Error-space estimator + LQR.** The EMA and Kalman filters now estimate the target's boresight
error in degrees, so the LQR setpoint ("target at boresight") is the zero vector and
`u = -K x` needs no explicit subtraction. The LQR's `u` acts on the error velocity in the plant
model, so the physical slew rate published is `-u = K x` -- slew *toward* the target to shrink the
error.

**Defense-in-depth limit enforcement.** The arbiter enforces the *mission* envelope (deadband,
rate limit, scan travel); the driver clamps the XD-C hardware envelope (−45° to +45°) and the
operational imaging envelope (0° to +45°), with the negative range reserved for SAFE stow. The wired deadband suppresses RATE commands below `min_deadband_px` and
escalates to `GIMBAL_RUNAWAY` above `max_deadband_px` after `max_deadband_strike_count` strikes.

**Encoder-based runaway.** `check_runaway` compares the measured encoder rate between consecutive
reads against the commanded rate (RATE mode only); sustained divergence over `runaway_strike_count`
checks raises `GIMBAL_RUNAWAY`. Outside RATE mode -- or when a read is missing or time does not
advance -- it resets rather than guessing (ABSOLUTE/STOW/HOME approach profiles are driver-internal).

**Latched SAFE with arbiter-issued stow.** `GIMBAL_FAULT` joins `SAFE_TRIGGERING_FAULTS`. On a
drained `ModeChangeMsg(SAFE)` (or any non-zero `mode_flags`), the arbiter transitions to SAFE and
returns a `STOW` request; SAFE latches (no further requests, blobs ignored) until a ground
`ModeChangeMsg(non-SAFE)` clears it back to IDLE. The app shell adds a fallback: if SAFE is
commanded while frame acquisition fails, it calls `stow()` directly -- a stalled camera must not
prevent mechanical safing.

**Single-axis scan and rate safety.** SCAN sweeps elevation over the imaging envelope. RATE
commands are quantized to 0.01°/s, stop below 0.005°/s, and reverse only after a confirmed
stop. A three-second command lease stops an unattended scan. Feedback velocity is derived
from successive EPOS/TIME samples (including controller timestamp wraparound), and repeated
TIME values become stale after 250 ms. The HV/UHV duty timer stops ordinary motion at 105 s,
reserving the final 15 s of the documented 120 s allowance for SAFE stow; the duty fault is
latched until explicit recovery.

**ROI crop re-enabled (deferral closed).** The sensor geometry moves to 1024x1024
(`ifov_deg_per_px = 0.02`, preserving the previous 512x0.04 field of view). Outside TRACKING the
full band plane is decimated to the inference input size (`scale_factor = 1/factor`); in TRACKING
with an initialized estimator a full-resolution ROI is cropped around the Kalman-estimated target
(`scale_factor = 1.0`). Quality flags always run on the full plane before the ROI is taken.

## Consequences

- **SIL exercises real dynamics.** The harness advances the shared `ManualClock` each step, so
  `SimGimbal`'s first-order dynamics integrate between steps and commanded motion actually moves
  the gimbal. The closed-loop tests assert: a thermal fault drives SAFE and the gimbal physically
  reaches the stow pose; a ground `ModeChangeMsg(IDLE)` un-latches SAFE; and TRACKING elevation
  rates point the gimbal toward vertical plume displacement while horizontal displacement alone
  produces no motion.

- **Static-scene honesty limit.** The SIL scene is still static (no `sim/twin` dynamics model), so
  the closed-loop tests assert command *direction* and *mechanism*, not photometric convergence on
  the plume. A real tracking-convergence test needs the dynamics twin (future work).

- **`PIXEL_TO_DEG` and the delta-command model are gone.** `rg "PIXEL_TO_DEG|az_delta_deg|send_command"`
  returns only doc/CONTEXT references. Pointing error is boresight-relative degrees everywhere.

- **Recovery is explicit.** Leaving SAFE requires a ground `ModeChangeMsg` with a non-SAFE mode;
  there is no automatic recovery, consistent with the single-latched-SAFE posture (ADR 0006).

- **Xeryon vendor provenance is explicit.** The v1.88 `Xeryon.py` source is vendored unchanged
  from Xeryon's website ZIP for private repository use; its archive digest and retrieval metadata
  are recorded with the vendor package. Public redistribution remains unresolved. HIL must verify
  the XD-C wiring, settings path, index/sign convention, and the manufacturer's cooldown rule.

- **`GimbalConfig` added to `PactConfig`.** XD-C axis/stage identity, hardware and operational
  envelopes, 86,400 counts/revolution, direction/index offset, settings path, lease/stale
  thresholds, and HV/UHV duty timing are typed config. Simulation dynamics remain shared config;
  obsolete azimuth and `counts_per_deg` fields are removed.
