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
`GimbalRequest(mode, az_deg, el_deg, reason)` -- a pure value, not a bus message. `GimbalCommandMode`
is `RATE` / `ABSOLUTE` / `STOW` / `HOME`. The app shell maps the request onto the HAL
(`set_rate` / `goto_angle` / `stow` / `home`) and publishes a `GimbalCommandMsg` *telemetry record*
of what it issued (`mode`, `az_value_deg`, `el_value_deg`, `state`, `reason`). The delta fields and
`GimbalActuator.send_command` are removed.

**Closed-loop HAL surface.** `GimbalActuator` is `goto_angle` / `set_rate` / `home` / `stow` /
`read_position` (now returning a timestamped `GimbalPosition`) / `read_stow_switch`. `SimGimbal`
implements first-order dynamics with lazy clock integration (every call advances the pose by the
elapsed clock time, so the one driver is honest under both the threaded `RealClock` flight loop and
the stepped `ManualClock` SIL), travel/slew clamps, seeded encoder noise, and a stow switch.
`RealGimbal` is a serial PTU ASCII driver (lazy `pyserial`); its verb set is a documented reference
assumption pending HIL bring-up.

**Boresight-relative pointing error via IFOV.** `boresight_error_deg` inverts the preprocess
crop/decimation transform (tensor pixel -> full-plane pixel via `crop_origin_px` and
`scale_factor`), measures the offset *from the plane center*, and scales by
`SensorConfig.ifov_deg_per_px`. Sign convention: image +x -> +azimuth, image +y (downward) ->
-elevation. `PIXEL_TO_DEG` is deleted.

**Error-space estimator + LQR.** The EMA and Kalman filters now estimate the target's boresight
error in degrees, so the LQR setpoint ("target at boresight") is the zero vector and
`u = -K x` needs no explicit subtraction. The LQR's `u` acts on the error velocity in the plant
model, so the physical slew rate published is `-u = K x` -- slew *toward* the target to shrink the
error.

**Defense-in-depth limit enforcement.** The arbiter enforces the *mission* envelope (deadband,
rate limit, scan travel +-30 deg); the driver clamps the *hardware* envelope (travel +-90/+-45 az/el,
max hardware slew). The wired deadband suppresses RATE commands below `min_deadband_px` and
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

**Reversing absolute SCAN raster.** SCAN issues `ABSOLUTE` pan commands that reverse direction at
+-30 deg (the old delta scan never reversed).

**ROI crop re-enabled (deferral closed).** The sensor geometry moves to 1024x1024
(`ifov_deg_per_px = 0.02`, preserving the previous 512x0.04 field of view). Outside TRACKING the
full band plane is decimated to the inference input size (`scale_factor = 1/factor`); in TRACKING
with an initialized estimator a full-resolution ROI is cropped around the Kalman-estimated target
(`scale_factor = 1.0`). Quality flags always run on the full plane before the ROI is taken.

## Consequences

- **SIL exercises real dynamics.** The harness advances the shared `ManualClock` each step, so
  `SimGimbal`'s first-order dynamics integrate between steps and commanded motion actually moves
  the gimbal. The closed-loop tests assert: a thermal fault drives SAFE and the gimbal physically
  reaches the stow pose; a ground `ModeChangeMsg(IDLE)` un-latches SAFE; and TRACKING rates point
  the gimbal toward the plume (+az, -el for a target right-of and below boresight).

- **Static-scene honesty limit.** The SIL scene is still static (no `sim/twin` dynamics model), so
  the closed-loop tests assert command *direction* and *mechanism*, not photometric convergence on
  the plume. A real tracking-convergence test needs the dynamics twin (future work).

- **`PIXEL_TO_DEG` and the delta-command model are gone.** `rg "PIXEL_TO_DEG|az_delta_deg|send_command"`
  returns only doc/CONTEXT references. Pointing error is boresight-relative degrees everywhere.

- **Recovery is explicit.** Leaving SAFE requires a ground `ModeChangeMsg` with a non-SAFE mode;
  there is no automatic recovery, consistent with the single-latched-SAFE posture (ADR 0006).

- **PTU verb set is unvalidated.** `RealGimbal`'s PP/TP/PS/TS ASCII protocol is a reference
  assumption (FLIR PTU E46-class) to be checked against the actual unit's manual at HIL bring-up;
  the fake-`pyserial` CI tests verify the lazy-import contract and the driver logic, not the wire
  protocol.

- **`GimbalConfig` added to `PactConfig`.** Travel/slew envelope, stow/home poses, sim dynamics
  constants (time constant, encoder noise, seed), and the serial link (port, baud, counts/deg) are
  typed config. `ControllerConfig` gains `runaway_rate_tolerance_deg_per_s` and
  `runaway_strike_count`.
