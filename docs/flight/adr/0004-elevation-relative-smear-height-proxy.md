# ADR-FLIGHT-0004: Elevation-relative smear cap and 2 km CoG height proxy

**Status:** Accepted
**Date:** 2026-09-11
**Topic:** feature-add
**Supersedes:** none
**Superseded-by:** none
**Related:** ADR-REPO-0008

## Context

The tracking loop commanded an absolute elevation rate and treated image smear as
a science-quality estimate that must not reduce control authority. REWIND slewed
at the hardware cap. CoG lock used a WGS-84 surface intersect, so plume height
entered the residual as unmodeled rate. Production hardware now tracks a leased
absolute rate command; the flight law no longer writes torque on that path.

Vertical smear is elevation-relative (`ω_t − ω_g`). Lateral smear from unactuated
azimuth (including Earth rotation at equator crossings) is a different pixel axis
and cannot be cancelled by the elevation motor.

Two-look CoG stereo is not a height measurement: the apparent centroid walks with
the visible mask, and near-nadir pairs have no altitude lever arm.

## Decision

- Lock the CoG as the pinhole ray intersected with a constant 2 km geodetic-height
  ellipsoid (`cog_height_m`). Re-intersect each accepted vision frame at that
  proxy. Do not estimate height from successive CoGs.
- `predict_los` returns elevation, elevation rate, and unactuated azimuth rate of
  a frozen ECEF point. Elevation rate includes ISS motion and Earth rotation.
  Azimuth rate is telemetry and tests only. It does not shrink the elevation
  smear budget and is not commanded.
- TRACKING commands `r = (ω_el + ω_res) + clip(K_p e, ±ω_sharp,el)` then hardware
  and science-window clips. Never smear-clip the matching scene rate.
- REWIND uses boresight intersect at 2 km as the scene rate, hunts at
  `ω_el + sign · ω_sharp,el` for `rewind_sharp_max_s`, then escapes at the
  hardware slew. Residual from a lost target is ignored in REWIND.
- `r` remains an absolute elevation rate for `set_rate` (production) or the
  detailed-plant inner PI (SIL).
- `MOTION_SMEAR` uses `|(ω_g − ω_scene,el)| Δt / IFOV`, not absolute slew and not
  a hypot with azimuth.

## Consequences

- Long exposures limit centering and sharp REWIND hunt, not feedforward match.
- A starved REWIND may fail to walk toward the limb until the escape timeout.
- Height error of the 2 km proxy remains in the residual at a small rate fraction.
- STE pages and the elevation-controller design brief must match this law.

## Alternatives considered

- Hypot-combine azimuth smear into the elevation budget — starves the only
  actuated axis for a lateral error it cannot cancel.
- Two-look CoG triangulation for height — centroid walk is not altitude.
- Surface WGS-84 lock — relief displacement loads the residual.
- Capping absolute `|r|` to smear — blurs a matched track and under-bounds REWIND
  relative smear.
