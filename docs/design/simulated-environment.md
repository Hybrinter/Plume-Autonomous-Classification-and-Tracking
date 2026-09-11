# Simulated environment

**Audience:** an implementation agent. This brief specifies the simulated world
that produces camera frames, plume geometry, and ISS kinematics. It is not a
model of the payload.

**First implementation scope:** `sim.environment` named models, `evaluate`,
and `EnvironmentConfig`. Driver-feed mutators and SIL bind land in follow-on
changes. Do not add SGP4, smear optics, advected plume, or GSE environment TOML
in this stack.

---

## Two selection spaces

Driver selection lives in flight `PactConfig.drivers` and profile TOML
`[drivers]`. Each axis is `sim` or `real`. Composition builds HAL drivers from
that table.

Environment selection lives in pact-sim `EnvironmentConfig` and
`packages/sim/config/environment.toml`. Each axis names a world model
(`wgs84_ellipsoid`, `circular_kepler`, `pinhole`, `oracle_mask`). Fidelity is
the named model. There is no `low` / `med` / `high` ladder.

## Contracts

Each model type is a `@runtime_checkable` Protocol in `sim.environment.models`.
Implementations are frozen dataclasses. The evaluate pipeline is a pure function
of time, shutter pose, RNG, and prior plume state. Environment objects do not
own a clock, bus, or gimbal.

## Binding

A SIL bind is opt-in on `SilHarness` and `ValidationHarness`. Default CI and GSE
keep pre-rendered `sim.scene.plume` frames. A harness that constructs an
`Environment` feeds mosaics or masks into sim drivers only when those fields are
not `None`.
