# sim.environment.models

**Source:** `packages/sim/src/sim/environment/models/`
**Kind:** package

## Purpose

The models package holds one Protocol per world axis and the named
implementations that satisfy it.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`earth`](models/earth.md) | module | `EarthModel`, `Wgs84Ellipsoid`, `SphereEarth` |
| [`orbit`](models/orbit.md) | module | `OrbitModel`, `CircularKepler` |
| [`wind`](models/wind.md) | module | `WindModel`, `StillWind`, `ConstantEcefWind` |
| [`plume`](models/plume.md) | module | `PlumeModel`, `BandplaneGaussian`, `EcefColumn`, `PoissonLatitude` |
| [`optics`](models/optics.md) | module | `OpticsModel`, `PinholeOptics` |
| [`appearance`](models/appearance.md) | module | `AppearanceModel`, `OracleMask` |

## Package interface

`models/__init__.py` is empty. Import from the per-axis modules.

## Interactions

Implementations call flight `geo` and `intersect`. They do not publish messages.

## Constraints

- Named models identify fidelity. There is no `low` / `med` / `high` field.
- Protocols are structural. Implementations do not subclass a base class.

## Related documents

- [`sim.environment`](../environment.md)
