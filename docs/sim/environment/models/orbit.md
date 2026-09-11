# sim.environment.models.orbit

**Source:** `packages/sim/src/sim/environment/models/orbit.py`
**Kind:** module

## Purpose

The orbit module supplies the `OrbitModel` Protocol and `CircularKepler`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `OrbitModel` | Protocol | `state_eci(utc_s) -> IssState` |
| `CircularKepler` | class | Circular two-body orbit in SI metres |

## Inputs and outputs

**`OrbitModel.state_eci(utc_s) -> IssState`**

- Input: UTC seconds.
- Output: ECI position metres and velocity metres per second.

**`CircularKepler.from_ephemeris_config(cfg) -> CircularKepler`**

- Input: `EphemerisConfig`.
- Output: mean elements copied from that config.

## Behavior

1. Mean motion converts revolutions per day to radians per second.
2. Semi-major axis follows `mu / n^2` to the one-third power.
3. At epoch the satellite is at the ascending node on the Greenwich meridian.
4. The kinematics match `SimIssEphemeris.read_state` at the same UTC.

## Errors and faults

None.

## Messages

None.

## Configuration

`inclination_deg`, `mean_motion_rev_per_day`, `mu_m3_s2`, `epoch_utc_s`.

## Constraints

Flight HAL keeps its own copy of these kinematics. This module does not import
`SimIssEphemeris`.

## Related documents

- [`sim.environment.models`](../models.md)
- [`sim.environment.evaluate`](../evaluate.md)
