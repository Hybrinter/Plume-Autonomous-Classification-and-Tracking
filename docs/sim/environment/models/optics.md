# sim.environment.models.optics

**Source:** `packages/sim/src/sim/environment/models/optics.py`
**Kind:** module

## Purpose

The optics module supplies the `OpticsModel` Protocol and `PinholeOptics`.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `OpticsModel` | Protocol | `project_centroid(...)` |
| `PinholeOptics` | class | Inverse of flight pinhole and mount maps |

## Inputs and outputs

**`OpticsModel.project_centroid(iss, shutter, cog_ecef_m, camera, omega, epoch)`**

- Inputs: ISS ECI state, true shutter pose, ECEF CoG metres, band-plane
  `CameraGeometry`, Earth rate, epoch UTC.
- Output: band-plane `(u, v)` pixels or `None`.

## Behavior

1. The CoG rotates into ECI.
2. The look vector maps into LVLH then the nadir camera frame at true
   elevation.
3. The pinhole inverse yields pixels. A point behind the camera returns `None`.

## Errors and faults

None. A miss or non-finite pixel returns `None`.

## Messages

None.

## Configuration

None. Focal length and pitch come from `CameraGeometry`.

## Constraints

Elevation is the true plant angle in `ShutterPose`, not encoder counts.

## Related documents

- [`sim.environment.models`](../models.md)
- [`sim.environment.evaluate`](../evaluate.md)
- [`sim.environment.records`](../records.md)
