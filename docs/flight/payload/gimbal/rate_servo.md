# flight.payload.gimbal.rate_servo

**Source:** `packages/flight/src/flight/payload/gimbal/rate_servo.py`
**Kind:** pure module

## Purpose

`rate_servo` is the inner two-axis rate loop. Two SISO PIs track a rate reference in
rad/s. Computed torque maps the PI output through the inertia and damping estimates.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `RateServoState` | dataclass | Integrators, last encoder angles, and filtered rates |
| `INITIAL_RATE_SERVO_STATE` | constant | Zeroed servo state |
| `reset_servo` | function | Returns a zeroed servo state |
| `step` | function | One PI tick: encoder in, torque out |

## Inputs and outputs

`step` takes the previous `RateServoState`, rate reference `(az, el)` in rad/s, encoder
angles in rad, `dt` in seconds, inertia `j` (2x2, kg m^2), damping `b` (2x2, N m s / rad),
gains `kp` and `ki`, torque clip `tau_max_nm`, filter time constant `ym_lpf_s`, and an
optional `travel_saturated` flag pair. It returns `(new_state, (tau_az_nm, tau_el_nm))`.

`reset_servo` takes an unused prior state and returns `INITIAL_RATE_SERVO_STATE`.

## Behavior

1. A `dt` of zero or less returns zero torque and the input state.
2. Encoder rate is `(theta - last_theta) / dt`. The first sample uses raw rate 0.
3. A first-order filter with time constant `ym_lpf_s` produces `y_m`.
4. Rate error is `r - y_m`. Each axis integrates error unless that axis is travel
   saturated or the torque command clips.
5. Desired acceleration is `v = Kp e + Ki I`. A travel-saturated axis sets `v` to 0.
6. Torque is `tau = J v + B y_m`, then clipped to `+-tau_max_nm` per axis.

## Errors and faults

None.

## Messages

None.

## Configuration

None. Callers pass `J`, `B`, gains, clip, and filter time constant.

## Constraints

- The module is pure. It does not import HAL, the bus, or a clock.
- Units are SI (rad, rad/s, N·m). Degree conversion lives in the app shell.
- Integrators freeze on torque clip and on a travel stop.

## Related documents

- [`flight.payload.gimbal`](../gimbal.md)
- [`flight.hal.interfaces.gimbal`](../../hal/interfaces/gimbal.md)
