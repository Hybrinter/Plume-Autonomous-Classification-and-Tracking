# flight.hal.interfaces.gimbal

**Source:** `packages/flight/src/flight/hal/interfaces/gimbal.py`
**Kind:** module

## Purpose

This module defines the elevation gimbal surface. Detailed-plant SIL retains the
legacy torque surface, while production adds a signed rate command with a mandatory
monotonic lease deadline. `GimbalPosition.timestamp_s` is mapped encoder sample time,
not host receipt time. Independent watchdog confirmation is required for physical
inhibit evidence.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `GimbalPosition` | dataclass | Elevation, mapped sample time, raw time, sequence, status, uncertainty |
| `GimbalHealth` | dataclass | Feedback, lease, requested/quantized rate, controller, duty, and inhibit evidence |
| `GimbalRateCommand` | dataclass | Signed rate plus valid-until monotonic deadline |
| `GimbalActuator` | Protocol | Legacy torque, pose targets, encoder, health, and stow switch |
| `GimbalRateActuator` | Protocol | Production rate, bounded stow-step, and shutdown surface |
| `ExternalWatchdogGate` | Protocol | Independent physical inhibit request/confirmation |

## Inputs and outputs

| Method | Inputs | Outputs |
| --- | --- | --- |
| `set_torque(tau_nm)` | Torque in N·m (detailed SIL only) | `Result[None, FaultCode]` |
| `set_rate(command)` | Signed rate and absolute lease deadline | `Result[None, FaultCode]` |
| `goto_angle(el_deg)` | Target elevation in degrees | `Result[None, FaultCode]` |
| `home()` | None | `Result[None, FaultCode]` |
| `stow()` | None | `Result[None, FaultCode]` |
| `read_position()` | None | `Result[GimbalPosition, FaultCode]` |
| `read_stow_switch()` | None | `Result[bool, FaultCode]` |

## Behavior

1. Detailed-plant tracking and STOW / HOME / GOTO may write torque through `set_torque`.
   Production tracking uses `GimbalRateCommand` through `set_rate`.
2. `stow`, `home`, and `goto_angle` latch a pose target and arm stow-switch logic.
   Detailed-plant SIL turns that target into a rate reference for its inner PI;
   production bounded stow uses leased rate commands.
3. The driver clips torque, rate, and travel to the hardware envelope.
4. `read_position()` returns encoder elevation with a mapped sample timestamp,
   controller time, sequence, status bits, and mapping uncertainty when available.
5. `read_stow_switch()` returns `True` when the mechanism is at the stow pose.

## Errors and faults

Driver implementations map hardware failures to `GIMBAL_FAULT` or related codes.
The Protocol itself does not fix fault values.

## Messages

None.

## Configuration

None at the Protocol level. Concrete drivers read poses, limits, and plant scalars
from `GimbalConfig`.

## Constraints

`GimbalPosition.timestamp_s` is in the injected application monotonic domain. The real
adapter rejects vendor samples without a qualified controller-time mapping. Production
inhibit is not established by serial acknowledgement alone.

## Related documents

- [`flight.hal.interfaces`](../interfaces.md)
- [`flight.hal.drivers_real.gimbal`](../drivers_real/gimbal.md)
- [`flight.hal.drivers_sim.gimbal`](../drivers_sim/gimbal.md)
- [`flight.payload.gimbal`](../../payload/gimbal.md)
