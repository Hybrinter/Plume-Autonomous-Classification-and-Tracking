# flight.hal.drivers_real.gimbal

**Source:** `packages/flight/src/flight/hal/drivers_real/gimbal.py`
**Kind:** driver

## Purpose

`RealGimbal` is a fail-closed XRT-U-40-109-HV/XD-C rate adapter. The vendored
Xeryon v1.88 module is imported with the driver; serial I/O waits until a rate
command connects with audited production prerequisites. Initial feedback
settings are `INFO=4` and `POLI=2 ms`; achieved cadence must be measured before
freezing them for flight.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `RealGimbal` | class | Fail-closed rate-command adapter and bounded stow scaffold |
| `DutyCreditBucket` | dataclass | Conservative HV on-time credit and full-recovery lockout |

## Inputs and outputs

Construction takes a `Clock`, optional `GimbalConfig`, and optional external watchdog,
vendor-factory, and controller-time mapper test seams.

| Method | Inputs | Outputs |
| --- | --- | --- |
| `set_torque(tau_nm)` | Torque in N·m | Always rejected (`GIMBAL_FAULT`) |
| `set_rate(command)` | Signed rate with lease deadline | Quantized rate command or fail-closed error |
| `goto_angle(el_deg)` | Target degrees | `Ok(None)` |
| `home()` | None | `Ok(None)` |
| `stow()` | None | `Ok(None)` |
| `read_position()` | None | Live `GimbalPosition`, or an error when disconnected/invalid |
| `read_stow_switch()` | None | `Ok(bool)` |

## Behavior

1. `set_rate` quantizes to 0.01 deg/s with a half-step deadband and commands that
   rate through vendored `setSpeed`. Direction changes are ordered stop, motor-off
   confirmation, `setSpeed`, and `startScan`. A zero rate, inhibit, or local fault
   sends `stopScan` then `stopMovements`.
2. On connect, the adapter selects degree units and programs `LLIM`/`HLIM` as
   min/max encoder counts from the hardware travel envelope.
3. `goto_angle` latches a travel-clamped target without changing encoder feedback.
4. `home` and `stow` latch the configured poses. `stow` also arms the switch.
5. `read_position` returns mapped controller feedback, not the latched target.
6. `read_stow_switch` is true only when stow was commanded and the encoder is near
   stow. On this stub that stays False.

## Errors and faults

The driver rejects absent prerequisites, stale/expired leases, invalid feedback,
unmappable controller time, thermal/safety status, exhausted duty credit, and
unconfirmed watchdog inhibition. Shutdown closes communication only after watchdog
confirmation; it never calls the vendor `stop()` helper, which may home the stage.

## Messages

None.

## Configuration

Reads `GimbalConfig` travel limits and stow/home poses plus `GimbalConfig.xeryon`:
86,400 controller counts/revolution, 109 µrad effective resolution, USB 115,200
baud or Jetson UART 76,800 baud, rate quantum/limits, `INFO`/`POLI`, timing bounds,
HV 120-second credit, and bounded stow rate/timeout. Production startup also
requires `settings_file_path` to name an existing settings export from the Xeryon
Windows interface; a missing file has no silent fallback. Motion remains disabled
until license/Python 3.14, watchdog, and stow bench audits are explicitly enabled.

## Constraints

Construction does not open a serial port. The amp interface is future work.

## Related documents

- [`flight.hal.interfaces.gimbal`](../interfaces/gimbal.md)
- [`flight.hal.drivers_real`](../drivers_real.md)
- [`flight.hal.drivers_sim.gimbal`](../drivers_sim/gimbal.md)
