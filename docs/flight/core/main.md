# flight.core.main

**Source:** `packages/flight/src/flight/core/main.py`
**Kind:** module

## Purpose

The production entry on the payload computer loads config, builds real HAL drivers, wires
every app, and runs them under the thread scheduler until shutdown.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `build_flight_system` | function | Load uplink key, select drivers, call `build_apps` |
| `main` | function | Load config, start scheduler, run startup health gate, supervise |

## Inputs and outputs

**`build_flight_system(config, bus, clock, calib) -> SystemApps`**

- Inputs: validated `PactConfig`, shared `MessageBus`, injected `Clock`, `MosaicCalibration`.
- Output: wired `SystemApps`.
- Raises `SystemExit` on uplink key load failure or real-sensor exposure/gain setup failure.

**`main(config_path="config/default.toml") -> None`**

- Input: path to the TOML config file.
- Raises `SystemExit` on config or calibration load failure.

## Behavior

1. Load config with `load_config`. Exit on `Err`.
2. Load mosaic calibration from `sensor.calibration_dir`, or build an identity calibration.
3. Create a bounded `MessageBus` with `default_bus_policy()`.
4. Select `RealClock` or `ManualClock` from `environment.clock`.
5. Subscribe to `HeartbeatMsg` before the scheduler starts.
6. Call `build_flight_system` to wire apps.
7. Register ten apps on the scheduler in fixed order: payload, fault, iss_iface, thermal,
   electrical, command_router, storage, downlink, mechanical, model_deploy.
8. Start the scheduler.
9. Run the startup health gate for `watchdog_interval_s * 3.0` seconds. Publish
   `ModeChangeMsg(SAFE)` when any monitored subsystem misses a first heartbeat.
10. Register a SIGTERM handler that sets a shutdown event.
11. Call `scheduler.supervise` until SIGTERM or `KeyboardInterrupt`.
12. Call `scheduler.stop` in a `finally` block.

## Errors and faults

Startup raises `SystemExit` for config load failure, calibration load failure, uplink key
load failure, or real-sensor exposure/gain command failure.

The startup health gate publishes `ModeChangeMsg(SAFE)` with `requested_by="startup_health_gate"`
when heartbeats are incomplete.

The scheduler publishes `FaultEventMsg(PROCESS_DIED)` when an app thread exhausts restart
attempts.

## Messages

**Subscribes:** `HeartbeatMsg` (startup health gate).

**Publishes:** `ModeChangeMsg` (failed startup health gate).

## Configuration

Reads the full `PactConfig` through `load_config`. Uses `sensor.calibration_dir`,
`environment.clock`, `command_ingress.hmac_key_path`, and `fault.watchdog_interval_s`.

## Constraints

- Real driver SDK modules load only inside `select_drivers`.
- The bus uses bounded queues per `default_bus_policy()`.
- `sim_inputs` is always `None` in flight. Every environment axis must be `real`.
- Daemon threads share the in-process bus.

## Related documents

- [`flight.core`](../core.md)
- [`flight.core.composition`](composition.md)
- [`flight.core.scheduler`](scheduler.md)
- [`flight.core.select_drivers`](select_drivers.md)
- [`flight.core.health`](health.md)
- [`flight.core.config_loader`](config_loader.md)
