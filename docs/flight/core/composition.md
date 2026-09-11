# flight.core.composition

**Source:** `packages/flight/src/flight/core/composition.py`
**Kind:** module

## Purpose

The driver-agnostic composition root wires every subsystem app on one shared bus and clock.
Flight and SIL call the same `build_apps` function with different driver bundles.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `MONITORED_SUBSYSTEMS` | constant | Nine heartbeat-emitting subsystem names |
| `Drivers` | class | Frozen bundle of HAL drivers, ISS ephemeris, and detector backend |
| `SystemApps` | class | Frozen bundle of all constructed apps and core services |
| `default_bus_policy` | function | Per-message-type queue bounds and overflow rules |
| `build_apps` | function | Construct every app from config, bus, clock, and drivers |

## Inputs and outputs

**`default_bus_policy() -> dict[type, QueuePolicy]`**

- Output: queue policy for each registered message type.

**`build_apps(config, bus, clock, drivers, monitored, calib, uplink_key) -> SystemApps`**

- Inputs: `PactConfig`, shared `MessageBus`, `Clock`, `Drivers`, monitored subsystem names,
  `MosaicCalibration`, uplink HMAC key bytes.
- Output: wired `SystemApps`.

## Behavior

1. `default_bus_policy` assigns `NEVER_DROP` (max 1024) to command, fault, ack, mode,
   upload, and storage-write message types.
2. `default_bus_policy` assigns `DROP_OLDEST` (max 8192) to telemetry, inference, link,
   heartbeat, product, downlink, model-deploy, and safety message types.
3. `build_apps` constructs `StorageService` first.
4. `build_apps` constructs payload, fault, iss_iface, thermal, electrical, command_router,
   downlink, mechanical, and model_deploy apps via each app's `from_config`.
5. `build_apps` passes `drivers.ephemeris` into `PayloadApp.from_config` with the gimbal,
   sensor, detector, and launch lock.
6. `build_apps` passes the same storage instance to payload, iss_iface, and model_deploy.

## Errors and faults

None at the library level. Driver construction errors occur in the caller.

## Messages

`build_apps` does not publish messages. Wired apps subscribe and publish per their own
modules.

Bus policy covers: `CommandMsg`, `RoutedCommandMsg`, `CommandAckMsg`, `FaultEventMsg`,
`ModeChangeMsg`, `ModelStagedMsg`, `UploadChunkMsg`, `StorageWriteMsg`,
`TelemetryEventMsg`, `ProcessedFrameMsg`, `InferenceResultMsg`, `LinkStateMsg`,
`LaunchLockStateMsg`, `GimbalCommandMsg`, `HeartbeatMsg`, `ProductRefMsg`,
`DownlinkItemMsg`, `ModelDeployStateMsg`, `SafetyStateMsg`.

## Configuration

`build_apps` reads the full `PactConfig` and passes typed sub-configs to each app.

## Constraints

- Imports HAL protocols and apps only. No concrete driver imports.
- `MONITORED_SUBSYSTEMS` is
  `("payload", "iss_iface", "thermal", "electrical", "command_router", "storage", "downlink",
  "mechanical", "model_deploy")`.
- The fault app receives the `monitored` tuple. It does not monitor itself.
- `Drivers.launch_lock` is always a sim stand-in. No real launch-lock driver exists.
- `Drivers.ephemeris` is the injected `IssEphemeris` (sim circular Keplerian or real stub).

## Related documents

- [`flight.core`](../core.md)
- [`flight.core.main`](main.md)
- [`flight.core.select_drivers`](select_drivers.md)
