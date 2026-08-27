# flight.core

**Source:** `packages/flight/src/flight/core/`
**Kind:** package

## Purpose

The compute and C&DH host loads configuration, wires subsystem apps, and runs them under a
thread scheduler. It is the flight composition root and the shared wiring entry for SIL.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`main`](core/main.md) | module | Production entry: load config, build drivers, run the scheduler |
| [`composition`](core/composition.md) | module | Driver-agnostic app wiring and bus queue policy |
| [`config_loader`](core/config_loader.md) | module | TOML merge, validation, and `PactConfig` mapping |
| [`scheduler`](core/scheduler.md) | module | Daemon-thread scheduler with crash supervision |
| [`select_drivers`](core/select_drivers.md) | module | Environment-axis driver selection |
| [`command_router`](core/command_router.md) | module | Command routing service shell |
| [`routing`](core/routing.md) | pure module | Pure command routing decisions |
| [`downlink`](core/downlink.md) | module | Prioritized AOS-gated downlink manager |
| [`storage`](core/storage.md) | module | Checksummed product store and fault ledger |
| [`model_deploy`](core/model_deploy.md) | module | Pair staging, activation, and rollback |
| [`health`](core/health.md) | pure module | Startup heartbeat health check |

## Package interface

`flight.core.__init__` re-exports:

| Name | Kind |
| --- | --- |
| `MONITORED_SUBSYSTEMS` | constant |
| `Drivers` | class |
| `SystemApps` | class |
| `build_apps` | function |
| `build_flight_system` | function |
| `load_config` | function |
| `main` | function |
| `RunnableApp` | Protocol |
| `Scheduler` | class |

## Interactions

The package constructs the shared `MessageBus` and `Clock`, selects HAL drivers, and wires
subsystem apps. Apps publish and subscribe on the bus only. HAL protocols come from
`flight.hal.interfaces`. Concrete real drivers are imported only in `main` and
`select_drivers`.

Monitored subsystems emit `HeartbeatMsg` on the bus. The fault app watches them. Core services
publish `RoutedCommandMsg`, `CommandAckMsg`, `DownlinkItemMsg`, `StorageWriteMsg`,
`ModelDeployStateMsg`, and `FaultEventMsg` as applicable.

## Constraints

- `build_apps` imports HAL protocols and apps only. It does not import concrete drivers.
- `main` is the flight entry. It imports real driver modules through `select_drivers`.
- One shared in-process bus connects all apps. The scheduler runs each app in a daemon
  thread.
- `MONITORED_SUBSYSTEMS` lists nine heartbeat-emitting subsystems. The fault app is not
  monitored.
- Preprocessing outputs stay inside the payload app. They are not bus messages.

## Related documents

- [`flight`](../flight.md)
- [`flight.core.main`](core/main.md)
- [`flight.core.composition`](core/composition.md)
