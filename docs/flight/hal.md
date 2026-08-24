# flight.hal

**Source:** `packages/flight/src/flight/hal/`
**Kind:** package

## Purpose

The hardware abstraction layer holds device Protocols and two parallel driver sets.
Apps depend on the Protocols. Composition roots inject a real or sim driver for each
axis.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`interfaces`](hal/interfaces.md) | package | Runtime-checkable device Protocols |
| [`drivers_real`](hal/drivers_real.md) | driver set | Real-hardware drivers |
| [`drivers_sim`](hal/drivers_sim.md) | driver set | Sim and SIL stand-in drivers |

## Package interface

None. The root `__init__.py` is empty.

## Interactions

Composition roots in `flight.core` and `sim.sil` construct drivers and pass them into
apps as Protocol types. Apps call driver methods directly. HAL code does not publish or
subscribe on the bus.

## Constraints

- App subsystems must not import `drivers_real` or `drivers_sim`.
- `drivers_real` and `drivers_sim` must not import each other.
- Real drivers import vendor SDKs inside `__init__`, not at module top.
- Drivers satisfy Protocols structurally. They do not subclass the Protocol classes.

## Related documents

- [`flight.hal.interfaces`](hal/interfaces.md)
- [`flight.hal.drivers_real`](hal/drivers_real.md)
- [`flight.hal.drivers_sim`](hal/drivers_sim.md)
