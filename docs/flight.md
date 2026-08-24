# flight

**Source:** `packages/flight/src/flight`
**Kind:** package

## Purpose

The `flight` package holds the PACT flight software. It groups subsystem apps, the
composition root, HAL drivers, and shared libraries. Apps communicate over the typed
`MessageBus`. The composition root wires the bus, clock, drivers, and apps at startup.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`core`](flight/core.md) | package | Composition root, scheduler, routing, storage, downlink |
| [`electrical`](flight/electrical.md) | package | Power monitoring app |
| [`fault`](flight/fault.md) | package | Fault detection, watchdog, and SAFE policy |
| [`hal`](flight/hal.md) | package | Device protocols and real/sim drivers |
| [`iss_iface`](flight/iss_iface.md) | package | Station link, command ingress, uplink |
| [`libs`](flight/libs.md) | package | Bus, messages, types, config, codecs, logging |
| [`mechanical`](flight/mechanical.md) | package | Launch-lock app |
| [`payload`](flight/payload.md) | package | Imaging, inference, gimbal control |
| [`thermal`](flight/thermal.md) | package | Thermal monitoring app |

## Package interface

None. The package has no top-level `__init__.py`. Callers import subpackages directly
(for example `flight.core.main`, `flight.libs.messages`).

## Interactions

Subsystem apps publish and subscribe on the in-process `MessageBus`. Apps call HAL
driver protocols for hardware. The composition root in `flight.core` constructs the bus,
clock, drivers, and apps. Raw sensor frames pass by direct call from the sensor driver to
the payload app; they do not use the bus.

## Constraints

- Subsystem apps do not import each other. Inter-app traffic uses bus message types only.
- Only composition roots construct the bus and concrete drivers.
- Pure decision cores take time as an argument and perform no I/O.
- Library code returns `Result[T, E]`; it does not raise for recoverable failures.

## Related documents

- [`flight.core`](flight/core.md)
- [`flight.libs`](flight/libs.md)
- [`flight.hal`](flight/hal.md)
- [`flight.payload`](flight/payload.md)
- [`flight.fault`](flight/fault.md)
- [`flight.iss_iface`](flight/iss_iface.md)
