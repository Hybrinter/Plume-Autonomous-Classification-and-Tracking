# flight

**Source:** `packages/flight/src/flight/`
**Kind:** package

## Purpose

The flight package holds the ISS-attached payload flight software. Subsystem apps run on a shared
message bus and talk to hardware through HAL drivers.

## Contents

| Item | Type | Description |
| --- | --- | --- |
| [`core`](flight/core.md) | package | Composition root, scheduler, config load, command routing, downlink, storage |
| [`payload`](flight/payload.md) | package | Imaging pipeline, inference, gimbal control, science products |
| [`fault`](flight/fault.md) | package | Fault detection, watchdog, SAFE mode policy |
| [`iss_iface`](flight/iss_iface.md) | package | Station link ingress and egress, command ACK, uplink |
| [`thermal`](flight/thermal.md) | package | Thermal housekeeping telemetry |
| [`electrical`](flight/electrical.md) | package | Power monitoring and limit enforcement |
| [`mechanical`](flight/mechanical.md) | package | Launch-lock state and hazardous release |
| [`hal`](flight/hal.md) | package | HAL protocols and sim/real drivers |
| [`libs`](flight/libs.md) | package | Shared types, messages, bus, config, CCSDS, commands, clock, logging |

## Package interface

`flight.__init__` is empty. Import subsystem packages directly.

## Interactions

The composition root in `flight.core` constructs the `MessageBus`, `Clock`, HAL drivers, and
subsystem apps. Apps publish and subscribe on the bus. Apps do not import each other. HAL
protocols live in `flight.hal.interfaces`. Concrete drivers are selected at startup.

Monitored subsystems emit `HeartbeatMsg`. The fault app watches them. Core services publish
command routing, downlink, storage, and model-deploy messages as applicable.

## Constraints

- Subsystem apps communicate only through typed bus messages in `flight.libs.messages`.
- Preprocessing outputs stay inside the payload app. They are not bus messages.
- Library code returns `Result[T, E]`. It does not raise for recoverable errors.
- Only composition roots construct the bus and HAL drivers.
- `build_apps` imports HAL protocols and apps only. It does not import concrete drivers.
- The Hatch wheel for `pact-flight` contains only the `flight` source tree.
- Optional extras `inference`, `camera`, and `gimbal` are named install roles. They currently
  declare no packages.
- A flight-only install does not include `pact-sim`, `pact-tools`, or `pact-gse`.

## Related documents

- [`flight.core`](flight/core.md)
- [`flight.libs`](flight/libs.md)
