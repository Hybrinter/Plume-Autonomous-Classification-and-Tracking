# `thermal` Subsystem Context

Non-obvious context not derivable from the individual files in this package.

## Defining Design Decision

`thermal` and `electrical` are deliberately minimal housekeeping apps. Their purpose is not
the thermal/power logic itself (the decision logic is a single `>` comparison) but to *prove
the node is wired into the topology*: every cycle exercises the full four-channel pattern --
heartbeat, telemetry publish, commandable-with-ack, and a threshold fault. Treat them as the
reference template for any new persistent subsystem app.

## thermal == electrical

`ThermalApp` and `ElectricalApp` are structurally identical -- byte-for-byte the same code
apart from `SUBSYSTEM`, the reading's unit/name, the `FaultCode`, and the limit field. Any
change to one almost certainly belongs in the other; they should be edited as a pair. They
are kept as separate copies rather than a shared base because peer apps cannot cross-import
(the `flight-layers` contract) and a shared base would have to sink below the app layer; at
this size, duplication is simpler than a new lower-layer abstraction. (The Rust-migration plan
that once motivated such choices is dropped -- see `docs/adr/0001-python-only-drop-rust.md`.)

## Shared HAL, consumer-owned units

Both apps depend on the *same* `ScalarSensor` Protocol. The sensor returns a bare `float`
with no unit -- meaning (Celsius vs. Watts) is owned by the consuming subsystem, not the
sensor. The same sim/real driver backs both nodes.

## Invariants / Gotchas

- Limits live in `cfg.fault` (`FaultConfig`), not a thermal-specific config: `thermal_limit_c`
  (default 80.0) and `power_limit_w` (default 55.0). The heartbeat cadence (`watchdog_interval_s`,
  default 5.0) is the *same field* that also paces the whole run loop -- one knob, two roles.
- A sensor read `Err` is a silent skip: no telemetry, no fault, no dedicated sensor-fault
  code. The design intent is that a read failure surfaces as *missing* telemetry, which the
  watchdog/ground infer. Do not add a sensor-fault code here without revisiting that contract.
- `handle_commands()` filters on `command.target == SUBSYSTEM` and acks via a `command_ack`
  *telemetry* event (there is no dedicated ack message type). Commands for other targets are
  drained and dropped, not re-queued.
