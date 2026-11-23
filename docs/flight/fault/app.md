# flight.fault.app

**Source:** `packages/flight/src/flight/fault/app.py`
**Kind:** app shell

## Purpose

`FaultApp` is the FDIR app shell. Each tick it drains heartbeats, routes fault events and
watchdog expiries through the SAFE-request policy, consumes activation records for
authorized recovery, and publishes the fault-owned `SafetyStateMsg` evidence. The app
never selects a system mode itself; requests go to the external mode authority.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SafetyLatch` | class | Mutable SAFE latch and recovery bookkeeping owned by the shell |
| `FaultApp` | class | Frozen FDIR app with config, bus, clock, epoch, and subscriptions |
| `FaultApp.from_config` | function | Builds the app and subscribes to heartbeats, faults, and activations |
| `FaultApp.initial_entries` | method | Seeds watchdog entries for all monitored subsystems |
| `FaultApp.tick` | method | Runs one watchdog, fault-routing, and safety-evidence cycle |
| `FaultApp.run` | method | Periodic loop until `stop_event` is set |

## Inputs and outputs

- `from_config(cfg, bus, clock, monitored, activation_epoch)` returns a `FaultApp` with
  fresh `HeartbeatMsg`, `FaultEventMsg`, and `SystemModeActivatedMsg` subscriptions.
- `initial_entries()` returns a `dict[str, WatchdogEntry]` keyed by monitored subsystem name.
- `tick(entries, now)` takes the current watchdog dict and monotonic seconds; it returns the
  updated entries dict.
- `run(stop_event)` runs until the event is set.

## Behavior

1. Drain all pending `HeartbeatMsg` values and reset miss counts for matching subsystems.
2. Drain all pending `FaultEventMsg` values and publish `SystemModeRequestMsg(SAFE)` for
   SAFE-triggering codes; latch SAFE and record the latch reason on each trigger.
3. Call `check_heartbeats` and route any `WATCHDOG_EXPIRE` faults through the same policy.
4. Drain `SystemModeActivatedMsg` records; the latch releases only when
   `recovery_authorized` accepts the record (authority-approved EXIT_SAFE recovery) and
   its request ID has not been consumed before. A consumed request ID is recorded on the
   outgoing evidence so the payload can match it exactly once.
5. Publish `SafetyStateMsg` with the latch state, this tick's active SAFE-triggering fault
   set, the latch reason, the injected evidence epoch, a per-epoch evidence sequence, and
   the observation time.

## Errors and faults

The app publishes `SystemModeRequestMsg(SAFE)` for faults in `SAFE_TRIGGERING_FAULTS`.
It does not raise at runtime.

## Messages

| Direction | Type |
| --- | --- |
| Subscribe | `HeartbeatMsg`, `FaultEventMsg`, `SystemModeActivatedMsg` |
| Publish | `SystemModeRequestMsg`, `SafetyStateMsg` |

## Configuration

Reads `FaultConfig` via `cfg.fault`:

| Field | Use |
| --- | --- |
| `watchdog_interval_s` | Tick and loop wait interval |
| `watchdog_max_miss_count` | Consecutive misses before `WATCHDOG_EXPIRE` |

The activation epoch arrives from the composition root, not from config.

## Constraints

- `FaultApp` is frozen; mutable state lives in `SafetyLatch` and the threaded entries dict.
- Heartbeats from subsystems not in `monitored` are ignored.
- `tick` takes `now` explicitly for deterministic tests.
- Requests never grant transitions by themselves; only an authority activation changes
  behavior, and recovery evidence is consumed exactly once per request ID.
- The loop uses `stop_event.wait(timeout=...)` for immediate shutdown.

## Related documents

- [`flight.fault`](../fault.md)
- [`flight.fault.watchdog`](watchdog.md)
- [`flight.fault.policy`](policy.md)
- [`flight.core.composition`](../core/composition.md)
