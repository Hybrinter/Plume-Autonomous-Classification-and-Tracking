# System modes

**Audience:** an implementation agent. This brief is the conceptual specification
for onboard operational modes. It is not as-built documentation.

**First implementation scope:** `SystemMode` members, the fault-app mode manager,
payload inner graphs (INIT homing, IDLE hold, OPERATE arbiter, SAFE inhibit,
STOW-to-rest), ground commands, and SIL/GSE coverage. Xeryon vendor index search,
motorized launch-lock removal, and thermal compare-to-SAFE are follow-on work.

The diagrams.net companion is [`system-modes.drawio`](system-modes.drawio).

---

## 1. How to use this document

1. Treat this file as the source of truth for *what to build* for system modes.
2. The mode manager in the fault app is the only publisher of `ModeChangeMsg`.
3. Each subsystem runs an inner graph selected by the current `SystemMode`.
   Inner graphs do not command other inner graphs.
4. Motor motion is allowed only in `INIT`, `OPERATE`, and `STOW`.
5. Ground test uses the same machine, commands, and homing numbers as orbit.
6. After the code exists, update STE-mirrored pages to match behavior. Do not
   leave this brief as the only description of as-built behavior.

---

## 2. Mission phases versus onboard modes

Unpowered intervals (pack, Dragon, Aegis transfer) are not onboard modes.
Flight software does not run without power.

| Physical phase | Power | Onboard mode |
| --- | --- | --- |
| Ground test | On | Same machine as orbit |
| Pack, Dragon, transfer | Off | None |
| Aegis ports live, computer boots | On | `SAFE` (latched at start) |
| Ground `ENTER_INIT` after faults clear | On | `INIT` then `IDLE` |
| Science | On | `OPERATE` |
| Fault or post-STOW park | On | `SAFE` |
| End of life | On, then off | `STOW` then `SAFE`, then unpowered |

The crew velcro strap is a procedure. It is not a software launch lock.
`INIT` and `SAFE` do not command strap engage or release.

---

## 3. SystemMode members

`SystemMode` has five members: `INIT`, `IDLE`, `OPERATE`, `SAFE`, `STOW`.

Drop unused members `ACTIVE`, `SCAN`, `MODEL_UPLINK`, and `DATA_DOWNLINK`.
`AOS` / `LOS` and `ModelDeployState` stay orthogonal to `SystemMode`.

---

## 4. System machine

Power-on starts **SAFE** with the latch set. A ground command is required to
leave SAFE. Successful STOW returns to SAFE so crew can re-strap.

| From | To | Trigger |
| --- | --- | --- |
| (boot) | `SAFE` | Aegis power; software starts latched |
| `SAFE` | `INIT` | `ENTER_INIT` ARM/EXECUTE; no SAFE-triggering fault this tick |
| `INIT` | `IDLE` | Homing complete and health ok |
| `INIT` | `SAFE` | Homing fail, health fail, or SAFE-triggering fault |
| `IDLE` | `OPERATE` | `ENTER_OPERATE`, or model activity done while `operate_suspended` |
| `OPERATE` | `IDLE` | `ENTER_IDLE`, or model upload / `ACTIVATE` |
| `IDLE` | `STOW` | `ENTER_STOW` ARM/EXECUTE |
| `STOW` | `SAFE` | Rest confirmed at the -45 deg hard stop |
| `INIT`, `IDLE`, `OPERATE`, `STOW` | `SAFE` | SAFE-triggering fault |

`SAFE` is halt in place. The payload drops the rate lease and does not seek
the rest pose. `IDLE` is a zero-rate hold. There is no autonomous `SAFE` to
`OPERATE` path.

A ground `ENTER_IDLE` is not `operate_suspended`. Model activity in commanded
IDLE stays in IDLE after the activity finishes.

---

## 5. Commands

| Command | From | Hazard | Effect |
| --- | --- | --- | --- |
| `ENTER_INIT` | `SAFE` | yes (ARM/EXECUTE) | Only command legal while SAFE-latched |
| `ENTER_OPERATE` | `IDLE` | no | Start closed-loop pointing |
| `ENTER_IDLE` | `OPERATE` | no | Stop pointing; hold |
| `ENTER_STOW` | `IDLE` | yes (ARM/EXECUTE) | Scripted slew to rest, then SAFE |

Retire payload `GIMBAL_STOW`, `GIMBAL_HOME`, and `GIMBAL_GOTO` as live ops.
Checkout motion lives in `INIT` and `STOW` only.

Replace `EXIT_SAFE` with `ENTER_INIT`. The command router SAFE-latch exception
is `ENTER_INIT`.

---

## 6. Payload inner graphs

The payload is the only subsystem with a rich inner machine. Entering a new
system mode aborts the previous inner graph and starts that mode's initial node.

### SAFE: `INHIBIT`

Halt. Drop rate authority. Do not command a stow slew.

### INIT: `CREEP_OUT` then `CREEP_BACK` then `DATUM`

Placeholder assumed-datum homing. No vendor index search.

1. Assume start on the -45 deg hard stop.
2. Creep toward 0 deg (science-window edge). Span is 45 deg.
3. Creep back and press into the hard stop.
4. Declare that contact as -45 deg.
5. Request `IDLE`. The mode manager publishes `ModeChangeMsg`.

An envelope trip is a gimbal fault and enters `SAFE`.

Config keys: `init_creep_rate_deg_per_s`, `init_creep_span_deg` (45), press
timeout. Ground profiles use the same numbers as flight.

### IDLE: `HOLD`

Rate command is 0. Reject pose commands.

### OPERATE: `TRACKING` / `REWIND`

The existing gimbal arbiter runs only while the system mode is `OPERATE`.

### STOW: `SLEW_REST` then `AT_REST`

Creep to the -45 deg rest pose. On confirmation, request `SAFE`.

---

## 7. Other subsystems

Thermal, electrical, and ISS iface keep housekeeping in every powered mode.
They do not grow parallel science FSMs.

- Electrical `POWER_OVER_LIMIT` requests `SAFE` from any mode.
- ISS iface `AOS` / `LOS` gates downlink only.
- Command ingress faults never enter `SAFE`.
- Model deploy lifecycle stays `ACTIVE` / `STAGED` / `ROLLBACK_AVAILABLE`.
  Upload or `ACTIVATE` requests IDLE and sets `operate_suspended` when the
  prior mode was `OPERATE`.
- The motorized launch-lock HAL remains in the tree. It is not on this graph.

---

## 8. Mode manager

A pure core `flight.fault.mode` maps `(SystemModeState, event)` to a new state
and an optional `ModeChangeMsg`. The fault app:

1. Boots `mode=SAFE` and `safe_latched=True`.
2. Publishes `SafetyStateMsg.mode` as the actual `SystemMode`.
3. Accepts homing-complete and stow-complete reports from payload. Only the
   mode manager turns those reports into `ModeChangeMsg`.

Sim gimbal initial pose is `stow_el_deg` (-45 deg) so SIL homing matches orbit.

---

## 9. Mode times function matrix

| Function | SAFE | INIT | IDLE | OPERATE | STOW |
| --- | --- | --- | --- | --- | --- |
| Camera / inference | off | off | off | on | off |
| Gimbal motion | halt | creep [-45, 0] | hold 0 | TRACKING / REWIND | slew to rest |
| Downlink | AOS, faults | AOS | AOS | AOS | AOS |
| Model `ACTIVATE` | no | no | yes | no (preempt IDLE) | no |
| Hazardous commands | `ENTER_INIT` | none | `ENTER_STOW` | none | none |
| Heartbeats / HK | yes | yes | yes | yes | yes |

---

## 10. Follow-on

- Xeryon vendor index search after a hardware investigation.
- Remove the motorized launch-lock command path.
- Thermal datasheet compare that requests `SAFE`.
