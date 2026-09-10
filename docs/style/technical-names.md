# Technical names

Use these domain terms with a fixed meaning. Do not invent synonyms on
descriptive pages.

## Nouns

| Name | Meaning |
| --- | --- |
| arbiter | Pure FSM that resolves gimbal mode and request type |
| band plane | Half-resolution spectral plane after demosaic |
| blob | Connected region extracted from a detection mask |
| boresight error | Angular offset from the optical axis to the target, in degrees |
| bus | Typed in-process `MessageBus` |
| composition root | Code that builds the bus, clock, drivers, and apps |
| deadband | Minimum command magnitude that the control law may emit |
| downlink | Prioritized path of telemetry and products to the station |
| gimbal | Single-axis elevation actuator |
| heartbeat | Periodic liveness message from a monitored subsystem |
| housekeeping | Thermal and electrical scalar telemetry |
| mosaic frame | Raw 2x2 filter mosaic image from the sensor |
| payload | Science subsystem that runs acquire through point |
| profile | Environment config that selects real or sim drivers per axis |
| SAFE | System mode that stows motion and waits for ground exit |
| scene | Synthetic imagery and readings used by SIL |
| SIL | Software-in-the-loop harness that steps flight apps |
| station link | Byte-level CCSDS transport to the ISS |
| venue | Validation setting such as SIL, PIL, or HIL |
| watchdog | FDIR check that counts missed heartbeats |

## Verbs

| Verb | Meaning |
| --- | --- |
| acquire | Read one frame from the imaging sensor |
| demosaic | Split a mosaic frame into spectral band planes |
| detect | Run the detector backend on a processed tensor |
| publish | Place a message on the bus |
| stow | Command the gimbal to the safe park pose |
| subscribe | Receive messages of one type from the bus |
| supersede | Replace a prior ADR with a new ADR |

## Mode names

Use the enum member text as written: `IDLE`, `TRACKING`, `REWIND`,
`SCAN`, `SAFE`, `ABSOLUTE`, `STOW`, `HOME`.
