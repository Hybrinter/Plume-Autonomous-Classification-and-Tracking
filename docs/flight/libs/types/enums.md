# flight.libs.types.enums

**Source:** `packages/flight/src/flight/libs/types/enums.py`
**Kind:** pure module

## Purpose

The module defines all flight enumerations used as discriminants, state values, and fault codes
across subsystems.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SystemMode` | enum | Top-level operational mode |
| `GimbalState` | enum | Gimbal arbiter state |
| `GimbalCommandMode` | enum | Gimbal command axis interpretation |
| `FaultCode` | enum | Enumerated fault conditions |
| `Band` | enum | Mosaic-filter band names |
| `FrameUsabilityTag` | enum | Per-frame quality classification |
| `MessageType` | enum | Bus message discriminant |
| `DownlinkPriority` | enum | Downlink queue priority |
| `ModelDeployState` | enum | Model deployment lifecycle state |
| `LinkState` | enum | Station link acquisition state |
| `LaunchLockState` | enum | Launch-lock mechanism state |
| `AckStatus` | enum | Command ingress outcome |
| `CommandId` | enum | Command dictionary opcode keys |
| `ParamKind` | enum | Command parameter primitive kind |

### SystemMode

| Member | Description |
| --- | --- |
| `IDLE` | Default idle mode |
| `ACTIVE` | Inference and gimbal running |
| `SCAN` | Nadir scan, no active target |
| `MODEL_UPLINK` | Model upload session |
| `DATA_DOWNLINK` | Data downlink session |
| `SAFE` | Fault-induced minimal activity |

### GimbalState

| Member | Description |
| --- | --- |
| `IDLE` | No gimbal activity |
| `ACQUIRING` | Seeking a target |
| `TRACKING` | Tracking a confirmed blob |
| `SCAN` | Nadir scan pattern |
| `SAFE` | Gimbal inhibited |

### GimbalCommandMode

| Member | Description |
| --- | --- |
| `RATE` | Axis values are rates in deg/s |
| `ABSOLUTE` | Axis values are target angles in degrees |
| `STOW` | Driver moves to configured stow pose |
| `HOME` | Driver moves to configured home pose |

### FaultCode

| Member | Description |
| --- | --- |
| `NONE` | No fault |
| `INFERENCE_TIMEOUT` | Inference exceeded latency budget |
| `INFERENCE_NAN` | Inference produced NaN output |
| `CAMERA_STALL` | Sensor stopped delivering frames |
| `STORAGE_FULL` | Storage capacity exhausted |
| `STORAGE_CORRUPT` | Storage integrity failure |
| `THERMAL_OVER_LIMIT` | Temperature above limit |
| `POWER_OVER_LIMIT` | Power draw above limit |
| `GIMBAL_RUNAWAY` | Gimbal rate or displacement violation |
| `COMM_TIMEOUT` | Communications timeout |
| `WATCHDOG_EXPIRE` | Subsystem heartbeat missed |
| `MODEL_CORRUPT` | Model artifact failed verification |
| `PROCESS_DIED` | App thread exited unexpectedly |
| `CALIBRATION_INVALID` | Startup calibration integrity failure |
| `FRAME_MALFORMED` | Per-frame geometry violation |
| `GIMBAL_FAULT` | Driver-level gimbal failure |
| `COMMAND_CRC_FAIL` | CCSDS packet CRC or length failure |
| `COMMAND_AUTH_FAIL` | HMAC authentication failure |
| `COMMAND_SEQ_ERROR` | Command sequence replay or ordering failure |
| `COMMAND_INVALID` | Unknown command or bad parameters |
| `COMMAND_UNROUTABLE` | Command target not in dictionary |
| `LAUNCH_LOCK_FAULT` | Launch-lock read or actuation failure |

### Band

| Member | Description |
| --- | --- |
| `BLUE` | Blue passband (~490 nm) |
| `GREEN` | Green passband (~560 nm) |
| `RED` | Red passband (~665 nm) |
| `NIR` | Near-infrared passband (~842 nm) |

### FrameUsabilityTag

| Member | Description |
| --- | --- |
| `TRAINING` | Suitable for training data |
| `TRACKING` | Suitable for tracking |
| `INVALID` | Unusable frame |
| `CLOUD_CONTAMINATED` | Cloud contamination detected |
| `SUNGLINT` | Sunglint detected |
| `SATURATED` | Saturation detected |
| `MOTION_SMEAR` | Motion smear detected |
| `INCOMPLETE_METADATA` | Missing metadata |

### MessageType

Discriminant for every bus message: `PROCESSED_FRAME`, `INFERENCE_RESULT`, `GIMBAL_COMMAND`,
`TELEMETRY_EVENT`, `FAULT_EVENT`, `HEARTBEAT`, `MODE_CHANGE`, `COMMAND`, `ROUTED_COMMAND`,
`SAFETY_STATE`, `STORAGE_WRITE`, `PRODUCT_REF`, `DOWNLINK_ITEM`, `UPLINK_CHUNK`, `COMMAND_ACK`,
`LINK_STATE`, `LAUNCH_LOCK_STATE`, `MODEL_STAGED`, `MODEL_DEPLOY`.

### DownlinkPriority

| Member | Value | Description |
| --- | --- | --- |
| `FAULT_EVENT` | 0 | Highest priority |
| `COMMAND_ACK` | 1 | Command acknowledgements |
| `HK_TELEMETRY` | 2 | Housekeeping telemetry |
| `SCIENCE_PRODUCT` | 3 | Lowest priority |

### ModelDeployState

| Member | Description |
| --- | --- |
| `ACTIVE` | Model is active |
| `STAGED` | Upload staged, awaiting activation |
| `ROLLBACK_AVAILABLE` | Rollback model available |

### LinkState

| Member | Description |
| --- | --- |
| `AOS` | Link up; downlink may drain |
| `LOS` | Link down; hold downlink |

### LaunchLockState

| Member | Description |
| --- | --- |
| `ENGAGED` | Pin engaged; gimbal motion inhibited |
| `RELEASED` | Pin released |
| `UNKNOWN` | Indeterminate read |

### AckStatus

| Member | Description |
| --- | --- |
| `ACCEPTED` | Command passed ingress validation |
| `REJECTED` | Command failed ingress validation |

### CommandId

| Member | Description |
| --- | --- |
| `PING` | Liveness check, no params |
| `NOOP` | Accepted no-op, no params |
| `SET_THERMAL_LIMIT` | Set thermal limit (`limit_c: float`) |
| `EXIT_SAFE` | Hazardous SAFE exit (`phase: str`) |
| `RELEASE_LAUNCH_LOCK` | Hazardous lock release (`phase: str`) |
| `UPLOAD_MODEL_CHUNK` | Chunked model upload params |
| `ACTIVATE_MODEL` | Activate staged model (`version: str`) |

### ParamKind

| Member | Description |
| --- | --- |
| `STR` | String parameter |
| `INT` | Integer parameter |
| `FLOAT` | Float parameter (accepts int or float) |
| `BOOL` | Boolean parameter |

## Inputs and outputs

Each enum member is a constant. Callers pass enum values into messages, config, and pure-core
functions.

## Behavior

1. Enum string values equal member names for log readability.
2. `DownlinkPriority` uses integer values `0..3`. Lower value means higher priority.
3. `CommandId` keys match entries in `flight.libs.commands.COMMAND_DICTIONARY`.

## Errors and faults

None. This module defines constants only.

## Messages

None.

## Configuration

None.

## Constraints

- No other flight module is imported here.
- `DownlinkPriority` is `enum.Enum`, not `IntEnum`. It does not serialize as a bare int into
  CCSDS packets.
- `MessageType.RAW_FRAME` does not exist. Raw frames use `MosaicFrame`, not a bus message.

## Related documents

- [`flight.libs.types`](../types.md)
- [`flight.libs.messages`](../messages.md)
