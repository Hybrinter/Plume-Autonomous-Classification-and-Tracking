# flight.libs.types.enums

**Source:** `packages/flight/src/flight/libs/types/enums.py`
**Kind:** pure module

## Purpose

This module defines enumerations used as discriminants and state values across flight
software.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `SystemMode` | enum | Top-level operational mode |
| `GimbalState` | enum | Gimbal arbiter state |
| `GimbalCommandMode` | enum | How gimbal axis values are interpreted |
| `FaultCode` | enum | Enumerated fault conditions |
| `Band` | enum | Mosaic filter band names |
| `FrameUsabilityTag` | enum | Per-frame quality classification |
| `MessageType` | enum | Bus message discriminant |
| `DownlinkPriority` | enum | Downlink queue priority (int values 0..3) |
| `ModelDeployState` | enum | Model deployment lifecycle state |
| `LinkState` | enum | Station link acquisition state |
| `LaunchLockState` | enum | Launch-lock mechanism state |
| `AckStatus` | enum | Command ingress outcome |
| `CommandId` | enum | Command dictionary opcode keys |
| `ParamKind` | enum | Command parameter primitive kind |

## Inputs and outputs

Enum members are referenced by name. String-valued enums use the member name as the value.
`DownlinkPriority` members hold integer values.

## Behavior

### `SystemMode`

`IDLE`, `ACTIVE`, `SCAN`, `MODEL_UPLINK`, `DATA_DOWNLINK`, `SAFE`.

### `GimbalState`

`IDLE`, `ACQUIRING`, `TRACKING`, `SCAN`, `SAFE`.

### `GimbalCommandMode`

`RATE` (axis values are deg/s), `ABSOLUTE` (axis values are target degrees), `STOW`, `HOME`
(axis values ignored).

### `FaultCode`

`NONE`, `INFERENCE_TIMEOUT`, `INFERENCE_NAN`, `CAMERA_STALL`, `STORAGE_FULL`,
`STORAGE_CORRUPT`, `THERMAL_OVER_LIMIT`, `POWER_OVER_LIMIT`, `GIMBAL_RUNAWAY`,
`COMM_TIMEOUT`, `WATCHDOG_EXPIRE`, `MODEL_CORRUPT`, `PROCESS_DIED`, `CALIBRATION_INVALID`,
`FRAME_MALFORMED`, `GIMBAL_FAULT`, `COMMAND_CRC_FAIL`, `COMMAND_AUTH_FAIL`,
`COMMAND_SEQ_ERROR`, `COMMAND_INVALID`, `COMMAND_UNROUTABLE`, `LAUNCH_LOCK_FAULT`.

### `Band`

`BLUE`, `GREEN`, `RED`, `NIR`.

### `FrameUsabilityTag`

`TRAINING`, `TRACKING`, `INVALID`, `CLOUD_CONTAMINATED`, `SUNGLINT`, `SATURATED`,
`MOTION_SMEAR`, `INCOMPLETE_METADATA`.

### `MessageType`

`PROCESSED_FRAME`, `INFERENCE_RESULT`, `GIMBAL_COMMAND`, `TELEMETRY_EVENT`,
`FAULT_EVENT`, `HEARTBEAT`, `MODE_CHANGE`, `COMMAND`, `ROUTED_COMMAND`, `SAFETY_STATE`,
`STORAGE_WRITE`, `PRODUCT_REF`, `DOWNLINK_ITEM`, `UPLINK_CHUNK`, `COMMAND_ACK`,
`LINK_STATE`, `LAUNCH_LOCK_STATE`, `MODEL_STAGED`, `MODEL_DEPLOY`.

### `DownlinkPriority`

`FAULT_EVENT=0`, `COMMAND_ACK=1`, `HK_TELEMETRY=2`, `SCIENCE_PRODUCT=3`. Lower integer
means higher priority.

### `ModelDeployState`

`ACTIVE`, `STAGED`, `ROLLBACK_AVAILABLE`.

### `LinkState`

`AOS` (link up), `LOS` (link down).

### `LaunchLockState`

`ENGAGED`, `RELEASED`, `UNKNOWN`.

### `AckStatus`

`ACCEPTED`, `REJECTED`.

### `CommandId`

`PING`, `SET_THERMAL_LIMIT`, `NOOP`, `EXIT_SAFE`, `RELEASE_LAUNCH_LOCK`,
`UPLOAD_MODEL_CHUNK`, `ACTIVATE_MODEL`.

### `ParamKind`

`STR`, `INT`, `FLOAT`, `BOOL`.

## Errors and faults

None.

## Messages

`MessageType` names every bus message class in `flight.libs.messages`.

## Configuration

None.

## Constraints

- String enum values mirror member names for log readability.
- `DownlinkPriority` is `enum.Enum` with integer values, not `IntEnum`.
- `CommandId` keys match entries in `flight.libs.commands.COMMAND_DICTIONARY`.

## Related documents

- [`flight.libs.types`](flight/libs/types.md)
- [`flight.libs.messages`](flight/libs/messages.md)
- [`flight.libs.commands`](flight/libs/commands.md)
