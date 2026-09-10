# flight.payload.tracking.residual

**Source:** `packages/flight/src/flight/payload/tracking/residual.py`

**Kind:** pure module

## Purpose

This module estimates elevation error `e` and residual target rate `omega_t_res`.
The physical state is `x = [e, omega_t_res]`. The outer estimator consumes
timestamped encoder angles and vision observations.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `ResidualState` | dataclass | State vector, covariance, and measurement flag |
| `ResidualFilter` | dataclass | Continuous process noise, measurement noise, and initial covariance |
| `EncoderSample` | dataclass | Stable ID, sample time, unwrapped angle, and angle variance |
| `NominalRateSample` | dataclass | Stable ID, sample time, and nominal target rate |
| `PredictorReferenceChange` | dataclass | Explicit old and replacement nominal rates |
| `VisionObservation` | dataclass | Frame ID, shutter time, relative error, and variance |
| `ResidualCheckpoint` | dataclass | Posterior and encoder anchor for replay |
| `ResidualHistory` | dataclass | Immutable checkpoint plus bounded raw event history |
| `ObservationDisposition` | enum | Event outcome such as accepted, future, or expired |
| `propagate_displacement` | function | Propagate with nominal and encoder angle deltas |
| `submit_event` | function | Deduplicate and append one immutable event |
| `estimate_at` | function | Replay the history to a reporting time |

The module also exports `ResidualSnapshot`, `predict`, `update`,
`rewind_update`, and `push_snapshot` for the detailed-plant transition. The
outer controller uses `ResidualHistory`, not the snapshot ring.

## Inputs and outputs

`ResidualFilter.from_config(cfg, dt_outer_s)` builds the estimator noise and
initial covariance. The continuous acceleration density is `q_a`. The process
covariance for an interval `dt` is:

```
q_a * [[dt^3 / 3, dt^2 / 2],
       [dt^2 / 2, dt]]
```

`propagate_displacement` takes `dt_s`, nominal angle displacement, encoder
angle displacement, endpoint variance, and reversal variance. It rejects a
negative interval. Encoder and reversal variance affect the elevation
covariance as uncertain inputs.

`submit_event` returns a new history and an `ObservationDisposition`. Every
event has a stable string ID. A repeated ID returns `duplicate` and does not
append a second event.

`estimate_at` returns an `EstimateResult` with the state, a returned history,
and event dispositions. The input history is not modified.

## Behavior

1. Encoder angles are unwrapped radians in the application monotonic time domain.
2. Nominal rates use a causal zero-order hold. A sample at `t` applies to
   intervals that start at `t`.
3. Events sort by `(event time, causal priority, event ID)`. At an equal time,
   propagation finishes first, a reference replacement follows, the encoder
   endpoint is established, and a vision update runs last.
4. A vision shutter time uses only an exact encoder sample or a valid bracket.
   The history does not extrapolate an encoder angle or use a cached angle.
5. Replay uses the net unwrapped displacement from the stable checkpoint anchor
   to each vision or reporting endpoint. It charges endpoint variance once for
   each replay span and adds one configured reversal term for each detected
   direction reversal.
6. Predictor-reference replacement changes `omega_t_res` by
   `old_rate - new_rate`. The total target rate stays continuous. Smooth nominal
   samples do not create a reference replacement.
7. Vision updates use Joseph-form covariance arithmetic. The covariance is
   symmetrized after propagation and update.
8. When the horizon expires, replay first creates a posterior at the new
   boundary. The history retains the predecessor encoder sample needed for a
   boundary bracket and does not cold-start from `P0`.
9. Repeated calls with the same input history are deterministic. Requesting a
   newer reporting time does not mutate the supplied history.

## Errors and faults

The pure API returns dispositions for duplicate, future, expired, invalid
encoder, stale timing, and missing-bracket events. It raises `ValueError` for a
non-finite replay time or a negative low-level propagation interval.

## Messages

None.

## Configuration

`ResidualConfig` supplies `q_accel_rad2_s3`, `R_v`, `P0_diag`, encoder and
reversal variances, reversal threshold, interpolation span, timing uncertainty,
and history horizon. `OuterLoopConfig.dt_s` supplies the normal reporting period.

## Constraints

All estimator functions are pure. The module performs no I/O, clock reads, or
logging. Encoder feedback is an uncertain input. It is not a second measurement
update. The compatibility snapshot helpers remain only during the detailed
plant transition.

## Related documents

- [`flight.payload.tracking`](../tracking.md)
- [`flight.payload.control`](../control.md)
