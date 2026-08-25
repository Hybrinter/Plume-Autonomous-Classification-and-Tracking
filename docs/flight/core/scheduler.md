# flight.core.scheduler

**Source:** `packages/flight/src/flight/core/scheduler.py`
**Kind:** module

## Purpose

The scheduler runs each subsystem app in a daemon thread and supervises unexpected thread
death with restart-then-give-up logic.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `RunnableApp` | Protocol | App with a blocking `run(stop_event)` loop |
| `SupervisionAction` | type alias | `"none"`, `"restart"`, or `"give_up"` |
| `next_supervision_action` | function | Pure restart-vs-give-up decision |
| `Scheduler` | class | Start, supervise, stop, and check app threads |

## Inputs and outputs

**`next_supervision_action(alive, stopping, restart_count, restart_limit) -> SupervisionAction`**

- Inputs: thread alive flag, scheduler stopping flag, current restart count, restart limit.
- Output: supervision action string.

**`Scheduler(apps, bus=None, restart_limit=3)`**

- Inputs: list of `(name, RunnableApp)` pairs; optional bus; restart limit per app.

**`Scheduler.start() -> None`** — launch all app threads.

**`Scheduler.check() -> None`** — run one supervision pass.

**`Scheduler.supervise(stop_event, poll_interval_s=1.0) -> None`** — loop `check` until
`stop_event` is set.

**`Scheduler.stop(timeout=5.0) -> None`** — set stop event and join threads.

**`Scheduler.is_running() -> bool`** — return true when any thread is alive.

**`Scheduler.restart_count(name) -> int`** — return restart count for one app.

**`Scheduler.gave_up_on(name) -> bool`** — return true after `PROCESS_DIED` was published.

## Behavior

1. `start` launches each app's `run(stop_event)` in a named daemon thread.
2. `check` inspects each registered app thread.
3. When a thread is dead and the scheduler is not stopping, `check` restarts it if restarts
   remain.
4. When restarts are exhausted, `check` publishes `FaultEventMsg(PROCESS_DIED)` once and
   stops restarting that app.
5. `supervise` calls `check` every `poll_interval_s` until the external stop event is set.
6. `stop` sets the shared stop event and joins each thread up to `timeout` seconds.
7. Dead threads remain in the internal map after `stop`. `is_running` reports liveness
   honestly.

## Errors and faults

Publishes `FaultEventMsg` with `FaultCode.PROCESS_DIED` when an app exhausts
`restart_limit` unexpected-death restarts. Detail names the app and restart count.

## Messages

**Publishes:** `FaultEventMsg(PROCESS_DIED)` on give-up.

## Configuration

None. The caller passes `restart_limit` at construction (default 3).

## Constraints

- Apps share one in-process bus via daemon threads.
- The scheduler owns one shared `threading.Event` passed to every app.
- `next_supervision_action` is a pure function with no bus access.
- When `bus` is `None`, give-up does not publish a fault.

## Related documents

- [`flight.core`](../core.md)
- [`flight.core.main`](main.md)
