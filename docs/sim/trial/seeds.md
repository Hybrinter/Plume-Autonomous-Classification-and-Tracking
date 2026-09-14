# sim.trial.seeds

**Source:** `packages/sim/src/sim/trial/seeds.py`
**Kind:** module

## Purpose

The seeds module spawns one numpy Generator per trial from a master seed.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `spawn_trial_rngs` | function | `SeedSequence(master_seed).spawn(n_trials)` |

## Inputs and outputs

**`spawn_trial_rngs(master_seed, n_trials) -> tuple[Generator, ...]`**

- Inputs: integer master seed, positive trial count.
- Output: one Generator per trial.

## Behavior

1. `numpy.random.SeedSequence` splits the master seed.
2. Each child seed constructs `default_rng`.

## Errors and faults

`ValueError` when `n_trials` is not positive.

## Messages

None.

## Configuration

None.

## Constraints

The streams do not alias `sim.scene.plume.build_frames` seeds.

## Related documents

- [`sim.trial`](../trial.md)
