"""Per-trial numpy Generators from a master seed. Not a world axis."""

from __future__ import annotations

import numpy as np


def spawn_trial_rngs(master_seed: int, n_trials: int) -> tuple[np.random.Generator, ...]:
    """Spawn one independent Generator per trial.

    Args:
        master_seed: Entropy for numpy SeedSequence.
        n_trials: Number of child streams. Must be positive.

    Returns:
        Tuple of Generators, index-aligned with trial_id 0 .. n_trials-1.

    Raises:
        ValueError: n_trials is not positive.
    """
    if n_trials <= 0:
        raise ValueError(f"n_trials must be positive, got {n_trials}")
    seq = np.random.SeedSequence(master_seed)
    children = seq.spawn(n_trials)
    return tuple(np.random.default_rng(child) for child in children)
